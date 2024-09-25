import torch
import torch.nn as nn
import torch.nn.functional as F
import utils as U
import pandas as pd
import random
import numpy as np
import wandb
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch


class CustomDataset(Dataset):
    def __init__(self, emb_ret, pmaps):
        self.emb_ret = emb_ret
        self.pmaps = pmaps

    def __len__(self):
        return len(self.emb_ret)

    def __getitem__(self, idx):
        data = self.emb_ret.iloc[idx]
        emb = torch.tensor(eval(data['embedding']), dtype=torch.float32)
        ret = torch.tensor(data['RET'], dtype=torch.float32)
        permno = data['PERMNO']
        pno = self.pmaps[permno]
        return emb, ret, pno

config=U.load_yaml('./config.yaml')
wandb.login(key=config['apikeys']['wandb'])
wandb.init(
    project="dapm",
    config={}
)

class Model(nn.Module):
    def __init__(self, config,expname,best=False):  
        super().__init__()
        mconfig=config['model']
        self.downsample=nn.Linear(mconfig['d_emb'],mconfig['d_model'])
        self.p_embs=nn.Embedding(mconfig['dsize'],mconfig['d_model'])
        self.net=nn.Sequential(
            nn.BatchNorm1d(mconfig['d_model']*2),
            nn.ReLU(),
            nn.Dropout(mconfig['dropout']),
            nn.Linear(mconfig['d_model']*2,mconfig['d_model']),
            nn.BatchNorm1d(mconfig['d_model']),
            nn.ReLU(),
            nn.Dropout(mconfig['dropout']),
            nn.Linear(mconfig['d_model'],mconfig['d_model']),
            nn.BatchNorm1d(mconfig['d_model']),
            nn.ReLU(),
            nn.Dropout(mconfig['dropout']),
            nn.Linear(mconfig['d_model'],1)
        )
        self.criterion=nn.MSELoss()
        self.bs=mconfig['batchsize']
        self.epochs=mconfig['epochs']

        ddir=config['dirs']['data']
        embs=pd.read_csv(U.pjoin(ddir,'daily_emb_dapm.csv'))
        ret=pd.read_csv(U.pjoin(ddir,'daily_ret_dapm.csv'))
        self.emb_ret=ret.merge(embs,on=['date'],how='inner').dropna()

        self.train_df=self.emb_ret[self.emb_ret['date']<'2022-07-01']
        self.test_df=self.emb_ret[self.emb_ret['date']>='2022-10-01']
        emb_ret_train=self.emb_ret[self.emb_ret['date']<'2022-10-01']
        self.dev_df=emb_ret_train[emb_ret_train['date']>='2022-07-01']

        libdir=U.pjoin(ddir,'library')
        self.index=pd.read_csv(U.pjoin(libdir,'index.csv'))
        self.pmaps={}
        for i, p in enumerate(self.emb_ret['PERMNO'].unique()):
            self.pmaps[p]=i

        self.train_loader = DataLoader(CustomDataset(self.train_df, self.pmaps), batch_size=self.bs, shuffle=True)
        self.eval_loader = DataLoader(CustomDataset(self.dev_df, self.pmaps), batch_size=self.bs*2, shuffle=True)
        self.test_loader = DataLoader(CustomDataset(self.test_df, self.pmaps), batch_size=self.bs*2, shuffle=False)

        self.ckptdir=U.pjoin(config['dirs']['ckpt'],expname)
        U.makedirs(self.ckptdir)
        self.current_epoch=0
        self.current_batch=0
        self.best_eval_loss=np.inf
        self.best_eval_epoch=0
        self.early_stop=mconfig['earlystop']
        self.cuda()
        self.optim=torch.optim.Adam(self.parameters(),lr=mconfig['lr'])
        self.load(best=best) 

    def forward(self, x, pnos):
        pes=self.p_embs(pnos).squeeze()
        x=self.downsample(x)
        x=torch.cat([x,pes],dim=1)
        x=F.relu(x)
        x=self.net(x)
        return x.squeeze()
    
    def eval(self, test=True):
        losses=0
        bar=tqdm(self.test_loader if test else self.eval_loader)
        for batch, (x, ret, pno) in enumerate(bar):
            x, ret = x.cuda(), ret.cuda()
            pno = pno.cuda()  # Assuming pno is a tensor of appropriate type and shape
            with torch.no_grad():
                pred=self(x,pno)
            loss=self.criterion(pred,ret)
            losses+=loss
            bar.set_description('Loss {:.5f}'.format(loss))
            wandb.log({"test loss": loss.item()}) if test else wandb.log({"eval loss": loss.item()})
        loss=losses/len(bar)
        return loss

    def trainloop(self):
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            self.train()
            bar = tqdm(self.train_loader)
            for batch, (x, ret, pno) in enumerate(bar):
                self.current_batch = batch
                x, ret = x.cuda(), ret.cuda()
                pno = pno.cuda()  # Assuming pno is a tensor of appropriate type and shape
                self.optim.zero_grad()
                pred = self(x, pno)
                loss = self.criterion(pred, ret)
                loss.backward()
                self.optim.step()
                bar.set_description('Epoch {} Batch {} Loss {:.5f}'.format(epoch, batch, loss.item()))
                wandb.log({"loss": loss.item()})
                if batch % 100 == 0: self.save()
            loss = self.eval(test=False)
            print('Epoch {} Dev Loss {}'.format(epoch, loss.item()))
            if loss<self.best_eval_loss:
                print(f'New best model at epoch {epoch}, improved from {self.best_eval_loss} to {loss}')
                self.best_eval_loss=loss
                self.best_eval_epoch=epoch
            if epoch-self.best_eval_epoch>self.early_stop:
                print('Early stopping at epoch {}'.format(epoch))
                break
            self.save(epoch)
        loss = self.eval(test=True)
        print('Test Loss {}'.format(loss))
    
    def save(self,epoch=None):
        state={
            'current_epoch':self.current_epoch, 
            'current_batch':self.current_batch,
            'state_dict':self.state_dict(),
            'optim':self.optim.state_dict(),
            'best_eval_loss': self.best_eval_loss,
            'best_eval_epoch': self.best_eval_epoch,
        }
        torch.save(state,U.pjoin(self.ckptdir,'model.pt'))
        if epoch is not None:
            torch.save(state,U.pjoin(self.ckptdir,f'model-{epoch}.pt'))

    def load(self,epoch=None,best=False):
        model='model.pt' if epoch is None else f'model-{epoch}.pt'
        if U.pexists(U.pjoin(self.ckptdir,model)):
            state=torch.load(U.pjoin(self.ckptdir,model))
            self.best_eval_loss=state['best_eval_loss']
            self.best_eval_epoch=state['best_eval_epoch']
            if best:
                best_state=torch.load(U.pjoin(self.ckptdir,'model.pt'))
                self.current_epoch=best_state['current_epoch']
                self.current_batch=best_state['current_batch']
                self.load_state_dict(best_state['state_dict'])
                self.optim.load_state_dict(best_state['optim'])
                print('Loaded best model from epoch {}'.format(self.best_eval_epoch))
            else:
                self.current_epoch=state['current_epoch']
                self.current_batch=state['current_batch']
                self.load_state_dict(state['state_dict'])
                self.optim.load_state_dict(state['optim'])
                print('Loaded model from epoch {} batch {}'.format(self.current_epoch, self.current_batch))

