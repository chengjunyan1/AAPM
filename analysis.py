import prompt as P
from openai import OpenAI
import pandas as pd
import os
import utils as U
from memdb import MemDB
from tqdm import tqdm
import sys

lib_dir='./Data/library/'

index=pd.read_csv(os.path.join(lib_dir, "index.csv"))

config=U.load_yaml("./config.yaml")
db=MemDB('default',config)

client = OpenAI(api_key=config['apikeys']['openai'])

def process(macro,emb,text,path,dt,max_steps=5,k_choices=3,reason_mode=3): 
    exids=[path]
    input='News: \n\n'+str(dt)+': '+str(text)
    messages=[]
    analysis=''
    if reason_mode==3: # do an initial analysis without relavant info
        analysis,messages=dialog(input,messages,macro=macro)
        if analysis=='SKIP': 
            texts=input
            emb=db.emb(input)
            return emb,texts,messages
    for i in range(max_steps):
        res,_=db.query(emb=emb,k=k_choices,filter=[exids],ret_emb=True,pad=True)
        docs=res['documents'][0]
        meta=res['metadatas'][0]
        rels=''
        for j in range(len(docs)):
            exids.append(res['ids'][0][j])
            if meta[j]['Type']=='News':
                head='News at '+meta[j]['Datetime']
            elif meta[j]['Type']=='Excert':
                head='Excerpt from '+meta[j]['Source']
            else: raise ValueError('Unknown type')
            rels+=head+': '+docs[j]+' \n\n'
        if reason_mode==3:
            try:
                analysis,messages=dialog(rels,messages,i==max_steps-1) 
            except:
                return None,None,None
        elif reason_mode==2 or (reason_mode==1 and i==max_steps-1):
            try:
                texts=input
                if analysis!='': texts+=' \n\nCurrent Analysis: \n\n'+analysis
                texts+=' \n\nRelevant information: \n\n'+rels
                analysis=reason(texts) # aug text
            except:
                return None,None,None
        texts=input+' \n\nAnalysis: \n\n'+analysis
        emb=db.emb(texts)
    
    macro=P.macro_update.format(date=dt,macro=macro,news=texts)
    return emb,texts,messages,macro

def reason(news):
    input_text=P.news_reason.format(news=news)  
    response = client.chat.completions.create(
        model=config['model']['news'],
        messages=[
            {"role": "system", "content": input_text}
        ]
    )
    return response.choices[0].message.content


def dialog(inputs,messages,last=False,macro=None):
    if messages==[]:
        if macro is not None:
            input_text=P.news_dialog_begin.format(inputs=inputs,macro=macro)
        else:
            input_text=P.news_dialog_begin.format(inputs=inputs)
        tools= [
            {
                "type": "function",
                "function": {
                    "name": "skip",
                    "description": "This news is completely not relevant to economics, investments, or finance now or future, skip it. Remember that call it only if there is no potential helpful information for investment at all.",
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "notskip",
                    "description": "This news contains potential helpful information for investment now or future.",
                }
            }
        ]
    else:
        prompt=P.news_dialog_end if last else P.news_dialog_cont
        input_text=prompt.format(inputs=inputs)
        tools=None
    messages.append({"role":'system',"content":input_text})
    if tools:
        response = client.chat.completions.create(
            model=config['model']['news'],
            messages=messages,
            tools=tools
        )
    else: 
        response = client.chat.completions.create(
            model=config['model']['news'],
            messages=messages
        )
    msg = response.choices[0].message
    ret = msg.content or "No response content."  # Ensure content is not None
    if msg.tool_calls:
        if msg.tool_calls[0].function.name=='skip':
            ret='SKIP'
        messages.append({"role": "function", "tool_call_id": msg.tool_calls[0].id, "name": msg.tool_calls[0].function.name, "content": ret})
    else:
        messages.append({"role": "assistant", "content": ret})
    return ret,messages


if __name__ == '__main__':
    reason_mode=3
    tail='_3' if reason_mode==3 else ''
    ne_dir=U.pjoin(lib_dir, "news_embedding",'bgel')
    stdir=U.pjoin(lib_dir,"news_augmented",'text'+tail)
    sedir=U.pjoin(lib_dir,"news_augmented",'emb'+tail)
    smdir=U.pjoin(lib_dir,"news_augmented",'msg'+tail)
    mcdir=U.pjoin(lib_dir,"news_augmented",'macro'+tail)
    U.makedirs(stdir)
    U.makedirs(sedir)
    U.makedirs(smdir)
    U.makedirs(mcdir)
    if len(sys.argv)<2: 
        start=0
        end=len(index)
    else:
        id=sys.argv[1] # 0 1 2 3 4
        chunksize=5000
        start=int(id)*chunksize
        end=min(len(index),(int(id)+1)*chunksize)
    macro=P.macro_init
    for i in tqdm(range(start,end),colour='green',desc='Processing'):
        path=index.iloc[i]['path'] 
        if U.pexists(U.pjoin(sedir,path.replace('/','_')+'.json')): continue 
        na=U.load_json(U.pjoin(lib_dir,"news_analysis",path.replace('/','_')+'.json'))
        meta={
            'Tickers':', '.join(na['Tickers']),
            'Topics':', '.join(na['Topics']),
            'Datetime':index.iloc[i,0],
            'Type': 'News',
        }
        text=na['Content']
        emb=U.load_json(U.pjoin(ne_dir,path.replace('/','_')+'.json'))['content']
        emb,text,msgs,macro=process(macro,emb,text,path,index.iloc[i,0],reason_mode=reason_mode)
        fname=path.replace('/','_')+'.json'
        db.add(text, meta, emb, path, overwrite=True) 
        U.save_json(U.pjoin(stdir,fname),{'content':text})
        U.save_json(U.pjoin(sedir,fname),{'content':emb})
        U.save_json(U.pjoin(smdir,fname),{'content':msgs})
        U.save_json(U.pjoin(mcdir,fname),{'content':macro})
