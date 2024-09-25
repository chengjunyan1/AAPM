import chromadb
import uuid
import utils as U
from chromadb.utils import embedding_functions
from FlagEmbedding import FlagModel
import numpy as np


class MemDB:
    def __init__(self, dbname, config):
        self.config = config
        sdir=U.pjoin(config['dirs']['ckpt'],config['name'])
        U.makedirs(sdir)
        self.dbname=dbname
        self.client = chromadb.PersistentClient(path=sdir)
        self.collection = self.client.create_collection(name=dbname, get_or_create=True)

        if self.config['model']['emb_type'] == 'openai':
            self.ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.config['apikeys']['openai'],
                model_name=self.config['model']['embed']
            )
        elif self.config['model']['emb_type'] == 'cohere':
            self.ef = embedding_functions.CohereEmbeddingFunction(
                api_key=self.config['apikeys']['cohere'],  
                model_name=self.config['model']['embed']
            )
        elif self.config['model']['emb_type'] == 'bge':
            self.ef = FlagModel(config['model']['embed'],
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                  use_fp16=False)

    def emb(self,doc,query=False):
        if self.config['model']['emb_type'] == 'openai':
            emb=self.ef(texts=doc)
        elif self.config['model']['emb_type'] == 'cohere':
            if query:
                emb=self.ef(texts=doc,input_type='search_query')
            else:
                emb=self.ef(texts=doc,input_type='search_document') # search_query
        elif self.config['model']['emb_type'] == 'bge':
            if query:
                emb=self.ef.encode_queries(doc).tolist()
            else:
                emb=self.ef.encode(doc).tolist()
        return emb
    
    def add(self, doc, meta, emb=None, path=None, overwrite=True): # add one doc one time, meta: at least type
        if emb is None:
            if isinstance(doc, str): doc=[doc] 
            emb=self.emb(doc)
        else: 
            if isinstance(emb, str): emb=[emb]
        if path is None:
            path=[str(uuid.uuid4()) for i in range(len(doc))]
        else:
            if isinstance(path, str): path=[path]
        if isinstance(meta, dict): meta=[meta]
        for m in meta:
            if m['Type']=='News':
                assert 'Datetime' in m
            elif m['Type']=='Excert':
                assert 'Source' in m
        method=self.collection.upsert if overwrite else self.collection.add
        method(embeddings=emb, documents=doc, metadatas=meta, ids=path)

    def query(self, query=None, emb=None, k=10, filter=None, ret_emb=False, pad=False): # filter: e.g. [[id1,id2,...],[...]]
        if emb is None:
            if isinstance(query, str): query=[query]
            emb=self.emb(query,query=True)
        if filter is not None:
            orignal_k=k
            k+=max([len(i) for i in filter])
        include=["metadatas", "documents", "distances"]
        if ret_emb: include.append("embeddings")
        res=self.collection.query(query_embeddings=emb,n_results=k,include=include)
        if filter is not None:
            new_res={}
            for key in res.keys():
                new_res[key]=[[] for _ in range(len(res[key]))] if res[key] is not None else None
            for i in range(len(filter)): # iterate each query
                count=0
                for j in range(len(res['ids'][i])): # iterate each id
                    id=res['ids'][i][j]
                    if id in filter[i]: continue
                    if count>=orignal_k: break
                    for key in res.keys():
                        if res[key] is not None:
                            new_res[key][i].append(res[key][i][j])
                    count+=1
            res=new_res 
            k=orignal_k
        if pad:
            masks=[]
            for i in range(len(res['ids'])):
                mask=np.zeros(k)
                mask[:len(res['ids'][i])]=1
                masks.append(mask.astype(bool).tolist())
                for _ in range(k-len(res['ids'][i])):
                    res['ids'][i].append('')
                    res['documents'][i].append('')
                    res['distances'][i].append(1e8)
                    res['metadatas'][i].append({})
                    if ret_emb:
                        res['embeddings'][i].append(np.zeros_like(res['embeddings'][i][0]).tolist())
            return res,masks
        return res  
    
    def remove_ids(self,ids, chunk_size=4000):
        allids=self.collection.get()['ids']
        if isinstance(ids,str): ids=[ids]
        to_delete=[]
        for id in ids:
            if id in allids:
                to_delete.append(id)
        chunks=[to_delete[i:i + chunk_size] for i in range(0, len(to_delete), chunk_size)]
        for chunk in chunks:
            self.collection.delete(ids=chunk)

    def reset_collection(self): # delete current collection
        self.client.delete_collection(name=self.dbname)
        self.collection = self.client.create_collection(name=self.dbname)

# config=U.load_yaml('./config.yaml')
# db = MemDB('test', config)
