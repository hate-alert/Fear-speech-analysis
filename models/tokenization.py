import torch
from transformers import AutoTokenizer
import math
from tqdm import tqdm 
import sys
sys.path.append("..") 
from utils.preprocess import preprocess_sent,preprocess_doc
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from langdetect import DetectorFactory
DetectorFactory.seed = 0
from langdetect import detect
from laserembeddings import Laser
from utils.emoji_extract import *

from sklearn.feature_extraction.text import CountVectorizer
from empath import Empath
import numpy as np

def encode_documents(documents,params,tokenizer):
    
    max_input_length=params['max_length']
    
    
    tokenized_documents = []
    for doc in documents:
        try:
            temp=tokenizer.tokenize(document)
        except AttributeError:
            temp=tokenizer.tokenize("dummy")
        tokenized_documents.append(temp)
   
    max_sequences_per_document = math.ceil(max(len(x)/(max_input_length-2) for x in tokenized_documents))
    #assert max_sequences_per_document <= 20, "Your document is to large, arbitrary size when writing"
    print("max sentences in documents",max_sequences_per_document)
    max_sequences_per_document=params['max_sentences_per_doc']
    if(params['model_path']=='xlm-roberta-base'):
        output= torch.zeros(size=(len(documents), max_sequences_per_document, 2, max_input_length),dtype=torch.long)
        output_2=torch.ones(size=(len(documents), max_sequences_per_document, 1, max_input_length),dtype=torch.long)
        output=torch.cat((output_2,output),dim=2)
        start_token='<s>'
        end_token='</s>'
        pad_id=1
    elif(params['model_path']=='bert-base-multilingual-cased'):
        output = torch.zeros(size=(len(documents), max_sequences_per_document, 3, max_input_length),dtype=torch.long)
        start_token='[CLS]'
        end_token='[SEP]'
        pad_id=0
    document_seq_lengths = [] #number of 
    
   
    
    
    for doc_index, tokenized_document in tqdm(enumerate(tokenized_documents)):
        max_seq_index = 0
        for seq_index, i in enumerate(range(0, len(tokenized_document), (max_input_length-2))):
            if(seq_index >= max_sequences_per_document):
                continue
            raw_tokens = tokenized_document[i:i+(max_input_length-2)]
            tokens = []
            input_type_ids = []
            tokens.append(start_token)
            input_type_ids.append(0)
            for token in raw_tokens:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append(end_token)
            input_type_ids.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_masks = [1] * len(input_ids)

            while len(input_ids) < max_input_length:
                input_ids.append(pad_id)
                input_type_ids.append(0)
                attention_masks.append(0)

            assert len(input_ids) == max_input_length and len(attention_masks) == max_input_length and len(input_type_ids) == max_input_length

            #we are ready to rumble
            output[doc_index][seq_index] = torch.cat((torch.LongTensor(input_ids).unsqueeze(0),
                                                    torch.LongTensor(input_type_ids).unsqueeze(0),
                                                    torch.LongTensor(attention_masks).unsqueeze(0)),
                                                          dim=0)
            
            max_seq_index = seq_index
        document_seq_lengths.append(max_seq_index+1)
    return output



def encode_documents_laser(documents,params,tokenizer=None):
    max_input_length=params['max_length']
    laser = Laser()
    output = torch.zeros(size=(len(documents), params['max_sentences_per_doc'], 3, 1024),dtype=torch.float)
    for doc_index, tokenized_document in tqdm(enumerate(documents)):
        lang_list=[]
        
        for ele in tokenized_document:
            try:
                lang_list.append(detect(ele))
            except:
                lang_list.append('en')
        
        embeddings = laser.embed_sentences(tokenized_document,lang=lang_list)  # lang is only used for tokenization
        for seq_index,embed in enumerate(embeddings):
            if(seq_index >= params['max_sentences_per_doc']):
                continue
            output[doc_index][seq_index][0]=torch.FloatTensor(embed)

    return output



def encode_documents_sent(documents,labels,params,tokenizer=None):
    max_input_length=params['max_length']
    
    if(params['model_path']=='xlm-roberta-base'):
        start_token='<s>'
        end_token='</s>'
        pad_id=1
    elif(params['model_path']=='bert-base-multilingual-cased'):
        start_token='[CLS]'
        end_token='[SEP]'
        pad_id=0
    
    inputs_ids_sent=[]
    att_masks_sent=[]
    labels_sent=[]
    print(documents[0:5])
    for tokenized_document, label in tqdm(zip(documents,labels),total=len(documents)):
        
        for sent in tokenized_document:
            
            tokenized_sent = tokenizer.tokenize(sent)
            tokens = []
            input_type_ids = []
            tokens.append(start_token)
            count=0
            for token in tokenized_sent:
                if(count >= max_input_length-2):
                    break
                count+=1
                tokens.append(token)    
            tokens.append(end_token)
            
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_masks = [1] * len(input_ids)
            
            while len(input_ids) < max_input_length:
                input_ids.append(pad_id)
                attention_masks.append(0)
            print(len(input_ids))
            assert len(input_ids) == max_input_length and len(attention_masks) == max_input_length 
            inputs_ids_sent.append(input_ids)
            att_masks_sent.append(attention_masks)
            labels_sent.append(label)

        
    return inputs_ids_sent, att_masks_sent,labels_sent




def dummy(string):
    return string

import json
import gensim
from gensim.test.utils import datapath
def get_feature_matrix(train,test):
  
    temp_file = datapath("lda_model_10")
    lda_model =  gensim.models.LdaModel.load(temp_file)
    dictionary = gensim.corpora.Dictionary.load("dictionary_lda.pkl")
    params={'remove_numbers':True,'remove_emoji':False,'remove_stop_words':True,'tokenize':True}
        
    imp_emoji=['🚩','🏹','⛳','🔱','🕉','🐚','👿','👺','🐖','🐷','😈','👹']

    with open('emoji_dict_fs.json') as json_file:
        fs_data = json.load(json_file)
    
    with open('emoji_dict_nfs.json') as json_file:
        nfs_data = json.load(json_file)
    
    
    
    lexicon=Empath()
    categories=["hate", "crime", "aggression", "suffering", "fight", "war", "weapon","negative_emotion","giving", "achievement", "fun"]
    
    
    
    
    emoji_list_train=[]
    empath_list_train=[]
    topic_list_train=[]
    
    emoji_list_test=[]
    empath_list_test=[]
    topic_list_test=[]
    
    for index,row in tqdm(train.iterrows(),total=len(train)):
        #emoji_list_train.append(extract_emojis(row['message_text']))
        
        
        #### topic vectors
        token_list=preprocess_sent(row['message_text'],params)
        vector=np.zeros(10)
        topic_preds=lda_model[dictionary.doc2bow(token_list)]
        for topic in topic_preds[0]:
            vector[topic[0]]=topic[1]
        topic_list_train.append(list(vector))
        
        
        
        temp_emoji_list=extract_emojis(row['message_text'])
        temp_val=[]
        for emoji in imp_emoji:
            if(emoji in temp_emoji_list):
                if(row['one_fear_speech']==1):
                    try:
                        temp_val.append(fs_data[emoji])
                    except KeyError:
                        temp_val.append(0)
                else:
                    try:
                        temp_val.append(nfs_data[emoji])
                    except KeyError:
                        temp_val.append(0)
        
            else:
                temp_val.append(0)
                
        emoji_list_train.append(temp_val)
                
        dict_lexicon=lexicon.analyze(row['translated'], normalize=True)
        try:
            temp=dict_lexicon.values()
        except AttributeError:
            dict_lexicon=lexicon.analyze("sample text", normalize=True)
        temp_list=[]
        for key in categories:
            temp_list.append(dict_lexicon[key])
        empath_list_train.append(temp_list)
    
    
    for index,row in tqdm(test.iterrows(),total=len(test)):
        #emoji_list_test.append(extract_emojis(row['message_text']))
        token_list=preprocess_sent(row['message_text'],params)
        vector=np.zeros(10)
        topic_preds=lda_model[dictionary.doc2bow(token_list)]
        for topic in topic_preds[0]:
            vector[topic[0]]=topic[1]
        topic_list_test.append(list(vector))
        
        temp_emoji_list=extract_emojis(row['message_text'])
        temp_val=[]
        for emoji in imp_emoji:
            if(emoji in temp_emoji_list):
                if(row['one_fear_speech']==1):
                    try:
                        temp_val.append(fs_data[emoji])
                    except KeyError:
                        temp_val.append(0)
                else:
                    try:
                        temp_val.append(nfs_data[emoji])
                    except KeyError:
                        temp_val.append(0)
        
            else:
                temp_val.append(0)
        
        emoji_list_test.append(temp_val)

        
        dict_lexicon=lexicon.analyze(row['translated'], normalize=True)
        try:
            temp=dict_lexicon.values()
        except AttributeError:
            dict_lexicon=lexicon.analyze("sample text", normalize=True)
        temp_list=[]
        for key in categories:
            temp_list.append(dict_lexicon[key])
        empath_list_test.append(temp_list)
    
    
        
    
    #vectorizer = CountVectorizer(stop_words=None,lowercase=False,tokenizer=dummy,preprocessor=None)
    
    
    #X_train=vectorizer.fit_transform(emoji_list_train)
    X_empath_train=np.array(empath_list_train)
    X_emoji_train=np.array(emoji_list_train)
    X_topic_train=np.array(topic_list_train)
    X_train_final=np.concatenate((X_emoji_train,X_empath_train,X_topic_train),axis=1)
    
    
    #X_test=vectorizer.transform(emoji_list_test)
    
    X_empath_test=np.array(empath_list_test)
    X_emoji_test=np.array(emoji_list_test)
    X_topic_test=np.array(topic_list_test)
    X_test_final=np.concatenate((X_emoji_test,X_empath_test,X_topic_test),axis=1)
    
    return X_train_final,X_test_final 



def encode_sent(documents,params,tokenizer=None):
    max_input_length=params['max_length']
    
    tokenized_documents = []
    for doc in documents:
        try:
            temp=tokenizer.tokenize(doc)
        except AttributeError:
            temp=tokenizer.tokenize("dummy")
        tokenized_documents.append(temp)
   
    #tokenized_documents = [tokenizer.tokenize(document) for document in documents]
    
    if(params['model_path']=='xlm-roberta-base'):
        start_token='<s>'
        end_token='</s>'
        pad_id=1
    elif(params['model_path']=='bert-base-multilingual-cased'):
        start_token='[CLS]'
        end_token='[SEP]'
        pad_id=0
    
    inputs_ids_sent=[]
    att_masks_sent=[]
    
    for doc in tokenized_documents:
        tokens=[]
        tokens.append(start_token)
        if(params['take_tokens_from']=='first'):
            count=0
            for token in doc:
                if(count >= max_input_length-2):
                    break
                count+=1
                tokens.append(token) 
        if(params['take_tokens_from']=='last'):
            count=0
            for token in doc[-(max_input_length-2):]:
                if(count >= max_input_length-2):
                    break
                count+=1
                tokens.append(token) 
        if(params['take_tokens_from']=='both'):
            max_input_length_sub=int(max_input_length/2)
            count=0
            for token in doc:
                if(count >= max_input_length_sub-2):
                    break
                count+=1
                tokens.append(token) 
            
            tokens.append(end_token)
            count=0
            
            for token in doc[-(max_input_length_sub-2):]:
                if(count >= max_input_length_sub-3):
                    break
                count+=1
                tokens.append(token) 
        tokens.append(end_token)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_masks = [1] * len(input_ids)
            
        while len(input_ids) < max_input_length:
            input_ids.append(pad_id)
            attention_masks.append(0)
        
        assert len(input_ids) == int(max_input_length) and len(attention_masks) == int(max_input_length) 
        inputs_ids_sent.append(input_ids)
        att_masks_sent.append(attention_masks)

        
    return inputs_ids_sent, att_masks_sent



def return_dataloader(document_tensors,labels,params,is_train=False):
    labels = torch.tensor(labels,dtype=torch.long)
    
    token_ids= document_tensors[:,:,0,:].squeeze()    
    attentions_mask= document_tensors[:,:,1,:].squeeze()
    tokens_types= document_tensors[:,:,2,:].squeeze()
    
    data=TensorDataset(token_ids,attentions_mask,tokens_types,labels)       
    if(is_train==False):
        sampler = SequentialSampler(data)
    else:
        sampler = RandomSampler(data)
        
    dataloader = DataLoader(data, sampler=sampler, batch_size=params['batch_size'])
    return dataloader


def return_dataloader_sent(inputs_ids, att_masks,labels,params,is_train=False):
    labels = torch.tensor(labels,dtype=torch.long)
    
    token_ids= torch.tensor(inputs_ids,dtype=torch.long)    
    attentions_mask= torch.tensor(att_masks,dtype=torch.long)
    tokens_types = torch.zeros(size=(token_ids.shape[0],token_ids.shape[1]),dtype=torch.long)

    data=TensorDataset(token_ids,attentions_mask,tokens_types,labels)       
    if(is_train==False):
        sampler = SequentialSampler(data)
    else:
        sampler = RandomSampler(data)
        
    dataloader = DataLoader(data, sampler=sampler, batch_size=params['batch_size'])
    return dataloader



def return_dataloader_sent_extra(inputs_ids, att_masks,extra_features,labels,params,is_train=False):
    labels = torch.tensor(labels,dtype=torch.long)
    token_ids= torch.tensor(inputs_ids,dtype=torch.long)  
    extra_features = torch.tensor(extra_features,dtype=torch.float)  
    attentions_mask= torch.tensor(att_masks,dtype=torch.long)
    tokens_types = torch.zeros(size=(token_ids.shape[0],token_ids.shape[1]),dtype=torch.long)

    data=TensorDataset(token_ids,attentions_mask,extra_features,tokens_types,labels)       
    if(is_train==False):
        sampler = SequentialSampler(data)
    else:
        sampler = RandomSampler(data)
        
    dataloader = DataLoader(data, sampler=sampler, batch_size=params['batch_size'])
    return dataloader
