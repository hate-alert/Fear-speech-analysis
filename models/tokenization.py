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



def encode_documents(documents,params,tokenizer):
    
    max_input_length=params['max_length']
    tokenized_documents = [tokenizer.tokenize(document) for document in documents]
    print(tokenized_documents[0][0:5])
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
    elif(params['model_path']=='bert-base-multilingual-cased'):
        output = torch.zeros(size=(len(documents), max_sequences_per_document, 3, max_input_length),dtype=torch.long)
        start_token='[CLS]'
        end_token='[SEP]'
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
                input_ids.append(0)
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
