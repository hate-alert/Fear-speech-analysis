import torch
from transformers import AutoTokenizer
import math
from tqdm import tqdm 
import sys
sys.path.append("..") 
from utils.preprocess import preprocess_sent
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def encode_documents(documents,params,tokenizer):
    
    max_input_length=params['max_length']
    tokenized_documents = [tokenizer.tokenize(document) for document in documents]
    print(tokenized_documents[0][0:5])
    max_sequences_per_document = math.ceil(max(len(x)/(max_input_length-2) for x in tokenized_documents))
    #assert max_sequences_per_document <= 20, "Your document is to large, arbitrary size when writing"
    print("max sentences in documents",max_sequences_per_document)
    max_sequences_per_document=params['max_sentences_per_doc']
    if(params['model_path']=='xlm-roberta-base'):
        output = torch.ones(size=(len(documents), max_sequences_per_document, 3, max_input_length),dtype=torch.long)
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
            output[doc_index][seq_index] = torch.cat((  torch.LongTensor(input_ids).unsqueeze(0),
                                                        torch.LongTensor(input_type_ids).unsqueeze(0),
                                                        torch.LongTensor(attention_masks).unsqueeze(0)),
                                                          dim=0)
            
            max_seq_index = seq_index
        document_seq_lengths.append(max_seq_index+1)
    return output, torch.LongTensor(document_seq_lengths)



def return_dataloader(document_tensors,labels,params,is_train=False):
    labels = torch.tensor(labels,dtype=torch.long)
    
    data = TensorDataset(document_tensors,labels)
    if(is_train==False):
        sampler = SequentialSampler(data)
    else:
        sampler = RandomSampler(data)
        
    dataloader = DataLoader(data, sampler=sampler, batch_size=params['batch_size'])
    return dataloader
