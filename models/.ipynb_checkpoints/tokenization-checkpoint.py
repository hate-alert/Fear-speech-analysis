import torch
from transformers import AutoTokenizer
import math
from tqdm import tqdm 
import sys
sys.path.append("..") 
from utils.preprocess import preprocess_sent

def encode_documents(documents,params):
    
    max_input_length=params['max_length']
    preprocess_params={'remove_numbers':False,'remove_emoji':True,'remove_stop_words':True,'tokenize':False}

    tokenizer = AutoTokenizer.from_pretrained(params['model_name'])
    tokenized_documents = [tokenizer.tokenize(preprocess_sent(document,preprocess_params)) for document in documents]
    max_sequences_per_document = math.ceil(max(len(x)/(max_input_length-2) for x in tokenized_documents))
    #assert max_sequences_per_document <= 20, "Your document is to large, arbitrary size when writing"
    output = torch.zeros(size=(len(documents), max_sequences_per_document, 3, max_input_length),dtype=torch.long)
    document_seq_lengths = [] #number of 
    
    for doc_index, tokenized_document in tqdm(enumerate(tokenized_documents)):
        max_seq_index = 0
        for seq_index, i in enumerate(range(0, len(tokenized_document), (max_input_length-2))):
            raw_tokens = tokenized_document[i:i+(max_input_length-2)]
            tokens = []
            input_type_ids = []

            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in raw_tokens:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
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

