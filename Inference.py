#!/usr/bin/env python
# coding: utf-8

# In[13]:


parent_path='../Data/New_Data_15-06-2020/'
import parmap
import pandas as pd
from tqdm import tqdm,tqdm_notebook
from models.tokenization import *
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from models.model_architecture import *
from models.train_eval import *
from models.model_utils import *
from models.tokenization import *


# In[2]:


whatsapp_data=pd.read_csv(parent_path+'Data_text_spam_removed_v03_preds.csv')
# temp=whatsapp_data[['group_id_anonymized','phone_num_anonymized','message_text','timestamp']]
# duplicateDFRow = temp[temp.duplicated()]
# whatsapp_data=whatsapp_data.drop(list(duplicateDFRow.index))



params={'model_path':'xlm-roberta-base',
        'preprocess_doc':False,
        'max_length':256,
        'batch_size':512,
        'hidden_size':128,
        'weights':[1.0,3.0],
        'seq_model':'lstm',
        'data_path':parent_path+'Fearspeech_data_final.pkl',
        'max_sentences_per_doc':5,
        'transformer_type':'normal_transformer',
        'take_tokens_from':'both',
        'device':'cuda',
        'learning_rate':2e-5,
        'epsilon':1e-8,
        'random_seed':2020,
        'epochs':10,
        'max_memory':0.7,
        'freeze_bert':False
       }


# In[36]:


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


def inference_phase(params,temp,fold):
    X_0 = np.array(list(temp['preprocessed']),dtype='object')
    y_0 = np.array([0]*len(temp))
    tokenizer = AutoTokenizer.from_pretrained(params['model_path'])
    if torch.cuda.is_available() and params['device']=='cuda':    
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        deviceID = get_gpu(params)
        torch.cuda.set_device(deviceID[0])
    else:
        print('Since you dont want to use GPU, using the CPU instead.')
        device = torch.device("cpu")
    params['load_saved']=True
    model=select_transformer_model(params['transformer_type'],params['model_path'],params,fold)
    model.cuda()
    model.eval()  
        
    logits_all_total=[]    
    for i in tqdm(range(0,len(X_0),10000)):  
        X_test,X_test_mask= encode_sent(X_0[i:i+10000],params,tokenizer)
        test_dataloader=return_dataloader_sent(X_test,X_test_mask,y_0[i:i+10000],params,is_train=True)
        #params['model_path']='Saved/'+params['model_path']+'_'+str(fold)
        
          
        logits_all=[]
        # Evaluate data for one epoch
        for batch in tqdm(test_dataloader,total=len(test_dataloader)):
            # Add batch to GPU
            b_input_ids = batch[0].to(device)
            b_att_masks = batch[1].to(device)
            b_token_types = batch[2].to(device)
            b_labels = batch[3].to(device)
            # Telling the model not to compute or store gradients,saving memory and
            # speeding up validation
            with torch.no_grad():        
                outputs = model(input_ids=b_input_ids, attention_mask=b_att_masks, token_type_ids=b_token_types)

            logits = outputs
            logits = logits.detach().cpu().numpy()


            logits_all+=list(logits)

        logits_all_final=[]
        for logits in logits_all:
            logits_all_final.append(softmax(logits)[1])
        
        logits_all_total+=logits_all_final
        
        
    temp['fold_'+str(fold)]=logits_all_total
    del model
    return temp


# In[42]:

import sys

fold = int(sys.argv[1])
whatsapp_data=inference_phase(params,whatsapp_data,fold)
whatsapp_data.to_csv(parent_path+'Data_text_spam_removed_v03_preds.csv',index=False)




