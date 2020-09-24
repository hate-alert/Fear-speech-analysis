#from transformers import BertForSequenceClassification,BertPreTrainedModel,RobertaPreTrainedModel,XLMRobertaForSequenceClassification
from transformers.modeling_bert import *
from transformers.modeling_roberta import *
from transformers.modeling_xlm_roberta import *
from torch import nn
import torch
from torch.nn import LSTM

def select_transformer_model(type_of_model,path,params):
    if(type_of_model=='lstm_transformer'):
        if (path=='bert-base-multilingual-cased'):
            model = DocumentBERTLSTM.from_pretrained(
            path, # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = 2,  
            params=params
            )
        elif (path=='xlm-roberta-base'):
            model = DocumentRobertaLSTM.from_pretrained(
            path, # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = 2, # The number of output labels--2 for binary classification             # You can increase this for multi-class tasks.   
            params=params
            )

    return model



class DocumentBERTLSTM(BertPreTrainedModel):
    """
    BERT output over document in LSTM
    """
    def __init__(self,config,params):
        super(DocumentBERTLSTM, self).__init__(config)
        self.bert = BertModel(config)
        print(params)
        self.num_labels = config.num_labels
        self.batch_size= params['batch_size']
        self.weights=params['weights']
        self.bert_batch_size=params['max_sentences_per_doc']
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.lstm = LSTM(config.hidden_size,config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(p=config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels),
            nn.Tanh()
        )

    #input_ids, token_type_ids, attention_masks
    def forward(self, document_batch, labels= None,device='cuda'):

        #contains all BERT sequences
        #bert should output a (batch_size, num_sequences, bert_hidden_size)
        bert_output = torch.zeros(size=(document_batch.shape[0],
                                              min(document_batch.shape[1],self.bert_batch_size),
                                              self.bert.config.hidden_size), dtype=torch.float, device=device)

        #only pass through bert_batch_size numbers of inputs into bert.
        #this means that we are possibly cutting off the last part of documents.

        for doc_id in range(document_batch.shape[0]):
            bert_output[doc_id][:self.bert_batch_size] = self.dropout(self.bert(document_batch[doc_id][:self.bert_batch_size,0],
                                            token_type_ids=document_batch[doc_id][:self.bert_batch_size,1],
                                            attention_mask=document_batch[doc_id][:self.bert_batch_size,2])[1])

        output, (_, _) = self.lstm(bert_output.permute(1,0,2))

        last_layer = output[-1]

        prediction = self.classifier(last_layer)

        assert prediction.shape[0] == document_batch.shape[0]
        
        
        if labels is not None:
            loss_funct = CrossEntropyLoss(weight=torch.tensor(self.weights).to(device))
            loss_logits =  loss_funct(prediction.view(-1, self.num_labels), labels.view(-1))
            loss= loss_logits
            output = [loss, output]

        return output
    
class DocumentRobertaLSTM(RobertaPreTrainedModel):
    """
    Roberta output over document in LSTM
    """
    def __init__(self,config,params):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta= RobertaModel(config)
        self.batch_size= params['batch_size']
        self.weights=params['weights']
        self.bert_batch_size=params['max_sentences_per_doc']
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.lstm = LSTM(config.hidden_size,config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(p=config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels),
            nn.Tanh()
        )
        self.init_weights()

    #input_ids, token_type_ids, attention_masks
    def forward(self, document_batch,labels= None,device='cuda'):

        #contains all BERT sequences
        #bert should output a (batch_size, num_sequences, bert_hidden_size)
        bert_output = torch.ones(size=(document_batch.shape[0],
                                              min(document_batch.shape[1],self.bert_batch_size),
                                              self.roberta.config.hidden_size), dtype=torch.float, device=device)

        #only pass through bert_batch_size numbers of inputs into bert.
        #this means that we are possibly cutting off the last part of documents.
        print(document_batch.shape)
        for doc_id in range(document_batch.shape[0]):
            temp=self.roberta(document_batch[doc_id][:self.bert_batch_size,0],
                                            token_type_ids=document_batch[doc_id][:self.bert_batch_size,1],
                                            attention_mask=document_batch[doc_id][:self.bert_batch_size,2])
            print(temp.shape)
            bert_output[doc_id][:self.bert_batch_size] = self.dropout(temp[1])

        output, (_, _) = self.lstm(bert_output.permute(1,0,2))

        last_layer = output[-1]

        prediction = self.classifier(last_layer)

        assert prediction.shape[0] == document_batch.shape[0]
        
        
        if labels is not None:
            loss_funct = CrossEntropyLoss(weight=torch.tensor(self.weights).to(device))
            loss_logits =  loss_funct(prediction.view(-1, self.num_labels), labels.view(-1))
            loss= loss_logits
            outputs = (loss,) + outputs

        return outputs
    
    
    