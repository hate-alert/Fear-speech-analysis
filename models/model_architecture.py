#from transformers import BertForSequenceClassification,BertPreTrainedModel,RobertaPreTrainedModel,XLMRobertaForSequenceClassification
from transformers.modeling_bert import *
from transformers.modeling_roberta import *
from transformers.modeling_xlm_roberta import *
from torch import nn
import torch
from torch.nn import LSTM,GRU

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
    if(type_of_model=='birnn_laser'):
        model=BiRNN(params)

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
        self.lstm = LSTM(config.hidden_size,params['hidden_size'])
        self.classifier = nn.Sequential(
            nn.Dropout(p=config.hidden_dropout_prob),
            nn.Linear(params['hidden_size'], config.num_labels),
            nn.Tanh()
        )

    #input_ids, token_type_ids, attention_masks
    def forward(self, input_ids=None, attention_mask=None,token_type_ids=None, labels= None,device=None):

        #contains all BERT sequences

        bert_output=self.dropout(self.bert(input_ids.view(-1,input_ids.shape[2]),attention_mask.view(-1,input_ids.shape[2]),
                                           token_type_ids.view(-1,input_ids.shape[2]))[1])
        
        
        bert_output=bert_output.view(input_ids.shape[0],self.bert_batch_size,-1)
        output, (_, _) = self.lstm(bert_output.permute(1,0,2))

        last_layer = output[-1]

        prediction = self.classifier(last_layer)

        assert prediction.shape[0] == input_ids.shape[0]
        
        
        if labels is not None:
            loss_funct = CrossEntropyLoss(weight=torch.tensor(self.weights).to(device))
            loss_logits =  loss_funct(prediction.view(-1, self.num_labels), labels.view(-1))
            loss= loss_logits
            prediction = [loss, prediction]

        return prediction
    
    
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def unfreeze_bert_encoder_last_layers(self):
        for name, param in self.bert.named_parameters():
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
    def unfreeze_bert_encoder_pooler_layer(self):
        for name, param in self.bert.named_parameters():
            if "pooler" in name:
                param.requires_grad = True

    
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
        self.lstm = LSTM(config.hidden_size,params['hidden_size'])
        self.classifier = nn.Sequential(
            nn.Dropout(p=config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels),
            nn.Tanh()
        )
        self.init_weights()

    #input_ids, token_type_ids, attention_masks
    def forward(self, input_ids=None, attention_mask=None,token_type_ids=None, labels= None,device=None):

        #contains all BERT sequences
        bert_output=self.dropout(self.roberta(input_ids.view(-1,input_ids.shape[2]),attention_mask.view(-1,input_ids.shape[2]))[1])
        
        
        bert_output=bert_output.view(input_ids.shape[0],self.bert_batch_size,-1)
        
        output, (_, _) = self.lstm(bert_output.permute(1,0,2))

        last_layer = output[-1]

        prediction = self.classifier(last_layer)

        assert prediction.shape[0] == input_ids.shape[0]
        
        
        if labels is not None:
            loss_funct = CrossEntropyLoss(weight=torch.tensor(self.weights).to(device))
            loss_logits =  loss_funct(prediction.view(-1, self.num_labels), labels.view(-1))
            loss= loss_logits
            prediction = [loss, prediction]

        return prediction
    
    def freeze_bert_encoder(self):
        for param in self.roberta.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.roberta.parameters():
            param.requires_grad = True

    def unfreeze_bert_encoder_last_layers(self):
        for name, param in self.roberta.named_parameters():
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
    def unfreeze_bert_encoder_pooler_layer(self):
        for name, param in self.roberta.named_parameters():
            if "pooler" in name:
                param.requires_grad = True

                
class BiRNN(nn.Module):  
    def __init__(self,args):
        super(BiRNN, self).__init__()
        self.hidden_size = args['hidden_size']
        self.batch_size = args['batch_size']
        self.seq_model_name=args["seq_model"]
        self.weights =args["weights"]
        if(args["seq_model"]=="lstm"):
            self.seq_model = LSTM(1024, self.hidden_size,bidirectional=True)
        elif(args["seq_model"]=="gru"):
            self.seq_model = GRU(1024, self.hidden_size,bidirectional=True) 
        self.linear1 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, 2)
        self.dropout_embed = nn.Dropout2d(0.5)
        self.dropout_fc = nn.Dropout(0.3)
        self.num_labels=2
    
    def forward(self, input_ids=None, attention_mask=None,token_type_ids=None, labels= None,device=None):
        batch_size=input_ids.size(0)
        if(self.seq_model_name=="lstm"):
            _, hidden = self.seq_model(input_ids.permute(1,0,2))
            hidden=hidden[0]
        else:
            _, hidden = self.seq_model(input_ids.permute(1,0,2))
        
        hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1) 
        hidden = self.dropout_fc(hidden)
        hidden = torch.relu(self.linear1(hidden))  #batch x hidden_size
        hidden = self.dropout_fc(hidden)
        logits = self.linear2(hidden)
        if labels is not None:
            loss_funct = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.weights).to(device),reduction='mean')
            loss_logits =  loss_funct(logits.view(-1, self.num_labels), labels.view(-1)) 
            return [loss_logits,logits] 
        return logits
    