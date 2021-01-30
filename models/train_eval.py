from .model_utils import fix_the_random,format_time,save_bert_model,get_gpu
from .model_architecture import *
from .tokenization import *
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from transformers import AdamW, BertConfig
from transformers import *
import time
from tqdm import tqdm,tqdm_notebook
from sklearn.metrics import confusion_matrix,make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import json

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


def pandas_classification_report(y_true, y_pred):
    metrics_summary = precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred)
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    
    avg = list(precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred,
            average='macro'))
    avg.append(accuracy_score(y_true, y_pred, normalize=True))
    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support','accuracy']
    list_all=list(metrics_summary)
    list_all.append(cm.diagonal())
    class_report_df = pd.DataFrame(
        list_all,
        index=metrics_sum_index)

    support = class_report_df.loc['support']
    total = support.sum() 
    avg[-2] = total

    class_report_df['avg / total'] = avg

    return class_report_df.T


def return_dataframe(path):
    with open(path, 'r') as fp:
        json_data=json.load(fp)
    text_tuples=[]
    for key in json_data.keys():
        element = json_data[key]
        count_fearspeech=element['annotation_list'].count('Fear speech')
        count_normal=element['annotation_list'].count('Normal')

        if(count_fearspeech>count_normal):
            one_fear_speech=1
        else:
            one_fear_speech=0
        text_tuples.append([element['message_text'],one_fear_speech])
    annotated_df = pd.DataFrame(text_tuples,columns=['message_text','one_fear_speech'])
    return annotated_df

def eval_phase(params,test_dataloader,which_files='test',model=None,device=None):
    model.eval()
    print("Running eval on ",which_files,"...")
    t0 = time.time()
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    # Tracking variables 
    eval_loss=0.0
    nb_eval_steps=0
    true_labels=[]
    pred_labels=[]
    logits_all=[]
    # Evaluate data for one epoch
    for batch in tqdm(test_dataloader,total=len(test_dataloader)):
        # Add batch to GPU
        b_input_ids = batch[0].to(device)
        b_att_masks = batch[1].to(device)
        b_token_types = batch[2].to(device)
        b_labels = batch[3].to(device)
        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():        
            outputs = model(input_ids=b_input_ids, attention_mask=b_att_masks, token_type_ids=b_token_types)

        logits = outputs
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # Accumulate the total accuracy.
        pred_labels+=list(np.argmax(logits, axis=1).flatten())
        true_labels+=list(label_ids.flatten())
        logits_all+=list(logits)
        # Track the number of batches
        nb_eval_steps += 1
    
    logits_all_final=[]
    for logits in logits_all:
        logits_all_final.append(softmax(logits)[1])
    
    print(pred_labels[0:5],true_labels[0:5])
    testf1=f1_score(true_labels, pred_labels, average='macro')
    testacc=accuracy_score(true_labels,pred_labels)
    testrocauc=roc_auc_score(true_labels, logits_all_final,average='macro')
    testprecision=precision_score(true_labels, pred_labels, average='binary')
    return testf1,testacc,true_labels,pred_labels,testrocauc,testprecision


def train_phase(params):
    annotated_df=return_dataframe(params['data_path'])
    params_preprocess={'remove_numbers': True, 'remove_emoji': True, 'remove_stop_words': False, 'tokenize': False}
    if(params['preprocess_doc']==True):
        list_sents=[preprocess_doc(ele,params=params_preprocess) for ele in tqdm(annotated_df['message_text'],total=len(annotated_df))]
    else:
        list_sents=[preprocess_sent(ele,params=params_preprocess) for ele in tqdm(annotated_df['message_text'],total=len(annotated_df))]
    X_0 = np.array(list_sents,dtype='object')
    y_0 = np.array(annotated_df['one_fear_speech'])
    if(params['weights']== [1.0,1.0]):
        params['weights']=list(class_weight.compute_class_weight("balanced", np.unique(y_0),y_0).astype(float))
    
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(params['model_path'])
    
    torch.autograd.set_detect_anomaly(True)
    if torch.cuda.is_available() and params['device']=='cuda':    
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        deviceID = get_gpu(params)
        torch.cuda.set_device(deviceID[0])
    else:
        print('Since you dont want to use GPU, using the CPU instead.')
        device = torch.device("cpu")
    
    list_val_accuracy=[]
    list_val_fscore=[]
    list_val_rocauc =[]
    list_val_precision=[]
    list_epoch=[]
    
    list_total_preds=[]
    list_total_truth=[]
    
    
    skf = StratifiedKFold(n_splits=5, random_state= 2020)
    count_skf=0
    for train_index, test_index in skf.split(X_0, y_0):
        print(train_index, test_index)
        count_skf+=1
        print("")
        print('======== Fold {:} / {:} ========'.format(count_skf,5))
        
        
        model=select_transformer_model(params['transformer_type'],params['model_path'],params)
        if(params['freeze_bert']==True):
            model.freeze_bert_encoder()
            model.unfreeze_bert_encoder_last_layers()

        if(params["device"]=='cuda'):
            model.cuda()
        
            
        optimizer = AdamW(model.parameters(),lr = params['learning_rate'], # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps = params['epsilon'] # args.adam_epsilon  - default is 1e-8.
                    )
        
        X_train, X_test = X_0[train_index], X_0[test_index]
        y_train, y_test = y_0[train_index], y_0[test_index]
        
        print(X_train.shape)
        if(params['transformer_type']=='birnn_laser'):
            X_train = encode_documents_laser(X_train,params)
            X_test= encode_documents_laser(X_test,params)
        elif(params['transformer_type']=='normal_transformer'):
            #X_train,X_train_mask,y_train = encode_documents_sent(X_train,y_train,params,tokenizer)
            #X_test,X_test_mask,y_test= encode_documents_sent(X_test,y_test,params,tokenizer)
            X_train,X_train_mask= encode_sent(X_train,params,tokenizer)
            X_test,X_test_mask= encode_sent(X_test,params,tokenizer)
        else:
            X_train = encode_documents(X_train,params,tokenizer)
            X_test= encode_documents(X_test,params,tokenizer)
        if(params['transformer_type']=='normal_transformer'):
            train_dataloader=return_dataloader_sent(X_train,X_train_mask,y_train,params,is_train=True)
            test_dataloader=return_dataloader_sent(X_test,X_test_mask,y_test,params,is_train=False)
        else:
            train_dataloader=return_dataloader(X_train,y_train,params,is_train=True)
            test_dataloader=return_dataloader(X_test,y_test,params,is_train=False)

        total_steps = len(train_dataloader) * params['epochs']

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(total_steps/10),num_training_steps = total_steps)
        fix_the_random(seed_val = params['random_seed'])
        # Store the average loss after each epoch so we can plot them.
        loss_values = []

        bert_model = params['model_path']
        best_val_fscore=0
        best_val_accuracy=0
        best_val_rocauc=0
        best_val_precision=0
        epoch_count=0
        best_true_labels=[]
        best_pred_labels=[]
        
        for epoch_i in range(0,params['epochs']):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, params['epochs']))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_loss = 0
            model.train()

            # For each batch of training data...
            for step, batch in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)
                # `batch` contains three pytorch tensors:
                #   [0]: tokens
                #   [1]: labels 
                b_input_ids = batch[0].to(device)
                b_att_masks = batch[1].to(device)
                b_token_types = batch[2].to(device)
                b_labels = batch[3].to(device)
                model.zero_grad()        
                outputs = model(input_ids=b_input_ids, attention_mask=b_att_masks, token_type_ids=b_token_types,labels=b_labels,device=device)
                loss = outputs[0]
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            avg_train_loss = total_loss / len(train_dataloader)
            train_fscore,train_accuracy,_,_,_,train_precision=eval_phase(params,train_dataloader,'train',model,device=device)
            print('avg_train_loss',avg_train_loss)
            print('train_fscore',train_fscore)
            print('train_accuracy',train_accuracy)
            print('train_precision',train_precision)
            # Store the loss value for plotting the learning curve.
            loss_values.append(avg_train_loss)
            val_fscore,val_accuracy,true_labels,pred_labels,val_rocauc,val_precision=eval_phase(params,test_dataloader,'val',model,device=device)
            print('val_fscore',val_fscore)
            print('val_accuracy',val_accuracy)
            print('val_rocauc',val_rocauc)
            print('val_precision',val_precision)
            #Report the final accuracy for this validation run.
            if(val_rocauc >= best_val_rocauc and epoch_i >= 1):
                print("current:",val_precision,'previous best:',best_val_precision)
                best_val_fscore=val_fscore
                best_val_accuracy=val_accuracy
                best_val_rocauc=val_rocauc
                best_val_precision=val_precision
                epoch_count=epoch_i
                best_pred_labels=pred_labels
                best_true_labels=true_labels
                save_bert_model(model,tokenizer,params,count_skf)
        list_total_preds+=best_pred_labels
        list_total_truth+=best_true_labels
        list_val_fscore.append(best_val_fscore)
        list_val_accuracy.append(best_val_accuracy)
        list_val_rocauc.append(best_val_rocauc)
        list_val_precision.append(best_val_precision)
        
        list_epoch.append(epoch_count)
       
    print("Accuracy: %0.2f (+/- %0.2f)" % (np.array(list_val_accuracy).mean(), np.array(list_val_accuracy).std() * 2))
    print("Fscore: %0.2f (+/- %0.2f)" % (np.array(list_val_fscore).mean(), np.array(list_val_fscore).std() * 2))
    print("AUC-ROC: %0.2f (+/- %0.2f)" % (np.array(list_val_rocauc).mean(), np.array(list_val_rocauc).std() * 2))
    print("(+ve) Precision: %0.2f (+/- %0.2f)" % (np.array(list_val_precision).mean(), np.array(list_val_precision).std() * 2))
    print("Epoch: %0.2f (+/- %0.2f)" % (np.array(list_epoch).mean(), np.array(list_epoch).std() * 2))
    print(pandas_classification_report(list_total_truth, list_total_preds))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    