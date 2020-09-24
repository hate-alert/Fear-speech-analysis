from .model_utils import fix_the_random,format_time,save_bert_model
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

def eval_phase(params,test_dataloader,which_files='test',model=None):
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
    # Evaluate data for one epoch
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():        
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)

        logits = outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # Accumulate the total accuracy.
        pred_labels+=list(np.argmax(logits, axis=1).flatten())
        true_labels+=list(label_ids.flatten())

        # Track the number of batches
        nb_eval_steps += 1
    print(pred_labels[0:5],true_labels[0:5])
    testf1=f1_score(true_labels, pred_labels, average='macro')
    testacc=accuracy_score(true_labels,pred_labels)
    


def train_phase(params,device):
    annotated_df=pd.read_pickle(params['data_path'])
    params_preprocess={'remove_numbers': True, 'remove_emoji': True, 'remove_stop_words': False, 'tokenize': False}
    list_sents=[preprocess_sent(ele,params=params_preprocess) for ele in tqdm_notebook(annotated_df['message_text'],total=len(annotated_df))]
    X_0 = np.array(list_sents,dtype='object')
    y_0 = np.array(annotated_df['one_fear_speech'])
    

    
    params['weights']=list(class_weight.compute_class_weight("balanced", np.unique(y_0),y_0).astype(float))
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(params['model_path'])
    
    
    
    
    skf = StratifiedKFold(n_splits=10, random_state= 2020)
    for train_index, test_index in skf.split(X_0, y_0):
        
        model=select_transformer_model(params['transformer_type'],params['model_path'],params)
        
        if(params["device"]=='cuda'):
            model.cuda()
        
        
        #### loading optimizer
        optimizer = AdamW(model.parameters(),
                      lr = params['learning_rate'], # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps = params['epsilon'] # args.adam_epsilon  - default is 1e-8.
                    )
        
        X_train, X_test = X_0[train_index], X_0[test_index]
        y_train, y_test = y_0[train_index], y_0[test_index]
        
        print(X_train.shape)
        X_train, X_train_length = encode_documents(X_train,params,tokenizer)
        X_test, X_test_length = encode_documents(X_test,params,tokenizer)
        print(X_train.shape,X_test.shape)
        train_dataloader=return_dataloader(X_train,y_train,params,is_train=True)
        test_dataloader=return_dataloader(X_test,y_test,params,is_train=False)
        
        total_steps = len(train_dataloader) * params['epochs']

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(total_steps/10),                     num_training_steps = total_steps)
        fix_the_random(seed_val = params['random_seed'])
        # Store the averaggit pull origin master --allow-unrelated-historiese loss after each epoch so we can plot them.
        loss_values = []

        bert_model = params['model_path']
        best_val_fscore=0
        best_val_accuracy=0
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
            for step, batch in tqdm(enumerate(train_dataloader)):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)
                # `batch` contains three pytorch tensors:
                #   [0]: document tuples 
                #   [1]: labels 
                b_input_ids = batch[0].to(device)
                b_labels = batch[1].to(device)
                model.zero_grad()        
                outputs = model(b_input_ids,labels=b_labels,device=device)
                loss = outputs[0]
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

        
        
