from models.tokenization import *
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from models.model_architecture import *
from models.train_eval import *
from models.train_eval_extra_features import *
from models.model_utils import *
from models.tokenization import *
from tqdm import tqdm, tqdm_notebook
parent_path='Data/'


#bert-base-multilingual-cased
#xlm-roberta-base
#lstm_transformer
params={'model_path':'xlm-roberta-base',
        'preprocess_doc':False,
        'max_length':256,
        'batch_size':16,
        'hidden_size':128,
        'weights':[1.0,1.0],
        'load_saved':False,
        'seq_model':'lstm',
        'data_path':parent_path+'fear_speech_data.json',
        'max_sentences_per_doc':5,
        'transformer_type':'normal_transformer',
        'take_tokens_from':'both',
        'device':'cuda',
        'learning_rate':2e-5,
        'epsilon':1e-8,
        'random_seed':2020,
        'epochs':10,
        'max_memory':0.6,
        'freeze_bert':False
       }



if __name__=='__main__': 
    #train_phase_held_out(params)
    train_phase(params)
