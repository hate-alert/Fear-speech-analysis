from models.tokenization import *
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from models.model_architecture import *
from models.train_eval import *
from models.model_utils import *
from models.tokenization import *
from tqdm import tqdm, tqdm_notebook
parent_path='../Data/New_Data_15-06-2020/'



params={'model_path':'bert-base-multilingual-cased',
        'max_length':128,
        'batch_size':16,
        'weights':[1.0,1.0],
        'data_path':parent_path+'Fearspeech_data_final.pkl',
        'max_sentences_per_doc':5,
        'transformer_type':'lstm_transformer',
        'device':'cuda',
        'learning_rate':2e-5,
        'epsilon':1e-8,
        'random_seed':2,
        'epochs':5,
        'max_memory':0.2
       }



if __name__=='__main__': 
    torch.autograd.set_detect_anomaly(True)
    if torch.cuda.is_available() and params['device']=='cuda':    
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        deviceID = get_gpu(params)
        torch.cuda.set_device(deviceID[0])
    else:
        print('Since you dont want to use GPU, using the CPU instead.')
        device = torch.device("cpu")
    
    train_phase(params,device = device)