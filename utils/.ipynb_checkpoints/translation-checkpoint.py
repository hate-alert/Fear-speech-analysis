import re
import emoji
from gtrans import translate_text, translate_html
# from whatsapp_analytics import preprocess_lib
import random
import pandas as pd
import numpy as np
from multiprocessing import  Pool
import time

def remove_emoji(text):
    return emoji.get_emoji_regexp().sub(u'', text)


def approximate_emoji_insert(string, index,char):
    if(index<(len(string)-1)):

        while(string[index]!=' ' ):
            if(index+1==len(string)):
                break
            index=index+1
        return string[:index] + ' '+char + ' ' + string[index:]
    else:
        return string + ' '+char + ' '

def extract_emojis(str1):
    try:
        return [(c,i) for i,c in enumerate(str1) if c in emoji.UNICODE_EMOJI]
    except AttributeError:
        return []

# Cell
def parallelize_dataframe(df, func, n_cores=4):
    '''parallelize the dataframe'''
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def translate(x,lang):
    '''provide the translation given text and the language'''
    emoji_list=extract_emojis(x)
    try:
        translated_text=""
        for k in range(0,len(x),4000):
            translated_text+=translate_text(x[k:k+4000],lang, 'en')
    except:
        translated_text=x
    if(len(emoji_list)!=0):
        for ele in emoji_list:
            translated_text=approximate_emoji_insert(translated_text, ele[1],ele[0])
    return translated_text


def add_features(df):
    '''adding new features to the dataframe'''
    translated_text=[]
    for index,row in df.iterrows():
        if(row['language']in ['en','unk']):
            translated_text.append(row['message_text'])
        else:
            translated_text.append(translate(row['message_text'],row['language']))
    df["translated"]=translated_text
    df=df.sort_values(['timestamp'], ascending=True)
    #print("done")
    return df