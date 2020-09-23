# Cell
import pandas as pd
import random
#from nltk.corpus import stopwords
#import demoji
import re
import string
import stopwordsiso as stopwords
stopwords.has_lang("th")  # check if there is a stopwords for the language
stopwords.langs()  # return a set of all the supported languages
#stopwords.stopwords("en")  # English stopwords
import emoji
from tqdm import tqdm_notebook
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import sent_tokenize
import re
import emoji
import nltk
from cltk.tokenize.word import WordTokenizer
#from nltk.corpus import stopwords
from cltk.stop.classical_hindi.stops import STOPS_LIST


# Cell
tok = WordTokenizer(language='multilingual')
## libraries that can be used
hi_stopwords=[]
with open('../Data/Data/hindi_stopwords.txt','r') as fp:
    for w in fp.readlines():
        hi_stopwords.append(str(w[:-1]))
puncts=[">","+",":",";","*","’","●","•","-",".","''","``","'","|","​","!",",","@","?","\u200d","#","(",")","|","%","।","=","``","&","[","]","/","'"]
stop_for_this=hi_stopwords+list(stopwords.stopwords(["en", "hi", "ta","te","bn"]))+["आएगा","गए","गई","करे","नही","हम","वो","follow","दे","₹","हर","••••","▀▄▀","नही","अब","व्हाट्सएप","॥","–","ov","डॉ","ॐॐॐॐॐॐॐॐॐॐॐॐॐॐॐॐॐॐॐॐ","क्या","जी","वो","╬═╬","_","backhand_index_pointing_down","backhand_index_pointing_right","link","subscribe","backhand_index_pointing_down_light_skin_tone","backhand_index_pointing_up","Whatsapp","Follow","Tweet","सब्सक्राइब","Link","\'\'","``","________________________________","_________________________________________"]

# Cell
def preprocess_sent(sent,params={'remove_numbers':False,'remove_emoji':True,'remove_stop_words':True,'tokenize':True}):
    '''This function should implememnt a multi-lingual tokenizer '''
    '''input: a document / sentence , params is a dict of control sequence'''
    '''output: should return a token list for the entire document/sentence'''

    s = sent
    s = emoji.demojize(s)
    s = re.sub(r"http\S+",'', s)
    if(params['remove_numbers']==True):
        s = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "",s)
    s = re.sub(r"/-", " ",s)
    s = re.sub(r"#,?,\,", " ",s)
    if(params['remove_emoji']==True):
        s = re.sub(r":\S+:", " ",s)
    else:
        s = re.sub(r"[:\*]", " ",s)

    s=re.sub('[' + re.escape(''.join(puncts)) + ']', '', s)
    s=s.lower()
    if(params['tokenize']==True):
        msg= tok.tokenize(s)
    else:
        msg=s

    if((params['tokenize']==True) and (params['remove_stop_words']==True)):
        msg_filtered =  [word for word in msg if word not in stop_for_this]
    else:
        msg_filtered=msg
    return msg_filtered


# Cell
def preprocess(df,params={'remove_numbers':False,'remove_emoji':True,'remove_stop_words':True,'tokenize':True}):
    '''This function should implememnt a multi-lingual tokenizer '''
    '''input: a document / sentence , params is a dict of control sequence'''
    '''output: should return a token list for the entire document/sentence'''
    list_tokenized=[]
    for index,row in tqdm_notebook(df.iterrows(),total=len(df)):
        list_tokenized.append(preprocess_sent(row['message_text'],params))

    df['tokenized']=list_tokenized
    return df
