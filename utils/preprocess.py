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
from cltk.tokenize.sentence import TokenizeSentence
import re
import emoji
import nltk
from cltk.tokenize.word import WordTokenizer
#from nltk.corpus import stopwords
from cltk.stop.classical_hindi.stops import STOPS_LIST


# Cell


from emoji import UNICODE_EMOJI

# search your emoji
def is_emoji(s):
    return s in UNICODE_EMOJI

# add space near your emoji
def add_space(text):
    return ''.join(' ' + char if is_emoji(char) else char for char in text).strip()


tok = WordTokenizer(language='multilingual')
## libraries that can be used
hi_stopwords=[]
with open('../Data/Data/hindi_stopwords.txt','r') as fp:
    for w in fp.readlines():
        hi_stopwords.append(str(w[:-1]))
        
from itertools import groupby 
from string import punctuation

puncts=[">","+",":",";","*","’","●","■","•","-",".","''","``","'","|","​","!",",","@","?","\u200d","#","(",")","|","%","।","=","``","&","[","]","/","'","”","‘","‘"]
stop_for_this=hi_stopwords+list(stopwords.stopwords(["en", "hi", "ta","te","bn"]))+["आएगा","गए","गई","करे","नही","हम","वो","follow","दे","₹","हर","••••","▀▄▀","नही","अब","व्हाट्सएप","॥","–","ov","डॉ","ॐॐॐॐॐॐॐॐॐॐॐॐॐॐॐॐॐॐॐॐ","क्या","जी","वो","╬═╬","_","backhand_index_pointing_down","backhand_index_pointing_right","link","subscribe","backhand_index_pointing_down_light_skin_tone","backhand_index_pointing_up","Whatsapp","Follow","Tweet","सब्सक्राइब","Link","\'\'","``","________________________________","_________________________________________"]

# Cell
def preprocess_sent(sent,params={'remove_numbers':False,'remove_emoji':True,'remove_stop_words':True,'tokenize':True}):
    '''This function should implememnt a multi-lingual tokenizer '''
    '''input: a document / sentence , params is a dict of control sequence'''
    '''output: should return a token list for the entire document/sentence'''

    s = sent
    s = re.sub(r"http\S+",' ', s)
    s = re.sub(r"www.\S+",' ', s)
    
    if(params['remove_numbers']==True):
        s = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", " ",s)
    s = re.sub(r"/-", " ",s)
    s = re.sub(r"#,?,\,", " ",s)
    if(params['remove_emoji']==True):
        s = emoji.demojize(s)
        s = re.sub(r":\S+:", " ",s)
    else:
        s = add_space(s)
        #s = re.sub(r"[:\*]", " ",s)
    
    punc = set(punctuation) - set('.')

    newtext = []
    for k, g in groupby(s):
        if k in punc:
            newtext.append(k)
        else:
            newtext.extend(g)

    s=''.join(newtext)

    
    s=re.sub('[' + re.escape(''.join(puncts)) + ']', '', s)
    s=s.lower()
    
    s = re.sub(' +', ' ', s) 
    
    if(params['tokenize']==True):
        msg= tok.tokenize(s)
    else:
        msg=s

    if((params['tokenize']==True) and (params['remove_stop_words']==True)):
        msg_filtered =  [word for word in msg if word not in stop_for_this+puncts]
    else:
        msg_filtered=msg
    return msg_filtered


def preprocess_doc(sent,params={'remove_numbers':False,'remove_emoji':True,'remove_stop_words':True,'tokenize':True}):
    '''This function should implememnt a multi-lingual tokenizer '''
    '''input: a document / sentence , params is a dict of control sequence'''
    '''output: should return a token list for the entire document/sentence'''

    
    sent = emoji.demojize(sent)
    sent = re.sub(r"http\S+",'', sent)
    sent = re.sub(r"www.\S+",'', sent)
    
    if(params['remove_numbers']==True):
        sent = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "",sent)
    sent = re.sub(r"/-", " ",sent)
    sent = re.sub(r"#,\,", " ",sent)
    tokenizer = TokenizeSentence('hindi')
    sents = tokenizer.tokenize(sent)
    all_sents=[]
    
    
    for s in sents:
        if(params['remove_emoji']==True):
            s = re.sub(r":\S+:", "",s)
        else:
            s = re.sub(r"[:\*]", "",s)

        punc = set(punctuation) - set('.')

        newtext = []
        for k, g in groupby(s):
            if k in punc:
                newtext.append(k)
            else:
                newtext.extend(g)

        s=''.join(newtext)


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
        if(len(msg_filtered)>0):
            all_sents.append(msg_filtered)
        
    return all_sents




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
