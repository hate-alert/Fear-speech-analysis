#export
import re
import emoji
from tqdm import tqdm_notebook
from gensim.models.phrases import Phrases, Phraser


# In[6]:


#export
def extract_emojis(str1):
    try:
        return [c for c in str1 if c in emoji.UNICODE_EMOJI]
    except AttributeError:
        return []

def extract_emojis_index(str1):
    try:
        return [(i,c) for i,c in enumerate(str1) if c in emoji.UNICODE_EMOJI]
    except AttributeError:
        return []


# # In[ ]:


# #export
# def get_emoji_dict(df,stop_emojis):
#     temp={}
#     for index,row in tqdm_notebook(df.iterrows(),total=df.shape[0]):
#         #print(row["message_text"])
#         list_emojis=set(extract_emojis(row['message_text']))
#         #print(list_emojis)

#         for emoji_this in list_emojis:
            
#             if(emoji_this in stop_emojis):
#                 #print("hello",emoji_this)
#                 continue
#             else:
#                 try:
#                     temp[emoji_this]+=1
#                 except KeyError:
#                     temp[emoji_this]=1
#     return temp


# # In[8]:


# #export
# # Date of conversion
# from datetime import datetime


# def convert(year,month,date):
#     return int(datetime(year, month, date, 0, 0, 0).timestamp()*1000)

# def convert_reverse(timestamp):
#     dt_object = datetime.fromtimestamp(timestamp/1000)
#     print("dt_object =", dt_object)
#     return dt_object


# # In[ ]:


# #export
# # Rank order difference
# def filter_dict1(dict1,threshold):
#     dict2={}
#     for key in dict1:
#         if(dict1[key]>threshold):
#             dict2[key]=dict1[key]
#     return dict2

# def rank_order_difference(dict1,dict2,max_rank=40,threshold=10):
#     dict1=filter_dict1(dict1,threshold)
#     dict2=filter_dict1(dict2,threshold)
    
#     tuples_A=sorted(dict1.items(), key=lambda item: item[1],reverse=True)
#     tuples_B=sorted(dict2.items(), key=lambda item: item[1],reverse=True)
    
#     dict_A={}
#     count=0
#     for ele in tuples_A:
#         if(count<max_rank):
#             count+=1
#         dict_A[ele[0]]=count
    
#     dict_B={}
#     count=0
#     for ele in tuples_B:
#         if(count<max_rank):
#             count+=1
#         dict_B[ele[0]]=count
    
#     rank_diff_A={}
#     for key in dict_A.keys():
#         try:
#             rank_diff_A[key]=dict_B[key]-dict_A[key]
#         except:
#             rank_diff_A[key]=max_rank-dict_A[key]
#     rank_diff_B={}
#     for key in dict_B.keys():
#         try:
#             rank_diff_B[key]=dict_A[key]-dict_B[key]
#         except:
#             rank_diff_B[key]=max_rank-dict_B[key]
            
#     return rank_diff_A,rank_diff_B 


# # In[10]:


# # export
# def add_keywords(df1,lexicon,type_):
#     occ_list=[]
#     phrase_present=[]
#     count_empty=0
#     for index,row in tqdm_notebook(df1.iterrows(),total=len(df1)):
#         temp=[]
#         x=row['tokenized']
#         for y in lexicon:
#             occ = [i for i,a in enumerate(x) if a == y[0]]
#             for b in occ:
#                   if x[b:b+len(y)] == y:
#                     temp.append([b,b+len(y)])
#         if(len(temp)>0):
#             temp.sort(key=lambda interval: interval[0])
#             merged = [temp[0]]
#             for current in temp:
#                 previous = merged[-1]
#                 if current[0] <= previous[1]:
#                     previous[1] = max(previous[1], current[1])
#                 else:
#                     merged.append(current)
#             occ_list.append(merged)
#             phrase_present.append(len(merged))
#         else:
#             occ_list.append([])
#             phrase_present.append(0)
#             count_empty+=1
        
#     df1[type_+'_phrases_count']=phrase_present
#     #df1[type_+'_phrases_to_check']=occ_list
#     return df1


# # In[1]:


# # export
# from urlextract import URLExtract
# import tldextract
# # ext = tldextract.extract('www.twitter.co/MF6mvzSBDs?ssr=true')
# # print(ext.subdomain, ext.domain, ext.suffix)


# extractor = URLExtract()
# # urls = extractor.find_urls("Text with URLs. Let's have URL janlipovsky.cz as an example.")
# # # print(urls)
# def return_domain_dict(df):
#     count=0
#     url_dict={}
#     for text in tqdm_notebook(df.message_text):
#         urls = extractor.find_urls(text,check_dns=False)
#         if(len(urls)>0):
#             for url in urls:
#                 splited=url.split('/')
#                 if(len(splited)>=3 and splited[2]=='youtu.be'):
#                     urlkey='youtube.com'
#                 elif(len(splited)>=3 and splited[2]=='t.co'):
#                     urlkey='telegram.com'
#                 elif(len(splited)>=3 and splited[2]=='g.co'):
#                     urlkey='google.com'
#                 elif(len(splited)>=3 and splited[2]=='wp.me'):
#                     urlkey='wordpress.com'
#                 elif(len(splited)>=3 and splited[2]=='bit.ly'):
#                     continue
#                 else:
#                     ext = tldextract.extract(url)
#                     urlkey=ext.domain+'.'+ext.suffix
#                 try:
#                     url_dict[urlkey.lower()]+=1
#                 except:
#                     url_dict[urlkey.lower()]=1
#             count+=1
#     print(count)
#     return url_dict


# def return_url_dict(df):
#     count=0
#     url_dict={}
#     for index,row in tqdm_notebook(df.iterrows(),total=len(df)):
#         url = row['media_url']
#         if(url!='None'):
# #             for url in urls:
#             #ext = tldextract.extract(url)
#             #urlkey=ext.domain+'.'+ext.suffix
#             try:
#                 url_dict[url]+=1
#             except KeyError:
#                 url_dict[url]=1
#             count+=1
#     print(count)
#     return url_dict


# # In[ ]:

# #export
# def get_lexicon(type):
#     lexicon_path='../Data/Important_lexicon/'
#     lexicon=[]
#     with open(lexicon_path+type+'_keywords_complete.txt', 'r') as f:
#         temp=f.read().splitlines()
#     for word in temp:
#         lexicon.append(word.split('_'))
#     return lexicon




# import time
# t=time.time()
# dict=findPOS(x)
# print(time.time()-t)


# def is_emoji(sentence,emojis):
#     count=0
#     for emoji in emojis:
#         if(emoji in sentence):
#             count+=1
#     if(count>0):
#         return True
#     return False


# def is_emoji_add(data,emojis):
#     list_is_emoji=[]
#     for index,row in data.iterrows():
#         flag=0
#         for emoji in emojis:
#             if(emoji in row['message_text']):
#                 flag=1
#                 break
#         if(flag==1):
#             list_is_emoji.append(1)
#         else:
#             list_is_emoji.append(0)
#     data['is_emoji']=list_is_emoji
#     return data


# # In[9]:


# #export
# from whatsapp_analytics.preprocess_lib import *

# def extract_pos(data,lexicon,emojis_set=None,language='en',num=30):
#     '''language is for the language to be used for stanza and
#         num is for the number of top ranked pos that we want to provide as output'''
#     data=data[data['language'].isin([language])]
#     if(emojis_set!=None):
#         emoji_filtered_data=data[is_emoji(data['message_text'],emojis_set)==True]
#     else:
#         emoji_filtered_data=data
#     ###getting the bogram and trigram phrasers####
#     phrase_path='../Data/phrase_models/'
#     bigrams = Phraser.load(phrase_path+"bigram_model.pkl")
#     trigrams= Phraser.load(phrase_path+"trigram_model.pkl")
#     ##tokeniziong the current messages
    
#     tokenized_data=preprocess(emoji_filtered_data.copy())
    
#     ####generating the 1grams,2grams and 3 gram phrasers
#     all_sentences=[]
#     for index,row in tqdm_notebook(tokenized_data.iterrows(),total=len(tokenized_data)):
#          if(len(row['tokenized']) >= 0):
#             all_sentences.append(row['tokenized'])
#     bigram_sents=[]
#     for sent in tqdm_notebook(all_sentences,total=len(all_sentences)):
#         bigram_sent=bigrams[sent]
#         bigram_sents.append(bigram_sent)
     
#     trigram_sents=[]
#     for sent in tqdm_notebook(bigram_sents,total=len(bigram_sents)):
#         trigram_sent=trigrams[sent]
#         trigram_sents.append(trigram_sent)
         
#     all_sents=[]
#     assert len(all_sentences)==len(bigram_sents) and len(all_sentences)==len(trigram_sents), 'the uni bi and tri are of different number of sents'
#     for s1,s2,s3 in tqdm_notebook(zip(all_sentences,bigram_sents,trigram_sents),total=len(all_sentences)):
#         all_sents.append(list(set(s1+s2+s3)))
#     ###filtering the matchings with the lexion
#     assert len(tokenized_data)==len(all_sents), 'tokenized data and all sents donot have same size'
#     totaltexts=[]
#     for (_,row),sent in tqdm_notebook(zip(tokenized_data.iterrows(),all_sents),total=len(all_sents)):     
#         count=0
#         for word in sent:
#             if word in lexicon:
#                 count+=1
#         if(count>0):
#             totaltexts.append(row['message_text'])
#     return findPOS(totaltexts,lang=language,k=num)       


# # In[10]:


# # export
# def extract_pos_lite(data,lexicon,emojis_set=None,num=30,language='en'):
#     '''language is for the language to be used for stanza and
#         num is for the number of top ranked pos that we want to provide as output'''
#     data=data[data[lexicon+'_phrases_count']>=1]
#     print(len(data))
#     data=data[data['language'].isin([language])]
#     data=is_emoji_add(data,emojis_set)
#     if(emojis_set!=None):
#         emoji_filtered_data=data[data['is_emoji']==1]
#     else:
#         emoji_filtered_data=data
   
#     tokenized_data=preprocess(emoji_filtered_data.copy())

#     return findPOS(list(tokenized_data['message_text']),lang=language,k=num)       


# # In[24]:


# is_emoji('This is a joke   asdasd🌞',['😁','🌞'])


# # In[ ]:


# import pandas as pd


# # In[17]:


# get_ipython().system('nbdev_build_lib')


# # In[20]:


# from empath import Empath
# lexicon = Empath()


# # In[27]:


# lexicon.analyze()


# # In[24]:


# from whatsapp_analytics import translation
# translation.translate('hello','hi',to_preprocess=False)


# # In[32]:


# from gtrans import translate_text, translate_html

# translate_text('prepare hindi alll','en','hi').strip()


# # In[ ]:


# from tqdm import tqdm_notebook
# dict_empath_hindi={}

# for key in tqdm_notebook(list(lexicon.cats.keys())):
#     list_temp=lexicon.cats[key]
#     list_temp_hindi=[]
#     for word in tqdm_notebook(list_temp):
#         word_hindi=translate_text(word,'en','hi').strip()
#         list_temp_hindi.append(word_hindi)
#     dict_empath_hindi[key]=list_temp_hindi


# # In[51]:


# import json
# with open('../Data/Important_lexicon/empath_hindi.json', 'w') as fp:
#     json.dump(dict_empath_hindi, fp, sort_keys=True, indent=4)


# # In[52]:


# with open('../Data/Important_lexicon/empath_hindi.json', 'r') as fp:
#     post_id_dict=json.load(fp)


# # In[ ]:




