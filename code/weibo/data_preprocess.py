# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 18:07:19 2017

@author: bowen
"""

import pandas as pd
import jieba
import re
import numpy as np

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer

mydata = pd.read_csv('weibo_data_labels_4000.txt',sep = '\t',encoding = 'utf-8')

print("my data size: %d"%len(mydata))
print(mydata.shape)

def sent2word(sentence):
    """
    Segment a sentence to words
    Delete stopwords
    """
    sentence = re.sub(r'\s+','',sentence)#delete space
    segList = jieba.cut(sentence)
    segResult = []
    for w in segList:
        segResult.append(w)
    # read stop words
    stopkey=[line.strip() for line in open('./data/stopword_chinese.txt',encoding='utf8').readlines()]
    stopwords = {}.fromkeys(stopkey)
    
    newSent = []
    for word in segResult:
        if word in stopwords:
            continue
        else:
            newSent.append(word)
    
    return newSent
 
corpus = []
for i in range(len(mydata)):
    corpus.append(' '.join(sent2word(mydata['data'][i])))

#print ("the lens of dict: %d"% len(corpus))

vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b',max_features=5000)
tfidf = vectorizer.fit_transform(corpus)
print('the size of dict: %d'%tfidf.shape[1])
print('the num of doc: %d'%tfidf.shape[0])
print(type(tfidf))

# show the tfidf of the word
words = vectorizer.get_feature_names()
for i in range(10):
    print ('----Document %d----' %i)
    for j in range(len(words)):
        if tfidf[i,j] > 1e-5:
            print (words[j], tfidf[i,j])