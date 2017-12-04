# -*- coding: utf-8 -*-
import sys
import numpy as np
import scipy
import random

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from scipy.sparse import coo_matrix, vstack

default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
	reload(sys)
	sys.setdefaultencoding(default_encoding)
# print sys.getdefaultencoding()


inputfile1 = './weibo_40/train_out.txt'
inputfile2 = './weibo_40/test_out.txt'

vectorizer = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')#保留长度为1的词，例如：好
corpus = []
text_id = []
label = []
xuhao_id_dict = {}

with open(inputfile1,'r') as fr:
	for line in fr:
		text_id.append(line.strip().split('\t')[0])
		label.append(int(line.strip().split('\t')[1]))
		text = line.strip().split('\t')[2]
		corpus.append(text)

with open(inputfile2,'r') as fr:
	for line in fr:
		text_id.append(line.strip().split('\t')[0])
		label.append(int(line.strip().split('\t')[1]))
		corpus.append(line.strip().split('\t')[2])

label = np.array(label)
text_id = np.array(text_id)

X = vectorizer.fit_transform(corpus)
word_dict = vectorizer.get_feature_names()
print 'length of word dict:',len(word_dict)
# X.toarray()
print 'shape of data vectors:',X.shape
print 'shape of labels:',label.shape
# print 'shape of text ids:',text_id.shape
# print 'length of id dict:',len(xuhao_id_dict)

# 将打印结果保存到文件
savedStdout = sys.stdout 
f_handler=open('./weibo_40/word_dict_weibo4000.log', 'w')
sys.stdout=f_handler

print ' '.join(word_dict)