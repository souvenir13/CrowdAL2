# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 15:04:04 2017

@author: bowen

active learning base on max entropy

"""
import re
import time
import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn import svm
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split

from sklearn.semi_supervised import label_propagation
from sklearn.metrics import f1_score,classification_report, accuracy_score
from sklearn.metrics import roc_auc_score
'''
load initial data and labels
'''
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
'''
word2vec by tfidf
'''
corpus = []
for i in range(len(mydata)):
    corpus.append(' '.join(sent2word(mydata['data'][i])))

vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b')
tfidf = vectorizer.fit_transform(corpus)
print('the num of doc: %d'%tfidf.shape[0])
print('the size of dict: %d'%tfidf.shape[1])

X = np.array(tfidf.todense(),dtype ='float32')#把稀疏矩阵输出成真实矩阵
y = np.array(mydata['label'])
y_crowd = np.array(mydata['crowd3'])
print(X.shape)

'''
disrupt the order of the data
'''
rng = np.random.RandomState(9)#创建随机数生成器
indices = np.arange(len(mydata))#生成固定长度数组
rng.shuffle(indices)#随机变换数组内元素顺序
X = X[indices[:]]
y = y[indices[:]]
y_crowd = y_crowd[indices[:]]

dataSize = len(mydata)# the total size of mydata
train_size = int(dataSize*0.005)# the initial train data size 20
initial_size = int(dataSize*0.005)# the initial train data size 20
batch_size = 10# the size of every add data 10
## show the data and the corresponding label
#for i in range(10):
#    print('%s\t%d'%(mydata.ix[indices[:train_size]]['data'][i:i+1],y[:train_size][i]))
model_acc_array = []
total_acc_array = []
f1_score_array = []
auc_array = []
running_time_array = []

unlabeled_indices = np.arange(dataSize)[train_size:]# the indices of unlabeled data
delete_indices = np.array([],dtype=np.int32)# the indices of selected data in every etr
selected_indices = np.array([],dtype=np.int32)# the indices of selected all data

X_train = X[:initial_size]
y_train = y[:initial_size]

start = time.clock()

for i in range(2): 
    X_train = np.concatenate((X_train,X[delete_indices]),axis=0)
    y_train = np.concatenate((y_train,y_crowd[delete_indices]),axis=0)
    print(y_train.shape)
    print(X_train.shape)
    
    X_test = X[unlabeled_indices]
    y_test = y[unlabeled_indices]
    print(X_test.shape)
    print(y_test.shape)
    start_train = time.clock()
    
    # start training model ...
    clf = svm.SVC(kernel='rbf',gamma=0.5,probability=True)# Gaussian Kernel
    clf.fit(X_train,y_train)
    y_results = clf.predict(X_test) # model predict labels
    
    end_train = time.clock()
    
    # record the running time
    running_time = end_train - start_train
    running_time_array.append(running_time)
    print('train and pred time:%f'%running_time)
    
    print('Iteration %i %s'%( i ,70 * '_'))
    print("SVM model: %d labeled & %d unlabeled (%d total)"
          % (train_size, dataSize - train_size, dataSize))
    
    # compute model acc
    model_acc = clf.score(X_test,y_test)
    model_acc_array.append(model_acc)
    print("model accuracy: %f" % model_acc)
    
    # compute model f1 score
    f1_score_value = f1_score(y_test,y_results)
    f1_score_array.append(f1_score_value)
    print("model f1 score: %f" % f1_score_value)
    
    # compute acc of all unlabeled data
    y_true = np.concatenate((y[selected_indices],y[unlabeled_indices]),axis=0) # the true label of all unlabeled data
    y_total = np.append(y_train[initial_size:], y_results, axis=0)# the crowd labels and model lobels
    total_acc = accuracy_score(y_true, y_total)
    total_acc_array.append(total_acc)
    print("total accuracy: %f" % total_acc)

    # compute AUC of the model
    y_score = clf.decision_function(X_test)# the predict prob of test data
    auc = roc_auc_score(y_test,y_score)
    auc_array.append(auc)
    print("model auc: %f" % auc)
    
    # select the most uncertainty unlabeled data
    y_margin_distance = np.abs(y_score)
    uncertainty_index = np.argsort(y_margin_distance)[:batch_size]
    
    delete_indices = np.array([],dtype=np.int32)
    
    for index,select_index in enumerate(uncertainty_index):
        delete_index = np.array(unlabeled_indices[select_index])
        delete_indices = np.append(delete_indices,delete_index)
    
    unlabeled_indices = np.delete(unlabeled_indices,uncertainty_index)
    selected_indices = np.concatenate((selected_indices,delete_indices),axis=0)

    # print the classification report
    print(classification_report(y_test, y_results))
    
#    # show the initial data
#    for j in range(batch_size):
#        print('%d\t%d\t%s'
#              %(y_crowd[train_size-batch_size:train_size][j],\
#                y[train_size-batch_size:train_size][j],\
#                mydata.ix[indices[train_size-batch_size:train_size]][j:j+1]))
    train_size += batch_size
    
end = time.clock()
print('running time is: %f'%(end-start))
#np.savetxt('./result/model_acc.txt',np.array(model_acc_array))
#np.savetxt('./result/total_acc.txt',np.array(total_acc_array))
#np.savetxt('./result/f1_score.txt',np.array(f1_score_array))
#np.savetxt('./result/model_auc.txt',np.array(auc_array))
#np.savetxt('./result/running_time.txt',np.array(running_time_array))

#print(mydata.ix[indices[:10]])