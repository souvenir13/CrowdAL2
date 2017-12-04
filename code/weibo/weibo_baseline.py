# -*- coding: utf-8 -*-
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

X = np.array(tfidf.todense())#把稀疏矩阵输出成真实矩阵
y = np.array(mydata['label'])
y_crowd = np.array(mydata['crowd3'])
print(X.shape)

'''
disrupt the order of the data
'''
rng = np.random.RandomState(0)#创建随机数生成器
indices = np.arange(len(mydata))#生成固定长度数组
rng.shuffle(indices)#随机变换数组内元素顺序
X = X[indices[:]]
y = y[indices[:]]
y_crowd = y_crowd[indices[:]]

dataSize = len(mydata)# the total size of mydata
train_size = int(dataSize*0.005)# the initial train data size 20
initial_size = int(dataSize*0.005)# the initial train data size 20
batch_size = int(dataSize*0.0025)# the size of every add data 10
## show the data and the corresponding label
#for i in range(10):
#    print('%s\t%d'%(mydata.ix[indices[:train_size]]['data'][i:i+1],y[:train_size][i]))
model_acc_array = []
total_acc_array = []
f1_score_array = []
auc_array = []
running_time_array = []
#unlabeled_indices = np.arange(dataSize)[train_size:]# the indices of unlabeled data

start = time.clock()

for i in range(2): 
    X_train = X[:train_size]
    y_train =np.append(y[:initial_size],y_crowd[initial_size:train_size],axis=0)
    print(y_train.shape)
    print(X_train.shape)
    
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    start_train = time.clock()
    
    # start training model ...
    clf = svm.SVC(kernel='rbf',gamma=0.5,probability=True)# Gaussian Kernel
    clf.fit(X_train,y_train)
    y_results = clf.predict(X_test) # model predict labels
    
    end_train = time.clock()
    
    # record the running time
    running_time = end_train - start_train
    running_time_array.append(running_time)
    
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
    y_true = y[initial_size:] # the true label of all unlabeled data
    y_total = np.append(y_train[initial_size:], y_results, axis=0)# the crowd labels and model lobels
    total_acc = accuracy_score(y_true, y_total)
    total_acc_array.append(total_acc)
    print("total accuracy: %f" % total_acc)

    # compute AUC of the model
    y_score = clf.decision_function(X_test)# the predict prob of test data
    auc = roc_auc_score(y_test,y_score)
    auc_array.append(auc)
    print("model auc: %f" % auc)
    
    # print the classification report
    print(classification_report(y_test, y_results))
#    for j in range(batch_size):
#        print('%d\t%d\t%s'
#              %(y_crowd[train_size-batch_size:train_size][j],\
#                y[train_size-batch_size:train_size][j],\
#                mydata.ix[indices[train_size-batch_size:train_size]][j:j+1]))
    train_size += batch_size
    
end = time.clock()
print('running time is: %f'%(end-start))
#np.savetxt('./result/model_acc9.txt',np.array(model_acc_array))
#np.savetxt('./result/total_acc9.txt',np.array(total_acc_array))
#np.savetxt('./result/f1_score9.txt',np.array(f1_score_array))
#np.savetxt('./result/model_auc9.txt',np.array(auc_array))
#np.savetxt('./result/running_time9.txt',np.array(running_time_array))
