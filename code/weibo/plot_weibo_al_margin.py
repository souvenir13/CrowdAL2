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

from sklearn import datasets, manifold

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

def plot_data_clf(X_plot,y_plot,h=0.2):
    x_min, x_max = X_plot[:,0].min() - 1, X_plot[:,0].max() + 1
    y_min, y_max = X_plot[:,1].min() - 1, X_plot[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # ravel()将多维数组降到一维
#    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
#    Z = Z.reshape(xx.shape)
#    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X_plot[:20, 0], X_plot[:20, 1],
            s=80, facecolors='none', edgecolors='k')
    plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, cmap=plt.cm.coolwarm, s=9, alpha=0.25)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

def X2dimension(X):
    # t-SNE embedding of the digits dataset
    print("Computing t-SNE embedding")           
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    t0 = time.clock()
    X_tsne = tsne.fit_transform(X)
    t1 = time.clock()
    print("running time is :%f"%(t1-t0))
    np.savetxt('./X2dimension.txt',X_tsne)
    return X_tsne

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

unlabeled_indices = np.arange(dataSize)[train_size:]# the indices of unlabeled data
delete_indices = np.array([],dtype=np.int32)# the indices of selected data in every etr
selected_indices = np.array([],dtype=np.int32)# the indices of selected all data

X_train = X[:initial_size]
y_train = y[:initial_size]

X_2d = np.loadtxt('X2dimension.txt')
start = time.clock()

for i in range(1): 
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
    
#    # return the error test data
#    y_error = y_results + y_test
#    error_indices = unlabeled_indices[y_error==0]
#    print(len(error_indices))
#    print('%s' %(mydata.ix[error_indices]['data']))
#    
#    # return the initial training data
#    print('initial training data %s'%('.'*7))
#    print('%s' %(mydata.ix[indices[:batch_size]]['data']))

    print('Iteration %i %s'%( i ,70 * '_'))
    print("SVM model: %d labeled & %d unlabeled (%d total)"
          % (train_size, dataSize - train_size, dataSize))
    
    # compute model acc
    model_acc = clf.score(X_test,y_test)
    print("model accuracy: %f" % model_acc)
    
    print('%s'%('+'*10))
    y_true = np.concatenate((y[selected_indices],y[unlabeled_indices]),axis=0) # the true label of all unlabeled data
    plot_data_clf(X_2d,y)
    plt.show()
    print("training time is : %f" % running_time)

    # compute AUC of the model
    y_score = clf.decision_function(X_test)# the predict prob of test data

    # select the most uncertainty unlabeled data
    y_margin_distance = np.abs(y_score)
    uncertainty_index = np.argsort(y_margin_distance)[:batch_size]
    
    delete_indices = np.array([],dtype=np.int32)
    
    for index,select_index in enumerate(uncertainty_index):
        delete_index = np.array(unlabeled_indices[select_index])
        delete_indices = np.append(delete_indices,delete_index)
    
    unlabeled_indices = np.delete(unlabeled_indices,uncertainty_index)
    selected_indices = np.concatenate((selected_indices,delete_indices),axis=0)
    
#    # show the initial data
#    for j in range(batch_size):
#        print('%d\t%d\t%s'
#              %(y_crowd[train_size-batch_size:train_size][j],\
#                y[train_size-batch_size:train_size][j],\
#                mydata.ix[indices[train_size-batch_size:train_size]][j:j+1]))
    train_size += batch_size
    
end = time.clock()

print('running time is: %f'%(end-start))

#print(mydata.ix[indices[:10]])