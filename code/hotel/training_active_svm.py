# -*- coding: utf-8 -*-
import sys
import numpy as np
import scipy
import random

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from scipy.sparse import coo_matrix, vstack

default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
	reload(sys)
	sys.setdefaultencoding(default_encoding)
# print sys.getdefaultencoding()
# 将打印结果保存到文件
savedStdout = sys.stdout 
f_handler=open('./predict_out/6000/predict_6000_active2.log', 'w')
sys.stdout=f_handler

inputfile1 = './uniform_hotel_30/train_u_out.txt'
inputfile2 = './uniform_hotel_30/test_u_out.txt'

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
trian_size = len(text_id)
# print trian_size
with open(inputfile2,'r') as fr:
	for line in fr:
		text_id.append(line.strip().split('\t')[0])
		label.append(int(line.strip().split('\t')[1]))
		corpus.append(line.strip().split('\t')[2])

label = np.array(label)
text_id = np.array(text_id)

X = vectorizer.fit_transform(corpus)
word_dict = vectorizer.get_feature_names()
size = len(word_dict)
print '----------------count word frquency-----------------'
print 'length of word dict:',size
# X.toarray()
print 'shape of data vectors:',X.shape
print 'shape of labels:',label.shape
# print 'shape of text ids:',text_id.shape
# print 'length of id dict:',len(xuhao_id_dict)

transformer = TfidfTransformer()
X_tfidf = transformer.fit_transform(X)
print '----------------count word tfidf-----------------'
print X_tfidf.shape

train_x = X_tfidf[:trian_size]
# print train_x_1
# train_x = scipy.sparse.vstack((train_x_1,train_x_2))
train_y = label[:trian_size]
# train_y = np.hstack((train_y_1,train_y_2))
test_x = X_tfidf[trian_size:]
test_y = label[trian_size:]
test_x_copy = test_x
test_y_copy = test_y
print 'train x shape:',train_x.shape
print 'train y shape:',train_y.shape
print 'test x shape:',test_x.shape
print 'test y shape:',test_y.shape
print '----------------training initial SVM model-----------------'
clf = SVC(kernel='linear')
clf.fit(train_x, train_y)
classes = clf.predict(test_x)
acc = accuracy_score(test_y,classes)
print 'acc:',acc
print classification_report(test_y,classes)
print '----------------save result-----------------'
# np.savetxt("./predict_out/predict_6000_no1.txt",classes)
# print 'predict no.3:'
# print np.array2string(classes)
# np.savetxt("./predict_out/predict_6000_no4.txt",classes)
def delete_rows_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]

add_num = 10
loop1 = 10
num_list = np.arange(len(test_y))
xuhao_id_dict = dict(zip(num_list,text_id[trian_size:]))
# print 'all items of id dict:',xuhao_id_dict.items()

test_left_xuhao = xuhao_id_dict.keys()
test_active_xuhao = test_left_xuhao
# 最不确定 主动学习模型
print '__________________start random active model__________________'
delete_all = []
for l in range(loop1):
	print '----------------select examples-----------------'
	delete_list = []
	distance_dict = {}
	distance_list = clf.decision_function(test_x)
	distance_abs = map(abs, distance_list)
	distance_dict = dict(zip(num_list,distance_abs))
	distance_dict_sorted = sorted(distance_dict.items(),lambda x,y:cmp(x[1],y[1]))
	examples = distance_dict_sorted[:add_num]
	for item in examples:
		delete_list.append(item[0])
	# 删除样本的序号
	print 'delete xuhao list:',delete_list
	delete_all[len(delete_all):len(delete_all)] = delete_list

	# 保存挑选的样本id
	save_per_delete_id = []
	for item in delete_list:
		save_per_delete_id.append(xuhao_id_dict[item])
		
		temp_x = test_x_copy[item]
		temp_y = test_y_copy[item]

		train_x = scipy.sparse.vstack((train_x,temp_x))
		train_y = np.hstack((train_y,temp_y))

	print 'delete id list:',save_per_delete_id
	test_x = delete_rows_csr(test_x_copy,delete_all)
	test_y = np.delete(test_y_copy,delete_all)
	test_active_xuhao = list(set(test_active_xuhao)^set(delete_list))

	print 'update train x shape:',train_x.shape
	print 'update train y shape:',train_y.shape
	print 'update test x shape:',test_x.shape
	print 'update test y shape:',test_y.shape
	print 'test left xuhao:',len(test_active_xuhao)
	print '----------------training no.'+str(l+1)+' SVM model-----------------'
	clf.fit(train_x, train_y)
	classes = clf.predict(test_x)
	acc = accuracy_score(test_y,classes)
	print 'active acc:',acc
	label_num = len(train_y) - trian_size
	acc_all = (acc*len(test_y)+label_num)/(len(test_y)+label_num)
	print 'active acc_all:',acc_all
	print classification_report(test_y,classes)
	print '----------------save result-----------------'
	# np.savetxt("./predict_out/predict_6000_no4.txt",classes)
	# np.savetxt("predict_no5.txt",classes)
	# 从test_out.txt中抽取delete id list中对应的文本
	# with open(inputfile2,'r') as fr:
	# 	for line in fr:
	# 		tt_id = line.strip().split('\t')[0]
	# 		if tt_id in save_per_delete_id:
	# 			print line.strip()

# # word_pos = [1030, 31375, 12048, 1298, 10261, 8982, 12061, 1438, 5155, 25253, 27942, 10158, 31049, 2110, 19903, 960, 10184, 19529, 16717, 23889, 1365, 1367, 1112, 13284, 26086, 30057, 13290, 1471, 1261, 28272, 23665, 12021, 7288, 11769, 16634, 1147, 1002, 1151]
# word_pos = [19082, 10741, 12433, 787, 15252, 5527, 924, 15263, 13608, 15273, 19373, 20786, 23220, 825, 13627, 9148, 1087, 19524, 838, 3271, 5705, 843, 10578, 992, 993, 21612, 22512, 14837, 24182, 13304, 3263, 9212]
# weight = 1.3
# size = len(word_dict)

# row = np.arange(size)
# col = np.arange(size)
# # data = np.ones(4)
# data_list = []
# for i in range(size):
# 	if i in word_pos:
# 		data_list.append(weight)
# 	else:
# 		data_list.append(1)
# data = np.array(data_list)
# # print data
# A = scipy.sparse.csr_matrix((data,(row,col)),shape = (size,size))
# # print A[1030,1030]
# X_tfidf_update = X_tfidf.dot(A)
# # print X_tfidf_update

# train_x = X_tfidf_update[:trian_size]
# # print train_x_1
# # train_x = scipy.sparse.vstack((train_x_1,train_x_2))
# train_y = label[:trian_size]
# # train_y = np.hstack((train_y_1,train_y_2))
# test_x = X_tfidf_update[trian_size:]
# test_y = label[trian_size:]
# test_x_copy = test_x
# test_y_copy = test_y
# print 'train x shape:',train_x.shape
# print 'train y shape:',train_y.shape
# print 'test x shape:',test_x.shape
# print 'test y shape:',test_y.shape


# # clf = SVC(kernel='linear')
# # clf.fit(train_x, train_y)
# # classes = clf.predict(test_x)
# # acc = accuracy_score(test_y,classes)
# print 'weight:',weight
# # print 'acc_add_weight:',acc

# for item in delete_list:
# 	# save_per_delete_id.append(xuhao_id_dict[item])
	
# 	temp_x = test_x_copy[item]
# 	temp_y = test_y_copy[item]

# 	train_x = scipy.sparse.vstack((train_x,temp_x))
# 	train_y = np.hstack((train_y,temp_y))

# # print 'delete id list:',save_per_delete_id
# test_x = delete_rows_csr(test_x_copy,delete_list)
# test_y = np.delete(test_y_copy,delete_list)
# # test_left_xuhao = list(set(test_left_xuhao)^set(delete_list))

# print 'update train x shape:',train_x.shape
# print 'update train y shape:',train_y.shape
# print 'update test x shape:',test_x.shape
# print 'update test y shape:',test_y.shape
# # print 'test left xuhao:',len(test_left_xuhao)
# clf.fit(train_x, train_y)
# classes = clf.predict(test_x)
# acc = accuracy_score(test_y,classes)

# print 'add_weight active acc:',acc
# acc_all = (acc*len(test_y)+add_num)/(len(test_y)+add_num)
# print 'add_weight active acc_all:',acc_all
# # np.savetxt("./predict_out/predict_6000_no5.txt",classes)
# # np.savetxt("predict_no6.txt",classes)
