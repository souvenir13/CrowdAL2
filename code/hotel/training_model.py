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
f_handler=open('./predict_out/weibo40/test.log', 'w')
sys.stdout=f_handler

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
# X_tfidf label 可以保存到硬盘

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

def choose_keywords(inputfile,maxline):
	pos = []
	linenum = 1
	with open(inputfile,'r') as fr:
		for line in fr:
			if linenum > maxline:
				break
			f_word = []
			f_word = line.strip().split(' ')
			linenum += 1
			for item in f_word:
				if item in word_dict:
					pos.append(word_dict.index(item))
	pos = list(set(pos))
	return pos


add_num = 10
loop1 = 1
num_list = np.arange(len(test_y))
xuhao_id_dict = dict(zip(num_list,text_id[trian_size:]))
# print 'all items of id dict:',xuhao_id_dict.items()

test_left_xuhao = xuhao_id_dict.keys()
test_random_xuhao = test_left_xuhao
# 随机主动学习模型
print '__________________start random active model__________________'
print 'loop:',loop1
print 'add examples:',add_num
delete_all = []
for l in range(loop1):
	print '----------------select examples-----------------'
	random.seed(l+1)# 设置随机种子每次挑出一样的序号

	delete_list = random.sample(test_random_xuhao, add_num)
	delete_all[len(delete_all):len(delete_all)] = delete_list
	# 删除样本的序号
	print 'delete xuhao list:',delete_list
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
	# 更新序号
	test_random_xuhao = list(set(test_random_xuhao)^set(delete_list))

	print 'update train x shape:',train_x.shape
	print 'update train y shape:',train_y.shape
	print 'update test x shape:',test_x.shape
	print 'update test y shape:',test_y.shape
	print 'test left xuhao:',len(test_random_xuhao)
	print '----------------training no.'+str(l+1)+' SVM model-----------------'
	clf.fit(train_x, train_y)
	classes = clf.predict(test_x)
	acc = accuracy_score(test_y,classes)

	print 'random acc:',acc
	label_num = len(train_y) - trian_size
	acc_all = (acc*len(test_y)+label_num)/(len(test_y)+label_num)
	print 'random acc_all:',acc_all
	# np.savetxt("predict_no5.txt",classes)
	print classification_report(test_y,classes)
	print '----------------save result-----------------'

	# np.savetxt("./predict_out/predict_6000_no2.txt",classes)

	# 从test_out.txt中抽取delete id list中对应的文本
	with open(inputfile2,'r') as fr:
		for line in fr:
			tt_id = line.strip().split('\t')[0]
			if tt_id in save_per_delete_id:
				print line.strip()
# print '__________________start random active model by adding weight__________________'
# delete_all = []
# test_addweight_xuhao = test_left_xuhao
# loop2 = 2
# weight = 2
# for l in range(loop2):
	
# 	print '----------------select examples-----------------'
# 	random.seed(l+1)# 设置随机种子每次挑出一样的序号
# 	delete_list = random.sample(test_addweight_xuhao, add_num)
# 	delete_all[len(delete_all):len(delete_all)] = delete_list
# 	# 删除样本的序号
# 	print 'delete xuhao list:',delete_list
# 	save_per_delete_id = []
# 	for item in delete_list:
# 		save_per_delete_id.append(xuhao_id_dict[item])
# 	with open(inputfile2,'r') as fr:
# 		for line in fr:
# 			tt_id = line.strip().split('\t')[0]
# 			if tt_id in save_per_delete_id:
# 				print line.strip()
# 	# word_pos = [3200, 16264, 15630, 5276, 21789, 1438, 10143, 25251, 4006, 9775, 432, 29106, 10981, 18997, 21321, 7354, 12895, 12095, 10828, 16451, 1350, 25927, 10184, 19529, 6732, 4046, 28261, 19409, 1112, 17375, 16352, 26085, 12369, 18280, 19305, 4971, 16108, 31087, 10355, 12021, 20585, 11769, 31098]
# 	# word_pos = [23179, 7950, 18925, 915, 15252, 6427, 1566, 21280, 5795, 20648, 10538, 23472, 23220, 17849, 14526, 15807, 16576, 8004, 8009, 24011, 10449, 17754, 9311, 3426, 3174, 8423, 16105, 10091, 2669, 10920, 16375, 763, 10748, 5757]
# 	word_pos = choose_keywords(feedback_file,add_num*(l+1))
# 	# print word_pos
# 	print '----------------change keywords weight-----------------'

# 	# size = len(word_dict)
# 	row = np.arange(size)
# 	col = np.arange(size)
# 	data_list = []
# 	for i in range(size):
# 		if i in word_pos:
# 			data_list.append(weight)
# 		else:
# 			data_list.append(1)
# 	data = np.array(data_list)
# 	# print data
# 	A = scipy.sparse.csr_matrix((data,(row,col)),shape = (size,size))
# 	X_tfidf_update = X_tfidf.dot(A)
# 	# print X_tfidf_update

# 	train_x = X_tfidf_update[:trian_size]
# 	# print train_x_1
# 	# train_x = scipy.sparse.vstack((train_x_1,train_x_2))
# 	train_y = label[:trian_size]
# 	# train_y = np.hstack((train_y_1,train_y_2))
# 	test_x = X_tfidf_update[trian_size:]
# 	test_y = label[trian_size:]
# 	test_x_copy = test_x
# 	test_y_copy = test_y
# 	print '----------------training no.'+str(l+1)+' add weight SVM model-----------------'
# 	print 'weight:',weight
# 	# print 'acc_add_weight:',acc

# 	for item in delete_list:
# 		# save_per_delete_id.append(xuhao_id_dict[item])
		
# 		temp_x = test_x_copy[item]
# 		temp_y = test_y_copy[item]

# 		train_x = scipy.sparse.vstack((train_x,temp_x))
# 		train_y = np.hstack((train_y,temp_y))

# 	print 'delete id list:',save_per_delete_id
# 	test_x = delete_rows_csr(test_x_copy,delete_all)
# 	test_y = np.delete(test_y_copy,delete_all)
# 	test_addweight_xuhao = list(set(test_addweight_xuhao)^set(delete_list))

# 	print 'update train x shape:',train_x.shape
# 	print 'update train y shape:',train_y.shape
# 	print 'update test x shape:',test_x.shape
# 	print 'update test y shape:',test_y.shape
# 	print 'test left xuhao:',len(test_addweight_xuhao)
# 	clf.fit(train_x, train_y)
# 	classes = clf.predict(test_x)
# 	acc = accuracy_score(test_y,classes)
# 	print 'add_weight random acc:',acc
# 	acc_all = (acc*len(test_y)+add_num)/(len(test_y)+add_num)
# 	print 'add_weight random acc_all:',acc_all
# 	print '----------------save result-----------------'
# 	# np.savetxt("./predict_out/predict_6000_no3.txt",classes)
# 	# np.savetxt("predict_no6.txt",classes)
