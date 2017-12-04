# -*- coding: utf-8 -*-
import numpy as np
from gensim.models.word2vec import Word2Vec

import os
import re
import sys
import math
import random
import pprint,pickle
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.svm import SVC

default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
	reload(sys)
	sys.setdefaultencoding(default_encoding)

def readDataSet(inputfile):
	text_list1 = []
	text_list2 = []
	with open(inputfile,'r') as fr:
		for line in fr:
			if line.strip().split('\t')[1] == '1':
				text_list1.append(line.strip())
			else:
				text_list2.append(line.strip())
	return text_list1,text_list2

def writeDataSet(writedata,outputfile,dirpath):
	if not os.path.exists(dirpath):
		os.mkdir(path)
	with open(os.path.join(dirpath,outputfile),'w') as fw:
		for i in range(len(writedata)):
			fw.write(writedata[i]+'\n')

def split_document(filename,train_num,validation_num):
	dataSets1,dataSets2 = readDataSet(filename)
	trainData1 = random.sample(dataSets1,train_num / 2)
	trainData2 = random.sample(dataSets2,train_num / 2)
	dataSets1.extend(dataSets2)
	dataSets = dataSets1
	trainData1.extend(trainData2)
	trainData = trainData1
	for i in range(len(trainData)):
		# print trainData[i]
		dataSets.remove(trainData[i])
	validationData = random.sample(dataSets,validation_num)
	for i in range(len(validationData)):
		dataSets.remove(validationData[i])
	print 'trainData length:',len(trainData)
	print 'validationData length:',len(validationData)
	print 'testData length:',len(dataSets)
	return trainData,validationData,dataSets
	# writeDataSet(trainData,file_w_random)
	# writeDataSet(dataSets,file_w)

def get_input_length(inputfile):
	with open(inputfile, 'r') as trf:
		len_arr = []
		for line in trf:
			cur_len = len(line.strip().split('\t')[2].split())
			len_arr.append(cur_len)

	len_arr = np.array(len_arr)
	return int(len_arr.mean())

def reshape_data(datasets):
	# reshape data
	datasets_reshape = []
	for i in range(datasets.shape[0]):
		max_vec = datasets[i].max(axis=0)
		mean_vec = np.mean(datasets[i],axis=0)
		min_vec = datasets[i].min(axis=0)
		reshape_vec = np.hstack([max_vec,mean_vec,min_vec])
		datasets_reshape.append(reshape_vec)
	datasets_reshape = np.array(datasets_reshape)
	return datasets_reshape

def load_data(file, model, max_len, dim):
	with open(file, 'r') as f:
		weiboId = []
		data = []
		label = []
		for line in f:
			sentence_vec = []
			# line:words \t label
			tmp_arr = line.strip().split('\t')
			weiboId.append(tmp_arr[0])
			label.append(tmp_arr[1])
			words = tmp_arr[2].split()
			for word in words:
				if len(sentence_vec) >= max_len:
					break
				try:
					word_vec = model[word.decode('UTF-8')]
					sentence_vec.append(word_vec)
				except:
					continue
			while len(sentence_vec) < max_len:
				sentence_vec.append([0.0 for i in range(dim)])
			# print sentence_vec
			# print np.array(sentence_vec)
			data.append(np.array(sentence_vec, 'float32'))
	data = np.array(data)
	new_label = []
	for item in label:
		if item == '-1':
			new_label.append(0)
		else:
			new_label.append(1)
	label = np.array(new_label)
	data = reshape_data(data)
	# pca = PCA(n_components=300)
	# data = pca.fit_transform(data)
	return data, label, weiboId

def documemts_to_tfidf(file1,file2,file3):
	# stopwords = {}.fromkeys([ line.rstrip() for line in open('stopword_chinese.txt') ])
	# print stopwords
	link = re.compile("\d+")
	with open(file1,'r') as fr1,open(file2,'r') as fr2, open(file3,'r') as fr3:
		corpus = []
		label = []
		for line in fr1:
			text = line.strip().split('\t')[2]
			text_del_num = re.sub(link,'',text)
			text_label = line.strip().split('\t')[1]
			corpus.append(text_del_num)
			label.append(text_label)
		print len(corpus)
		for line in fr2:
			text = line.strip().split('\t')[2]
			text_del_num = re.sub(link,'',text)
			text_label = line.strip().split('\t')[1]
			corpus.append(text_del_num)
			label.append(text_label)
		print len(corpus)
		for line in fr3:
			text = line.strip().split('\t')[2]
			text_del_num = re.sub(link,'',text)
			text_label = line.strip().split('\t')[1]
			corpus.append(text_del_num)
			label.append(text_label)
		print len(corpus)
		# print corpus
	new_label = []
	for item in label:
		if item == '-1':
			new_label.append(0)
		else:
			new_label.append(1)
	label = np.array(new_label)
	print label.shape
	vectorizer = CountVectorizer(max_features=2000)
	X = vectorizer.fit_transform(corpus)
	X = X.toarray()
	features = vectorizer.get_feature_names()
	transformer = TfidfTransformer()
	tfidf = transformer.fit_transform(X)
	weight = tfidf.toarray()
	pca = PCA(n_components=300)
	new_X = pca.fit_transform(weight)
	dataSets1 = new_X[:500]
	dataSets2 = new_X[500:1500]
	dataSets3 = new_X[1500:]
	label1 = label[:500]
	label2 = label[500:1500]
	label3 = label[1500:]
	return dataSets1,label1,dataSets2,label2,dataSets3,label3


def save_to_pkl(dataSet,filename,path):
	output = open(os.path.join(path,filename),'wb')
	pickle.dump(dataSet, output, -1)
	output.close()

def read_from_pkl(filename,path):
	dataSet = pickle.load(open(os.path.join(path,filename), 'rb'))
	print dataSet.shape
	return dataSet

def aclearning_model(train_x, train_y, test_x, test_y, times, add_num, qs):
	acc = []
	# print 'the first training ...'
	clf = SVC(kernel='linear')
	clf.fit(train_x, train_y)
	classes = clf.predict(test_x)
	acc.append(accuracy_score(test_y,classes))
	# print 'acc:',acc
	if times==0:
		return acc
	if qs == 'mg':
		print 'simple margin select ...'	
	if qs == 'rd':
		print 'random select ...'
	for i in range(times):
		if qs == 'mg':
			distance_dict = {}
			num_list = np.arange(len(validate_y))
			distance_list = clf.decision_function(validate_x)
			distance_abs = map(abs, distance_list)
			distance_dict = dict(zip(num_list,distance_abs))
			distance_dict_sorted = sorted(distance_dict.items(),lambda x,y:cmp(x[1],y[1]))
			examples = distance_dict_sorted[:add_num]
			# update train and test
			delete = []
			for item in examples:
				temp_x = test_x[item[0]]
				temp_y = test_y[item[0]]
				temp_x = temp_x.reshape((1,-1))

				train_x = np.append(train_x,temp_x,axis=0)
				train_y = np.append(train_y,temp_y)
				delete.append(item[0])

		if qs == 'rd':
			delete = random.sample(range(len(test_x)), add_num)	
			for item in delete:
				temp_x = test_x[item]
				temp_y = test_y[item]
				temp_x = temp_x.reshape((1,-1))

				train_x = np.append(train_x,temp_x,axis=0)
				train_y = np.append(train_y,temp_y)
		test_x = np.delete(test_x, delete,axis=0)
		test_y = np.delete(test_y,delete)
		clf.fit(train_x, train_y)
		classes = clf.predict(test_x)
		acc.append(accuracy_score(test_y,classes))

	return acc

def readDeleteText(d_list,text):
	print 'text length:',len(text)
	list_sorted = sorted(d_list)
	print list_sorted
	offset = 0
	print 'delete text:'
	for index in list_sorted:
		print index,':',text[index-offset]
		del text[index-offset]
		offset = offset + 1
	return text

def validate_model(train_x, train_y, validate_x, validate_y, test_x, test_y, times, add_num, qs):
	# read validation data ID
	acc = []
	hotel_text = []
	filename = 'E:\Document\Python\SVM_weibo\hotel500\\val_1000.txt'
	cnt = 0
	print 'lalala'
	with open(filename,'r') as fr:
		line = ''
		for line in fr:
			hotel_text.append(line.strip())
	print '-----------------------------------'
	print 'hotel text length:',len(hotel_text)
	print '-----------------------------------'
	print 'the first training ...'
	clf = SVC(kernel='linear')
	clf.fit(train_x, train_y)
	classes = clf.predict(test_x)
	acc.append(accuracy_score(test_y,classes))
	# print 'acc:',acc
	if times==0:
		return acc
	if qs == 'mg':
		print 'simple margin select ...'	
	if qs == 'rd':
		print 'random select ...'
	for i in range(times):
		if qs == 'mg':
			distance_dict = {}
			num_list = np.arange(len(validate_y))
			# record the distance of test data
			# decision_funtion 注意传入的数据类型
			
			distance_list = clf.decision_function(validate_x)
			distance_abs = map(abs, distance_list)
			distance_dict = dict(zip(num_list,distance_abs))
			distance_dict_sorted = sorted(distance_dict.items(),lambda x,y:cmp(x[1],y[1]))
			examples = distance_dict_sorted[:add_num]
			# print examples
			# update train and validation
			delete = []
			for item in examples:
				temp_x = validate_x[item[0]]
				temp_y = validate_y[item[0]]
				temp_x = temp_x.reshape((1,-1))

				train_x = np.append(train_x,temp_x,axis=0)
				# train_x = np.append(train_x,temp_x,axis=0)
				train_y = np.append(train_y,temp_y)
				# train_y = np.append(train_y,temp_y)
				# if temp_y == 1:
				# 	train_x = np.append(train_x,temp_x,axis=0)
				# 	train_y = np.append(train_y,temp_y)
				delete.append(item[0])
		if qs == 'rd':
			delete = random.sample(range(len(validate_x)), add_num)	
			for item in delete:
				temp_x = validate_x[item]
				temp_y = validate_y[item]
				temp_x = temp_x.reshape((1,-1))

				train_x = np.append(train_x,temp_x,axis=0)
				train_y = np.append(train_y,temp_y)
		print '........................................'
		print 'delete list:',delete
		hotel_text = readDeleteText(delete,hotel_text)
		print '........................................'
		validate_x = np.delete(validate_x, delete,axis=0)
		validate_y = np.delete(validate_y,delete)
		clf.fit(train_x, train_y)
		classes = clf.predict(test_x)
		acc.append(accuracy_score(test_y,classes))
	print 'train x shape:',train_x.shape
	print 'validate x shape:',validate_x.shape
	return acc

#### split document
dataName = 'hotel'
representation = '_tfidf'

path = './'+dataName+'500'

# file_r = 'hotel_split.txt'
# train_n = 500
# valid_n = 1000
# train_wf = 'train_'+str(train_n)+'.txt'
# valid_wf = 'val_'+str(valid_n)+'.txt'
# test_wf = 'test.txt'
#### write txt file
# train_data,val_data,test_data = split_document(file_r,train_n,valid_n)
# writeDataSet(train_data,train_wf,path)
# writeDataSet(val_data,valid_wf,path)
# writeDataSet(test_data,test_wf,path)

# input
input_dim = 300#weibo词向量是300维的
train_size = [20]
# input_length = 15
add_num_all = [100]
times = [2,5,10]


# input_file = 'input_file2.txt'
# input_file = 'hotel_split.txt'
vecfile = 'ChineseWeibo300.bin'
# vecfile = 'vecmodel.bin'
trian_file = './'+dataName+'500/train_500.txt'
validate_file = './'+dataName+'500/val_1000.txt'
test_file = './'+dataName+'500/test.txt'

# input_length = 32
# input_length = get_input_length(input_file)
# print 'input length:',input_length
# print 'load word2vec model'
# w2c_model = Word2Vec.load_word2vec_format(vecfile, binary=False)
# w2c_model = Word2Vec.load_word2vec_format(vecfile, binary=True)

########################计算tfidf值###############################
# print 'calculate tfidf value ...'
# trainSet_X,trainSet_y,valSet_X,valSet_y,test_X,test_y = documemts_to_tfidf(trian_file,validate_file,test_file)
#############################分割线################################
print 'load train data ...'
# trainSet_X, trainSet_y, trainSet_id = load_data(trian_file, w2c_model, input_length, input_dim)
# save_to_pkl(trainSet_X,'trainSet_X'+representation+'.pkl',path)
# save_to_pkl(trainSet_y,'trainSet_y'+representation+'.pkl',path)

trainSet_X = read_from_pkl('trainSet_X'+representation+'.pkl',path)
trainSet_y = read_from_pkl('trainSet_y'+representation+'.pkl',path)

print 'load validation data ...'
# valSet_X,valSet_y,valSet_id = load_data(validate_file, w2c_model, input_length, input_dim)
# save_to_pkl(valSet_X,'valSet_X'+representation+'.pkl',path)
# save_to_pkl(valSet_y,'valSet_y'+representation+'.pkl',path)
valSet_X = read_from_pkl('valSet_X'+representation+'.pkl',path)
valSet_y = read_from_pkl('valSet_y'+representation+'.pkl',path)

print 'load test data ...'
# test_X, test_y, test_id = load_data(test_file, w2c_model, input_length, input_dim)
# save_to_pkl(test_X,'test_X'+representation+'.pkl',path)
# save_to_pkl(test_y,'test_y'+representation+'.pkl',path)
test_X = read_from_pkl('test_X'+representation+'.pkl',path)
test_y = read_from_pkl('test_y'+representation+'.pkl',path)


for tr_size in train_size:
	print 'split train data and test data ...'
	X_train_init, X_left, y_train_init, y_left = train_test_split(trainSet_X, trainSet_y, train_size=tr_size, random_state=5)
	print '======================================='
	print 'initial train data shape:',X_train_init.shape
	print '======================================='

	for add_size in add_num_all:
		print 'training passive learning model ...'
		print '======================================='
		print 'the total num of add samples:',add_size
		print '======================================='
		acc_score_no = []
		for i in range(100):
			X_train_add, X_test_add, y_train_add, y_test_add = train_test_split(valSet_X, valSet_y, train_size=add_size)
			X_train_all = np.row_stack((X_train_init,X_train_add))
			y_train_all = np.hstack((y_train_init,y_train_add))
			acc_score_no.append(aclearning_model(X_train_all, y_train_all, test_X, test_y, 0, 0,'rd')[0])
		print 'all train data shape:',X_train_all.shape
		# print acc_score_no
		acc_noactive_mean = sum(acc_score_no)/len(acc_score_no)
		# acc_noactive_mean = 0.8718
		print 'no active mean value: ',acc_noactive_mean

		for time in times:
			print '................start iteration..........................'
			acc_score_mg = []
			acc_score_rd = []
			if time <= add_size:
				print 'iteration times:',time
				add_per_num = add_size/time
				print 'training active learning model ...'
				acc_score_mg = validate_model(X_train_init,y_train_init,valSet_X,valSet_y,test_X,test_y,time, add_per_num,'mg')
				print '-------------accuracy margin-----------'
				print acc_score_mg[0],acc_score_mg[len(acc_score_mg)-1]
				print '---------------------------------------'
				# acc_score_rd = validate_model(X_train_init,y_train_init,valSet_X,valSet_y,test_X,test_y,time, add_per_num,'rd')
				# print '-------------accuracy random-----------'
				# print acc_score_rd[0],acc_score_rd[len(acc_score_rd)-1]
				# print '---------------------------------------'

				print '======================================='
				# 随机挑选
				# for i in range(len(acc_score_rd)):
				# 	if acc_score_rd[i] > acc_noactive_mean:
				# 		print 'random select:',i
				# 		break

				for i in range(len(acc_score_mg)):
					if acc_score_mg[i] > acc_noactive_mean:
						print 'simple margin:',i
						break
				print '======================================='

				query_num_1 = np.arange(0, add_size+add_per_num , add_per_num)
				acc_score_no_np = np.ones(len(query_num_1))*acc_noactive_mean
				# print acc_score_2_np
				# fig = plt.figure()
				# plt.plot(query_num_1, acc_score_mg, 'r', label='margin')
				# plt.plot(query_num_1, acc_score_rd, 'b', label='random')
				# plt.plot(query_num_1, acc_score_no_np, 'g', label='no active')
				# plt.xlabel('Number of Queries')
				# plt.ylabel('Acc')
				# plt.title('Experiment Result')
				# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
				#                fancybox=True, shadow=True, ncol=5)
				# 必须在show之前保存图片
				# path_img = './'+dataName+'500/'+representation+'_2/'+str(add_size+tr_size)

				# figname = dataName+'_'+str(tr_size)+'_'+str(time)+'.png'
				# if not os.path.exists(path_img):
				# 	os.makedirs(path_img)
				# plt.savefig(os.path.join(path_img,figname),format='png')
				# plt.close('all')
			print '................end iteration..........................'

				# plt.show()
