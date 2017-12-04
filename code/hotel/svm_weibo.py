#!coding:utf-8
import numpy as np
from gensim.models.word2vec import Word2Vec
import sys
import math

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.layers import LSTM
from keras.utils import np_utils, generic_utils
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import neighbors

from sklearn.metrics import classification_report, confusion_matrix
np.set_printoptions(threshold=np.nan)

default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
	reload(sys)
	sys.setdefaultencoding(default_encoding)

def get_input_length(trainf, testf):
	with open(trainf, 'r') as trf, open(testf, 'r') as tef:
		len_arr = []
		for line in trf:
			cur_len = len(line.strip().split('\t')[2].split(' '))
			len_arr.append(cur_len)

		for line in tef:
			cur_len = len(line.strip().split('\t')[2].split(' '))
			len_arr.append(cur_len)
	len_arr = np.array(len_arr)
	return int(len_arr.mean())


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
			data.append(np.array(sentence_vec, 'float32'))
	data = np.array(data)
	new_label = []
	for item in label:
		if item == '-1':
			new_label.append(0)
		else:
			new_label.append(1)
	label = np.array(new_label)
	return data, label, weiboId

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
#parm
# input_length = 50
input_dim = 300
out_dim = 150 
drop = 0.5
epoch = 20
batch_size = 32
vecfile = 'ChineseWeibo300.bin'
#原始600条数据
trainfile = './dataSets/weibo_600_split.txt'
testfile = './dataSets/weibo_text_unselect_split.txt'
# trainfile = 'train_20.txt'
# testfile = 'test_20.txt'
# trainfile = './dataSets/sentiment/training_vstrong_data_100.txt'
# testfile = './dataSets/sentiment/test_vstrong_data_100.txt'
# trainfile = './dataSets/uncertainty/uncertainty_second100.txt'
# testfile = './dataSets/uncertainty/uncertainty_second100_left.txt'
# trainfile = './dataSets/random/weibo_600_split_random_100.txt'
# testfile = './dataSets/random/weibo_text_unselect_split_left_3988.txt'
# candidatefile = 'weibo_short.txt'

input_length = get_input_length(trainfile, testfile)
print 'input length:',input_length
print 'load word2vec model'
w2c_model = Word2Vec.load_word2vec_format(vecfile, binary=True)

print 'load train data'
train_x, train_y, train_id = load_data(trainfile, w2c_model, input_length, input_dim)
# train_y = np_utils.to_categorical(train_y, 2)
print train_x.shape, 'samples'
print train_y.shape, 'labels'
print len(train_id), 'samples id'

print 'load test data'
test_x, test_y, test_id = load_data(testfile, w2c_model, input_length, input_dim)
print test_x.shape, 'samples'
print test_y.shape, 'labels'
print len(test_id), 'samples id'

train_x_reshape = reshape_data(train_x)
test_x_reshape = reshape_data(test_x)
print train_x_reshape.shape,'train_x reshape'
print test_x_reshape.shape,'test_x reshape'

# print train_y
###################
#training svm model
###################
print 'training svm model...'
clf_svm = SVC(kernel='linear')
clf_svm.fit(train_x_reshape, train_y) 
classes_svm = clf_svm.predict(test_x_reshape)
dec = clf_svm.decision_function(test_x_reshape)

print clf_svm
print classification_report(test_y,classes_svm)
###############
# calculate accuracy
###############
svm_acc = np_utils.accuracy(classes_svm, test_y)
print 'svm accuracy: ', svm_acc
###################
#training GaussianNB model
###################
print 'training GaussianNB model ...'
clf_nb1 = GaussianNB()
clf_nb1.fit(train_x_reshape,train_y)
classes_nb1 = clf_nb1.predict(test_x_reshape)
print classification_report(test_y,classes_nb1)
###############
# calculate accuracy
###############
nb1_acc = np_utils.accuracy(classes_nb1, test_y)
print 'GaussianNB accuracy: ', nb1_acc
###################
#training BernoulliNB model
###################
print 'training BernoulliNB model ...'
clf_nb2 = BernoulliNB()
clf_nb2.fit(train_x_reshape,train_y)
classes_nb2 = clf_nb2.predict(test_x_reshape)
print classification_report(test_y,classes_nb2)
###############
# calculate accuracy
###############
nb2_acc = np_utils.accuracy(classes_nb2, test_y)
print 'BernoulliNB accuracy: ', nb2_acc
###################
#training knn model
###################
print 'training KNN model ... '
clf_knn = neighbors.KNeighborsClassifier()
clf_knn.fit(train_x_reshape,train_y)
classes_knn = clf_knn.predict(test_x_reshape)
print classification_report(test_y,classes_knn)
###############
# calculate accuracy
###############
knn_acc = np_utils.accuracy(classes_knn, test_y)
print 'KNN accuracy: ', knn_acc
############ end ###########
#model.evaluate(test_x, test_y, batch_size = batch_size)

# for i, classes_index in enumerate(classes):
# 	if i > 10:
# 		break
# 	print prob[i, classes_index]
# print classes


# print 'input_length: ', input_length
# print 'input_dim: ', input_dim
# print 'out_dim: ', out_dim
# print 'drop: ', drop
# print 'epoch: ', epoch
# print 'batch_size: ', batch_size
# print 'vecfile: ', vecfile
# print 'trainfile: ', trainfile
# print 'testfile: ', testfile
with open('result_uncertainty100_svm_2.txt','w') as fw:
	for i in range(len(test_id)):
		fw.write(str(test_id[i])+'\t'+str(test_y[i])+'\t'+str(classes_svm[i])+'\t'+str(dec[i])+'\n')

