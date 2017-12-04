# -*- coding: utf-8 -*-
import os
import re
import sys
import math
import random

default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
	reload(sys)
	sys.setdefaultencoding(default_encoding)

def readDataSet(inputfile):
	text_list = []
	with open(inputfile,'r') as fr:
		for line in fr:
			text_list.append(line.strip())
	return text_list

def writeDataSet(writedata,outputfile,dirpath):
	if not os.path.exists(dirpath):
		os.mkdir(dirpath)
	with open(os.path.join(dirpath,outputfile),'w') as fw:
		for i in range(len(writedata)):
			fw.write(writedata[i]+'\n')

def split_document(filename,train_num):
	dataSets = readDataSet(filename)
	print len(dataSets)
	trainData = random.sample(dataSets,train_num)
	for i in range(len(trainData)):
		# print trainData[i]
		dataSets.remove(trainData[i])
	print 'trainData length:',len(trainData)
	print 'testData length:',len(dataSets)
	return trainData,dataSets
	# writeDataSet(trainData,file_w_random)
	# writeDataSet(dataSets,file_w)

# file_r = 'uniform_hotel_data.txt'
file_r = 'weibo_data.txt'
train_n = 20
data_path = './weibo_'+str(train_n)
train_data,test_data = split_document(file_r,train_n)

writeDataSet(train_data,'train.txt',data_path)
writeDataSet(test_data,'test.txt',data_path)

# writeDataSet(test_data,test_wf,path)
