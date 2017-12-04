# -*- coding: utf-8 -*-
import os
import re
import sys
import math
import random

import jieba
import jieba.analyse
# import jieba.posseg as pseg

default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
	reload(sys)
	sys.setdefaultencoding(default_encoding)
print sys.getdefaultencoding()

def readAndWrite(inputfile,outputfile):
	text_id = []
	text_label = []
	text_content = []
	with open(inputfile,'r') as fr:
		for line in fr:
			text_id.append(line.strip().split('\t')[0])
			text_label.append(line.strip().split('\t')[1])
			text_content.append(dataPreprocess(line.strip().split('\t')[2]))
	with open(outputfile,'w') as fw:
		for i in range(len(text_id)):
			fw.write(text_id[i]+'\t'+text_label[i]+'\t'+text_content[i]+'\n')
#读取停用词
stopkey=[line.strip().decode('utf-8') for line in open('stopword_chinese.txt').readlines()]
stopwords = {}.fromkeys(stopkey)

def dataPreprocess(textString):
	print 'source text ...'
	print textString
	textString = re.sub(r'\s+','',textString)
	re_words = re.compile(u"[\u4e00-\u9fa5]+")
	res = re.findall(re_words, textString.decode('utf8'))
	text_del_num = ''.join(res)
	segs_exact = jieba.lcut(text_del_num,cut_all=False)
	text_final = ''
	for seg in segs_exact:
		if seg not in stopwords:
			text_final += seg
	print 'del stopwords...'
	print text_final
	seg_list = jieba.lcut(text_final, cut_all=False)
	return ' '.join(seg_list)
	# return string

savedStdout = sys.stdout 
f_handler=open('./preprocess_out/weibo_out_test_u.log', 'w')
sys.stdout=f_handler

# key_words = jieba.analyse.extract_tags(final,topK=3)
# print 'key words:'
# print ' '.join(key_words)
input_file = './weibo_40/test.txt'
output_file = './weibo_40/test_out.txt'

readAndWrite(input_file,output_file)
