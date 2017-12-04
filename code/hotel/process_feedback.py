# -*- coding: utf-8 -*-

import os
import sys

default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
	reload(sys)
	sys.setdefaultencoding(default_encoding)

feedback_file = './answer/feedback_6000.txt'
word_dict_file = 'word_dict_6000.log'

word_dict = []
with open(word_dict_file,'r') as fr:
	for line in fr:
		word_dict = line.strip().split(' ')

print 'the num of words:',len(word_dict)

pos = []
maxline = 10
linenum = 1
with open(feedback_file,'r') as fr:
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
print pos
print 'the num of feedback words:',len(pos)
for item in pos:
	print word_dict[item]
