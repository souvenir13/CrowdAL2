#!coding:utf-8  
  
import re  
import sys  
import scipy
import numpy as np
import jieba
import jieba.analyse
from scipy.sparse import coo_matrix, vstack
reload(sys)  
sys.setdefaultencoding('utf8')

# savedStdout = sys.stdout 
# f_handler=open('test.log', 'w')
# sys.stdout=f_handler

# s=""" а е р т ⅰ ⅲ ⅸ ⑴ ⑸ ⑺ 
#             en: Regular expression is a powerful tool for manipulating text.  
#             zh: 汉语是世界上最优美4324324的语言，正则表达式是一个很有用的工具 
#             jp: 正規表現は非常に役に立つツールテキストを操作することです。  
#             jp-char: あアいイうウえエおオ  
#             kr:정규 표현식은 매우 유용한 도구 텍스트를 조작하는 것입니다.  
#             该酒店楼层比较高，所以对自驾游的朋友来说，酒店容易找的到。我住的是20楼，那个位置基本可以俯瞰整个横店，观赏性比较好。酒店设施比较好，比较新。如果不是自驾游的话，那有麻烦的，因为酒店在市中心以外，很难叫到出租车的。补充点评 2008年7月6日 ： 地下是专用车位，游客不能停车
#             """  
# #unicode  
# s = unicode(s)  
# print "原始unicode字符"  
# print "--------"  
# print repr(s)  
# print "--------\n" 
# #unicode chinese  
# re_words = re.compile(u"[\u4e00-\u9fa5]+")  
# m =  re_words.search(s,0)  
# print "unicode 中文"  
# print "--------"  
# print m  
# print m.group()  
# res = re.findall(re_words, s)       # 查询出所有的匹配字符串  
# # if res:  
# #     print "There are %d parts:\n"% len(res)   
# #     for r in res:   
# #         print "\t",r   
# #         print   
# # print "--------\n" 
# # print ' '.join(res)
# # res = re.findall(re_words, textString.decode('utf8'))
# text_del_num = ' '.join(res)
# print 'text:',text_del_num
# segs_exact = jieba.lcut(text_del_num,cut_all=False)
# print ''.join(segs_exact)

# 处理稀疏矩阵
# def delete_rows_csr(mat, indices):
#     """
#     Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
#     """
#     if not isinstance(mat, scipy.sparse.csr_matrix):
#         raise ValueError("works only for CSR format -- use .tocsr() first")
#     indices = list(indices)
#     mask = np.ones(mat.shape[0], dtype=bool)
#     mask[indices] = False
#     return mat[mask]


# array_sparse = scipy.sparse.csr_matrix([[10, 20, 30, 40, 50],[15, 25, 35, 45, 55],[95, 96, 97, 98, 99]])
# array_update = delete_rows_csr(array_sparse,[0,2])
# print array_update
# row = np.arange(4)
# col = np.arange(4)
# # data = np.ones(4)
# data_list = []
# weight_list = [0,2]
# for i in range(4):
# 	if i in weight_list:
# 		data_list.append(1.5)
# 	else:
# 		data_list.append(1)
# data = np.array(data_list)
# print data
# A = scipy.sparse.csr_matrix((data,(row,col)),shape = (4,4)).todense()
# print A

# 创建均匀分布的数据集
# zero_num = 0
# one_num = 0
# # with open('uniform_hotel_data.txt','w') as fw:
# with open('uniform_hotel_data.txt','r') as fr:
# 	for line in fr:
# 		text = line.strip()
# 		label = line.strip().split('\t')[1]
# 		if label == '1':
# 			one_num += 1
# 		if label == '0':
# 			zero_num += 1
# 			# fw.write(text+'\n')
# 			# if one_num == zero_num:
# 				# break

# print zero_num
# print one_num

a = np.zeros(3)
b = np.ones(3)
a = np.array([4,0])
print a.shape
a = a.reshape((1,len(a)))
print a.shape
c = np.append(a,a,axis=0)
# c = np.append(c,b,axis=0)
print c
c = c.astype(np.bool)
print c