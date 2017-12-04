写这篇readme，目的是希望你能够快速入手——（基于众包的）主动学习模型。

目录：
1 背景知识点
2 数据集说明
3 代码说明
4 相关工具包说明
5 学习建议
6 待做事项

1 背景知识点

	众包技术
	主动学习
	文本分类
		贴几个链接：基础的文本处理方式
		http://www.jianshu.com/p/4cfcf1610a73 （Python做文本情感分析之情感极性分析）
		http://blog.csdn.net/liugallup/article/details/51164962 （使用word2vec对新浪微博进行情感分析和分类）
	机器学习算法
		如果你想深入了解一些机器学习算法原理可以参考《统计学习方法》李航 著

2 数据集说明
	
	1）整理了4000条微博情感评论集（二分类）包含：正确标签（0,1）、众包收集的标签（共9个）、随机选三个的多数标签、随机选5个的多数标签、原始文本数据。
	     数据来源：COAE2014
	     众包标签来源：阿里众包
	     weibo/weibo_data_lablels_4000.txt 
	2）微博解释性数据集。
	     来源：阿里众包 
	     weibo/weibo_feedback.xls
	     weibo/weibo_feedback_1.xls
	3）twitter数据集——还未整理
	     twitter/情感标注_Twitter1000_9.xlsx
	4）停用词——中文分词需要用到
	     stopword_chinese.txt
	5）模拟解释性反馈
	     CEF/ex0.txt

3 代码说明
	
	data_preprocess.py 
		文本预处理。对文本进行预处理并用TFIDF转换为词向量。
		input：
			原文本数据集 weibo_data_lablels_4000.txt 
			停用词集：stopword_chinese.txt；
		output：词向量 X

	weibo_baseline.py 
		基于随机抽样的主动学习模型。
		input：
			原文本数据集 weibo_data_lablels_4000.txt
		output：各项评测指标

	weibo_al_margin.py
		基于边缘抽样的主动学习模型。
		input：同上；output：同上

	CEF_AL_weibo.py
		基于众包解释性反馈的主动学习模型
		input： 
			weibo_data_labels_4000.txt – 原始数据
			jieba_data.txt – 分词数据
  		        ex0.txt – 人工挑选的解释性反馈信息
		output：
			mydata_disrupt.txt – 乱序后的原始数据
			selected_data.txt – 挑选样本（原始）
			selected_data_jieba.txt – 挑选样本（分词后）
			各项评测指标

	plot_weibo_al_margin.py
		微博文本可视化。把样本映射到二维平面，并标出挑选的样本。但没分析出啥来，有兴趣的话可以参考代码，网上matplotlib例子也很多。

4 相关工具包说明
	
	numpy，scipy 便于矩阵运算
	pandas 便于数据存取
	jieba 中文分词
	sklearn 机器学习算法
	matplotlib 数据可视化
	pickle 持久化模型工具
	libact 这是一个主动学习算法工具包

5 学习建议

	了解主动学习算法基本流程；
	了解处理文本分类问题的基本流程；
	了解支持向量机SVM；
	然后再看懂data_preprocess.py；weibo_baseline.py ；weibo_al_margin.py
	最后再看我论文中提出的方法CEF_AL_weibo.py。

6 待做事项 
	
	1）将反馈数据集换成阿里众包收集数据，做一遍
	2）数据集twitter做一遍实验

PS：
代码略渣，欢迎提建议。



														by cbw