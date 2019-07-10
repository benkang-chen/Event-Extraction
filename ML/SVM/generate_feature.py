# coding=utf8
import os
import jieba.analyse
from ML.SVM.common import *


# 自定义字典
jieba.add_word('林雷', 15, 'n')

# 若特征数据集文件不存在，那么创建一个
if not os.path.exists(train_data_filename):
	fp = open(train_data_filename,'w')
	fp.close()

# 遍历主题词段落
for w, line in ThemeLineIterator(text_filename):
	# 触发成功
	# 构造特征向量,每句话中包含的触发词个数
	feature = [line.count(feature_tag) for feature_tag in feature_tag_list]
	# any(x)判断x对象是否为空对象，如果都为空、0、false，则返回false，如果不都为空、0、false，则返回true
	# 排除没有触发词的
	if not any(feature[:-1]):
		continue

	# 输出这句话
	print(line)

	# 输出关键词
	print('Top 10 key:')
	tags = jieba.analyse.extract_tags(line, topK=10)
	print(','.join(tags))
	
	print('Feature:')
	print(feature)

	# 构造特征数据集
	# 构造特征向量，存入本地
	# 输入用于标注
	label = input()
	if label != '' and int(label) in range(1, len(event_type) + 1):
		# 是预定事件类别
		label = int(label)
		feature.append(label)
		print('Feature && label:')
		print(feature)
		print('')
		ft = open(train_data_filename, 'a')
		ft.write('%s\n'%(','.join([str(x) for x in feature])))
		ft.close()

fp.close()
