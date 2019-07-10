# coding=utf8
import jieba.posseg as pseg
import codecs

feature_tag_list = [
    '腰部',
    '发酸',
    '锻炼',
    '修炼',
    '酸痛',
    '歇息',
    '打熬',
    '说道',
    '笑道',
    '询问',
    '感到',
    '高兴',
    '心中',
    '忐忑',
    '心底',
    '坚定',
]

# 设置主题词
theme = [
    '林雷',
]

# 设置事件类别
event_type = {
    1: '修炼',
    2: '对话',
    3: '心理活动',
}

# 文件名
text_filename = 'panlong.txt'
train_data_filename = 'feature.txt'		# 字符串存储特征向量和对应标签


# 预处理部分
# 全部转换为 unicode 存储
# feature_tag_list = [x for x in feature_tag_list]
# theme = [w.encode('utf8') for w in theme]
# event_type = {k:v.encode('utf8') for k,v in event_type.items()}

# 文件名转换为 gbk 编码
text_filename = '../../data/panlong.txt'
s = codecs.open(text_filename, 'rb', encoding='utf-8')


# 归一化处理函数
def MaxMinNormalization(x, Max, Min):
    """归一化"""
    x = (x - Min) / (Max - Min)
    return x


# 主题段落迭代器
class ThemeLineIterator(object):
    """主题段落迭代器"""
    # 文件符
    fp = None

    def __init__(self, filename):
        '''构造函数'''
        self.fp = codecs.open(filename, 'rb', encoding='utf-8')

    def __next__(self):
        """返回下一个包含主题词的段落"""
        for line in self.fp:
            # 提取每句话
            line = line.strip()
            if len(line) == 0:
                continue

            # 对每句话进行词性标注
            res = pseg.cut(line)
            words = []
            flags = []
            for word, flag in res:
                words.append(word)
                flags.append(flag)
            # 匹配触发词
            # isTrigger = {k:False for k in theme}
            for i, w in enumerate(words):
                if w in theme and (
                                (i+1 < len(words) and flags[i+1] == 'v')
                                or
                                (i+2 < len(words) and flags[i+2] == 'v')
                              ) and \
                              (i-1 >= 0 and 'v' not in flags[i-1]):
                    # 这句话包含主题词 且 主题词后面两个词其中之一为动词
                    # 这个主题词没触发过
                    # if isTrigger[w]:
                    #   continue

                    # isTrigger[w] = True
                    return w, line

        # fp全部迭代结束
        self.fp.close()
        raise StopIteration

    def __iter__(self):  
        """返回迭代器自身"""
        return self

    def __del__(self):
        """析构函数"""
        if self.fp != None:
            self.fp.close()


if __name__ == '__main__':
    for w, line in ThemeLineIterator(text_filename):
        print(w, line)
        feature = [line.count(feature_tag) for feature_tag in feature_tag_list]
        print(feature)
