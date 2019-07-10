#! -*- coding: utf-8 -*-

import json
from tqdm import tqdm
import os
import pandas as pd

from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam

mode = 0
min_count = 2
char_size = 128
maxlen = 256

# 读取数据，排除“其他”类型, 排除文本长度过大的
D = pd.read_csv('../../data/ccks2019_event_entity_extract/event_type_entity_extract_train.csv',
                encoding='utf-8', header=None)
D = D[D[2] != u'其他']
D = D[D[1].str.len() <= maxlen]

'''
[
    {
        "0":"资金账户风险",
        "1":"涉嫌欺诈",
        "2":"业绩下滑",
        "3":"信批违规",
        "4":"涉嫌传销",
        "5":"交易违规",
        "6":"财务造假",
        "7":"评级调整",
        "8":"重组失败",
        "9":"实控人股东变更",
        "10":"不能履职",
        "11":"涉嫌违法",
        "12":"涉嫌非法集资",
        "13":"资产负面",
        "14":"歇业停业",
        "15":"提现困难",
        "16":"高管负面",
        "17":"投诉维权",
        "18":"失联跑路",
        "19":"产品违规",
        "20":"公司股市异常"
    },
    {
        "资金账户风险":0,
        "涉嫌欺诈":1,
        "业绩下滑":2,
        "信批违规":3,
        "涉嫌传销":4,
        "交易违规":5,
        "财务造假":6,
        "评级调整":7,
        "重组失败":8,
        "实控人股东变更":9,
        "不能履职":10,
        "涉嫌违法":11,
        "涉嫌非法集资":12,
        "资产负面":13,
        "歇业停业":14,
        "提现困难":15,
        "高管负面":16,
        "投诉维权":17,
        "失联跑路":18,
        "产品违规":19,
        "公司股市异常":20
    }
]
'''
if not os.path.exists('../classes.json'):
    id2class = dict(enumerate(D[2].unique()))
    class2id = {j: i for i, j in id2class.items()}
    json.dump([id2class, class2id], open('./classes.json', 'w'))
else:
    id2class, class2id = json.load(open('./classes.json'))

train_data = []
# t是文本，c是事件类型，n是主题词
for t, c, n in zip(D[1], D[2], D[3]):
    start = t.find(n)
    # 文本内容必须包含事件主题词
    if start != -1:
        train_data.append((t, c, n))

# 生成词频文件
if not os.path.exists('./all_chars_me.json'):
    chars = {}
    for d in tqdm(iter(train_data)):
        for c in d[0]:
            # 统计每个词出现的次数
            chars[c] = chars.get(c, 0) + 1
    # 过滤掉词频数过低的词
    chars = {i: j for i, j in chars.items() if j >= min_count}
    # 0: mask, 1: padding
    id2char = {i + 2: j for i, j in enumerate(chars)}
    char2id = {j: i for i, j in id2char.items()}
    json.dump([id2char, char2id], open('./all_chars_me.json', 'w'))
else:
    id2char, char2id = json.load(open('./all_chars_me.json'))

# 记录乱序的训练语句索引号
if not os.path.exists('./random_order_train.json'):
    random_order = list(range(len(train_data)))
    np.random.shuffle(random_order)
    json.dump(
        random_order,
        open('./random_order_train.json', 'w'),
        indent=4
    )
else:
    random_order = json.load(open('./random_order_train.json'))

# 训练数据中能被9整除的
dev_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 == mode]
# 训练数据中不能被9整除的
train_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 != mode]

# 读取测试数据
D = pd.read_csv('../../data/ccks2019_event_entity_extract/event_type_entity_extract_eval.csv',
                encoding='utf-8', header=None)
test_data = []
for id, t, c in zip(D[0], D[1], D[2]):
    test_data.append((id, t, c))


# 补0使每条句子的长度相等
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    # 记录最大句子的长度
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class DataGenerator:
    def __init__(self, data, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            """
                X表示每个句子的中词的词频[ [1,2,2,2,...], ... ]
                C是该条句子的类别[ [1], [2], ...]
                S1记录的是该条句子主题词起使位置为1的其余为0的数组[ [0,1,0,0,...], ...]
                S2记录的是该条句子主题词结束位置为1的其余为0的数组[ [0,0,0,1,...], ...]
            """
            X, C, S1, S2 = [], [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0]
                x = [char2id.get(c, 1) for c in text]
                c = class2id[d[1]]
                s1, s2 = np.zeros(len(text)), np.zeros(len(text))
                start = text.find(d[2])
                end = start + len(d[2]) - 1
                s1[start] = 1
                s2[end] = 1
                X.append(x)
                C.append([c])
                S1.append(s1)
                S2.append(s2)
                if len(X) == self.batch_size or i == idxs[-1]:
                    X = seq_padding(X)
                    C = seq_padding(C)
                    S1 = seq_padding(S1)
                    S2 = seq_padding(S2)
                    yield [X, C, S1, S2], None
                    X, C, S1, S2 = [], [], [], []


class Attention(Layer):
    """
        多头注意力机制
    """

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.out_dim = nb_head * size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        q_in_dim = input_shape[0][-1]
        k_in_dim = input_shape[1][-1]
        v_in_dim = input_shape[2][-1]
        self.q_kernel = self.add_weight(name='q_kernel',
                                        shape=(q_in_dim, self.out_dim),
                                        initializer='glorot_normal')
        self.k_kernel = self.add_weight(name='k_kernel',
                                        shape=(k_in_dim, self.out_dim),
                                        initializer='glorot_normal')
        self.v_kernel = self.add_weight(name='w_kernel',
                                        shape=(v_in_dim, self.out_dim),
                                        initializer='glorot_normal')

    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            # ndim获取轴数，expand_dims在指定的轴上扩展一维
            for _ in range(K.ndim(x) - K.ndim(mask)):
                mask = K.expand_dims(mask, K.ndim(mask))
            if mode == 'mul':
                return x * mask
            else:
                return x - (1 - mask) * 1e10

    def call(self, inputs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变化
        qw = K.dot(q, self.q_kernel)
        kw = K.dot(k, self.k_kernel)
        vw = K.dot(v, self.v_kernel)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.nb_head, self.size_per_head))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.nb_head, self.size_per_head))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.nb_head, self.size_per_head))
        # 维度置换，permute_dimension重排张量的轴
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))
        # Attention
        a = K.batch_dot(qw, kw, [3, 3]) / self.size_per_head ** 0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = self.mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = K.softmax(a)
        # 完成输出
        o = K.batch_dot(a, vw, [3, 2])
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.mask(o, q_mask, 'mul')
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


# 待识别句子输入
x_in = Input(shape=(None,))
# 事件类型
c_in = Input(shape=(1,))
# 实体左边界（标签）
s1_in = Input(shape=(None,))
# 实体右边界（标签）
s2_in = Input(shape=(None,))

x, c, s1, s2 = x_in, c_in, s1_in, s2_in
# cast(
#     x,
#     dtype,
#     name=None
# )  将 x 的数据格式转化成 dtype. 例如，原来 x 的数据格式是 bool，那么将其转化成 float 以后，就能够将其转化成 0 和 1 的序列。反之也可以.
# 想要维度增加一维，可以使用tf.expand_dims(input, dim, name=None)函数
# greater 比较两个数，返回一个 bool 类型的张量
# 输入数据经过pading处理后补0，这一步的操作是将大于2的数字变为1
x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x)
# 参数个数 3173（词汇表的大小） * 128
x = Embedding(len(id2char) + 2, char_size)(x)
# 参数个数 21（类别个数）* 128
c = Embedding(len(class2id), char_size)(c)
# 操作结果是，x中每个词的向量都是该类别的词向量，输出尺寸变成x
c = Lambda(lambda x: x[0] * 0 + x[1])([x, c])
# 在每个词向量上加上对应的类别向量
x = Add()([x, c])
x = Dropout(0.2)(x)

# 将0填充的字符的词向量置为0
x = Lambda(lambda x: x[0] * x[1])([x, x_mask])

# GPU
# x = Bidirectional(CuDNNLSTM(char_size // 2, return_sequences=True))(x)
x = Bidirectional(LSTM(char_size // 2, return_sequences=True))(x)
x = Lambda(lambda x: x[0] * x[1])([x, x_mask])
# x = Bidirectional(CuDNNLSTM(char_size // 2, return_sequences=True))(x)
x = Bidirectional(LSTM(char_size // 2, return_sequences=True))(x)
x = Lambda(lambda x: x[0] * x[1])([x, x_mask])

xo = x
x = Attention(8, 16)([x, x, x, x_mask, x_mask])
x = Lambda(lambda x: x[0] + x[1])([xo, x])

x = Concatenate()([x, c])

x1 = Dense(char_size, use_bias=False, activation='tanh')(x)
ps1 = Dense(1, use_bias=False)(x1)
ps1 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps1, x_mask])

x2 = Dense(char_size, use_bias=False, activation='tanh')(x)
ps2 = Dense(1, use_bias=False)(x2)
ps2 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps2, x_mask])

model = Model([x_in, c_in], [ps1, ps2])

train_model = Model([x_in, c_in, s1_in, s2_in], [ps1, ps2])

loss1 = K.mean(K.categorical_crossentropy(s1_in, ps1, from_logits=True))
loss2 = K.mean(K.categorical_crossentropy(s2_in, ps2, from_logits=True))
loss = loss1 + loss2

train_model.add_loss(loss)
train_model.compile(optimizer=Adam(1e-3))
train_model.summary()


def extract_entity(text_in, c_in):
    """解码函数，应自行添加更多规则，保证解码出来的是一个公司名
    """
    if c_in not in class2id:
        return 'NaN'
    _x = [char2id.get(c, 1) for c in text_in]
    _x = np.array([_x])
    _c = np.array([[class2id[c_in]]])
    _ps1, _ps2 = model.predict([_x, _c])
    start = _ps1[0].argmax()
    end = _ps2[0][start:].argmax() + start
    return text_in[start: end + 1]


class Evaluate(Callback):
    def __init__(self):
        self.ACC = []
        self.best = 0.

    def on_epoch_end(self, epoch, logs=None):
        acc = self.evaluate()
        self.ACC.append(acc)
        if acc > self.best:
            self.best = acc
            train_model.save_weights('best_model.weights')
        print('acc: %.4f, best acc: %.4f\n' % (acc, self.best))

    def evaluate(self):
        A = 1e-10
        for d in tqdm(iter(dev_data)):
            R = extract_entity(d[0], d[1])
            if R == d[2]:
                A += 1
        return A / len(dev_data)


def test(test_data):
    """注意官方页面写着是以\t分割，实际上却是以逗号分割
    """
    F = open('result.txt', 'w')
    for d in tqdm(iter(test_data)):
        s = u'"%s","%s"\n' % (d[0], extract_entity(d[1].replace('\t', ''), d[2]))
        s = s.encode('utf-8')
        F.write(s)
    F.close()


evaluator = Evaluate()
train_D = DataGenerator(train_data)

train_model.fit_generator(train_D.__iter__(),
                          steps_per_epoch=len(train_D),
                          epochs=120,
                          callbacks=[evaluator]
                          )
