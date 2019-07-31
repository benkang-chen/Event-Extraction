from DL.OpenNRE import nrekit
import numpy as np
import tensorflow as tf
import sys
import os

dataset_name = 'reprocess'
dataset_dir = os.path.join('./', dataset_name)
if not os.path.isdir(dataset_dir):
    raise Exception("[ERROR] Dataset dir %s doesn't exist!" % dataset_dir)

# The first 3 parameters are train / test data file name,
# word embedding file name and relation-id mapping file name respectively.
train_loader = nrekit.data_loader.JsonFileDataLoader(os.path.join(dataset_dir, 'train.json'),
                                                        os.path.join(dataset_dir, 'word_vec.json'),
                                                        os.path.join(dataset_dir, 'rel2id.json'), 
                                                        mode=nrekit.data_loader.JsonFileDataLoader.MODE_RELFACT_BAG,
                                                        shuffle=True)
test_loader = nrekit.data_loader.JsonFileDataLoader(os.path.join(dataset_dir, 'test.json'),
                                                       os.path.join(dataset_dir, 'word_vec.json'),
                                                       os.path.join(dataset_dir, 'rel2id.json'), 
                                                       mode=nrekit.data_loader.JsonFileDataLoader.MODE_ENTPAIR_BAG,
                                                       shuffle=False)

framework = nrekit.framework.ReFramework(train_loader, test_loader)


class Model(nrekit.framework.ReModel):
    encoder = "pcnn"
    selector = "att"

    def __init__(self, train_data_loader, batch_size, max_length=120):
        nrekit.framework.ReModel.__init__(self, train_data_loader, batch_size, max_length=max_length)
        self.mask = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name="mask")
        
        # Embedding
        with tf.name_scope('embedding'):
            x = nrekit.network.embedding.word_position_embedding(self.word, self.word_vec_mat, self.pos1, self.pos2)

        # Encoder
        with tf.name_scope('encoder'):
            if Model.encoder == "pcnn":
                x_train = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=0.5)
                x_test = nrekit.network.encoder.pcnn(x, self.mask, keep_prob=1.0)
            elif Model.encoder == "cnn":
                x_train = nrekit.network.encoder.cnn(x, keep_prob=0.5)
                x_test = nrekit.network.encoder.cnn(x, keep_prob=1.0)
            elif Model.encoder == "rnn":
                x_train = nrekit.network.encoder.rnn(x, self.length, keep_prob=0.5)
                x_test = nrekit.network.encoder.rnn(x, self.length, keep_prob=1.0)
            elif Model.encoder == "birnn":
                x_train = nrekit.network.encoder.birnn(x, self.length, keep_prob=0.5)
                x_test = nrekit.network.encoder.birnn(x, self.length, keep_prob=1.0)
            else:
                raise NotImplementedError

        # Selector
        with tf.name_scope('selector'):
            if Model.selector == "att":
                self._train_logit, train_repre = nrekit.network.selector.bag_attention(x_train, self.scope,
                                                                                       self.ins_label, self.rel_tot,
                                                                                       True, keep_prob=0.5)
                self._test_logit, test_repre = nrekit.network.selector.bag_attention(x_test, self.scope, self.ins_label,
                                                                                     self.rel_tot, False, keep_prob=1.0)
            elif Model.selector == "ave":
                self._train_logit, train_repre = nrekit.network.selector.bag_average(x_train, self.scope, self.rel_tot,
                                                                                     keep_prob=0.5)
                self._test_logit, test_repre = nrekit.network.selector.bag_average(x_test, self.scope, self.rel_tot,
                                                                                   keep_prob=1.0)
                self._test_logit = tf.nn.softmax(self._test_logit)
            elif Model.selector == "one":
                self._train_logit, train_repre = nrekit.network.selector.bag_one(x_train, self.scope, self.label,
                                                                                 self.rel_tot, True, keep_prob=0.5)
                self._test_logit, test_repre = nrekit.network.selector.bag_one(x_test, self.scope, self.label,
                                                                               self.rel_tot, False, keep_prob=1.0)
                self._test_logit = tf.nn.softmax(self._test_logit)
            elif Model.selector == "cross_max":
                self._train_logit, train_repre = nrekit.network.selector.bag_cross_max(x_train, self.scope,
                                                                                       self.rel_tot, keep_prob=0.5)
                self._test_logit, test_repre = nrekit.network.selector.bag_cross_max(x_test, self.scope, self.rel_tot,
                                                                                     keep_prob=1.0)
                self._test_logit = tf.nn.softmax(self._test_logit)
            else:
                raise NotImplementedError
        
        # Classifier
        with tf.name_scope('classifier'):
            self._loss = nrekit.network.classifier.softmax_cross_entropy(self._train_logit, self.label, self.rel_tot,
                                                                         weights_table=self.get_weights())
 
    def loss(self):
        return self._loss

    def train_logit(self):
        return self._train_logit

    def test_logit(self):
        return self._test_logit

    def get_weights(self):
        with tf.variable_scope("weights_table", reuse=tf.AUTO_REUSE):
            print("Calculating weights_table...")
            _weights_table = np.zeros(self.rel_tot, dtype=np.float32)
            for i in range(len(self.train_data_loader.data_rel)):
                _weights_table[self.train_data_loader.data_rel[i]] += 1.0 
            _weights_table = 1 / (_weights_table ** 0.05)
            weights_table = tf.get_variable(name='weights_table', dtype=tf.float32, trainable=False,
                                            initializer=_weights_table)
            print("Finish calculating")
        return weights_table


use_rl = False
# pcnn cnn rnn birnn
model.encoder = "pcnn"
# att ave one cross_max
model.selector = "att"

if use_rl:
    rl_framework = nrekit.rl.rl_re_framework(train_loader, test_loader)
    rl_framework.train(model, nrekit.rl.policy_agent, model_name=dataset_name + "_" + model.encoder + "_"
                       + model.selector + "_rl", max_epoch=60, ckpt_dir="checkpoint")
else:
    framework.train(model, model_name=dataset_name + "_" + model.encoder + "_" + model.selector,
                    max_epoch=60, ckpt_dir="checkpoint", gpu_nums=1)
