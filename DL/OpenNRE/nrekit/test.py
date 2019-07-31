import tensorflow as tf
import numpy as np
with tf.device('/gpu:1'):
    v1 = tf.constant([1.0, 2.0, 3.0], shape=[3], name='v1')
    v2 = tf.constant([1.0, 2.0, 3.0], shape=[3], name='v2')
    sumV12 = v1 + v2
    pos1_embedding = tf.get_variable('real_pos1_embedding', [240, 5], dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())
    # pos1_embedding = tf.concat([tf.zeros((1, pos_embedding_dim), dtype=tf.float32), real_pos1_embedding], 0)
    pos2_embedding = tf.get_variable('real_pos2_embedding', [240, 5], dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())
    # pos2_embedding = tf.concat([tf.zeros((1, pos_embedding_dim), dtype=tf.float32), real_pos2_embedding], 0)

    input_pos1 = tf.nn.embedding_lookup(pos1_embedding, [1, 2])
    input_pos2 = tf.nn.embedding_lookup(pos2_embedding, [1, 2])
    x = tf.concat([input_pos1, input_pos2], -1)

    num_filters = 2
    kernel_size = 2
    batch_size = 1
    seq_length = 4
    embedding_dim = 5

    embedding_inputs = tf.constant(-1.0, shape=[batch_size, seq_length, embedding_dim], dtype=tf.float32)

    with tf.name_scope("cnn"):
        conv = tf.layers.conv1d(embedding_inputs, num_filters, kernel_size, name='conv', padding="same")

    mask_embedding = tf.constant([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    mask = tf.constant([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int32)
    inp = tf.constant([[[0, 0, 0, 1.0], [0, 0, 0, 1.0], [0, 0, 0, 1.0]],
                       [[0, 0, 0, 1.0], [0, 0, 0, 1.0], [0, 0, 0, 1.0]],
                       [[0, 0, 0, 1.0], [0, 0, 0, 1.0], [0, 0, 0, 1.0]],
                       [[0, 0, 0, 1.0], [0, 0, 0, 1.0], [0, 0, 0, 1.0]]], dtype=np.float32)
    mask = tf.nn.embedding_lookup(mask_embedding, mask)
    mask_ex = tf.expand_dims(mask * 100, 2)
    inp_ex = tf.expand_dims(inp, 3)
    # hidden_size = x.shape[-1]
    add_mask = tf.expand_dims(mask * 100, 2) + tf.expand_dims(inp, 3)
    # mask_out = tf.reduce_max(tf.expand_dims(mask * 100, 2) + tf.expand_dims(inp, 3), axis=1) - 100
    m = np.array([[
        [
           [1, 0, 0],
           [0, 1, 0]
        ],
        [
           [0, 0, 1],
           [0, 1, 0]
        ]
    ]])
    n = tf.Variable(m)
with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        # print(sess.run(input_pos1))
        # print("______")
        # print(sess.run(input_pos2))
        # print("______")
        # print(sess.run(x))
        # print(sess.run(conv).shape)
        # out = tf.reduce_max(conv, axis=-2)
        # print(sess.run(out).shape)
        print(sess.run(mask_ex).shape)
        print("*" * 15)
        print(sess.run(inp_ex).shape)
        print("*" * 15)
        print(sess.run(add_mask).shape)
        # print(sess.run(n).shape)
        # n_out = tf.reduce_max(n, axis=1)
        # print(sess.run(n_out))
