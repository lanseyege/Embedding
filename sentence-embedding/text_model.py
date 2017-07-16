import tensorflow as tf

class SE():
    def __init__(self, sen_len, vocab_size, embedding_size, embedding, batch_size, num_epochs, emd, neg_num):
        self.input_1 = tf.placeholder(tf.int32, [None, sen_len], name = 'input_1')
        self.input_2 = tf.placeholder(tf.int32, [None, sen_len], name = 'input_2')
        self.input_3 = tf.placeholder(tf.int32, [None, sen_len], name = 'input_3')
        self.input_4 = tf.placeholder(tf.int32, [None, neg_num, sen_len], name = 'input_4')
        a = [0.0 for ll in range(neg_num)]
        a.insert(0, 1.0)
        input_y = tf.constant([a for l in range(batch_size)])
        if emd:
            self.em_pos = tf.get_variable(name = 'em_pos', shape = embedding.shape, initializer = tf.constant_initializer(embedding), regularizer = tf.contrib.layers.l2_regularizer(0.001))
        else:
            self.em_pos = tf.get_variable(name = 'em_pos', shape = [vocab_size, embedding_size], initializer = tf.truncated_normal_initializer(stddev = 0.01), regularizer = tf.contrib.layers.l2_regularizer(0.001))
        with tf.name_scope('emlayer'):
            em1 = tf.nn.embedding_lookup(self.em_pos, self.input_1)
            em2 = tf.nn.embedding_lookup(self.em_pos, self.input_2)
            em3 = tf.nn.embedding_lookup(self.em_pos, self.input_3)
            em4 = tf.nn.embedding_lookup(self.em_pos, self.input_4)

            mask1 = tf.reshape(tf.sign(self.input_1), [-1])
            em1 = tf.transpose(tf.transpose(tf.reshape(em1, [-1, embedding_size])) * tf.cast(mask1, tf.float32))
            em1 = tf.reshape(em1, [-1, sen_len, embedding_size])
            mask2 = tf.reshape(tf.sign(self.input_2), [-1])
            em2 = tf.transpose(tf.transpose(tf.reshape(em2, [-1, embedding_size])) * tf.cast(mask2, tf.float32))
            em2 = tf.reshape(em2, [-1, sen_len, embedding_size])
            mask3 = tf.reshape(tf.sign(self.input_3), [-1])
            em3 = tf.transpose(tf.transpose(tf.reshape(em3, [-1, embedding_size])) * tf.cast(mask3, tf.float32))
            em3 = tf.reshape(em3, [-1, sen_len, embedding_size])
            mask4 = tf.reshape(tf.sign(self.input_4), [-1])
            em4 = tf.transpose(tf.transpose(tf.reshape(em4, [-1, embedding_size])) * tf.cast(mask4, tf.float32))
            em4 = tf.reshape(em4, [-1, neg_num, sen_len, embedding_size])

            ave1 = tf.reduce_sum(em1, 1)
            ave2 = tf.reduce_sum(em2, 1)
            ave3 = tf.reduce_sum(em3, 1)
            ave4 = tf.reduce_sum(em4, 2)
            sum_ = tf.add(ave1, ave2) / 2
        with tf.name_scope('loss'):
            res3 = tf.reduce_mean(tf.mul(sum_, ave3), 1, keep_dims = True)
            sum_ = tf.expand_dims(sum_, 2)
            res4 = tf.reshape(tf.batch_matmul(ave4, sum_), [-1, neg_num])/sen_len
            res = tf.concat(1, [res3, res4])
            losses = tf.nn.softmax_cross_entropy_with_logits(res, input_y)
            self.loss = tf.reduce_mean(losses)
        self.trains = tf.train.AdamOptimizer().minimize(self.loss)
