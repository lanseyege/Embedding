import tensorflow as tf
import numpy as np
from data_help import get_vocab, load_book, batch_model
from text_model import SE
import time 

sen_len = 120
batch_size = 10
num_epochs = 10
neg_num = 2

print 'get vocab ....'
vocab_size, embedding_size, embedding, vocab_w2i, vocab_i2w = get_vocab()
print 'get book ...'
doc = load_book(vocab_w2i, sen_len)

print 'run ...'

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #sess = tf.Session()
    with sess.as_default():
        se = SE(sen_len, vocab_size, embedding_size, embedding, batch_size, num_epochs, True, neg_num)
        sess.run(tf.initialize_all_variables())
        def train_step(input_1, input_2, input_3, input_4):
            feed_dict = {
                se.input_1:input_1,
                se.input_2:input_2,
                se.input_3:input_3,
                se.input_4:input_4,
                #se.input_y:input_y

            }
            sess.run([se.trains], feed_dict)
        batches = batch_model(doc, num_epochs, batch_size, neg_num, vocab_size, sen_len)
        for input_1, input_2, input_3, input_4 in batches:
            train_step(input_1, input_2, input_3, input_4)
        em_pos = sess.run(se.em_pos)
        fw = open('store embedding file', 'w')
        for i in range(vocab_size):
            fw.write(vocab_i2w[i] + ' ')
            for l in em_pos[i]:
                fw.write(str(l) + ' ')
            fw.write('\n')
        fw.close()

