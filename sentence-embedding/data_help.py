import numpy as np
import os
import random
import time 

def get_vocab():
    file_em = 'vocab and inital embedding'
    f1 = open(file_em)
    line = f1.readline()
    line = line.strip().split()
    if len(line) != 2:
        print('wrong first line...')
        exit()

    vocab_size = int(line[0]) + 2
    embedding_size = int(line[1])
    print("vocab_size: "+str(vocab_size)+" embedding_size: "+str(embedding_size))
    embedding = np.zeros(vocab_size * embedding_size).reshape(vocab_size, embedding_size).astype(np.float32)
    vocab_w2i = {}
    vocab_i2w = {}
    vocab_w2i['-unknown-'] = 1
    vocab_i2w[1] = '-unknown-'
    vocab_w2i['-padding_zero-'] = 0
    vocab_i2w[0] = '-padding_zero-'

    i = 2
    for line in f1:
        line = line.strip().split()
        vocab = np.array(line[1:])
        vocab_w2i[line[0]] = i
        vocab_i2w[i] = line[0]
        for j in range(embedding_size):
            embedding[i][j] = vocab[j]
            #print(embedding[i][j])
        i += 1
    f1.close()
    return vocab_size, embedding_size, embedding, vocab_w2i, vocab_i2w

def load_book(vocab_w2i, sen_len):
    file1 = 'book'
    f1 = open(file1)
    doc = []
    for line in f1:
        line = line.strip().split()
        li = [0] * sen_len
        i = 0
        for l in line:
            if l in vocab_w2i:
                li[i] = vocab_w2i[l]
            else:
                li[i] = 1
            i += 1
        doc.append(li)
    f1.close()
    return doc
def batch_book(data, batch_size, num_epochs, neg_num):
    doc_len = len(data) - 2
    batch = int(doc_len / batch_size) 
    label = np.zeros(batch_size *(2+neg_num)).reshape(batch_size, 2 + neg_num).astype(np.float32)
    for i in range(2 + neg_num):
        label[i][0] = 0.5
        label[i][1] = 0.5
    for epoch in range(num_epochs):
        print 'epoch ' + str(epoch)+' ...'
        i = 1
        for j in range(batch):
            dd = []
            for k in range(batch_size):
                d = []
                d.append(data[i-1+k])
                d.append(data[i+1+k])
                A = random.sample(range(1,doc_len-1), neg_num)
                for a in A:
                    d.append(data[a])
                dd.append(d)
            yield np.array(data[i:i+batch_size]), np.array(dd), label
            i += batch_size

def batch_iter(data, batch_size, num_epochs, neg_num, sequence_length):
    data = np.array(data)
    data_size = len(data)-2
    num_batch_per_epoch = int((len(data)-2)/batch_size)+1
    input_y = np.zeros(batch_size * (neg_num + 2)).reshape(batch_size, neg_num+2)
    input_1 = np.array([[0 for l in range(sequence_length)] for j in range(batch_size)])
    input_2 = np.array([[[0 for k in range(sequence_length)] for l in range(neg_num + 2)] for j in range(batch_size)])
    
    for epoch in range(num_epochs):
        i = 1
        print("epoch: "+str(epoch))
        input_y.fill(0)
        input_1.fill(0)
        input_2.fill(0)
        for batch_num in range(num_batch_per_epoch):
            #start_time = time.time()
            for bat in range(batch_size):
                if i < data_size:
                    #input_1[bat] = data[i]
                    for j in range(len(data[i])):
                        input_1[bat][j] = data[i][j]
                    #input_2[bat][0] = data[i-1]
                    for j in range(len(data[i-1])):
                        input_2[bat][0][j] = data[i-1][j]
                    #input_2[bat][1] = data[i+1]
                    for j in range(len(data[i+1])):
                        input_2[bat][1][j] = data[i+1][j]
                    A = random.sample(range(1, data_size-2), neg_num)
                    i += 1
                    input_y[bat][0] = 0.5
                    input_y[bat][1] = 0.5
                    for a in range(neg_num):
                        #input_2[bat][a+2] = data[A[a]]
                        for j in range(len(data[A[a]])):
                            if len(data[A[a]]) > sequence_length:
                                print(len(data[A[a]]))
                            #s = data[A[a]][j]
                            input_2[bat][a+2][j] = data[A[a]][j]
                        input_y[bat][a+2] = 0.0
                else:
                    i += 1
                    input_y[bat][0] = 0.5
                    input_y[bat][1] = 0.5
            #print("%s "%(time.time()-start_time))
            yield input_1, input_2, input_y
           
def batch_model(data, num_epoch, batch_size, neg_num, vocab_size, sen_len):
    n = len(data)
    lens = []
    for a in data:
        lens.append(np.sum(np.sign(a)))
    neg_sample = [] 
    flg1 = 1.0 * 88355 / 2000000
    flg2 = 1.0 * 104342 / 24950760
    for k in range(num_epoch):
        neg_s = []
        for l in range(1, n):
            neg_s_ = []
            for j in range(neg_num):
                A = np.zeros(sen_len, dtype = np.int32)
                A[0:lens[l]] = np.random.randint(1, vocab_size-1, lens[l])
                if lens[l] < sen_len - 2:
                    if np.random.rand(1)[0] < flg1:
                        A[lens[l]] = 1
                    if np.random.rand(1)[0] < flg2:
                        A[lens[l] + 1] = 1
                neg_s_.append(A)
            neg_s.append(neg_s_)
        neg_sample.append(neg_s)
    #data = np.array(data)
    for i in range(num_epoch):
        print str(i) + ' th ... '
        for k in range(n-1):
            if k % 100000 == 0:
                print k
            if k + batch_size < n-1:
                yield data[k:k+batch_size], data[k+2:k+2+batch_size], data[k+1:k+1+batch_size], neg_sample[i][k:k+batch_size]
                k += batch_size
 

