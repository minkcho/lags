
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf
import sys

start_idx = 1
bulk = 1
if len(sys.argv) == 3:
    start_idx = int(sys.argv[1])
    bulk = int(sys.argv[2])


# 하루 일과에서 찾아낼 패턴의 길이
n_size = 4
seq_length = n_size
data_dir = '/export/data/cert-data/processed/4.2/'
flag_additional_loss = True


# for ground-truth 
self_user_dict = pd.read_csv(data_dir + 'dictionary.csv', sep=',')
self_user_dict = self_user_dict['user']
self_answer = pd.read_csv(data_dir + 'answer_r4.2_all.csv', sep=',')
self_answer['date'] = pd.to_datetime(self_answer['date'])

def is_anomaly(date_x, n):
    user_n = self_user_dict[n]
    return not (self_answer[(self_answer['date']==date_x) & (self_answer['user']==user_n)]).empty


# train set에서 나올 수 있는 모든 패턴을 찾아낸다
def get_patterns(data_set, size):
    patterns = np.random.rand(1, n_size)
    for i in range(0,len(data_set)):
        sequence = data_set[i]
        sequence = np.array(sequence[~np.isnan(sequence)])

        if len(sequence) <= size:
            print('extend sequence')
            sequence = np.lib.pad(sequence, (0, (size - len(sequence))), 'edge')
            patterns = np.append(patterns, [sequence], axis=0)

        for j in range(0,(len(sequence)-size)):
            patterns = np.append(patterns, [sequence[j:(j+size)]], axis=0)

    patterns = np.delete(patterns, 0, 0)
    patterns = np.vstack({tuple(row) for row in patterns})
    return patterns

def train_seq(patterns):
    X = patterns
    Y = X[:]

    X = np.array(X, dtype=int).T
    Y = np.array(Y, dtype=int).T
    # print(X.shape)
    feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
    feed_dict.update({labels[t]: Y[t] for t in range(seq_length)})
    # print(feed_dict)

    _, loss_t = sess.run([train_op, loss], feed_dict)
    return loss_t

def compute_dist_events(m_train_set):
    # compute distribution of each event
    m_mean = np.zeros(8)
    m_std  = np.zeros(8)
    for event_idx in range(1,8):
        counter = []
        for events in m_train_set:
            event_cnt = len(events[events == event_idx])
            counter.append(event_cnt)

        m_mean[event_idx] = np.mean(counter)
        m_std[event_idx]  = np.std(counter) + 10e-3
        
    return m_mean, m_std


# In[2]:



# RNN Autoencoder를 위한 변수 선언
seq_length = n_size
batch_size = 40
vocab_size = 8
embedding_dim = 50
memory_dim = 100
#memory_dim = 16 # for 2d


# In[4]:


with tf.Graph().as_default():
    enc_inp = [tf.placeholder(tf.int32, shape=(None,), name="inp%i" % t) for t in range(seq_length)]
    labels  = [tf.placeholder(tf.int32, shape=(None,), name="labels%i" % t) for t in range(seq_length)]
    weights = [tf.ones_like(labels_t, dtype=tf.float32) for labels_t in labels]
    dec_inp = ([tf.zeros_like(enc_inp[0], dtype=np.int32)] + enc_inp[:-1])
    prev_mem = tf.zeros((batch_size, memory_dim))

    cell = tf.nn.rnn_cell.GRUCell(memory_dim)
    # cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(memory_dim) for _ in range(2)])

    #dec_outputs, dec_memory = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(enc_inp, dec_inp, cell, vocab_size, vocab_size,
    #                                                                      embedding_size=100)
    dec_outputs, dec_memory = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(enc_inp, dec_inp, cell, vocab_size, vocab_size,
                                                                  embedding_size=100, feed_previous=True, num_heads=3)

    loss = tf.contrib.legacy_seq2seq.sequence_loss(dec_outputs, labels, weights, vocab_size)
    learning_rate = 0.05
    momentum = 0.9
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    train_op = optimizer.minimize(loss)

    # saver = tf.train.Saver(max_to_keep=1000)
    anomalies = np.array([['name', 'date', 'loss']])
    userList = pd.read_csv(data_dir + 'dictionary.csv', sep=',')
    userList = userList['user']
    
    
    total_true_positive = 0
    total_true_negative = 0
    total_false_positive = 0
    total_other = 0
    
    for n in range(start_idx, start_idx+bulk, 1):

        user_true_positive = 0
        user_true_negative = 0
        user_false_positive = 0
        user_other = 0
        
        #유저별 파일을 하나씩 가져와서 실행
        file = 'user/' + 'daily_f_u{:03}.csv'.format(n)
        user = pd.read_csv(data_dir + file, sep=',')

        date = user['date']

        print("begin user ", n)
        #필요 없는 column 제거 후 60일간의 trainig set와 전체 일의 test set 구성.
        cols = user.columns.tolist()
        cols = cols[3:-1] + cols[-1:]
        user = np.array(user[cols])
        train_set = user[0:60]
        test_set = user

        m_mean, m_std = compute_dist_events(train_set)
        train_patterns = get_patterns(train_set, n_size)
        
        X = train_patterns
        Y = X[:]
        X = np.array(X, dtype=int).T
        Y = np.array(Y, dtype=int).T
        feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
        feed_dict.update({labels[t]: Y[t] for t in range(seq_length)})
                
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        with tf.Session(config=config) as sess:
            
            # phase1 : training
            sess.run(tf.global_variables_initializer())
                      
            for t in range(501):
                _, loss_train = sess.run([train_op, loss], feed_dict)
                if t % 100 == 0:
                    print("step: {0}  loss: {1:.8f}".format(t, loss_train))
        
            print("user {0}  trained.  loss: {1:.8f}".format(n, loss_train))
            #save_path = saver.save(sess, '/home/minkcho/src/InsiderThreat/saver/saver_' + file + '.ckpt')
            loss_arr = [[loss_train]]
            
            # phase2 : testing
            loss_arr = np.zeros(len(test_set), dtype=float)
            ans_arr  = np.zeros(len(test_set), dtype=int)

            print("Testing...")
            for x in range(len(test_set)):  # for each day 
                seq_len = len(np.array(test_set[x][~np.isnan(test_set[x])]))
                if seq_len >= 4:
                    seq = [test_set[x]]
                    test_patterns = get_patterns(seq, n_size)
                    tX = test_patterns
                    tX = np.array(tX, dtype=int).T
                    tY = tX
                    t_feed_dict = {enc_inp[t]: tX[t] for t in range(seq_length)}
                    t_feed_dict.update({labels[t]: tY[t] for t in range(seq_length)})
                    
                    ploss = sess.run(loss, t_feed_dict)
                    
                    # compute additional loss based on standard deviations
                    s = test_set[x]
                    additional_loss = 0.
                    std_score = np.zeros(8)
                    for event_idx in range(1,8):
                        # std_score[event_idx] = np.rint(len(s[s == event_idx]) - m_mean[event_idx]) / m_std[event_idx]
                        std_score[event_idx] = (len(s[s == event_idx]) - m_mean[event_idx]) / m_std[event_idx]
                        if(std_score[event_idx] >= 2.):
                            additional_loss = additional_loss + (std_score[event_idx] * 0.1)

                    if flag_additional_loss:
                        ploss = ploss + additional_loss
                else:
                    ploss = 0.


                # 0.9 이상이면 이상행동에 추가
                flag_answer = False
                flag_prediction = False
                flag_answer = is_anomaly(date[x], n)

                loss_arr[x] = ploss
                ans_arr[x]  = flag_answer

                if ploss >= 0.9:
                    flag_prediction = True
                
                #if (flag_answer or flag_prediction):
                print('[{0}] DAY {1} : {2:.8f} => (ans,pred) {3}:{4}]'.format(n, date[x], ploss, flag_answer, flag_prediction))
                    
                # compute stat
                if flag_answer and flag_prediction:
                    user_true_positive  = user_true_positive +1
                elif flag_answer:
                    user_true_negative  = user_true_negative +1
                elif flag_prediction:
                    user_false_positive = user_false_positive +1
                else:
                    user_other = user_other+1

        print("user {} : TP{}, TN{}, FP{}, OT{} : Acc({}) Recall({}) DONE".format(n, user_true_positive, user_true_negative, user_false_positive, user_other,
                                                  user_true_positive / (user_true_positive+user_false_positive+1e-6), 
                                                  user_true_positive / (user_true_positive+user_true_negative+1e-6)))

        df_loss = pd.DataFrame({'y':ans_arr, 'loss':loss_arr})
        df_loss.to_csv('loss/{:03}.csv'.format(n),sep=',',index=False)
        # np.savetxt('loss/{:03}.csv'.format(n), [p for p in zip(ans_arr, loss_arr)], delimiter=',')


        total_true_positive = total_true_positive + user_true_positive
        total_true_negative = total_true_negative + user_true_negative
        total_false_positive = total_false_positive + user_false_positive
        total_other = total_other + user_other       

    print("Total TP{}, TN{}, FP{}, OT{} DONE".format(total_true_positive, total_true_negative, total_false_positive, total_other))

