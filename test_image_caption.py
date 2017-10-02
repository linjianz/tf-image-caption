#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Creat Time: 2017-10-02 14:27:57
Program: 
Description: 
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Creat Time: 2017-09-26 16:44:48
Program:
Description:
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import random
import os
import shutil
import time
import pickle
import numpy as np

flags = tf.app.flags
flags.DEFINE_string('dir0', '20170930', 'directory name to save the model/log')
flags.DEFINE_string('net_name', 'image_caption/', 'project name')
flags.DEFINE_integer('batch_size', 256, 'batch size')
flags.DEFINE_integer('epoch_max', 100, 'max epoch number')
flags.DEFINE_integer('epoch_save', 10, 'save model every # epoch')
flags.DEFINE_float('lr_base', 1e-3, 'init learning rate')
flags.DEFINE_integer('epoch_lr_decay', 500, 'every # epoch, lr decay 0.1')
flags.DEFINE_integer('img_dim', 1024, 'image feature vector dimension')
flags.DEFINE_integer('embed_dim', 512, 'word embed dimension')
flags.DEFINE_integer('hidden_dim', 512, 'lstm hidden dimension')
flags.DEFINE_integer('num_layer', 3, 'number of lstm layers')
flags.DEFINE_bool('use_gpu_1', False, 'whether to use gpu 1')
FLAGS = flags.FLAGS

dir_restore = '../model/'
dir_data = '../data/'
dir_models = 'model/' + FLAGS.net_name
dir_logs = 'log/' + FLAGS.net_name
dir_model = dir_models + FLAGS.dir0
dir_log_train = dir_logs + FLAGS.dir0 + '_train'
dir_log_val = dir_logs + FLAGS.dir0 + '_val'
if not os.path.exists(dir_models):
    os.mkdir(dir_models)
if not os.path.exists(dir_logs):
    os.mkdir(dir_logs)
if os.path.exists(dir_model):
    shutil.rmtree(dir_model)
if os.path.exists(dir_log_train):
    shutil.rmtree(dir_log_train)
if os.path.exists(dir_log_val):
    shutil.rmtree(dir_log_val)

os.mkdir(dir_model)
os.mkdir(dir_log_train)
os.mkdir(dir_log_val)
########################################


def load_data():
    with open('data/index2token.pkl','r') as f:
        index2token = pickle.load(f)
    with open('data/preprocessed_train_captions.pkl','r') as f:
        train_captions, train_caption_id2sentence, train_caption_id2image_id = pickle.load(f)
    with open('data/train_82783.pkl','r') as f:
        train_image_id2feature = pickle.load(f)

    return index2token, train_captions, train_caption_id2sentence, train_caption_id2image_id, train_image_id2feature


def get_batches(train_captions, bs=FLAGS.batch_size):
    train_batches = []
    for sent_length, caption_set in train_captions.items():
        caption_set = list(caption_set)
        random.shuffle(caption_set)
        num_captions = len(caption_set)
        num_batches = num_captions // bs
        for i in range(num_batches+1):
            end_idx = min((i+1)*bs, num_captions)
            new_batch = caption_set[(i*bs):end_idx]
            if len(new_batch) == bs:  # omit the tail data whose length less than a batch
                train_batches.append((new_batch, sent_length))
    random.shuffle(train_batches)
    return train_batches, len(train_batches)


def generate_batch(batches, caption_id2sentence, caption_id2image_id, image_id2feature):
    bs = FLAGS.batch_size
    img_dim = FLAGS.img_dim
    for batch_item in batches:
        (caption_ids, sent_length) = batch_item  # caption_ids: list
        num_captions = len(caption_ids)  # 256
        sentences = np.array([caption_id2sentence[k] for k in caption_ids])  # [256, 15]
        images = np.array([image_id2feature[caption_id2image_id[k]] for k in caption_ids])  # [256, 1024]
        targets = sentences[:, 1:]  # [256, 14]

        sentences_template = np.zeros([bs, sent_length])  # [256, 15]
        images_template = np.zeros([bs, img_dim])  # [256, 1024]
        targets_template = 3591*np.ones([bs, sent_length])  # [256, 16] -> [-1, w1, w2, ..., wn, -1]

        sentences_template[range(num_captions), :] = sentences
        images_template[range(num_captions), :] = images
        targets_template[range(num_captions), 1:sent_length] = targets
        # assert (targets_template[:, [0, -1]] == -1).all()  # front and back should be padded with -1

        yield sentences_template, images_template, targets_template


class Net(object):
    def __init__(self, use_gpu_1=False, vocab_size=None):
        self.x1 = tf.placeholder(tf.int32, [None, None], name='x1')  # sentence [bs, 15]
        self.x2 = tf.placeholder(tf.float32, [None, FLAGS.img_dim], name='x2')  # image vector [bs, 1024]
        self.x3 = tf.placeholder(tf.int32, [None, None], name='x3')  # target  [bs, 16]
        self.lr = tf.placeholder(tf.float32, [], name='lr')  # lr
        self.kp = tf.placeholder(tf.float32, [], name='kp')  # keep_prob
        self.vocab_size = vocab_size

        with tf.variable_scope('image_fc'):
            fc1 = slim.fully_connected(self.x2, FLAGS.embed_dim, activation_fn=tf.nn.sigmoid, scope='fc1')
            img_input = tf.expand_dims(fc1, 1)  # [bs, 1, 512]
        with tf.variable_scope('sentence_embedding'):
            word_embeddings = tf.get_variable('word_embeddings', shape=[self.vocab_size, FLAGS.embed_dim])
            sent_inputs = tf.nn.embedding_lookup(word_embeddings, self.x1)  # [bs, 15, 512]
            lstm_inputs = tf.concat(1, [img_input, sent_inputs])  # [bs, 16, 512]
        with tf.variable_scope('lstm'):
            cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.hidden_dim)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.kp, output_keep_prob=self.kp)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * FLAGS.num_layer)
            # initial_state = cell.zero_state(FLAGS.batch_size, tf.float32)
            output, _ = tf.nn.dynamic_rnn(cell, lstm_inputs, dtype=tf.float32)  # [bs, steps, 512]
            output = output[:, 0:-1, :]
            output = tf.reshape(output, [-1, FLAGS.hidden_dim])  # [?, 512]
        with tf.variable_scope('softmax'):
            logits = slim.fully_connected(output, self.vocab_size, activation_fn=None, scope='softmax')

        self.t_vars = tf.trainable_variables()
        # self.variables_names = [v.name for v in self.t_vars]  #  turn on if you want to check the variables

        # loss
        x3_reshaped = tf.reshape(self.x3, [-1])  # bs*16
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=x3_reshaped))
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)

        # tensor board
        loss_summary = tf.summary.scalar('loss', self.loss)
        self.summary_merge = tf.summary.merge([loss_summary])

        # gpu configuration
        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True
        if use_gpu_1:
            self.tf_config.gpu_options.visible_device_list = '1'

        # last but no least, get all the variables
        self.init_all = tf.initialize_all_variables()

    def generate_caption(self, session, img_feature):
        img_feature = np.zeros([])
        sent_pred = np.ones([FLAGS.batch_size, 1]) * 3591  # <SOS>
        while sent_pred[0, -1] != 3339 and (sent_pred.shape[1] - 1) < 50:
            feed_dicts_t = {self.x1: sent_pred, model.x2: x2_t, model.x3: x3_t, model.lr: lr, model.kp: 0.75}


def main(_):
    index2token, captions, caption_id2sentence, caption_id2image_id, image_id2feature = load_data()
    batches, len_batches = get_batches(captions)
    print 'vocab size: ', len(index2token)
    model = Net(use_gpu_1=FLAGS.use_gpu_1, vocab_size=len(index2token))
    saver = tf.train.Saver(max_to_keep=FLAGS.epoch_max // FLAGS.epoch_save)
    with tf.Session(config=model.tf_config) as sess:
        writer_train = tf.summary.FileWriter(dir_log_train, sess.graph)
        # writer_val = tf.summary.FileWriter(dir_log_val, sess.graph)

        # 1. train from scratch
        sess.run(model.init_all)

        # 2. restore
        # model.saver.restore(sess, dir_restore)
        global_iter = 0
        for epoch in range(FLAGS.epoch_max):
            lr_decay = 0.1 ** (epoch / FLAGS.epoch_lr_decay)
            lr = FLAGS.lr_base * lr_decay
            for iteration, (x1_t, x2_t, x3_t) in enumerate(generate_batch(batches, caption_id2sentence,
                                                                          caption_id2image_id, image_id2feature)):
                time_start = time.time()
                feed_dicts_t = {model.x1: x1_t, model.x2: x2_t, model.x3: x3_t, model.lr: lr, model.kp: 0.75}
                sess.run(model.train_op, feed_dicts_t)

                # display
                if not (iteration+1) % 1:
                    merged_out_t, loss_out_t = sess.run([model.summary_merge, model.loss], feed_dicts_t)
                    writer_train.add_summary(merged_out_t, global_iter+1)
                    hour_per_epoch = len_batches * ((time.time() - time_start) / 3600)
                    print('%.2f h/epoch, epoch %03d/%03d, iter %04d/%04d, lr %.5f, loss: %.5f' %
                          (hour_per_epoch, epoch+1, FLAGS.epoch_max, iteration+1, len_batches, lr, loss_out_t))

                # if not (iteration+1) % 10:
                #     feed_dicts_v = {model.x1: x1_v, model.x2: x2_v, model.kp: 1.0}
                #     merged_out_v, loss_out_v = sess.run([model.summary_merge, model.loss], feed_dicts_v)
                #     writer_val.add_summary(merged_out_v, global_iter + 1)
                #     print('****val loss**** {:.5f}'.format(loss_out_v))

                # save
                if not (epoch+1) % FLAGS.epoch_save:
                    saver.save(sess, (dir_model + '/model'), global_step=epoch+1)

                global_iter += 1


if __name__ == "__main__":
    tf.app.run()
