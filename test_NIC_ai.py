#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2017-10-04 19:52:43
Program: 
Description: 
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pickle
import json
import numpy as np
from tqdm import tqdm
import hashlib
import sys
import jieba

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 2, 'batch size')
flags.DEFINE_integer('img_dim', 1024, 'image feature vector dimension')
flags.DEFINE_integer('embed_dim', 512, 'word embed dimension')
flags.DEFINE_integer('hidden_dim', 512, 'lstm hidden dimension')
flags.DEFINE_integer('num_layer', 3, 'number of lstm layers')
flags.DEFINE_bool('use_gpu_1', True, 'whether to use gpu 1')
FLAGS = flags.FLAGS


class Net(object):
    def __init__(self, use_gpu_1=False, index2token=None):
        self.x1 = tf.placeholder(tf.int32, [None, None], name='x1')  # sentence 15 [SOS, w1, w2, ..., w13, EOS]
        self.x2 = tf.placeholder(tf.float32, [None, FLAGS.img_dim], name='x2')  # image [bs, 1024]
        self.lr = tf.placeholder(tf.float32, [], name='lr')  # lr
        self.kp = tf.placeholder(tf.float32, [], name='kp')  # keep_prob
        self.index2token = index2token
        self.vocab_size = len(index2token)

        with tf.variable_scope('image_fc'):
            fc1 = slim.fully_connected(self.x2, FLAGS.embed_dim, activation_fn=tf.nn.sigmoid, scope='fc1')
            img_input = tf.expand_dims(fc1, 1)  # [bs, 1, 512]
        with tf.variable_scope('sentence_embedding'):
            word_embeddings = tf.get_variable('word_embeddings', shape=[self.vocab_size, FLAGS.embed_dim])
            sent_inputs = tf.nn.embedding_lookup(word_embeddings, self.x1)  # [bs, 15, 512]
            lstm_inputs = tf.concat(1, [img_input, sent_inputs])  # [bs, 16, 512]-->16 [img, SOS, w1, w2, ..., w13, EOS]
        with tf.variable_scope('lstm'):
            cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.hidden_dim)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.kp, output_keep_prob=self.kp)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * FLAGS.num_layer)
            # initial_state = cell.zero_state(FLAGS.batch_size, tf.float32)
            output, _ = tf.nn.dynamic_rnn(cell, lstm_inputs, dtype=tf.float32)  # [bs, 16, 512]-->16 [x, w1, w2, ..., w13, EOS, x]
            output = tf.reshape(output, [-1, FLAGS.hidden_dim])  # [bs*ts, 512]
        with tf.variable_scope('softmax'):
            self.logits = slim.fully_connected(output, self.vocab_size, activation_fn=None, scope='softmax')  # [bs*ts, vs]

        self.predictions = tf.reshape(tf.argmax(self.logits, 1), [FLAGS.batch_size, -1])  # [bs, 16] --> for test
        self.t_vars = tf.trainable_variables()

        # logits for loss
        logits_reshape = tf.reshape(self.logits, [FLAGS.batch_size, -1, self.vocab_size])  # [bs, 16, vs]
        logits_final = logits_reshape[:, 1: -1, :]  # [bs, 14, vs]-->[w1, w2, ..., w13, EOS]
        self.logits_for_loss = tf.reshape(logits_final, [-1, self.vocab_size])  # [bs*14, vs]

        # loss
        target = self.x1[:, 1:]  # remove SOS
        target_reshaped = tf.reshape(target, [-1])  # [bs*14, ]
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_for_loss, labels=target_reshaped))
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

    def generate_caption(self, sess, img_feature):
        # <SOS>: 3074
        # <EOS>: 2824
        img_template = np.zeros([FLAGS.batch_size, FLAGS.img_dim])
        img_template[0, :] = img_feature
        sent_input = np.ones([FLAGS.batch_size, 1]) * 3074  # <SOS>  # [bs, 1]-> [bs, 2]
        while sent_input[0, -1] != 2824 and (sent_input.shape[1] - 1) < 50:
            feed_dicts_t = {self.x1: sent_input, self.x2: img_template, self.kp: 1}
            predicted_total = sess.run(self.predictions, feed_dicts_t)  # [bs, 2]->[bs, 3]
            predicted_next = predicted_total[:, -1:]  # [bs, 1]->[bs, 1]
            sent_input = np.concatenate([sent_input, predicted_next], 1)  # [bs, 2]
        predicted_sentence = ''.join(self.index2token[idx] for idx in sent_input[0, 1: -1])
        return predicted_sentence


def test_train_set():
    print 'Load index2token...'
    with open('data/ai/index2token.pkl', 'r') as f:
        index2token = pickle.load(f)

    print 'Load image vector & captions...'
    with open('data/ai/img_vector_train_210000.pkl', 'r') as f:
        image_id2feature = pickle.load(f)
    ks = [k for k, _ in sorted(image_id2feature.items(), key=lambda item: item[0])]
    vs = [v for _, v in sorted(image_id2feature.items(), key=lambda item: item[0])]

    with open('data/ai/preprocessed_captions_train.pkl', 'r') as f:
        _, caption_id2sentence, caption_id2image_id = pickle.load(f)

    model = Net(use_gpu_1=FLAGS.use_gpu_1, index2token=index2token)
    saver = tf.train.Saver()
    with tf.Session(config=model.tf_config) as sess:
        saver.restore(sess, dir_restore)
        for i in [0, 1, 2, 3, 4]:
            img_vector = vs[i]
            img_id = ks[i]
            print 'image id: ', img_id
            caption_ids = [k for k, v in caption_id2image_id.items() if v == img_id]
            captions = [caption_id2sentence[caption_id] for caption_id in caption_ids]
            print 'gt: '
            for sentence_id in captions:
                sentence = ' '.join(index2token[idx] for idx in sentence_id[1: -1])
                print sentence

            caption = model.generate_caption(sess, img_vector)
            print 'predicted: '
            print caption
            print '=========='


def get_prediction_val(dir_restore, dir_save):
    print 'Load index2token...'
    with open('data/ai/index2token.pkl', 'r') as f:
        index2token = pickle.load(f)

    print 'Load image vector & captions...'
    with open('data/ai/img_vector_val_30000.pkl', 'r') as f:
        image_id2feature = pickle.load(f)
    ks = [k for k, _ in sorted(image_id2feature.items(), key=lambda item: item[0])]
    vs = [v for _, v in sorted(image_id2feature.items(), key=lambda item: item[0])]

    model = Net(use_gpu_1=FLAGS.use_gpu_1, index2token=index2token)
    saver = tf.train.Saver()
    with tf.Session(config=model.tf_config) as sess:
        saver.restore(sess, dir_restore)
        ret_json = []
        for i in tqdm(range(len(ks))):
            ret_one_img = dict()
            img_vector = vs[i]
            img_jpg = ks[i]
            img_jpg_split = img_jpg.split('.')
            ret_one_img['image_id'] = img_jpg_split[0]
            caption = model.generate_caption(sess, img_vector)
            ret_one_img['caption'] = caption
            ret_json.append(ret_one_img)

        with open(dir_save, 'w') as f:
            json.dump(ret_json, f)


def get_reference_val():
    """
    Run it just only once
    :return:
        {"annotations": list --> {"caption", "id", "image_id"}
        "images": list --> {"file_name", "id" = "image_id"}
        "type": "captions"
        "license":
        "info": }
    """
    dir0 = '/media/csc105/Data/dataset/AI-challenger/ai_challenger_caption_validation_20170910/'
    with open(dir0 + 'caption_validation_annotations_20170910.json', 'r') as f:
        data = json.load(f)
    data = sorted(data, key=lambda item: item['image_id'])
    reference = dict()
    annotations = []
    images = []
    count = 1
    for i in tqdm(range(len(data))):
        file_name = data[i]['image_id'].split('.')[0]
        image_hash = int(int(hashlib.sha256(file_name).hexdigest(), 16) % sys.maxint)
        for caption in data[i]['caption']:
            annotation = dict()
            caption_split = jieba.cut(caption.replace('\n', '').strip())
            annotation['caption'] = ' '.join(caption_split)  # notice that caption should be cut first!!!
            annotation['id'] = count
            annotation['image_id'] = image_hash
            annotations.append(annotation)
            count += 1

            image = dict()
            image['file_name'] = file_name
            image['id'] = image_hash
            images.append(image)

    reference['annotations'] = annotations
    reference['images'] = images
    reference['type'] = 'captions'
    reference['licenses'] = 'https://github.com/linjian93'
    info = dict()
    info['contributor'] = 'JiAnge Zhang'
    reference['info'] = info
    with open('test/val_30000_reference.json', 'w') as f:
            json.dump(reference, f)


def get_prediction_test(dir_restore, dir_save):
    print 'Load index2token...'
    with open('data/ai/index2token.pkl', 'r') as f:
        index2token = pickle.load(f)

    print 'Load image vector...'
    with open('data/ai/img_vector_test_30000.pkl', 'r') as f:
        image_id2feature = pickle.load(f)
    ks = [k for k, _ in sorted(image_id2feature.items(), key=lambda item: item[0])]
    vs = [v for _, v in sorted(image_id2feature.items(), key=lambda item: item[0])]

    model = Net(use_gpu_1=FLAGS.use_gpu_1, index2token=index2token)
    saver = tf.train.Saver()
    with tf.Session(config=model.tf_config) as sess:
        saver.restore(sess, dir_restore)
        ret_json = []
        for i in tqdm(range(len(ks))):
            ret_one_img = dict()
            img_vector = vs[i]
            img_jpg = ks[i]
            img_jpg_split = img_jpg.split('.')
            ret_one_img['image_id'] = img_jpg_split[0]
            caption = model.generate_caption(sess, img_vector)
            ret_one_img['caption'] = caption
            ret_json.append(ret_one_img)

        with open(dir_save, 'w') as f:
            json.dump(ret_json, f)


if __name__ == "__main__":
    # dir_restore = 'model/image_caption_ai/20171004_1/model-20'
    dir_restore = 'model/image_caption_ai/20171005_1/model-50'

    # 1. val
    # dir_save = 'test/val_30000_20171005_model_40.json'
    # get_prediction_val(dir_restore, dir_save)

    # 2. test
    dir_save = 'test/ret_20171005_model_50.json'
    get_prediction_test(dir_restore, dir_save)
