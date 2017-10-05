#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2017-10-05 14:25:39
Program: convert image to vector
Description:
Convert all the images in train/validation/test set to vectors.
Use the pre-trained model of googlenet by caffe.
the result is saved as a list of dict, such as {image_id: image_vector}
"""
import numpy as np
import os
import sys
from tqdm import tqdm
import json
import pickle

caffe_root = '/home/jiange/dl/caffe-master/'
model_prototxt = caffe_root + 'models/bvlc_googlenet/deploy.prototxt'
model_trained = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'

# File containing the class labels
imagenet_labels = caffe_root + 'data/ilsvrc12/synset_words.txt'

# Path to the mean image (used for input processing)
mean_path = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'

# Name of the layer we want to extract
layer_name = 'pool5/7x7_s1'

sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu()
net = caffe.Classifier(model_prototxt, model_trained,
                       mean=np.load(mean_path).mean(1).mean(1),
                       channel_swap=(2, 1, 0),
                       raw_scale=255,
                       image_dims=(256, 256))
# print [(k, v.data.shape) for k, v in net.blobs.items()]

# Loading class labels
with open(imagenet_labels) as f:
    labels = f.readlines()


def forward_cnn(image_path):
    image_path = image_path.strip()
    input_image = caffe.io.load_image(image_path)
    prediction = net.predict([input_image], oversample=False)
    print os.path.basename(image_path), ' : ', labels[prediction[0].argmax()].strip(), ' (', prediction[0][prediction[0].argmax()], ')'
    image_vector = net.blobs[layer_name].data[0].reshape(1, -1)
    return image_vector


def convert_image_to_vector_train():
    dir0 = '/media/csc105/Data/dataset/AI-challenger/ai_challenger_caption_train_20170902/'
    with open(dir0 + 'caption_train_annotations_20170902.json', 'r') as f:
        data = json.load(f)

    image_ids = [data[i]['image_id'] for i in range(len(data))]

    image_id2feature = dict()
    for image_id in tqdm(sorted(image_ids)):
        input_image = caffe.io.load_image(dir0 + 'caption_train_images_20170902/' + image_id)
        prediction = net.predict([input_image], oversample=False)
        image_vector = np.array(net.blobs[layer_name].data[0]).reshape([-1])  # remember transfer into array first!
        image_id2feature[image_id] = image_vector

    pickle.dump(image_id2feature, open('data/ai/img_vector_train_{:d}.pkl'.format(len(image_ids)), 'wb'), -1)


def convert_image_to_vector_val():
    dir0 = '/media/csc105/Data/dataset/AI-challenger/ai_challenger_caption_validation_20170910/'
    with open(dir0 + 'caption_validation_annotations_20170910.json', 'r') as f:
        data = json.load(f)

    image_ids = [data[i]['image_id'] for i in range(len(data))]

    image_id2feature = dict()
    for image_id in tqdm(sorted(image_ids)):
        input_image = caffe.io.load_image(dir0 + 'caption_validation_images_20170910/' + image_id)
        prediction = net.predict([input_image], oversample=False)
        image_vector = np.array(net.blobs[layer_name].data[0]).reshape([-1])  # remember transfer into array first!
        image_id2feature[image_id] = image_vector

    pickle.dump(image_id2feature, open('data/ai/img_vector_val_{:d}.pkl'.format(len(image_ids)), 'wb'), -1)


def convert_image_to_vector_test():
    dir0 = '/media/csc105/Data/dataset/AI-challenger/ai_challenger_caption_test1_20170923/caption_test1_images_20170923/'
    image_path = os.listdir(dir0)
    image_path.sort()
    image_id2feature = dict()
    for image_id in tqdm(image_path):
        input_image = caffe.io.load_image(dir0 + image_id)
        prediction = net.predict([input_image], oversample=False)
        image_vector = np.array(net.blobs[layer_name].data[0]).reshape([-1])  # remember transfer into array first!
        image_id2feature[image_id] = image_vector

    pickle.dump(image_id2feature, open('data/ai/img_vector_test_{:d}.pkl'.format(len(image_path)), 'wb'), -1)


if __name__ == "__main__":
    # convert_image_to_vector_train()
    # convert_image_to_vector_val()
    convert_image_to_vector_test()
