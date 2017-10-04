import numpy as np
import os, sys, getopt
from tqdm import tqdm
import glob
import scipy.io as sio
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
    """
    :return: a list of dict
    """
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

    pickle.dump(image_id2feature, open('data/ai/train_{:d}.pkl'.format(len(image_ids)), 'wb'), -1)


def convert_image_to_vector_val():
    """
    :return: a list of dict --> {image_id: image_feature}
    """
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

    pickle.dump(image_id2feature, open('data/ai/val_{:d}.pkl'.format(len(image_ids)), 'wb'), -1)


if __name__ == "__main__":
    convert_image_to_vector_train()
    convert_image_to_vector_val()
