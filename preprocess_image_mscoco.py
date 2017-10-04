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

caffe.set_mode_cpu()
net = caffe.Classifier(model_prototxt, model_trained,
                       mean=np.load(mean_path).mean(1).mean(1),
                       channel_swap=(2, 1, 0),
                       raw_scale=255,
                       image_dims=(256, 256))
# print [(k, v.data.shape) for k, v in net.blobs.items()]

# Loading class labels
with open(imagenet_labels) as f:
    labels = f.readlines()

dir0 = '/media/csc105/Data/dataset/ms-coco/'


def forward_cnn(image_path):
    image_path = image_path.strip()
    input_image = caffe.io.load_image(image_path)
    prediction = net.predict([input_image], oversample=False)
    print os.path.basename(image_path), ' : ', labels[prediction[0].argmax()].strip(), ' (', prediction[0][prediction[0].argmax()], ')'
    image_vector = net.blobs[layer_name].data[0].reshape(1, -1)
    return image_vector


def transform_train():
    img_list = glob.glob(dir0 + 'train2014/*.jpg')
    img_list.sort()
    image2vector = np.zeros([len(img_list), 1024])
    print 'images of train: ', len(img_list)
    for i, image_path in enumerate(tqdm(img_list)):
        input_image = caffe.io.load_image(image_path)
        net.predict([input_image], oversample=False)
        image2vector[i, :] = net.blobs[layer_name].data[0].reshape(1024)
    sio.savemat('data/train.mat', {'1': image2vector})


def transform_val():
    img_list = glob.glob(dir0 + 'val2014/*.jpg')
    img_list.sort()
    image2vector = np.zeros([len(img_list), 1024])
    print 'images of val: ', len(img_list)
    for i, image_path in tqdm(enumerate(img_list)):
        input_image = caffe.io.load_image(image_path)
        prediction = net.predict([input_image], oversample=False)
        # print os.path.basename(image_path), ' : ', labels[prediction[0].argmax()].strip(), ' (', prediction[0][prediction[0].argmax()], ')'
        image2vector[i, :] = net.blobs[layer_name].data[0].reshape(1024)
    sio.savemat('data/val.mat', {'1': image2vector})


def transform_train_pickle():
    with open(dir0 + 'annotations/captions_train2014.json', 'r') as f:
        data = json.load(f)
    image_ids = set()  # 40504
    for caption_data in data['annotations']:
        image_id = caption_data['image_id']
        image_ids.add(image_id)

    image_id2feature = dict()
    for image_id in tqdm(sorted(image_ids)):
        input_image = caffe.io.load_image(dir0 + 'train2014/COCO_train2014_' + str("{0:012d}".format(image_id)+'.jpg'))
        prediction = net.predict([input_image], oversample=False)
        image_vector = np.array(net.blobs[layer_name].data[0]).reshape([-1])  # remember transfer into array first!
        image_id2feature[image_id] = image_vector

    pickle.dump(image_id2feature, open('data/train_{:d}.pkl'.format(len(image_ids)), 'wb'), -1)


def transform_val_pickle():
    with open(dir0 + 'annotations/captions_val2014.json', 'r') as f:
        data = json.load(f)
    image_ids = set()  # 40504
    for caption_data in data['annotations']:
        image_id = caption_data['image_id']
        image_ids.add(image_id)

    image_id2feature = dict()
    for image_id in tqdm(sorted(image_ids)):
        input_image = caffe.io.load_image(dir0 + 'val2014/COCO_val2014_' + str("{0:012d}".format(image_id)+'.jpg'))
        prediction = net.predict([input_image], oversample=False)
        image_vector = np.array(net.blobs[layer_name].data[0]).reshape([-1])  # remember transfer into array first!
        image_id2feature[image_id] = image_vector

    pickle.dump(image_id2feature, open('data/val_{:d}.pkl'.format(len(image_ids)), 'wb'), -1)


if __name__ == "__main__":
    # transform_train_pickle()
    image_vector = forward_cnn('images/pizza.jpg')
    pickle.dump(image_vector, open('data/image-features-test/pizza.pkl', 'wb'), -1)

