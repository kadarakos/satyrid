# Derived from https://github.com/jazzsaxmafia/show_attend_and_tell.tensorflow/blob/master/make_flickr_dataset.py

import numpy as np
import os
import cPickle
from extract_conv_feats import *
import string
exclude = string.punctuation + "-"
from collections import defaultdict
from scipy.sparse import csr_matrix
import tables
import json
import argparse

# Change these to point to Caffe on your syste
VGG_MODEL = '/home/delliott/src/caffe/models/vgg_19/VGG_ILSVRC_19_layers.caffemodel'
VGG_DEPLOY = '/home/delliott/src/caffe/models/vgg_19/VGG_ILSVRC_19_layers_deploy.prototxt'

w_dict = defaultdict(int)
cnn = CNN(model=VGG_MODEL, deploy=VGG_DEPLOY, width=224, height=224,
          batch_size=100)

def get_cnn_features(image_list, split, batch_size, relu=False):
    hdf5_path = "%s-%s" % (split, "cnn_features.hdf5")
    hdf5_file = tables.open_file(hdf5_path, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    data_storage = hdf5_file.createEArray(hdf5_file.root, 'feats',
                                          tables.Float32Atom(),
                                          shape=(0, 100352),
                                          filters=filters,
                                          expectedrows=len(image_list))
    for start, end in zip(range(0, len(image_list)+batch_size, batch_size),
                          range(batch_size, len(image_list)+batch_size, batch_size)):
        print("Processing %s images %d-%d / %d" 
              % (split, start, end, len(image_list)))
        batch_list = image_list[start:end]
        feats = cnn.get_features(batch_list, layers='conv5_4', layer_sizes=[512,14,14])
        # transpose and flatten feats to prepare for reshape(14*14, 512)
        feats = np.array(map(lambda x: x.T.flatten(), feats))
        if relu:
            feats = np.clip(feats, a_min=0., a_max=np.inf, out=feats) # RELU
        data_storage.append(feats)
    print("Finished processing %d images" % len(data_storage))
    hdf5_file.close()

def get_data_by_split(json_data, split):
    data = dict()
    data['sents'] = []
    data['files'] = []

    num = 0
    # loop over the data in the JSON file and-
    for img in json_data['images']:
        data_split = img['split']
        for sent in img['sentences']:
            text = sent['raw']
            text = text.replace('\n','')
            text = text.lower()
            text = ''.join(ch for ch in text if ch not in exclude)
            text = text.strip()
            if data_split == split:
                data['sents'].append((text, num))
            if split == "train":
                # we only collect the vocabulary over the training data
                for w in text.split():
                    w_dict[w] += 1
        if data_split == split:
            data['files'].append(img['filename'])
            num += 1

    print("%s: collected %d sents %d images"
          % (split, len(data['sents']), len(data['files'])))
    return data

def process_json(json_path):
    json_data = json.load(open(json_path))

    data = dict()
    data['train'] = get_data_by_split(json_data, 'train')
    data['val'] = get_data_by_split(json_data, 'val')
    data['test'] = get_data_by_split(json_data, 'test')

    with open('train.pkl', 'wb') as f:
        cPickle.dump(data['train']['sents'], f, protocol=2)
    with open('dev.pkl', 'wb') as f:
        cPickle.dump(data['val']['sents'], f, protocol=2)
    with open('test.pkl', 'wb') as f:
        cPickle.dump(data['test']['sents'], f, protocol=2)

    return data

def make_dataset(args):
    # get the filenames of the images in the JSON file
    data = process_json(args.json_path)
    for split in data:
        files = ['%s/%s' % (args.images_path, x) for x in data[split]['files']]
        get_cnn_features(files, split, args.batch_size, relu=args.relu)

    # Sort dictionary in descending order
    sorted_dict = sorted(w_dict, key=lambda x: w_dict[x], reverse=True)
    # Start at 2 because 0 and 1 are reserved
    numbered_dict = [(w, idx+2) for idx,w in enumerate(sorted_dict)]
    d_dict = dict(numbered_dict)

    with open('dictionary.pkl', 'wb') as f:
        cPickle.dump(d_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create the dataset bundles to train or test a model")

    parser.add_argument("--splits", default="train,dev,test",
                        help="Comma-separated list of the splits to process")
    parser.add_argument("--relu", action="store_true",
                        help="ReLU the extracted visual features?")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Minibatch size for processing images")
    parser.add_argument("--images_path", type=str,
                        help="Path to the directory containing the images",
                        default="data/flickr30k/")
    parser.add_argument("--json_path", type=str,
                        help="Path to the JSON file to build the bundles",
                        default="data/flickr30k")

    arguments = parser.parse_args()

    make_dataset(arguments)
