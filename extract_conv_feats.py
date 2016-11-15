# Derived from https://github.com/asampat3090/anandlib/blob/master/dl/caffe_cnn.py
# Modifications:
#  * accepts a batch size for the CNN
#  * fixed value for mean BGR (as recommended by Simonyan and Zisserman)
#  * handles black & white images

import sys
sys.path.append("/home/delliott/src/caffe/python")
import caffe
import numpy as np
import skimage
import skimage.transform
import scipy

class CNN(object):

    def __init__(self, deploy, model, batch_size=100, width=224, height=224):

        self.deploy = deploy
        self.model = model

        self.batch_size = batch_size
        self.net, self.transformer = self.get_net()
        self.net.blobs['data'].reshape(self.batch_size, 3, height, width)

        self.width = width
        self.height = height

    def get_net(self):
        caffe.set_mode_gpu()
        net = caffe.Net(self.deploy, self.model, caffe.TEST)

        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        # mean from https://gist.github.com/ksimonyan/3785162f95cd2d5fee77
        mean_value = np.array([103.939, 116.779, 123.68])

        transformer.set_mean('data', mean_value)
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2,1,0))
        return net, transformer

    def get_features(self, image_list, layers='fc7', layer_sizes=[4096]):
        iter_until = len(image_list) + self.batch_size
        all_feats = np.zeros([len(image_list)] + layer_sizes)

        batch_count = 0
        for start, end in zip(range(0, iter_until, self.batch_size), \
                              range(self.batch_size, iter_until, self.batch_size)):

            image_batch_file = image_list[start:end]
            image_batch = np.array(map(lambda x: self.crop_image(x, target_width=self.width, target_height=self.height), image_batch_file))

            caffe_in = np.zeros(np.array(image_batch.shape)[[0,3,1,2]], dtype=np.float32)

            for idx, in_ in enumerate(image_batch):
                caffe_in[idx] = self.transformer.preprocess('data', in_)

            out = self.net.forward_all(blobs=[layers], **{'data':caffe_in})
            feats = out[layers]

            all_feats[start:end] = feats
            batch_count += 1
        return all_feats


def crop_image(self, image_path, target_height=224, target_width=224):
        """Reshape image shorter and crop, keep aspect ratio."""
        image = skimage.img_as_float(skimage.io.imread(image_path)).astype(np.float32)

        if len(image.shape) == 2:
            image = np.tile(image[:, :, np.newaxis], (1, 1, 3))
        height, width, rgb = image.shape

        if width == height:
            resized_image = skimage.transform.resize(image,
                                                     (target_height,
                                                      target_width))
        elif height < width:
            w = int(width * float(target_height)/height)
            resized_image = skimage.transform.resize(image, (target_height, w))
            crop_length = int((w - target_width) / 2)
            resized_image = resized_image[:, crop_length:crop_length+target_width]

        else:
            h = int(height * float(target_width) / width)
            resized_image = skimage.transform.resize(image, (h, target_width))
            crop_length = int((h - target_height) / 2)
            resized_image = resized_image[crop_length:crop_length+target_height, :]
        return resized_image
