from __future__ import absolute_import

import os
import logging
import re

import tensorflow as tf

from six import b


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def generate(annotations_path, output_path, log_step=5000,
             force_uppercase=False, save_filename=False):

    logging.info('Building a dataset from %s.', annotations_path)
    logging.info('Output file: %s', output_path)

    writer = tf.python_io.TFRecordWriter(output_path)
    longest_label = ''
    idx = 0

    with open(annotations_path, 'r') as annotations:
        for idx, line in enumerate(annotations):
            line = line.rstrip('\n')

            # Split the line on the first whitespace character and allow empty values for the label
            # NOTE: this does not allow whitespace in image paths
            line_match = re.match(r'([^,]*),(.*)', line)
            #line_match = re.match(r'([^:]*):(.*)', line)
            #line_match = re.match(r'([^:]*):(.*):', line)
            if line_match is None:
                logging.error('missing filename or label, ignoring line %i: %s', idx+1, line)
                continue
            (img_path, label) = line_match.groups()
            base_dir = os.path.dirname(annotations_path)
            #base_dir = '/dl_data/arthur/tempholdings_test_set/'
            #base_dir = '/dl_data/arthur/half-kana-test-set/'
            #base_dir = '/dl_data/arthur/open-num-padding/'
            #base_dir = './out/'
            #base_dir = './'
            #base_dir = '/dl_data/arthur/numbers-training-parentheses-2018-12-21/'
            #base_dir = '/dl_data/arthur/aocr-training-numbers-2019-02-19_2/'
            #base_dir = '/dl_data/arthur/aocr-training-numbers-2019-02-21_1/'
            #base_dir = '/dl_data/arthur/aocr-training-numbers-2019-02-21_2/'
            #base_dir = '/dl_data/arthur/easy/'
            #base_dir = '/dl_data/arthur/half-width-katakana-training-2018-12-03/'
            #base_dir = './out-2018-11-16/'
            #base_dir = './out-small-numbers-training-2018-11-16/'
            #base_dir = './liusample/'
            #base_dir = '/dl_data/arthur/numbers-training-2018-10-22/'
            #base_dir = '/dl_data/arthur/random_crop_dataset_2019-03-01/'
            #base_dir = '/dl_data/arthur/random_crop_dataset_2019-03-07/'
            #base_dir = '/dl_data/arthur/random_crop_dataset_2019-03-08/'

            img_path = os.path.join(base_dir,img_path) 

            with open(img_path, 'rb') as img_file:
                img = img_file.read()

            if force_uppercase:
                label = label

            if len(label) > len(longest_label):
                longest_label = label
                print(longest_label)

            feature = {}
            feature['image'] = _bytes_feature(img)
            feature['label'] = _bytes_feature(b(label))
            if save_filename:
                feature['comment'] = _bytes_feature(b(img_path))

            example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())

            #print(example.SerializeToString())
            if idx % log_step == 0:
                logging.info('Processed %s pairs.', idx+1)

    if idx:
        logging.info('Dataset is ready: %i pairs.', idx+1)
        logging.info('Longest label (%i): %s', len(longest_label), longest_label)

    writer.close()
