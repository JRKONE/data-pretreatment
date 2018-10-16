# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from nets import nets_factory
import argparse
import os.path
import re
import sys
import tarfile
import numpy as np
import numpy as np
from six.moves import urllib
import tensorflow as tf
import pandas as pd
import time
import cv2
from preprocessing import preprocessing_factory

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')
tf.app.flags.DEFINE_string(
    'test_dir', '.', 'Test image directory.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_string(
    'model_name', 'inception_v4', 'The name of the architecture to evaluate.')
FLAGS = tf.app.flags.FLAGS
def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(FLAGS.model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def preprocess_for_eval(image, height, width,
                        central_fraction=0.875, scope=None):
  with tf.name_scope(scope, 'eval_image', [image, height, width]):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    if central_fraction:
      image = tf.image.central_crop(image, central_fraction=central_fraction)

    if height and width:
      # Resize the image to the specified height and width.
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(image, [height, width],
                                       align_corners=False)
      image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image

def main(_):
    imn=[]
    label=[]
  
    with tf.Graph().as_default():
        
   
        network_fn = nets_factory.get_network_fn(FLAGS.model_name,num_classes=5,is_training=False)
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name,is_training=False)
    
        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
          checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
          checkpoint_path = FLAGS.checkpoint_path
        batch_size = 16
        tensor_input = tf.placeholder(tf.float32, [None, 299, 299, 3])
        logits, _ = network_fn(tensor_input)
        logits = tf.nn.top_k(logits, 1)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        rootpath="/home/math10/fl/yunshibie/test"
        test_ids=os.listdir(rootpath)
        tot = len(test_ids)
       
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
          sess.run(tf.global_variables_initializer())
          saver = tf.train.Saver()
          saver.restore(sess, checkpoint_path)
          for idx in range(0, tot, batch_size):
              images = list()
              idx_end = min(tot, idx + batch_size)
              print(idx)
              for i in range(idx, idx_end):
                  image_id = test_ids[i]
                  imn.append(test_ids[i])
                  test_path = os.path.join(FLAGS.test_dir, image_id)
                  
                  image = open(test_path, 'rb').read()
                  image = tf.image.decode_jpeg(image, channels=3)
                  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
                  processed_image = image_preprocessing_fn(image, 299, 299)
                  
                  processed_image = tf.subtract(image, 0.5)
                  processed_image = tf.multiply(processed_image, 2.0)
                  processed_image = sess.run(processed_image)
                  images.append(processed_image)
              images = np.array(images)
              predictions = sess.run(logits, feed_dict = {tensor_input : images}).indices
              label.extend(predictions)
     
    csvpath="/home/math10/fl/yunshibie/"+'test.csv'
    dataframe = pd.DataFrame({'filenames':imn,'labels':label})
    dataframe.to_csv(csvpath,sep=',')    
if __name__ == '__main__':
    tf.app.run()
