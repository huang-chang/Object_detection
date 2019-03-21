# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

r"""Convert the Oxford pet dataset to TFRecord for object_detection.

See: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
     Cats and Dogs
     IEEE Conference on Computer Vision and Pattern Recognition, 2012
     http://www.robots.ox.ac.uk/~vgg/data/pets/

Example usage:
    ./create_pet_tf_record --data_dir=/home/user/pet \
        --output_dir=/home/user/pet/output
"""
import io
import logging
import os
import re

from lxml import etree
import PIL.Image
import tensorflow as tf

from utils import dataset_util
from utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/data1/object/DATA', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '/data1/object/tf_record', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', '/data1/object/DATA/object_label.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS


def get_class_name_from_filename(file_name):
  """Gets the class name from a file.

  Args:
    file_name: The file name to get the class name from.
               ie. "american_pit_bull_terrier_105.jpg"

  Returns:
    example: The converted tf.Example.
  """
  match = re.match(r'([A-Za-z_]+)(_[0-9]+\.jpg)', file_name, re.I)
  return match.groups()[0]

def dict_to_tf_example(data,label_map_dict,example,ignore_difficult_instances=False):
     try:
         dirname = os.path.dirname(example)
         basename = os.path.basename(example)
         filename = os.path.splitext(basename)[0]
         img_path = '{}.jpg'.format(os.path.join(dirname,filename))
         object_name = img_path.split('/')[-2]
     except:
         print('error')
         return 0
     try:
         if os.path.isfile(img_path):
             with tf.gfile.GFile(img_path, 'rb') as fid:
                 encoded_jpg = fid.read()
             encoded_jpg_io = io.BytesIO(encoded_jpg)
             image = PIL.Image.open(encoded_jpg_io)
             signal = 0
         else:
             signal = 1
     except:
         print('error')
         return 0
     if signal == 0:
         if image.format != 'JPEG':
             print('image format is not jpeg')
             return 0
         else:
             try:
                 width = int(data['size']['width'])
                 height = int(data['size']['height'])
                 xmin = []
                 ymin = []
                 xmax = []
                 ymax = []
                 classes = []
                 classes_text = []
                 for obj in data['object']:
                     xmin.append(float(obj['bndbox']['xmin']) / width)
                     ymin.append(float(obj['bndbox']['ymin']) / height)
                     xmax.append(float(obj['bndbox']['xmax']) / width)
                     ymax.append(float(obj['bndbox']['ymax']) / height)
                     class_name = object_name
                     classes_text.append(class_name)
                     classes.append(label_map_dict[class_name])
             except:
                 return 0
         try:
             example = tf.train.Example(features=tf.train.Features(feature={
             'image/height': dataset_util.int64_feature(height),
             'image/width': dataset_util.int64_feature(width),
             'image/filename': dataset_util.bytes_feature(data['filename'].encode('utf8')),
             'image/source_id': dataset_util.bytes_feature(data['filename'].encode('utf8')),
             'image/encoded': dataset_util.bytes_feature(encoded_jpg),
             'image/format': dataset_util.bytes_feature(b'jpg'),
             'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
             'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
             'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
             'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
             'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
             'image/object/class/label': dataset_util.int64_list_feature(classes),}))
             return example
         except:
             return 0

def create_tf_record(output_filename,
                     label_map_dict,
                     image_dir,
                     examples):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    label_map_dict: The label map dictionary.
    annotations_dir: Directory where annotation files are stored.
    image_dir: Directory where image files are stored.
    examples: Examples to parse and save to tf record.
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  for idx, example in enumerate(examples):
    if idx % 5000 == 0:
        print('processed {} images'.format(idx))
    if idx % 100 == 0:
      logging.info('On image %d of %d', idx, len(examples))
    path = example

    if not os.path.exists(path):
      logging.warning('Could not find %s, ignoring example.', path)
      continue
    try:
        with tf.gfile.GFile(path, 'r') as fid:
          xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    except:
        continue
    try:
      tf_example = dict_to_tf_example(data, label_map_dict,example)
      if tf_example == 0:
        print('the bad image string')
      else:
        writer.write(tf_example.SerializeToString())
    except:
      pass

  writer.close()


# TODO: Add test for pet/PASCAL main files.
def main(_):
  data_dir = FLAGS.data_dir
  print('run the here')
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)


  logging.info('Reading from Pet dataset.')
  image_dir = data_dir
#  annotations_dir = os.path.join(data_dir, 'annotations')
  examples_path = os.path.join(data_dir, 'xml_random.txt')
  examples_list = dataset_util.read_examples_list(examples_path)
#
#  # Test images are not included in the downloaded data set, so we shall perform
#  # our own split.
#  random.seed(42)
#  random.shuffle(examples_list)
  num_examples = len(examples_list)
  num_train = int(1 * num_examples)
  train_examples = examples_list[:num_train]
#  val_examples = examples_list[num_train:]
#  logging.info('%d training and %d validation examples.',
#               len(train_examples), len(val_examples))
#
  train_output_path = os.path.join(FLAGS.output_dir, 'object_train.record')
  print('train_output_path: {}'.format(train_output_path))
  print('image_dir: {}'.format(image_dir))
  
#  val_output_path = os.path.join(FLAGS.output_dir, 'pet_val.record')
  len_train_examples = 'train_examples number: {}'.format(len(train_examples))
  print(len_train_examples)
  create_tf_record(train_output_path, label_map_dict,
                   image_dir, train_examples)
#  create_tf_record(val_output_path, label_map_dict, annotations_dir,
#                   image_dir, val_examples)

if __name__ == '__main__':
  tf.app.run()
