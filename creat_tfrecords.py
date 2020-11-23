import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import os
from PIL import Image
import random

objects = ['cat','dog']#'cat'0,'dog'1
#path='../../SecondProject/dog_and_cat_200'

train_percent = 0.90

filename_train="./data/train.tfrecords"
filename_test="./data/test.tfrecords"
writer_train= tf.python_io.TFRecordWriter(filename_train)
writer_test= tf.python_io.TFRecordWriter(filename_test)

tf.app.flags.DEFINE_string(
    'data', 'None', 'where the datas?.')
FLAGS = tf.app.flags.FLAGS

if(FLAGS.data == None):
    os._exit(0)

data_path = FLAGS.data
total = os.listdir(data_path)
num = len(total)
list_ = range(num)
train_num = int(num * train_percent)
test_num = num-train_num

train_index = random.sample(list_,train_num)

dim = (224,224)
object_path = data_path
for index in list_:
    if index in train_index:
        img_path=os.path.join(object_path,total[index])
        print(img_path)
        img=Image.open(img_path)
        img=img.resize(dim)
        img_raw=img.tobytes()
        if 'cat' in total[index]:
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
            index_ = 0
        elif'dog' in total[index]:
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
            index_ = 1
        writer_train.write(example.SerializeToString())  #序列化为字符串
        print([index_,total[index]])
    else:
        img_path=object_path+total[index]
        print(img_path)
        img=Image.open(img_path)
        img=img.resize(dim)
        img_raw=img.tobytes()
        if 'cat' in total[index]:
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
            index_ = 0
        elif'dog' in total[index]:
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
            index_ = 1
        writer_test.write(example.SerializeToString())  #序列化为字符串
        print([index_,total[index]])
writer_train.close()
writer_test.close()
