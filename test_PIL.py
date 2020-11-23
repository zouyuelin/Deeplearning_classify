# -*- coding: utf-8 -*-

"""
    姓名：邹岳霖\马国语
    学校：天津大学
    专业：机械工程
    学院：机械工程学院
    designer by zouyuelin 
    Student number is :2020201082
    Tianjin University(TJU)
    tel 18222927836
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os
from tensorflow.python.platform import gfile
import cv2 
import matplotlib.pyplot as plt
from skimage import transform

model_path = './model/'
model_name = 'classify.pb'

tf.app.flags.DEFINE_string(
    'model', 'None', 'need meta file.')
tf.app.flags.DEFINE_string(
    'image', 'None', 'type of the training nets.')

FLAGS = tf.app.flags.FLAGS

sess = tf.Session()
#利用模型meta data checkpoint
if(FLAGS.model != 'None'):
    saver = tf.train.import_meta_graph(FLAGS.model)#.meta
    saver.restore(sess,tf.train.latest_checkpoint(model_path))#checkpoint
    graph = tf.get_default_graph()
    x = sess.graph.get_tensor_by_name("Input:0")
    op_to_restore = sess.graph.get_tensor_by_name("predictions/Softmax:0")
    is_training = sess.graph.get_tensor_by_name("is_training:0")
#利用模型pb文件
else:
    with gfile.FastGFile(model_path+model_name,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def,name='')

    x = sess.graph.get_tensor_by_name("Input:0")
    op_to_restore = sess.graph.get_tensor_by_name("predictions/Softmax:0")
    is_training = sess.graph.get_tensor_by_name("is_training:0")

if(FLAGS.image != 'None'):
    img = Image.open(FLAGS.image)
    image = img.resize((224,224))
    image = np.array(image)*(1./255)
    #img = np.array(Image.open(FLAGS.image))*(1./255)
    #image = transform.resize(img,(224,224,3))
    img = np.reshape(image,(1,224,224,3))
else:
    os._exit(0)
print(img)
feed_dict ={x:img,is_training:False}

result_index = sess.run(op_to_restore,feed_dict)
print("-----------------------------predict:---------------------------\n")
print(result_index)

if(np.argmax(result_index)==0):
    print("It is a cat")
else:
    print("It is a dog")

plt.figure("animals")
plt.imshow(image)
plt.show()

