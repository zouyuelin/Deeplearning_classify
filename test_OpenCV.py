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

import cv2
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
import os


model_path = './model/'
model_name = 'classify.pb'
tf.app.flags.DEFINE_string(
    'image', 'None', 'type of the training nets.')
tf.app.flags.DEFINE_string(
    'model', 'None', 'need meta file.')

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
    img = cv2.imread(FLAGS.image)
    dst = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)#
    rgb = cv2.cvtColor(dst,cv2.COLOR_BGR2RGB)
else:
    os._exit(0)
	
gb_ = np.array(rgb)*(1./255)
gb = gb_.reshape(1,224,224,3)

print(gb)
feed_dict ={x:gb,is_training:False}

result_index = sess.run(op_to_restore,feed_dict)

cv2.imshow('animals',dst)
cv2.waitKey(1000)
print("-----------------------------predict:---------------------------\n")
print(result_index)


if(np.argmax(result_index)==0):
    print("It is a cat")
else:
    print("It is a dog")


'''#图像测试模型
image_total = os.listdir(images_path)
for index_animals in image_total:
    src = cv2.imread(images_path+index_animals)
    dst = cv2.resize(src,(224,224),interpolation=cv2.INTER_CUBIC)#
    cv2.imshow('animals',dst)
    cv2.waitKey(300)
    print('this is %s'%(index_animals),end=' ')
    rgb = cv2.cvtColor(dst,cv2.COLOR_BGR2RGB)
    gb = rgb.reshape(1,224,224,3)
    feed_dict ={x_datas:rgb}
    test_output = sess.run([predictions],feed_dict=feed_dict)
    index_max = np.argmax(test_output)
    if(index_max == 0):
        print('prediction:cat\n')
    else:
        print('prediction:dog\n')
cv2.dnn.readNetFromTensorflow('model/classify.pb')'''