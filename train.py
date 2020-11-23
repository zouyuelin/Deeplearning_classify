# -*- coding: utf-8 -*-

"""
    姓名：邹岳霖,马国语
    学校：天津大学
    专业：机械工程
    学院：机械工程学院
    designer by zouyuelin & maguoyu
    Tianjin University(TJU)
    tel 18222927836
"""

import tensorflow as tf
import numpy as np
import vgg16
import resnet_v2
from PIL import Image
import os
import cv2
import time

slim = tf.contrib.slim
vggnet = vgg16.vgg16_WithoutArgvs
images_path = '../../SecondProject/data/'
tfrecords_path = './data/train.tfrecords'#train_200.tfrecords
test_path = './data/test.tfrecords'#test_200.tfrecords
model_path = './model/model.ckpt'
log_path = './log/log'
pb_path = './model/classify.pb'
pbtxt_path = './model/'
labels = ['cat','dog']

#set the nets' argvs
#相关参数，学习率，衰减率
batch_size = 64
num_steps = 100
LEARNING_RATE_BASE = 0.003
LRARNING_RATE_DECAY = 0.96
dataset_num = 25000
decay_step = 300
num_classes = 2
dim = (224,224)


tf.app.flags.DEFINE_string(
    'typenets', 'resnet101', 'type of the training nets.')

FLAGS = tf.app.flags.FLAGS
    
#读取tfrecord
def read_and_decode(tfrecords_path):
    filename_queue = tf.train.string_input_producer([tfrecords_path],shuffle=True) 
    reader = tf.TFRecordReader()
    _,  serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={
        'label': tf.FixedLenFeature([], tf.int64),
        'img_raw' : tf.FixedLenFeature([], tf.string),})

    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image,[224,224,3])#reshape 200*200*3
    image = tf.cast(image,tf.float32)*(1./255)#image张量可以除以255，*(1./255)
    label = tf.cast(features['label'], tf.int32)
    return image,label

#定义one_hot函数,int = 1 表示孔标签
def one_hot(labels,Label_class):
    one_hot_label = np.array([[int(i == int(labels[j])) for i in range(Label_class)] for j in range(len(labels))])   
    return one_hot_label

#加载图
def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open(model_file,"rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph

#set the forward nets
#较为复杂的前向传播
def forward(datas_train,is_training):
    if FLAGS.typenets == 'vggnet16':
        net = vggnet(datas_train)
        net = slim.fully_connected(net,num_classes,activation_fn=None,scope='fc1')
        #the last layer
        outputs = slim.softmax(net, scope='predictions')
        #outputs = tf.nn.softmax(net，namescope='output')
    elif FLAGS.typenets == 'resnet50':
        with slim.arg_scope(resnet_v2.resnet_arg_scope(is_training=is_training)):
            net,_ = resnet_v2.resnet_v2_50(datas_train,num_classes=num_classes)
            net = slim.flatten(net,scope='flat2')
            outputs = slim.softmax(net, scope='predictions')
    elif FLAGS.typenets == 'resnet101':  
        with slim.arg_scope(resnet_v2.resnet_arg_scope(is_training=is_training)):
            net,_ = resnet_v2.resnet_v2_101(datas_train,num_classes=num_classes)
            net = slim.flatten(net,scope='flat2')
            outputs = slim.softmax(net, scope='predictions')
    elif FLAGS.typenets == 'resnet152':  
        with slim.arg_scope(resnet_v2.resnet_arg_scope(is_training=is_training)):
            net,_ = resnet_v2.resnet_v2_152(datas_train,num_classes=num_classes)
            net = slim.flatten(net,scope='flat2')
            outputs = slim.softmax(net, scope='predictions')
    elif FLAGS.typenets == 'resnet200':
        with slim.arg_scope(resnet_v2.resnet_arg_scope(is_training=is_training)):
            net,_ = resnet_v2.resnet_v2_200(datas_train,num_classes=num_classes)
            net = slim.flatten(net,scope='flat2')
            outputs = slim.softmax(net, scope='predictions')
    elif FLAGS.typenets == 'simple':
        outputs = forward_simple(datas_train,is_training)
    return outputs
    
#简单的神经网络-------------------designed by zouyuelin(Tianjin university)
#使用简单的神经网络即可达到很高的检测准确率
def forward_simple(datas_train,is_training):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.conv2d(datas_train,64 , [3, 3])
        net = slim.max_pool2d(net, [2, 2])
        net = slim.conv2d(net,128 , [3, 3])
        net = slim.max_pool2d(net, [2, 2])
        net = slim.conv2d(net,128 , [3, 3])
        net = slim.max_pool2d(net, [2, 2])
        net = slim.flatten(net)
        net = slim.fully_connected(net, 1024, scope='fc1')
        net = slim.dropout(net, 0.5, scope='dropout1')
        net = slim.fully_connected(net, 512, scope='fc2')
        net = slim.dropout(net, 0.5, scope='dropout2')
        net = slim.fully_connected(net, 64, activation_fn=None, scope='fc3')
        net = slim.dropout(net, 0.5, scope='dropout3')
        net = slim.fully_connected(net,num_classes)
        outputs = slim.softmax(net, scope='predictions')
    return outputs

#set the backward nets:
#反向传播参数
def backward():
    x_datas = tf.placeholder(tf.float32,[None,224,224,3],name='Input')
    y_labels = tf.placeholder(tf.float32,[None,2],name='labels')
    is_training = tf.placeholder(dtype=tf.bool,name='is_training')
    predictions = forward(x_datas,is_training)
    
    #损失函数
    #ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions,labels=tf.argmax(y_labels,1))
	#cem = tf.reduce_mean(ce)
	#loss = cem+tf.add_n(tf.get_collection('losses'))
    
    loss = slim.losses.softmax_cross_entropy(predictions, y_labels)
    with tf.name_scope('cross_entropy'):
        loss = slim.losses.get_total_loss(add_regularization_losses=True)
        tf.summary.scalar('cross_entropy',loss)
    
    #准确率
    correct_prediction = tf.equal(tf.argmax(predictions,1),tf.argmax(y_labels,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy',accuracy)

    #指数衰减学习率
    global_step = tf.Variable(0,trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
        global_step,
        decay_step, #dataset_num/batch_size,
        LRARNING_RATE_DECAY,
        staircase=True)

    #优化算法:梯度下降方法可以找到最优解
    #梯度下降优化器
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #train_step = optimizer.minimize(loss,global_step=global_step)
    
    #Adams优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.01)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops): #保证train_op在update_ops执行之后再执行。
        train_step = optimizer.minimize(loss)
    
    print('---------------------------------Load the image data------------------------------------')
    #--------------------------------------------------------------tfrecords-------------------------------------------------
    #读取数据
	#capacity>=min_after_dequeue+num_threads*batch_szie
    dsets, dlabel = read_and_decode(tfrecords_path)
    
    img_batch, label_batch = tf.train.batch([dsets,dlabel],
        batch_size=batch_size,
        num_threads=4,
        capacity= 640)
    
    sets_test, label_test = read_and_decode(test_path)
    dsets_test, dlabel_test = tf.train.batch([sets_test, label_test],
                                             batch_size=16,
                                             num_threads=3,
                                            capacity = 64)
    print('---------------------------------Load the image data successful------------------------------------')


	#写入日志log
    merged = tf.summary.merge_all()
    
    #训练数据
    print("***************************************start the gpu with kernel function*********************************************\n")
    with tf.Session() as sess:
        print("###########the training is start#########")
        
        #初始化日志
        summary_writer = tf.summary.FileWriter(log_path,sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator() 
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)
        
        #处理BN层，获取滑动平均值
        var_list = tf.trainable_variables() 
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        saver = tf.train.Saver(var_list=var_list,max_to_keep=2,keep_checkpoint_every_n_hours=1)
        
        for i in range(num_steps):
            #计算时间
            start_time = time.time()
            img_trains,label_trains = sess.run([img_batch, label_batch])
            
            #简单的测试代码------------------------------------可以去掉注释测试看看-----------------------------------------
            #img = np.reshape(img_trains[0],(224,224,3))
            #cv2.imshow('l',cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            #print(sess.run([predictions],feed_dict={x_datas:np.reshape(img_trains[0],[1,224,224,3])}))
            #print(label_trains[0])
            #cv2.waitKey(200)

            label_= one_hot(label_trains,num_classes)
            _,acc,los,summary= sess.run([train_step,accuracy,loss,merged],feed_dict={x_datas:img_trains,y_labels:label_,is_training:True})
			#添加日志
            summary_writer.add_summary(summary,i)
            
            #测试已经训练的模型
            img_test,label_test = sess.run([dsets_test, dlabel_test])
            label_test_one = one_hot(label_test,2)
            y_test,acc_test = sess.run([predictions,accuracy],feed_dict={x_datas:img_test,y_labels:label_test_one,is_training:False})
            #print(y_test)
            duration = time.time()-start_time
			#打印输出信息
            if i % 5 == 0:
                print("the accuracy is : %.2f%%,the loss is : [%.8f],the total step is :[%i] " %(acc*100,los,i),end='')
                print("test accuracy is : %.2f%%,%.3fs/step"%(acc_test*100,duration))
                if i % 1000 == 0:
                #中间保存模型
                    saver_path = saver.save(sess,model_path,global_step = i,write_meta_graph=True)
                    
        #保存最后模型 ckpt,meta,data       
        saver_path = saver.save(sess,model_path,global_step = num_steps,write_meta_graph=True)
        #关闭线程协调器
        coord.request_stop()
        coord.join(threads)
        
		#保存模型pb
        constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['predictions/Softmax'])
        with tf.gfile.FastGFile(pb_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())
			
        with tf.gfile.FastGFile(pb_path, mode='rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
		#保存模型pbtxt
        tf.train.write_graph(graph_def, pbtxt_path, 'classify.pbtxt', as_text=True)
        
        '''
        var_list = tf.global_variables()
        constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, 
                                                                      output_node_names=[var_list[i].name for i in range(len(var_list))])
        
        tf.train.write_graph(constant_graph, pbtxt_path, pb_name, as_text=False)
        
        tf.train.write_graph(constant_graph, pbtxt_path, pbtext_name, as_text=True)
        '''

    summary_writer.close()
	
def main():
    backward()
main()
