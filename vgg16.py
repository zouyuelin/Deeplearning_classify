import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

def vgg16_WithoutArgvs(inputs):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      normalizer_fn=slim.batch_norm,
                      weights_initializer=slim.variance_scaling_initializer(),
                      normalizer_params= 
                          { 
                          'is_training': True,
                          'decay': 0.997,                     # batch_norm_decay=0.997 
                          'epsilon': 1e-5,                    #  BN的epsilon默认1e-5
                          'scale': True,                      # BN的scale默认值
                          'updates_collections': tf.GraphKeys.UPDATE_OPS,
                          },
                      weights_regularizer=slim.l2_regularizer(0.0005)):
        # 主体网络结构
        # Layer  1
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], trainable=True, scope='conv1')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')

        # Layer 2
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable=True, scope='conv2')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')

        # Layer 3
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], trainable=True, scope='conv3')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')

        # Layer 4
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=True, scope='conv4')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')

        # Layer 5
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=True, scope='conv5')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool5')
    
        #fully connected
        net = slim.flatten(net,scope='flat5')
        net = slim.fully_connected(net, 4096, scope='fc6')
        net = slim.dropout(net, 0.5, scope='dropout6')
        
        net = slim.fully_connected(net, 4096, scope='fc7')
        net = slim.dropout(net, 0.5, scope='dropout7')
        
        net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
    return net
        