
快速预测：
       python test_OpenCV.py --image [要预测的图像(例如 ../../SecondProject/data/cat.9001.jpg)
快速训练:
       python creat_tfrecords.py --data [图像数据集路径]
       python train.py

该demo代码由邹岳霖、马国语共同完成
作者介绍：
姓名：邹岳霖
学校：天津大学
学院：机械工程学院
学号: 2020201082

姓名：马国语
学校：天津大学
学院：机械工程学院
学号：2020201065

‘’‘
Introduce:
该demo的任务主要是图像分类，在该任务中主要用于二分类，识别猫和狗

主要的开源库为tensorflow以及contrib中的slim库

输入的图像大小为224*224*3，将所有的图像resize后传入神经网络

总共设置了几个网络，VGGNet16，ResNet50,ResNet101,ResNet152,ResNet200以及一个自己设计的简单、浅层的simple网络

训练默认的网络为ResNet101网络,以下是训练的归一化方法：
    with slim.arg_scope(resnet_v123.resnet_arg_scope()):
    
batch_size = 64
迭代次数为20000次
初始学习率设置为0.003
学习率指数衰减下降率 0.96

Tips:
    测试和训练的精度可达99%

硬件设备：
显卡：NVIDIA TITAN V，11GB显存
内存RAM： 16GB
处理器：Inter i9-9900 3.10GHz x 16
磁盘：固态硬盘 1.3TB
操作系统：Ubuntu 20.04.1 LTS
类型：64位
内核版本：5.4.0
python 版本：3.6.0 && 2.7.18
Tensorflow 版本：GPU版 1.13.1
cuda : 10.0
cudnn: 7.6.4
gcc/g++版本：7.5.0 

’‘’
quickly to start the network:

1.-----------------------------------------------------------制作tfrecord格式的数据集：--------------------------------

tfrecord格式的图像对内存友好，在使用tensorflow时因首先考虑该格式

---修改creat_tfrecords.py中的filename_train和filename_test,默认为train_200.tfrecords

---python creat_tfrecords.py --data [图像数据集路径]

--####--
        python creat_tfrecords.py --data ../../SecondProject/data/

2.--------------------------------------------------------------------直接训练-------------------------------------------------
---修改train.py里的tfrecords路径，学习率等

---python train.py --typenets [网络类型(默认resnet101)]

        可供选择的网络类型:vggnet16,resnet50,resnet101,resnet152
        
---python train.py (默认的网络为resnet101)
        训练完后将生成
        model.ckpt-20000.meta,
        model.ckpt-20000.data,
        checkpoint,
        classify.pb等文件
        
3.--------------------------------------------------------------------------测试模型----------------------------------------
---包含两个test_OpenCV.py 和 test_PIL.py
    
    
    两个模型的使用方法一样:
    
--------python test_PIL.py --model [meta文件如model/model.ckpt-20000.meta] --image [要预测的图像(例如 ../../SecondProject/data/cat.9001.jpg)]
        
--------python test_OpenCV.py --model [meta文件如model/model.ckpt-20000.meta] --image [要预测的图像(例如 ../../SecondProject/data/cat.9001.jpg)]

        #也可以使用固化的pb模型
--------python test_OpenCV.py --image [要预测的图像(例如 ../../SecondProject/data/cat.9001.jpg)]

        python test_OpenCV.py --model model/model.ckpt-20000.meta --image ../../SecondProject/data/cat.9001.jpg
        
        python test_OpenCV.py --image ../../SecondProject/data/cat.9001.jpg

4.----------------------------------------------------------------查看训练精度过程-------------------
--------tensorboard --logdir=./log/log
        
        
        

遇到的问题和解决方案：
1.在使用cat_and_dog_200的样本时出现过拟合

    即在训练集上准确率高（可高达100%），在测试样本上表现较差（50%左右），几乎没有效果
    
    训练集高达100%，至少可以证明resnet101网络在获取图像特征上能力较强，准确性高，适合改分类问题
    
    方法是加大数据集

2.训练振荡问题
    
    训练下降速度太慢，准确率不断振荡，或者很早就出现振荡
    
    将学习率调整为0.003，之前为0.05，同时利用指数下降学习率的方法逐渐减小学习率
    
3.BN层测试和训练的问题
    
    训练表现的非常好，单张图测试表现效果极差
    
    原因是BN属于global_varible,直接保存的模型并没有保存参数
    
    方法是利用节点信息直接保存BN的滑动平均值作为参数，并且训练时将is_training设置为True，测试时将is_training设置为False
    
    
        