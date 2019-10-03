import numpy as np
import cv2
import glob
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.misc import imread, imresize, imsave
import PIL

batch_size = 10
stddev = 0.02
loss_EG_list =[]
loss_Dz_list =[]
loss_Di_list =[]


size_image=128  # size the input images
size_kernel=5  # size of the kernels in convolution and deconvolution
size_batch=10  # mini-batch size for training and testing, must be square of an integer
num_input_channels=3  # number of channels of input images
num_encoder_channels=64 # number of channels of the first conv layer of encoder
num_z_channels=50  # number of channels of the layer z (noise or code)
num_categories=10 # number of categories (age segments) in the training dataset
num_gen_channels=1024  # number of channels of the first deconv layer of generator
enable_tile_label=True # enable to tile the label
tile_ratio=1.0  # ratio of the length between tiled label and z
is_training=True  # flag for training or testing mode
save_dir='./save'  # path to save checkpoints, samples, and summary
dataset_name='UTKFace'  # name of the dataset in the folder ./data
image_value_range = (-1,1)

   
    
def init_weights_Encoder():
    
    weights_Encoder = {
    "w1" : tf.get_variable("wc1", shape=[4, 4, 3, 64],initializer=tf.truncated_normal_initializer(stddev=stddev)),
    "w2" : tf.get_variable("wc2", shape=[4, 4, 64, 128],initializer=tf.truncated_normal_initializer(stddev=stddev)),
    "w3" : tf.get_variable("wc3", shape=[4, 4, 128, 256],initializer=tf.truncated_normal_initializer(stddev=stddev)),
    "w4" : tf.get_variable("wc4", shape=[4, 4, 256, 512],initializer=tf.truncated_normal_initializer(stddev=stddev)),
    #'wd1': tf.get_variable("wd1", shape=[8192, 512],initializer=tf.random_normal_initializer(stddev=stddev)),
    'out': tf.get_variable("out", shape=[32768,50],initializer=tf.random_normal_initializer(stddev=stddev))
     }

    biases_Encoder = {
            'bc1': tf.get_variable("bc1", shape=[64],initializer=tf.constant_initializer(0.0)),
            'bc2': tf.get_variable("bc2", shape=[128],initializer=tf.constant_initializer(0.0)),
            'bc3': tf.get_variable("bc3", shape=[256],initializer=tf.constant_initializer(0.0)),
            'bc4': tf.get_variable("bc4", shape=[512],initializer=tf.constant_initializer(0.0)),
            #'bd1': tf.get_variable("bd1", shape=[512],initializer=tf.constant_initializer(0.0)),
            'bout': tf.get_variable("bout", shape=[50],initializer=tf.constant_initializer(0.0))
    }
    
    return weights_Encoder,biases_Encoder


def Encoder(input_image_1,weights_Encoder,biases_Encoder):
    
    input_image_1 = input_image_1
    weights_Encoder = weights_Encoder
    biases_Encoder = biases_Encoder
    ##Layer 1
    conv1 = tf.nn.conv2d(input = input_image_1, filter = weights_Encoder["w1"], strides=[1, 2, 2, 1], padding='SAME', name = "convolution1")
    conv1 = tf.nn.bias_add(conv1, biases_Encoder["bc1"])
    #conv1 = tf.layers.batch_normalization(conv1)
    conv1 = tf.nn.relu(conv1)
    ##print(conv1.shape)
    #maxpool1 = tf.nn.max_pool(value = conv1, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1],padding='SAME', name = "maxpool1")

    ##Layer 2
    conv2 = tf.nn.conv2d(input = conv1, filter = weights_Encoder["w2"], strides=[1, 2, 2, 1], padding='SAME', name = "convolution2")
    conv2 = tf.nn.bias_add(conv2, biases_Encoder["bc2"])
    #conv2 = tf.layers.batch_normalization(conv2)
    conv2 = tf.nn.relu(conv2)
    ##print(conv2.shape)
    #maxpool2 = tf.nn.max_pool(value = conv2, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1],padding='SAME', name = "maxpool2")

    ##Layer 3
    conv3 = tf.nn.conv2d(input = conv2, filter = weights_Encoder["w3"], strides=[1, 2, 2, 1], padding='SAME', name = "convolution3")
    conv3 = tf.nn.bias_add(conv3, biases_Encoder["bc3"])
    #conv3 = tf.layers.batch_normalization(conv3)
    conv3 = tf.nn.relu(conv3)
    ##print(conv3.shape)
    #maxpool3 = tf.nn.max_pool(value = conv3, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1],padding='SAME', name = "maxpool3")

    ##Layer 4
    conv4 = tf.nn.conv2d(input = conv3, filter = weights_Encoder["w4"], strides=[1, 2, 2, 1], padding='SAME', name = "convolution4")
    conv4 = tf.nn.bias_add(conv4, biases_Encoder["bc4"])
    #conv4 = tf.layers.batch_normalization(conv4)
    conv4 = tf.nn.relu(conv4)
    ##print(conv4.shape)
    #maxpool4 = tf.nn.max_pool(value = conv4, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1],padding='SAME', name = "maxpool4")


    ## Linear layer
    fc1 = tf.contrib.layers.flatten(conv4)
    ##print(fc1.shape)
        #fc1 = tf.reshape(conv5, [-1, weights['wd1'].get_shape().as_list()[0]])
    #fc2 = tf.matmul(fc1, weights_Encoder['wd1']) +  biases_Encoder['bd1']
    #fc2 = tf.layers.batch_normalization(fc2)
    #fc2 = tf.nn.relu(fc2,name ='finalrelu')
    ###print(fc2.shape)

    logits = tf.matmul(fc1, weights_Encoder['out']) + biases_Encoder['bout']
    logits_tan = tf.nn.tanh(logits,name = "finaltan")
    ##print("Shape",logits.shape)

    return logits_tan

def init_weights_Generator():

    weights_Gen = {
    "w0_Gen" : tf.get_variable("wc0_Gen", shape=[3, 3, 3 , 64],initializer=tf.random_normal_initializer(stddev=stddev)),
    "w1_Gen" : tf.get_variable("wc1_Gen", shape=[4, 4, 64, 128],initializer=tf.random_normal_initializer(stddev=stddev)),
    "w2_Gen" : tf.get_variable("wc2_Gen", shape=[4, 4, 128, 256],initializer=tf.random_normal_initializer(stddev=stddev)),
    "w3_Gen" : tf.get_variable("wc3_Gen", shape=[4, 4, 256, 512],initializer=tf.random_normal_initializer(stddev=stddev)),
    "w4_Gen" : tf.get_variable("wc4_Gen", shape=[4, 4, 512, 1024],initializer=tf.random_normal_initializer(stddev=stddev)),
    'wd1_Gen': tf.get_variable("wd1_Gen", shape=[150, 1024*4*4],initializer=tf.random_normal_initializer(stddev=stddev))
    }

    biases_Gen = {
            'bc0_Gen': tf.get_variable("bc0_Gen", shape=[3],initializer=tf.constant_initializer(0.0)),
            'bc1_Gen': tf.get_variable("bc1_Gen", shape=[64],initializer=tf.constant_initializer(0.0)),
            'bc2_Gen': tf.get_variable("bc2_Gen", shape=[128],initializer=tf.constant_initializer(0.0)),
            'bc3_Gen': tf.get_variable("bc3_Gen", shape=[256],initializer=tf.constant_initializer(0.0)),
            'bc4_Gen': tf.get_variable("bc4_Gen", shape=[512],initializer=tf.constant_initializer(0.0)),
            'bd1_Gen': tf.get_variable("bd1_Gen", shape=[1024*4*4],initializer=tf.constant_initializer(0.0))
    }

    return weights_Gen,biases_Gen

def concat_label(x, label, duplicate=1):
    x_shape = x.get_shape().as_list()
    if duplicate < 1:
        return x
    # duplicate the label to enhance its effect, does it really affect the result?
    label = tf.tile(label, [1, duplicate])
    label_shape = label.get_shape().as_list()
    if len(x_shape) == 2:
        return tf.concat(axis=1, values=[x, label])
    elif len(x_shape) == 4:
        label = tf.reshape(label, [x_shape[0], 1, 1, label_shape[-1]])
        return tf.concat(axis=3, values=[x, label*tf.ones([x_shape[0], x_shape[1], x_shape[2], label_shape[-1]])])

def Generator(z,age,gender,weights_Gen,biases_Gen):

    ###print(logits)
    ###print((one_hot_encoding_age_p1))
    ###print(gender_var_p1)

    z = z
    age = age
    gender = gender
    weights_Gen = weights_Gen
    biases_Gen = biases_Gen
    enable_tile_label = True

    if enable_tile_label:
            duplicate = int(50 * 1.0 / 10)
    else:
        duplicate = 1
    y = age
    z1 = concat_label(z, y, duplicate=duplicate)
    if enable_tile_label:
        duplicate = int(num_z_channels * tile_ratio / 2)
    else:
        duplicate = 1
    z2 = concat_label(z1, gender, duplicate=duplicate)

    

    Gen_fc = tf.matmul(z2, weights_Gen['wd1_Gen']) +  biases_Gen['bd1_Gen']
    Gen_fc = tf.reshape(Gen_fc, [-1, 4, 4, 1024])
    Gen_fc =  tf.nn.relu(Gen_fc)

    
    #Gen_fc = tf.nn.relu(Gen_fc)

    denconv4 = tf.nn.conv2d_transpose(Gen_fc,filter = weights_Gen["w4_Gen"],output_shape = [batch_size,8,8,512], strides=[1, 2, 2, 1], padding='SAME', name = "Deconv4")
    denconv4 = tf.nn.bias_add(denconv4, biases_Gen["bc4_Gen"])
    denconv4 = tf.nn.relu(denconv4)
    ##print(denconv4.shape)


    denconv3 = tf.nn.conv2d_transpose(denconv4,filter = weights_Gen["w3_Gen"],output_shape = [batch_size,16,16,256], strides=[1, 2, 2, 1], padding='SAME', name = "Deconv3")
    denconv3 = tf.nn.bias_add(denconv3, biases_Gen["bc3_Gen"])
    denconv3 = tf.nn.relu(denconv3)
    ##print(denconv3.shape)

    denconv2 = tf.nn.conv2d_transpose(denconv3,filter = weights_Gen["w2_Gen"], output_shape = [batch_size,32,32,128],strides=[1, 2, 2, 1], padding='SAME', name = "Deconv2")
    denconv2 = tf.nn.bias_add(denconv2, biases_Gen["bc2_Gen"])
    denconv2 = tf.nn.relu(denconv2)
    ##print(denconv2.shape)

    denconv1 = tf.nn.conv2d_transpose(denconv2,filter = weights_Gen["w1_Gen"], output_shape = [batch_size,64,64,64],strides=[1, 2, 2, 1], padding='SAME', name = "Deconv1")
    denconv1 = tf.nn.bias_add(denconv1, biases_Gen["bc1_Gen"])
    denconv1 = tf.nn.relu(denconv1)
    ##print(denconv1.shape)


    denconv0 = tf.nn.conv2d_transpose(denconv1,filter = weights_Gen["w0_Gen"], output_shape = [batch_size,128,128,3],strides=[1, 2, 2, 1], padding='SAME', name = "Deconv0")
    denconv0 = tf.nn.bias_add(denconv0, biases_Gen["bc0_Gen"])
    out = tf.nn.tanh(denconv0)
    ##print(out.shape)
    return out


def init_weights_Disc_Encoder():

    
    weights_Disc_Encoder = {
        "wc1_Disc_Encoder" : tf.get_variable("wc1_Disc_Encoder", shape=[ 50, 64],initializer=tf.random_normal_initializer(stddev=stddev)),
        "wc2_Disc_Encoder" : tf.get_variable("wc2_Disc_Encoder", shape=[ 64, 32],initializer=tf.random_normal_initializer(stddev=stddev)),
        "wc3_Disc_Encoder" : tf.get_variable("wc3_Disc_Encoder", shape=[ 32, 16],initializer=tf.random_normal_initializer(stddev=stddev)),
        #"wc4_Disc_Encoder" : tf.get_variable("wc4_Disc_Encoder", shape=[ 16, 16],initializer=tf.random_normal_initializer(stddev=stddev)),
        "wc5_Disc_Encoder" : tf.get_variable("wc5_Disc_Encoder", shape=[ 16, 1],initializer=tf.random_normal_initializer(stddev=stddev))

    }

    biases_Disc_Encoder = {
                'bc1_Disc_Encoder': tf.get_variable("bc1_Disc_Encoder", shape=[64],initializer=tf.constant_initializer(0.0)),
                'bc2_Disc_Encoder': tf.get_variable("bc2_Disc_Encoder", shape=[32],initializer=tf.constant_initializer(0.0)),
                'bc3_Disc_Encoder': tf.get_variable("bc3_Disc_Encoder", shape=[16],initializer=tf.constant_initializer(0.0)),
                #'bc4_Disc_Encoder': tf.get_variable("bc4_Disc_Encoder", shape=[16],initializer=tf.constant_initializer(0.0)),
                'bc5_Disc_Encoder': tf.get_variable("bc5_Disc_Encoder", shape=[1],initializer=tf.constant_initializer(0.0))

    }

    return weights_Disc_Encoder,biases_Disc_Encoder

def Disc_Encoder():

    ##print(z.shape)

    fc1_Disc_Encoder = tf.matmul(z, weights_Disc_Encoder['wc1_Disc_Encoder']) +  biases_Disc_Encoder['bc1_Disc_Encoder']
    fc1_Disc_Encoder = tf.contrib.layers.batch_norm(fc1_Disc_Encoder)
    fc1_Disc_Encoder = tf.nn.relu(fc1_Disc_Encoder,name ='relu1_Disc_Encoder')

    ##print("Fc1",fc1_Disc_Encoder.shape)

    fc2_Disc_Encoder = tf.matmul(fc1_Disc_Encoder, weights_Disc_Encoder['wc2_Disc_Encoder']) +  biases_Disc_Encoder['bc2_Disc_Encoder']
    fc2_Disc_Encoder = tf.contrib.layers.batch_norm(fc2_Disc_Encoder)
    fc2_Disc_Encoder = tf.nn.relu(fc2_Disc_Encoder,name ='relu2_Disc_Encoder')
    ##print("Fc2",fc2_Disc_Encoder.shape)

    fc3_Disc_Encoder = tf.matmul(fc2_Disc_Encoder, weights_Disc_Encoder['wc3_Disc_Encoder']) +  biases_Disc_Encoder['bc3_Disc_Encoder']
    fc3_Disc_Encoder = tf.contrib.layers.batch_norm(fc3_Disc_Encoder)
    fc3_Disc_Encoder = tf.nn.relu(fc3_Disc_Encoder,name ='relu3_Disc_Encoder')
    ##print("Fc3",fc3_Disc_Encoder.shape)

    #fc4_Disc_Encoder = tf.matmul(fc3_Disc_Encoder, weights_Disc_Encoder['wc4_Disc_Encoder']) +  biases_Disc_Encoder['bc4_Disc_Encoder']
    #fc4_Disc_Encoder = tf.nn.sigmoid(fc4_Disc_Encoder,name ='relu4_Disc_Encoder')
    ###print("Fc4",fc4_Disc_Encoder.shape)

    fc4_Disc_Encoder = tf.matmul(fc3_Disc_Encoder, weights_Disc_Encoder['wc5_Disc_Encoder']) +  biases_Disc_Encoder['bc5_Disc_Encoder']
    ##print("Fc4",fc4_Disc_Encoder.shape)

    return tf.nn.sigmoid(fc4_Disc_Encoder),fc4_Disc_Encoder



def init_weights_Disc_Generator():

    weights_Disc_Gen = {
    "wc1_Disc_Gen" : tf.get_variable("wc1_Disc_Gen", shape=[4, 4, 3, 16],initializer=tf.truncated_normal_initializer(stddev=stddev)),
    "wc3_Disc_Gen" : tf.get_variable("wc3_Disc_Gen", shape=[4, 4, 36, 32],initializer=tf.truncated_normal_initializer(stddev=stddev)),
    "wc4_Disc_Gen" : tf.get_variable("wc4_Disc_Gen", shape=[4, 4, 32, 64],initializer=tf.truncated_normal_initializer(stddev=stddev)),
    'wc5_Disc_Gen': tf.get_variable("wc5_Disc_Gen", shape=[4,4,64, 128],initializer=tf.truncated_normal_initializer(stddev=stddev)),
    'wcd1_Disc_Gen': tf.get_variable("wcd1_Disc_Gen", shape=[8192,1024],initializer=tf.random_normal_initializer(stddev=stddev)),
    #'wcd2_Disc_Gen': tf.get_variable("wcd2_Disc_Gen", shape=[1024,512],initializer=tf.random_normal_initializer(stddev=stddev)),
    'wd3_Disc_Gen': tf.get_variable("wd3_Disc_Gen", shape=[1024,1],initializer=tf.random_normal_initializer(stddev=stddev)),

     }


    biases_Disc_Gen = {
            'bc1_Disc_Gen': tf.get_variable("bc1_Disc_Gen", shape=[16],initializer=tf.constant_initializer(0.0)),
            'bc3_Disc_Gen': tf.get_variable("bc3_Disc_Gen", shape=[32],initializer=tf.constant_initializer(0.0)),
            'bc4_Disc_Gen': tf.get_variable("bc4_Disc_Gen", shape=[64],initializer=tf.constant_initializer(0.0)),
            'bc5_Disc_Gen': tf.get_variable("bc5_Disc_Gen", shape=[128],initializer=tf.constant_initializer(0.0)),
            'bd1_Disc_Gen': tf.get_variable("bd1_Disc_Gen", shape=[1024],initializer=tf.constant_initializer(0.0)),
            #'bd2_Disc_Gen': tf.get_variable("bd2_Disc_Gen", shape=[512],initializer=tf.constant_initializer(0.0)),
            'bd3_Disc_Gen': tf.get_variable("bd3_Disc_Gen", shape=[1],initializer=tf.constant_initializer(0.0))

    }

    return weights_Disc_Gen,biases_Disc_Gen

def Disc_Generator():

    conv1_Disc_Gen = tf.nn.conv2d(input = input_image_1, filter = weights_Disc_Gen["wc1_Disc_Gen"], strides=[1, 2, 2, 1], padding='SAME', name = "convolution1_Disc_Gen")
    conv1_Disc_Gen = tf.nn.bias_add(conv1_Disc_Gen, biases_Disc_Gen["bc1_Disc_Gen"])
    conv1_Disc_Gen = tf.contrib.layers.batch_norm(conv1_Disc_Gen)
    conv1_Disc_Gen =  tf.nn.relu(conv1_Disc_Gen)

    y = age
    conv1_Disc_Gen = concat_label(conv1_Disc_Gen, y)
    conv1_Disc_Gen = concat_label(conv1_Disc_Gen, gender, int(num_categories / 2))

    ##print(conv1_Disc_Gen.shape)

    conv2_Disc_Gen = tf.nn.conv2d(conv1_Disc_Gen, filter = weights_Disc_Gen["wc3_Disc_Gen"], strides=[1, 2, 2, 1], padding='SAME', name = "convolution2_Disc_Gen")
    conv2_Disc_Gen = tf.nn.bias_add(conv2_Disc_Gen, biases_Disc_Gen["bc3_Disc_Gen"])
    conv2_Disc_Gen = tf.contrib.layers.batch_norm(conv2_Disc_Gen)
    conv2_Disc_Gen =  tf.nn.relu(conv2_Disc_Gen)

    ##print(conv2_Disc_Gen.shape)

    conv3_Disc_Gen = tf.nn.conv2d(conv2_Disc_Gen, filter = weights_Disc_Gen["wc4_Disc_Gen"], strides=[1, 2, 2, 1], padding='SAME', name = "convolution3_Disc_Gen")
    conv3_Disc_Gen = tf.nn.bias_add(conv3_Disc_Gen, biases_Disc_Gen["bc4_Disc_Gen"])
    conv3_Disc_Gen = tf.contrib.layers.batch_norm(conv3_Disc_Gen)
    conv3_Disc_Gen =  tf.nn.relu(conv3_Disc_Gen)

    ##print(conv3_Disc_Gen.shape)

    conv4_Disc_Gen = tf.nn.conv2d(conv3_Disc_Gen, filter = weights_Disc_Gen["wc5_Disc_Gen"], strides=[1, 2, 2, 1], padding='SAME', name = "convolution4_Disc_Gen")
    conv4_Disc_Gen = tf.nn.bias_add(conv4_Disc_Gen, biases_Disc_Gen["bc5_Disc_Gen"])
    conv4_Disc_Gen = tf.contrib.layers.batch_norm(conv4_Disc_Gen)
    conv4_Disc_Gen =  tf.nn.relu(conv4_Disc_Gen)
    ##print(conv4_Disc_Gen.shape)



    ## Fully conected Layer 

    #fc1_Disc_Gen = tf.contrib.layers.flatten(conv4_Disc_Gen)
    ###print(fc1_Disc_Gen.shape)
    fc1_Disc_Gen = tf.reshape(conv4_Disc_Gen, [size_batch, -1])
    fc2_Disc_Gen = tf.matmul(fc1_Disc_Gen, weights_Disc_Gen['wcd1_Disc_Gen']) +  biases_Disc_Gen['bd1_Disc_Gen']
    fc2_Disc_Gen = tf.nn.relu(fc2_Disc_Gen,name ='finalrelu_Disc_Gen')

    ##print(fc2_Disc_Gen.shape)

    ## Sigmoid on Fullly Conncted

    #fc3_Disc_Gen = tf.matmul(fc2_Disc_Gen, weights_Disc_Gen['wcd2_Disc_Gen']) +  biases_Disc_Gen['bd2_Disc_Gen']
    #fc3_Disc_Gen = tf.nn.relu(fc3_Disc_Gen,name ='finalsigmoid_Disc_Gen')
    ###print("Fc",fc3_Disc_Gen.shape)


    #3 Softmax on Fully Connected

    fc4_Disc_Gen = tf.matmul(fc2_Disc_Gen, weights_Disc_Gen['wd3_Disc_Gen']) +  biases_Disc_Gen['bd3_Disc_Gen']
    fc5_Disc_Gen = tf.nn.sigmoid(fc4_Disc_Gen,name ='finalsigmoid_Disc_Gen')

    return fc5_Disc_Gen,fc4_Disc_Gen

        


def test():


    image_path = "C:\\Users\\ganch\\Desktop\\GANS\\Output"
    num_epochs = 26
    sess = tf.Session()
    width = 128
    height = 128
    with sess.as_default():

        #data,one_hot_encoding_age,gender_var = data_load()
        weights_Encoder,biases_Encoder = init_weights_Encoder()
        weights_Gen,biases_Gen = init_weights_Generator()
        weights_Disc_Encoder,biases_Disc_Encoder = init_weights_Disc_Encoder() 
        weights_Disc_Gen,biases_Disc_Gen = init_weights_Disc_Generator()

        

        input_image_1 = tf.placeholder(tf.float32,[size_batch, size_image, size_image, num_input_channels],name='input_images')

        age = tf.placeholder(
            tf.float32,
            [size_batch, num_categories],
            name='age_labels'
        )
        gender = tf.placeholder(
            tf.float32,
            [size_batch, 2],
            name='gender_labels'
        )
        z_prior = tf.placeholder(
            tf.float32,
            [size_batch, num_z_channels],
            name='z_prior'
        )

        z = Encoder(input_image_1,weights_Encoder,biases_Encoder)

        reconst = Generator(z,age,gender,weights_Gen,biases_Gen)
        labeled_data = "C:\\Users\\ganch\\Desktop\\DeepLearningProject\\Demo\\static\\test\\input\\pic"
        #test_data = "C:\\Users\\ganch\\Desktop\\GANS\\UTKface_Aligned_cropped\\labeled\\Output\\test7.jpg"
        data_path = os.path.join(labeled_data,'*g')
        files = glob.glob(data_path)
        print(data_path)

        #data = []
        for f1 in files:
            img = imread(f1,mode='RGB')
            #data.append(img)
        #img = imread(files,mode='RGB')
        print(img.shape)
        img = imresize(img,(128,128))
   
        norm_image = img.astype(np.float32) * (image_value_range[-1] - image_value_range[0]) / 255.0 + image_value_range[0]

        init = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        sess.run(init)
        sess.run(init_l)
        saver = tf.train.Saver()
        checkpoint_dir = "C:\\Users\\ganch\\Desktop\\GANS\\Final_Results\\EpochTest"
        checkpoints = tf.train.get_checkpoint_state("C:\\Users\\ganch\\Desktop\\GANS\\Final_Results\\EpochTest")
        print(checkpoints)

        if checkpoints and checkpoints.model_checkpoint_path:
            checkpoints_name = os.path.basename(checkpoints.model_checkpoint_path)

        print(checkpoints_name)

        saver.restore(sess, os.path.join(checkpoint_dir, checkpoints_name))

        #saver.restore(sess, "C:\\Users\\ganch\\Desktop\\GANS\\UTKface_Aligned_cropped\\labeled\\save\\save\\checkpoint\\model-5900.ckpt")
        graph = tf.get_default_graph()

        print("Model restored.")
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(graph, cmd='op', options=opts)

        sample_images = np.array(norm_image).astype(np.float32)

        #sample_files = file_names[0:size_batch]

        sample_label_age = np.ones(
            shape=(1, num_categories),
            dtype=np.float32
        ) * image_value_range[0]

        sample_label_gender = np.ones(
            shape=(1, 2),
            dtype=np.float32
        ) * image_value_range[0]
    
        imsave("C:\\Users\\ganch\\Desktop\\DeepLearningProject\\Demo\\static\\input_image.png",sample_images)
        
        #images = np.expand_dims(images,axis=0)
        
        query_labels = np.ones(
            shape=(10,  10),
            dtype=np.float32
        ) * image_value_range[0]
        #print(query_labels)
        for i, l in enumerate(query_labels):
            l[i // 1] = 1

        #print(query_labels)
        # for i in range(query_labels.shape[0]):
        #     query_labels[i, labels[i]] = image_value_range[-1]
        query_images = np.tile(sample_images, [num_categories, 1, 1, 1])
        #print(query_images.shape)
        #print(query_labels)

        print("Gender",gender)
        query_gender = np.tile(sample_label_gender, [num_categories, 1])
        

        z1, G1 = sess.run([z, reconst],feed_dict={
                input_image_1: query_images,
                age: query_labels,
                gender: query_gender
            }
        )

        
        #print("G1.shape",G1.shape)
        #print(G1)

        images1 = (G1 - image_value_range[0]) / (image_value_range[-1] - image_value_range[0])
        #images1 = G1
        #print(images1.shape)
        #print(images1)
        vis = np.concatenate((images1[0], images1[1],images1[2],images1[3],images1[4],images1[5],images1[6],images1[7],images1[8],images1[9]), axis=1)
        imsave("C:\\Users\\ganch\\Desktop\\DeepLearningProject\\Demo\\static\\output.png",vis)
        #for k in range(10):
        #     print(images1[k].shape)
        #     #G[k] = G[k].reshape(-1,1)
        #     #G[k] = (sigmoid(G[k]))*255
        #     #images1[k] = ((images1[k]))*255
        #     #G[k] = G[k].squeeze().astype(np.uint8)
         #     imsave("C:\\Users\\ganch\\Desktop\\GANS\\UTKface_Aligned_cropped\\labeled\\Output"+str(k)+".png",images1[k])
        # if size_frame is None:
        #     auto_size = int(np.ceil(np.sqrt(images.shape[0])))
        #     size_frame = [auto_size, auto_size]
        # size_frame = int(np.sqrt(size_batch))
        # size_frame=[size_frame, size_frame]
        # img_h, img_w = G.shape[1], G.shape[2]
        # frame = np.zeros([img_h * size_frame[0], img_w * size_frame[1], 3])
        # for ind, image2 in enumerate(images1):
        #     ind_col = ind % size_frame[1]
        #     ind_row = ind // size_frame[1]
        #     frame[(ind_row * img_h):(ind_row * img_h + img_h), (ind_col * img_w):(ind_col * img_w + img_w), :] = image2



            #test(sample_images, sample_label_gender)


