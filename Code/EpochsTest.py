import numpy as np
import cv2
import glob
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.misc import imread, imresize, imsave

batch_size = 100
stddev = 0.02
loss_EG_list =[]
loss_Dz_list =[]
loss_Di_list =[]
EG_loss_list =[]
D_z_loss_prior_list = []
D_z_loss_z_list = []
E_z_loss_list = []
D_img_loss_input_list = []
D_img_loss_G_list = []
G_img_loss_list = []
tv_loss_list =[]


size_image=128  # size the input images
size_kernel=5  # size of the kernels in convolution and deconvolution
size_batch=100  # mini-batch size for training and testing, must be square of an integer
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

def data_load():
    data = []
    one_hot_encoding_age = -np.ones([23705,10])
    gender_var = -np.ones([23705,2])
    width = 128
    height = 128
    j = 0
    folder_labeled_images = "C:\\Users\\ganch\\Desktop\\GANS\\UTKface_Aligned_cropped\\labeled\\"
    for i in range(20):
        if (i == 0):
            labeled_data = folder_labeled_images+"0"
            label = 0
        elif(i == 1):
            labeled_data = folder_labeled_images+"1"
            label = 1
        elif(i == 2):
            labeled_data = folder_labeled_images+"2"
            label = 2
        elif( i == 3 ):
            labeled_data = folder_labeled_images+"3"
            label = 3
        elif( i == 4):
            labeled_data = folder_labeled_images+"4"
            label = 4
        elif(i == 5 ):
            labeled_data = folder_labeled_images+"5"
            label = 5
        elif(i == 6):
            labeled_data = folder_labeled_images+"6"
            label = 6
        elif(i == 7):
            labeled_data = folder_labeled_images+"7"
            label = 7
        elif(i == 8):
            labeled_data = folder_labeled_images+"8"
            label = 8
        elif(i==9 ):
            labeled_data = folder_labeled_images+"9"
            label = 9
        elif( i == 10):
            labeled_data = folder_labeled_images+"10"
            label = 10
        elif(i == 11):
            labeled_data = folder_labeled_images+"11"
            label = 11                    
        elif(i == 12):
            labeled_data = folder_labeled_images+"12"
            label = 12
        elif(i == 13):
            labeled_data = folder_labeled_images+"13"
            label = 13
        elif( i == 14):
            labeled_data = folder_labeled_images+"14"
            label = 14
        elif( i == 15):
            labeled_data = folder_labeled_images+"15"
            label = 15
        elif(i == 16):
            labeled_data = folder_labeled_images+"16"
            label = 16
        elif(i == 17):
            labeled_data = folder_labeled_images+"17"
            label = 17
        elif(i == 18):
            labeled_data = folder_labeled_images+"18"
            label = 18
        else:
            labeled_data = folder_labeled_images+"19"
            label = 19
            
            
        data_path = os.path.join(labeled_data,'*g')
        files = glob.glob(data_path)
        for f1 in files:
            # img = cv2.imread(f1)
            # img = cv2.resize(img,(width,height))
            # #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # #img = img.reshape(1,width*height)
            # norm_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            ###print(img.shape)
            #cv2.imshow('image',img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            data.append(f1)
            ###print(label)
            ###print(j)
            ###print(label)
            if(label == 0):
                one_hot_encoding_age[j,0] = 1
                gender_var[j,0] = -1
                gender_var[j,1] = 1
                ###print("Update0")
            if(label == 1):
                one_hot_encoding_age[j,0] = 1
                gender_var[j,0] = 1
                gender_var[j,1] = -1
                #("Update1")
            if(label == 2):
                one_hot_encoding_age[j,1] = 1
                gender_var[j,0] = -1
                gender_var[j,1] = 1
                ###print("Update2")
            if(label == 3):
                one_hot_encoding_age[j,1] = 1
                gender_var[j,0] = 1
                gender_var[j,1] = -1
                ###print("Update3")
            if(label == 4):
                one_hot_encoding_age[j,2] = 1
                gender_var[j,0] = -1
                gender_var[j,1] = 1
                ###print("Update4")
            if(label == 5):
                one_hot_encoding_age[j,2] = 1
                gender_var[j,0] = 1
                gender_var[j,1] = -1
                ###print("Update5")
            if(label == 6):
                one_hot_encoding_age[j,3] = 1
                gender_var[j,0] = -1
                gender_var[j,1] = 1
                ###print("Update6")
            if(label == 7):
                one_hot_encoding_age[j,3] = 1
                gender_var[j,0] = 1
                gender_var[j,1] = -1
                ###print("Update7")
            if(label == 8):
                one_hot_encoding_age[j,4] = 1
                gender_var[j,0] = -1
                gender_var[j,1] = 1
                ###print("Update8")
            if(label == 9):
                one_hot_encoding_age[j,4] = 1
                gender_var[j,0] = 1
                gender_var[j,1] = -1
                ###print("Update9")
            if(label == 10):
                one_hot_encoding_age[j,5] = 1
                gender_var[j,0] = -1
                gender_var[j,1] = 1
                ###print("Update10")
            if(label == 11):
                one_hot_encoding_age[j,5] = 1
                gender_var[j,0] = 1
                gender_var[j,1] = -1
                ###print("Update11")
            if(label == 12):
                one_hot_encoding_age[j,6] = 1
                gender_var[j,0] = -1
                gender_var[j,1] = 1
                ###print("Update12")
            if(label == 13):
                one_hot_encoding_age[j,6] = 1
                gender_var[j,0] = 1
                gender_var[j,1] = -1
                ###print("Update13")
            if(label == 14):
                one_hot_encoding_age[j,7] = 1
                gender_var[j,0] = -1
                gender_var[j,1] = 1
                ###print("Update14")
            if(label == 15):
                one_hot_encoding_age[j,7] = 1
                gender_var[j,0] = 1
                gender_var[j,1] = -1
                ###print("Update15")
            if(label == 16):
                one_hot_encoding_age[j,8] = 1
                gender_var[j,0] = -1
                gender_var[j,1] = 1
                ###print("Update16")
            if(label == 17):
                one_hot_encoding_age[j,8] = 1
                gender_var[j,0] = 1
                gender_var[j,1] = -1
                ###print("Update17")
            if(label == 18):
                one_hot_encoding_age[j,9] = 1
                gender_var[j,0] = -1
                gender_var[j,1] = 1
                ###print("Update18")
            if(label == 19):
                one_hot_encoding_age[j,9] = 1
                gender_var[j,0] = 1
                gender_var[j,1] = -1
                ###print("Update19")
            ###print(one_hot_encoding_age[j])
            ###print(one_hot_encoding[j])
            
            j += 1
        
    return data,one_hot_encoding_age,gender_var
    
    
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


def Encoder():
    

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

def Generator():

    ###print(logits)
    ###print((one_hot_encoding_age_p1))
    ###print(gender_var_p1)

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









if __name__ == '__main__':

    image_path = "C:\\Users\\ganch\\Desktop\\GANS\\GANTest\\"
    num_epochs = 26
    sess = tf.Session()
    width = 128
    height = 128
    with sess.as_default():

        data,one_hot_encoding_age,gender_var = data_load()
        weights_Encoder,biases_Encoder = init_weights_Encoder()
        weights_Gen,biases_Gen = init_weights_Generator()
        weights_Disc_Encoder,biases_Disc_Encoder = init_weights_Disc_Encoder() 
        weights_Disc_Gen,biases_Disc_Gen = init_weights_Disc_Generator()

        Y_concat = np.concatenate((one_hot_encoding_age,gender_var),axis=1)

        train_size = int(len(data))
        X_train = data[0:train_size]
        Y_train = Y_concat[0:train_size]
        #x_test = data[train_size:len(data)]
        #y_test = Y_concat[train_size:len(data)]

        X_train = np.array(X_train)

        xtrain_dataset = tf.data.Dataset.from_tensor_slices(X_train)
        ytrain_dataset = tf.data.Dataset.from_tensor_slices(Y_train)
        numberOfBatches = int(X_train.shape[0]/size_batch)

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

        z = Encoder()

        reconst = Generator()

        ###############################First Loss################################################################

        EG_loss = tf.reduce_mean(tf.abs(input_image_1 - reconst))

        #############################Second Loss####################################################################

        ##Call with z_prior
        D_z_prior,D_z_prior_logits = Disc_Encoder()
        D_z_loss_prior = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_z_prior_logits, labels=tf.ones_like(D_z_prior_logits)))

        ############################################################################################################

        ##Call with z
        D_z,D_z_logits = Disc_Encoder()
        D_z_loss_z = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_z_logits, labels=tf.zeros_like(D_z_logits)))

        ###########################################################################################################

        E_z_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_z_logits, labels=tf.ones_like(D_z_logits)))

        #########################################################################################################

        D_input_logits_sig,D_input_logits = Disc_Generator()
        D_img_loss_input = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_input_logits, labels=tf.ones_like(D_input_logits)))

        ############################################################################################################

        D_G_logits_sig,D_G_logits = Disc_Generator()
        D_img_loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_G_logits, labels=tf.zeros_like(D_G_logits)))

        ###############################################################################################################

        G_img_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_G_logits, labels=tf.ones_like(D_G_logits)))

        #####################################################################################################################

        tv_y_size = 128
        tv_x_size = 128
        tv_loss = ((tf.nn.l2_loss(reconst[:, 1:, :, :] - reconst[:, :127, :, :]) / tv_y_size) +(tf.nn.l2_loss(reconst[:, :, 1:, :] - reconst[:, :, :127, :]) / tv_x_size)) / batch_size

        ######################################################################################################################


        loss_EG = EG_loss + 0.0001 * G_img_loss + 0 * E_z_loss +  0* tv_loss
        loss_Dz = D_z_loss_prior + D_z_loss_z
        loss_Di = D_img_loss_input + D_img_loss_G

        #####################################################################################################################

        loss_EG_OP = tf.train.AdamOptimizer(learning_rate = 0.0002,beta1 = 0.5).minimize(loss_EG)
        loss_Dz_OP = tf.train.AdamOptimizer(learning_rate = 0.0002,beta1 = 0.5).minimize(loss_Dz)
        loss_Di_OP = tf.train.AdamOptimizer(learning_rate = 0.0002,beta1 = 0.5).minimize(loss_Di)
        init = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        sess.run(init)
        sess.run(init_l)
        #saver = tf.train.Saver()
        saver = tf.train.Saver()

        start = time.time()

        ##print("X_train",X_train.shape)

        z_prior_1 = np.random.uniform(-1,1,[batch_size,50]).astype(np.float32)

        for j in range(2):
            combindedTrainDataset = tf.data.Dataset.zip((xtrain_dataset, ytrain_dataset)).shuffle(X_train.shape[0]).batch(batch_size)
            iterator = combindedTrainDataset.make_initializable_iterator()
            next_element = iterator.get_next()
            sess.run(iterator.initializer)
            for i in range(1):
                image_batch = []
                val = sess.run(next_element)
                one_hot_age = val[1][:,0:10]
                gender_1 = val[1][:,10:12]
                a1 = one_hot_age
                b1 = gender_1
                ##print("B1",b1.shape)

                for f1 in val[0]:
                    #img = cv2.imread(f1.decode("utf-8"))
                    img = imread(f1,mode='RGB')
                    #img = cv2.resize(img,(128,128))
                    img = imresize(img,(128,128))
                    #norm_image = cv2.normalize(img,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype = cv2.CV_32F)
                    norm_image = img.astype(np.float32) * (image_value_range[-1] - image_value_range[0]) / 255.0 + image_value_range[0]
                    image_batch.append(norm_image)
                image_batch = np.array(image_batch)
                ##print(image_batch.shape)
                ##print(type(image_batch))
                Out_z = sess.run(z,feed_dict={input_image_1:image_batch})
                ##print(Out_z.shape)
                ##print(z_prior.shape)
                ##print(type(Out_z))
                out_G = sess.run(reconst, feed_dict={z_prior:Out_z,age:a1,gender:b1,input_image_1:image_batch})
                sess.run(D_z_prior_logits, feed_dict={z_prior:z_prior_1,input_image_1:np.array(image_batch)})
                sess.run(D_z_logits, feed_dict={z_prior:Out_z,input_image_1:np.array(image_batch)})
                sess.run(D_input_logits, feed_dict={input_image_1:image_batch,age:a1,gender:b1,input_image_1:image_batch})
                sess.run(D_G_logits, feed_dict={input_image_1:out_G,age:a1,gender:b1,input_image_1:image_batch})

                sess.run(loss_EG_OP,feed_dict={age:a1,gender:b1,input_image_1:np.array(image_batch),z_prior:z_prior_1})
                sess.run(loss_Dz_OP,feed_dict={age:a1,gender:b1,input_image_1:np.array(image_batch),z_prior:z_prior_1})
                sess.run(loss_Di_OP,feed_dict={age:a1,gender:b1,input_image_1:np.array(image_batch),z_prior:z_prior_1})

                loss_EG_1 = sess.run(loss_EG, feed_dict={age:a1,gender:b1,input_image_1:np.array(image_batch),z_prior:z_prior_1})
                loss_Dz_1 = sess.run(loss_Dz, feed_dict={age:a1,gender:b1,input_image_1:np.array(image_batch),z_prior:z_prior_1})
                loss_Di_1 = sess.run(loss_Di, feed_dict={age:a1,gender:b1,input_image_1:np.array(image_batch),z_prior:z_prior_1})

                EG_loss_1 = sess.run(EG_loss, feed_dict={age:a1,gender:b1,input_image_1:np.array(image_batch),z_prior:z_prior_1})

                D_z_loss_prior_1 = sess.run(D_z_loss_prior, feed_dict={age:a1,gender:b1,input_image_1:np.array(image_batch),z_prior:z_prior_1})

                D_z_loss_z_1 = sess.run(D_z_loss_z, feed_dict={age:a1,gender:b1,input_image_1:np.array(image_batch),z_prior:z_prior_1})

                E_z_loss_1 = sess.run(E_z_loss, feed_dict={age:a1,gender:b1,input_image_1:np.array(image_batch),z_prior:z_prior_1})

                D_img_loss_input_1 =  sess.run(D_img_loss_input, feed_dict={age:a1,gender:b1,input_image_1:np.array(image_batch),z_prior:z_prior_1})

                D_img_loss_G_1 =  sess.run(D_img_loss_G, feed_dict={age:a1,gender:b1,input_image_1:np.array(image_batch),z_prior:z_prior_1})

                G_img_loss_1 =  sess.run(G_img_loss, feed_dict={age:a1,gender:b1,input_image_1:np.array(image_batch),z_prior:z_prior_1})

                tv_loss_1 =  sess.run(tv_loss, feed_dict={age:a1,gender:b1,input_image_1:np.array(image_batch),z_prior:z_prior_1})

                
                loss_EG_list.append(loss_EG_1)
                loss_Dz_list.append(loss_Dz_1)
                loss_Di_list.append(loss_Di_1)

                EG_loss_list.append(EG_loss_1)
                D_z_loss_prior_list.append(D_z_loss_prior_1)
                D_z_loss_z_list.append(D_z_loss_z_1)
                E_z_loss_list.append(E_z_loss_1)
                D_img_loss_input_list.append(D_img_loss_input_1)
                D_img_loss_G_list.append(D_img_loss_G_1)
                G_img_loss_list.append(G_img_loss_1)
                tv_loss_list.append(tv_loss_1)



                
                print("Loss_EG",loss_EG_list )
                print("loss_Dz",loss_Dz_list )
                print("loss_Di",loss_Di_list )
                print("EG_Loss",EG_loss_1)
                print("Dz_Loss",D_z_loss_prior_1)
                print("Dz_Z",D_z_loss_z_1)
                print("Ez Loss",E_z_loss_1)
                print("D_img_loss_input",D_img_loss_input_1)
                print("D_img_loss_G",D_img_loss_G_1)
                print("G_img_loss",G_img_loss_1)
                print("Tv_loss",tv_loss_1)

                print("Batch processing",i)
            print("Epochs Completed",j)
            if((j>1) and ((j % 25) == 0)):
                save_path1 = saver.save(sess, image_path+"/Output"+str(j)+"/saved_model"+str(j)+".ckpt")
                print("Model saved after 25 epochs in path: %s" % save_path1)
            
        save_path2 = saver.save(sess, image_path+"saved_modelfinal.ckpt")
        print("Model saved after 50 epochs in path: %s" % save_path2)


plt.xlabel("Iterations")
plt.ylabel("Encoder and Generator Loss")
plt.plot(loss_EG_list)
plt.savefig(image_path+"loss_EG_list.png")
plt.figure()
plt.xlabel("Iterations")
plt.ylabel("Discriminator on Encoder Loss")
plt.plot(loss_Dz_list)
plt.savefig(image_path+"loss_Dz_list.png")
plt.figure()
plt.xlabel("Iterations")
plt.ylabel("Discriminator on Images Loss")
plt.plot(loss_Di_list)
plt.savefig(image_path+"loss_Di_list.png")
pd.DataFrame(loss_EG_list).to_excel(image_path+"loss_EG.xlsx")
pd.DataFrame(loss_Dz_list).to_excel(image_path+"loss_Dz.xlsx")
pd.DataFrame(loss_Di_list).to_excel(image_path+"loss_Di.xlsx")
pd.DataFrame(EG_loss_list).to_excel(image_path+"EG_loss.xlsx")
pd.DataFrame(D_z_loss_prior_list).to_excel(image_path+"D_z_loss_prior.xlsx")
pd.DataFrame(D_z_loss_z_list).to_excel(image_path+"D_z_loss_z.xlsx")
pd.DataFrame(E_z_loss_list).to_excel(image_path+"E_z_loss.xlsx")
pd.DataFrame(D_img_loss_input_list).to_excel(image_path+"D_img_loss_input.xlsx")
pd.DataFrame(D_img_loss_G_list).to_excel(image_path+"D_img_loss_G.xlsx")
pd.DataFrame(G_img_loss_list).to_excel(image_path+"G_img_loss.xlsx")
pd.DataFrame(tv_loss_list).to_excel(image_path+"tv_loss.xlsx")



                

                

















        

