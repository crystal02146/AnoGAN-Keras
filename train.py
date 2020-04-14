# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:22:44 2019

@author: Tseng Yi-Yao
"""

from keras.utils. generic_utils import Progbar
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose, Flatten
from keras.layers.core import Activation
from keras.optimizers import Adam
import keras.backend as K
import math, cv2
import numpy as np
import os
import tensorflow as tf

from DCGAN import DCGAN

# train
if __name__ == '__main__':

    #設定神經網路參數
    
    batch_size = 16 #一次訓練資料數量
    epochs = 100   # 歷經全部資料集次數
    input_dim = 100 #輸入隱維度z大小
    g_optim = Adam(lr=0.0001, beta_1=0.5, beta_2=0.999) #Adam優化器　lr 學習率 　
    d_optim = Adam(lr=0.0001, beta_1=0.5, beta_2=0.999)
    save_image_epoch = 10 # 每多少Epoch就儲存訓練過程圖到result資料夾
    
 
    # 加載資料集
    
    #from keras.datasets import mnist
    #from keras.datasets import cifar10
    #from keras.datasets import fashion_mnist
    
    #(x_train, y_train), (x_test, y_test) = mnist.load_data() #數字資料集
    #(x_train, y_train), (x_test, y_test) = cifar10.load_data()　#飛機，汽車，鳥類，貓，鹿，狗，青蛙，馬，輪船和卡車　資料集
    #(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()  # 衣服　鞋子　包包　資料集 
    
    import dataloader
    
    x_train, y_train, x_test, y_test = dataloader.load_data() #自訂資料集
    
    x_train = x_train.reshape(x_train.shape[0], 64, 64, 3)#訓練集數量,影像長度,影像寬度,影像頻段
    x_train = x_train.astype('float32') / 255#正規化訓練集資料
    

    # 創建訓練資料
    x_train_normal = []
    for i in range(len(x_train)):
        if y_train[i] == 0: #標籤為0 為Normal data
           x_train_normal.append(x_train[i].reshape((64, 64, 3))) 
    x_train_normal = np.array(x_train_normal)
    image_shape = x_train_normal[0].shape
    
    print("trainnig data number:",len(x_train_normal)) #顯示訓練資料數量
    
    
    
    dcgan = DCGAN(input_dim, image_shape)
    
    #model summary 
    
    dcgan.g.summary()
    dcgan.d.summary()
    
    # train generator & discriminator
    dcgan.compile(g_optim, d_optim)
    g_losses, d_losses = dcgan.train(epochs, batch_size, x_train_normal,save_image_epoch) 
    
    