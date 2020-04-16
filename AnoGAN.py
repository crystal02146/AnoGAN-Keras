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

#殘差Loss值
def sum_of_residual(y_true, y_pred):
    
    
    return K.sum(K.abs(y_true - y_pred))
    
    
#AnoGAN (原始DCGAN新增FC全連結層並串接生成器層 只訓練FC連結層其餘凍結不進行訓練)
class ANOGAN(object):

    def __init__(self, input_dim, g):
        self.input_dim = input_dim #輸入雜訊Z
        self.g = g #生成器
        g.trainable = False #生成器不進行訓練
        
        anogan_in = Input(shape=(input_dim,))#與輸入層相同的維度
        g_in = Dense((input_dim), activation='tanh', trainable=True)(anogan_in)#新增FC全連結層進行訓練
        g_out = g(g_in) #生成器輸出維度大小
        self.model = Model(inputs=anogan_in, outputs=g_out)#宣告AnoGAN Model 
        
    #AnoGAN 設定優化器 Loss值 
    def compile(self, optim):
        
        self.model.compile(loss=sum_of_residual, optimizer=optim)
        K.set_learning_phase(0)
 
    def compute_anomaly_score(self, x, iterations):
        
        
        z = np.random.uniform(-1, 1, size=(1, self.input_dim))
       
        #訓練 Model z為輸入維度 x為輸出維度 batch_size為一張影像訓練 不顯示訓練結果
        loss = self.model.fit(z, x, batch_size=1, epochs=iterations, verbose=0)
        loss = loss.history['loss'][-1]
        
        similar_data = self.model.predict(z)
        
        return loss, similar_data
 

    
    