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


# Generator
class Generator(object):
    def __init__(self, input_dim, image_shape):
    
        INITIAL_CHANNELS = 128
        INITIAL_SIZE = 16
 
        inputs = Input((input_dim,))
        fc1 = Dense(input_dim=input_dim, units=INITIAL_CHANNELS * INITIAL_SIZE * INITIAL_SIZE)(inputs)
        fc1 = BatchNormalization()(fc1)
        fc1 = LeakyReLU(0.2)(fc1)
        fc2 = Reshape((INITIAL_SIZE, INITIAL_SIZE, INITIAL_CHANNELS), input_shape=(INITIAL_CHANNELS * INITIAL_SIZE * INITIAL_SIZE,))(fc1)
        up1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(fc2)
        conv1 = Conv2D(64, (3, 3), padding='same')(up1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        up2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv1)
        conv2 = Conv2D(image_shape[2], (5, 5), padding='same')(up2)
        outputs = Activation('tanh')(conv2)
 
        self.model = Model(inputs=[inputs], outputs=[outputs])
 
    def get_model(self):
        return self.model
 
# Discriminator
class Discriminator(object):
    def __init__(self, input_shape):
        inputs = Input(input_shape)
        conv1 = Conv2D(64, (5, 5), padding='same')(inputs)
        conv1 = LeakyReLU(0.2)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, (5, 5), padding='same')(pool1)
        conv2 = LeakyReLU(0.2)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        fc1 = Flatten()(pool2)
        fc1 = Dense(1)(fc1)
        outputs = Activation('sigmoid')(fc1)
 
        self.model = Model(inputs=[inputs], outputs=[outputs])
 
    def get_model(self):
        return self.model
 
# DCGAN
class DCGAN(object):
    def __init__(self, input_dim, image_shape):
        self.input_dim = input_dim
        self.d = Discriminator(image_shape).get_model()
        self.g = Generator(input_dim, image_shape).get_model()
 
    def compile(self, g_optim, d_optim):
        self.d.trainable = False
        self.dcgan = Sequential([self.g, self.d])
        self.dcgan.compile(loss='binary_crossentropy', optimizer=g_optim)
        self.d.trainable = True
        self.d.compile(loss='binary_crossentropy', optimizer=d_optim)
        K.set_learning_phase(1)
    #載入權重
    def load_weights(self, g_weight, d_weight):
        self.g.load_weights(g_weight)
        self.d.load_weights(d_weight)
        
    def train(self, epochs, batch_size, X_train,save_image_epoch):
        g_losses = []
        d_losses = []
        for epoch in range(epochs):
            np.random.shuffle(X_train)# 打亂訓練資料
            n_iter = X_train.shape[0] // batch_size #計算疊代次數
            
            progress_bar = Progbar(target=n_iter)
            
            for index in range(n_iter):
                # create random noise -> N latent vectors
                noise = np.random.uniform(-1, 1, size=(batch_size, self.input_dim))
                
                # load real data & generate fake data
                image_batch = X_train[index * batch_size:(index + 1) * batch_size]
                
                #圖片翻轉
                '''
                for i in range(batch_size):
                    if np.random.random() > 0.75:
                        image_batch[i] = np.fliplr(image_batch[i])
                    if np.random.random() < 0.5:
                        image_batch[i] = np.flipud(image_batch[i])
                '''
                generated_images = self.g.predict(noise)
                
                # attach label for training discriminator
                X = np.concatenate((image_batch, generated_images))
                y = np.array([1] * batch_size + [0] * batch_size)
 
                # training discriminator
                d_loss = self.d.train_on_batch(X, y)
 
                # training generator
                g_loss = self.dcgan.train_on_batch(noise, np.array([1] * batch_size))
 
                progress_bar.update(index, values=[('g', g_loss), ('d', d_loss)])
                
                image = self.combine_images(generated_images)
                image = image * 255.0
                cv2.imwrite('./result/Iteration.png', image)
                
            g_losses.append(g_loss)
            d_losses.append(d_loss)
            
            #保存訓練過程圖
            
            if (epoch+1)%save_image_epoch == 0:
                image = self.combine_images(generated_images)
                image = image * 255.0
                cv2.imwrite('./result/Epoch_' + str(epoch+1) + ".png", image)
            
            

                
                
            print('\nEpoch ' + str(epoch+1) + " end")
 
            # save weights for each epoch
            if epoch == epochs-1 :
            
                self.g.save_weights('model/generator.h5')
                self.d.save_weights('model/discriminator.h5')
                
                
        return g_losses, d_losses
        
    #保存訓練過程圖
    def combine_images(self, generated_images):
        num = generated_images.shape[0]
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num) / width))
        shape = generated_images.shape[1:4]
        image = np.zeros((height * shape[0], width * shape[1], shape[2]),
                         dtype=generated_images.dtype)
        for index, img in enumerate(generated_images):
            i = int(index / width)
            j = index % width
            image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], :] = img[:, :, :]
        return image
 

    
    
