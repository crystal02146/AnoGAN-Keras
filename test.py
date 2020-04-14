# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:22:44 2020

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
from os import listdir

import tensorflow as tf
import time

#限定GPU使用量

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

from DCGAN import DCGAN
from AnoGAN import ANOGAN


#反正規化 0~1 變成 0~255之間

def denormalize(X):

    for i in range (0,X.shape[1]): #Height 
        for j in range (0,X.shape[0]): # Width
        
            if(X[i,j,0] < 0):
                X[i,j,0] = 0
            if(X[i,j,1] < 0):
                X[i,j,1] = 0
            if(X[i,j,2] < 0):
                X[i,j,2] = 0

    
    return (X * 255.0).astype(dtype=np.uint8)


#AnoGAN model 
class anogan_model(object):
    
    #K.clear_session()
    
    def __init__(self,model_path,input_dim,image_shape):
    
        anogan_optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)#AnoGAN 優化器
        
        
        dcgan = DCGAN(input_dim, image_shape)#宣告Generator 與 descriminater 空model
        
        # load weights
        dcgan.load_weights(model_path+"Generator.h5",model_path+"Discriminator.h5")#載入權重

        self.anogan = ANOGAN(input_dim, dcgan.g)#輸入雜訊維度z 輸入生成器模型 
        self.anogan.compile(anogan_optim) #設定優化器 Loss值

    
    def detection(self,img,iterations):
        
        h,w,c = img.shape 
        
        img = img.reshape((h, w, c)) #重新調整維度大小
        img = img.astype('float32') / 255 #數值正規化0到1之間
        test_img = img[np.newaxis,:,:,:] #由於只有一筆資料 將(64,64,3)變成(1,64,64,3)維度

        anomaly_score, generated_img = self.anogan.compute_anomaly_score(test_img, iterations)#計算異常分數 跟 生成影像
        
        
        test_img = denormalize(test_img[0])  #數值反正規化將0~1的資料轉成0~255之間
        gen_img = denormalize(generated_img[0])#數值反正規化將0~1的資料轉成0~255之間
        

        
        return anomaly_score , test_img , gen_img
    

    
    
#自我測試程式
if __name__ == '__main__':
        
        file_path = "./dataset/test/abnormal/"#檔案路徑
        files = listdir(file_path)
        
        w = 64
        h = 64
        c = 3 
        
        image_shape = ((h,w,c))
        input_dim = 100
        iterations = 100 #迭代次數
        
        
        AnoGAN = anogan_model("./model/",input_dim,image_shape)
        
        
        #show summury
        AnoGAN.anogan.model.summary()
        

        
        for f in files:
            
            img = cv2.imread(file_path+f)
            img = cv2.resize(img, (w, h))
            
            tStart = time.time()#計時開始

            anomaly_score , test_img , gen_img = AnoGAN.detection(img,iterations)

            tEnd = time.time()#計時結束
            
            #print infomation to screen
            
            print("\n")
            print("File name : "+str(f))
            print("Anomaly Score : "+str(anomaly_score))
            print("Cost time : "+str(tEnd - tStart))
            
            #show image to screen 
            
            imgs = np.concatenate((test_img , gen_img), axis=1) #合併圖片
            imgs = cv2.resize(imgs, (w*8, h*4))
            
            cv2.imshow('Windows', imgs)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            