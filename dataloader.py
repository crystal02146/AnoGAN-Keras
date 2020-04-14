# -*- codiabnormal: utf-8 -*-
"""
Created on Tue Oct 15 14:22:44 2019

@author: Tseng Yi-Yao
"""

from os import listdir
import numpy as np
import cv2

def load_file(file_path,x_data,y_label,label):
    
    #Resize image
    w = 64
    h = 64
    
    
    files = listdir(file_path)
    
    for f in files:
        
        img = cv2.imread(file_path+f)
        #print(f) # show file name 
        img = cv2.resize(img, (w, h))
        x_data.append(img)
        y_label.append(label)
        
    
    return x_data , y_label 

# Train data
def load_data():

    # train data list
    x_train = []
    y_train = []
    
    x_train , y_train = load_file("./dataset/train/normal/",x_train,y_train,0)
    x_train , y_train = load_file("./dataset/train/abnormal/",x_train,y_train,1)
    
    x_train  = np.asarray(x_train)
    y_train  = np.asarray(y_train)
    

    
    # test data list
    x_test = []
    y_test = []
    
    x_test , y_test = load_file("./dataset/test/normal/",x_test,y_test,0)
    x_test , y_test = load_file("./dataset/test/abnormal/",x_test,y_test,1)
    
    x_test  = np.asarray(x_test)
    y_test  = np.asarray(y_test)
    
    # show data shape 
    print()
    print("Training data shape : "+str(x_train.shape))
    print("Training label shape : "+str(y_train.shape))
    print("Testing data shape : "+str(x_test.shape))
    print("Testing label shape : "+str(y_test.shape))
    print()
    
    return x_train,y_train,x_test,y_test
    
if __name__ == '__main__':
    
    x_train, y_train, x_test, y_test = load_data()
