# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:22:44 2019

@author: Tseng Yi-Yao
"""

import matplotlib.pyplot as plt
import random
import seaborn as sns

#Read txt file

file = open('anomaly_score.txt')
text = []

for line in file:
    text.append(line)

# Text array to data list

normal = []
abnormal = []
label =[]
filename =[]
for i in range (len(text)):

    string = text[i].split(" ")
    
    filename.append(string[0])
    label.append(string[1])
    
    if label[i] ==  "normal":
    
        normal.append(int(float(string[2])))
    else:
        abnormal.append(int(float(string[2])))


#x 軸數量
if len(normal) > len(abnormal):
    bins = len(normal)
else:
    bins = len(abnormal)

#plt.hist([normal,abnormal], bins=bins,label = ['Normal','Abnormal'],color=['blue','red'],alpha=0.5)

sns.distplot(normal,label="Normal",color="blue")
sns.distplot(abnormal,label="Abnormal",color="red")


plt.title('Anomaly detection',fontsize=14)
plt.xlabel('Anomaly Score')
plt.ylabel('Probability')

                       
plt.legend()
plt.show()
 