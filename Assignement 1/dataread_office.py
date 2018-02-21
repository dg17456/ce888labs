# -*- coding: utf-8 -*-
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt


def PIL2array(img):
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)
    



root_path=["","",""]
root_path[0]='/Users/Didac/Documents/DSDM/Original_images/amazon/images/'
root_path[1]='/Users/Didac/Documents/DSDM/Original_images/dslr/images/'
root_path[2]='/Users/Didac/Documents/DSDM/Original_images/webcam/images/'

num_labels=np.int64([0])
num_labels[0]=len(list(os.walk(root_path[0])))-1
directories=['back_pack','bike','bike_helmet','bookcase','bottle','calculator',
             'desk_chair','desk_lamp', 'desktop_computer', 'file_cabinet', 
             'headphones', 'keyboard', 'laptop_computer', 'letter_tray', 
             'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 
             'phone', 'printer', 'projector', 'punchers', 'ring_binder', 'ruler', 
             'scissors','speaker', 'stapler', 'tape_dispenser', 'trash_can' ]

#A=np.array=(3,32,100,300,300,3)
target_set=[]
target_set_label=[]

target_train=[]

source_set=[]
source_label=[]

for j in range(np.shape(root_path)[0]):
    if j==2: #if dslr => test, target database
        for i in range(num_labels[0]):
            subpath=os.path.join(root_path[j],directories[i])
            for k in range(len(os.listdir(subpath))):
                path= os.path.join(root_path[j],directories[i],"frame_%04d.jpg" %(k+1))
                im=(Image.open(path))
                if k< 10:
                    target_set.append(im)
                    target_set_label.append(directories[i]) #for computing accuracy
                else:
                    target_train.append(im)
            
    elif j==0: #if amazon => train, source database
        for i in range(num_labels[0]):
            subpath=os.path.join(root_path[j],directories[i])
            for k in range(len(os.listdir(subpath))):
                path= os.path.join(root_path[j],directories[i],"frame_%04d.jpg" %(k+1))
                im=(Image.open(path))
                source_set.append(im)
                source_label.append(directories[i])
            
            
train_set=np.sum([source_set,target_train])
#test_set=target_set


 
