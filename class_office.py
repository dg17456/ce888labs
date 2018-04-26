# -*- coding: utf-8 -*-

from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing

from timeit import default_timer as timer
import numpy as np
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from snes import SNES
from keras.applications.densenet import DenseNet121
from tensorflow.examples.tutorials.mnist import input_data
import cPickle as pkl
import urllib
import os
import tarfile
import skimage
import skimage.io
import skimage.transform
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# use just a small sample of the train set to test
SAMPLE_SIZE = 1024
# how many different sets of weights ask() should return for evaluation
POPULATION_SIZE = 10
# how many times we will loop over ask()/tell()
GENERATIONS = 30
##FUNCTIONS    

# input image dimensions
img_rows, img_cols = 80, 80
num_classes = 10


def ponder(acc):
    accuracy = 4*(np.square(acc-0.5))
    return accuracy

def ponder2(acc):
    accuracy = acc*acc
    return accuracy

def train_classifier(model, X, y):
    X_features = model.predict(X)
    pca=PCA(.95)
    pca.fit(X_features)
    X_features=pca.transform(X_features)
    
    #clf = ExtraTreesClassifier(n_estimators=100, n_jobs=4)
    clf = DecisionTreeClassifier()

    clf.fit(X_features, y)
    y_pred = clf.predict(X_features)
    return clf, y_pred,pca


def predict_classifier(model, clf, X,pca):
    X_features = model.predict(X)
    X_features=pca.transform(X_features)
    return clf.predict(X_features)

class NNWeightHelper:
    def __init__(self, model):
        self.model = model
        self.init_weights = K.get_session().run(self.model.trainable_weights)


    def _set_trainable_weight(self, model, weights):
        """Sets the weights of the model.

        # Arguments
            model: a keras neural network model
            weights: A list of Numpy arrays with shapes and types matching
                the output of `model.trainable_weights`.
        """
        tuples = []
        for layer in model.layers:
            num_param = len(layer.trainable_weights)
            layer_weights = weights[:num_param]
            for sw, w in zip(layer.trainable_weights, layer_weights):
                tuples.append((sw, w))
            weights = weights[num_param:]
        K.batch_set_value(tuples)


    def set_weights(self, weights):
        new_weights = []
        total_consumed = 0
        for w in self.init_weights:
            layer_shape, layer_size = w.shape, w.size
            chunk = weights[total_consumed:total_consumed + layer_size]
            total_consumed += layer_size
            new_weights.append(np.array(chunk.reshape(layer_shape)))

        self._set_trainable_weight(self.model, new_weights)


    def get_weights(self):
        W_list = K.get_session().run(self.model.trainable_weights)
        W_flattened_list = [k.flatten() for k in W_list]
        W = np.concatenate(W_flattened_list)
        return W

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
target_label=[]

target_train=[]

source_label=[]
source_set=[]
for j in range(np.shape(root_path)[0]):
    if j==2: #if dslr => test, target database
        for i in range(num_labels[0]):
            subpath=os.path.join(root_path[j],directories[i])
            for k in range(len(os.listdir(subpath))):
                path= os.path.join(root_path[j],directories[i],"frame_%04d.jpg" %(k+1))
                im=(Image.open(path))
                im = im.resize((80,80),resample=0)
                arra=PIL2array(im)
                target_set=np.append(target_set,arra)
                target_label.append(directories[i]) #for computing accuracy
                im.close()

            
    elif j==0: #if amazon => train, source database
        for i in range(num_labels[0]):
            subpath=os.path.join(root_path[j],directories[i])
            for k in range(len(os.listdir(subpath))):
                A=[]
                path= os.path.join(root_path[j],directories[i],"frame_%04d.jpg" %(k+1))
                im=(Image.open(path))
                im = im.resize((80,80),resample=0)
                ara=PIL2array(im)
                source_set=np.append(source_set,ara)
                source_label.append(directories[i])
                im.close()

            
#im=Image.open(path)
#x_train = np.arange(np.shape(source_set)[0]*im.size[0]*im.size[1]*3).reshape((np.shape(source_set)[0], im.size[0], im.size[1], 3))            
#for i in range(x_train.shape[0]):
#    x_train[i,:,:,:]=
source_set=np.reshape(source_set,(source_set.shape[0]/19200,80,80,3),order='C')
target_set=np.reshape(target_set,(target_set.shape[0]/19200,80,80,3),order='C')               
    
target_set=target_set.astype(np.uint8)
source_set=source_set.astype(np.uint8)
       
x_train=source_set
y_train=source_label
train_dom=np.zeros(len(x_train))

x_test=target_set
y_test=target_label
test_dom=np.ones(len(x_test))

le = preprocessing.LabelEncoder()

y_train=le.fit_transform(y_train)
y_test=le.transform(y_test)



if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#model = Sequential()
#model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
## model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Flatten())
#model.add(Dense(6, activation='relu'))


model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
#model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Flatten())
#model.add(Dense(128, activation='relu'))
model.add(Dense(1024, activation='relu'))


# this is irrelevant for what we want to achieve
model.compile(loss="mse", optimizer="adam")
print("compilation is over")

nnw = NNWeightHelper(model)
weights = nnw.get_weights()
ori_weights=weights.copy()


def main():
    print("Total number of weights to evolve is:", weights.shape)

    all_examples_indices = list(range(x_train.shape[0]))


    clf, _,pca = train_classifier(model, x_train, y_train)
    y_pred = predict_classifier(model, clf, x_test,pca)

    clf_dom, _,pca = train_classifier(model, x_train, train_dom)
    y_pred_dom = predict_classifier(model, clf_dom, x_test,pca)


    print(y_test.shape, y_pred.shape,y_pred_dom.shape)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_accuracy_dom=accuracy_score(test_dom,y_pred_dom)

    print('Non-trained NN Test accuracy:', test_accuracy)
    print('Non-trained NN Test Domain accuracy:', test_accuracy_dom)

    # print('Test MSE:', test_mse)

    snes = SNES(weights, 1, POPULATION_SIZE)
    for i in range(0, GENERATIONS):
        start = timer()
        asked = snes.ask()

        # to be provided back to snes
        told = []
        # use a small number of training samples for speed purposes
        subsample_indices = np.random.choice(all_examples_indices, size=SAMPLE_SIZE, replace=False)

        # evaluate on another subset
        subsample_indices_valid = np.random.choice(all_examples_indices, size=SAMPLE_SIZE + 1, replace=False)

        # iterate over the population
        for asked_j in asked:
            # set nn weights
            nnw.set_weights(asked_j)
            # train the classifer and get back the predictions on the training data
            clf, _,pca = train_classifier(model, x_train[subsample_indices], y_train[subsample_indices])
            y_pred = predict_classifier(model, clf, x_train[subsample_indices_valid],pca)

            clf_dom, _,pca = train_classifier(model, x_train[subsample_indices], train_dom[subsample_indices])
            # calculate the predictions on a different set
            y_pred_dom = predict_classifier(model, clf_dom, x_train[subsample_indices_valid],pca)
            
            score_clas = accuracy_score(y_train[subsample_indices_valid], y_pred)
            score_dom = accuracy_score(train_dom[subsample_indices_valid], y_pred_dom)
            score=ponder2(score_clas)-ponder(score_dom)
            # clf, _ = train_classifier(model, x_train, y_train)
            # y_pred = predict_classifier(model, clf, x_test)
            # score = accuracy_score(y_test, y_pred)
            # append to array of values that are to be returned
            told.append(score)
            print('scls:',score_clas,'p2:',ponder2(score_clas),' dm:',score_dom,'p:',ponder(score_dom),'sc:',score)

        snes.tell(asked, told)
        end = timer()
        print("It took", end - start, "seconds to complete generation", i + 1)

    nnw.set_weights(snes.center)

    clf,_,pca = train_classifier(model, x_train, y_train)
    y_pred = predict_classifier(model, clf, x_test,pca)

    clf_dom,_,pca = train_classifier(model, x_train, train_dom)
    y_pred_dom = predict_classifier(model, clf_dom, x_test,pca)


    print(y_test.shape, y_pred.shape,y_pred_dom.shape)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_accuracy_dom = accuracy_score(test_dom, y_pred_dom)
    
    


    print('Test accuracy target:', test_accuracy)
    print('Test accuracy target domain:', test_accuracy_dom)
    


if __name__ == '__main__':
    main()


 
