


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


#%matplotlib inline
print ("PACKAGES LOADED")

filelink = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz'
filename = 'BSR_bsds500.tgz'
if os.path.isfile(filename):
    print ("[%s] ALREADY EXISTS." % (filename))
else:
    print ("DOWNLOADING %s ..." % (filename))
    urllib.urlretrieve(filelink, filename)
    print ("DONE")


# use just a small sample of the train set to test
SAMPLE_SIZE = 1024
# how many different sets of weights ask() should return for evaluation
POPULATION_SIZE = 10
# how many times we will loop over ask()/tell()
GENERATIONS = 60
##FUNCTIONS    
def compose_image(digit, background):
    """Difference-blend a digit and a random patch from a background image."""
    w, h, _ = background.shape
    dw, dh, _ = digit.shape
    x = np.random.randint(0, w - dw)
    y = np.random.randint(0, h - dh)
    bg = background[x:x+dw, y:y+dh]
    return np.abs(bg - digit).astype(np.uint8)

def mnist_to_img(x):
    """Binarize MNIST digit and convert to RGB."""
    x = (x > 0).astype(np.float32)
    d = x.reshape([28, 28, 1]) * 255
    return np.concatenate([d, d, d], 2)

def create_mnistm(X):
    """
    Give an array of MNIST digits, blend random background patches to
    build the MNIST-M dataset as described in
    http://jmlr.org/papers/volume17/15-239/15-239.pdf
    """
    X_ = np.zeros([X.shape[0], 28, 28, 3], np.uint8)
    for i in range(X.shape[0]):
        bg_img = rand.choice(background_data)
        d = mnist_to_img(X[i])
        d = compose_image(d, bg_img)
        X_[i] = d
    return X_

def unison_shuffled_copies(a, b, c):
    assert len(a) == len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]

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


# input image dimensions
img_rows, img_cols = 28, 28
num_classes = 10
print ("FUNCTIONS READY")
mnistm_name = 'mnistm_data.pkl'
if os.path.isfile(mnistm_name):
    print ("[%s] ALREADY EXISTS. " % (mnistm_name))
else:
    mnist = input_data.read_data_sets('data')
    # OPEN BSDS500
    f = tarfile.open(filename)
    train_files = []
    for name in f.getnames():
        if name.startswith('BSR/BSDS500/data/images/train/'):
            train_files.append(name)
    print ("WE HAVE [%d] TRAIN FILES" % (len(train_files)))
    # GET BACKGROUND
    print ("GET BACKGROUND FOR MNIST-M")
    background_data = []
    for name in train_files:
        try:
            fp = f.extractfile(name)
            bg_img = skimage.io.imread(fp)
            background_data.append(bg_img)
        except:
            continue
    print ("WE HAVE [%d] BACKGROUND DATA" % (len(background_data)))
    rand = np.random.RandomState(42)
    print ("BUILDING TRAIN SET...")
    train = create_mnistm(mnist.train.images)
    print ("BUILDING TEST SET...")
    test = create_mnistm(mnist.test.images)
    print ("BUILDING VALIDATION SET...")
    valid = create_mnistm(mnist.validation.images)
    # SAVE
    print ("SAVE MNISTM DATA TO %s" % (mnistm_name))
    with open(mnistm_name, 'w') as f:
        pkl.dump({ 'train': train, 'test': test, 'valid': valid }, f, -1)
    print ("DONE")
###MAIN
print ("LOADING MNIST")
mnist        = input_data.read_data_sets('data', one_hot=True)
mnist_train  = (mnist.train.images > 0).reshape(55000, 28, 28, 1).astype(np.uint8) * 255
mnist_train  = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
mnist_test   = (mnist.test.images > 0).reshape(10000, 28, 28, 1).astype(np.uint8) * 255
mnist_test   = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
mnist_train_label = mnist.train.labels
mnist_test_label = mnist.test.labels
print ("LOADING MNIST-M")
mnistm_name  = 'mnistm_data.pkl'
mnistm       = pkl.load(open(mnistm_name))
mnistm_train = mnistm['train']
mnistm_test  = mnistm['test']
mnistm_valid = mnistm['valid']
mnistm_train_label = mnist_train_label
mnistm_test_label = mnist_test_label
mnist_train_dom=np.zeros(len(mnist_train))
mnist_test_dom=np.zeros(len(mnist_test))
mnistm_train_dom=np.ones(len(mnistm_train))
mnistm_test_dom=np.ones(len(mnistm_test))

train_label=[]
for i in range(len(mnistm_train_label[:,0])):
    for j in range(len(mnistm_train_label[0,:])):
        if mnistm_train_label[i,j]==1:
            train_label=np.append(train_label,j)
            
mnist_train_label=train_label.copy()        
mnistm_train_label=train_label.copy()    

test_label=[]
for i in range(len(mnistm_test_label[:,0])):
    for j in range(len(mnistm_test_label[0,:])):
        if mnistm_test_label[i,j]==1:
            test_label=np.append(test_label,j)
            
mnist_test_label=test_label.copy()
mnistm_test_label=test_label.copy()        
       
X_train_dom=np.concatenate([mnist_train,mnistm_train])
Y_train=np.concatenate([mnist_train_label,mnistm_train_label])   
Y_train_dom=np.concatenate([mnist_train_dom,mnistm_train_dom])

X_train_dom_dw=X_train_dom[::2]
Y_train_dw=Y_train[::2]
Y_train_dom_dw=Y_train_dom[::2]
#X_test=np.concatenate([mnist_test,mnistm_test])
#Y_test=np.concatenate([mnist_test_label,mnistm_test_label])
#Y_test_dom=np.concatenate([mnist_test_dom,mnistm_test_dom])

#[x_test,y_test,y_test_dom]=unison_shuffled_copies(X_test,Y_test,Y_test_dom)
[x_train_dom,y_train,y_train_dom]=unison_shuffled_copies(X_train_dom_dw,Y_train_dw,Y_train_dom_dw)

x_train=x_train_dom
#y_train=mnist_train_label
#y_train_dom=mnist_train_dom

x_test=mnistm_test
y_test=mnistm_test_label
y_test_dom=mnistm_test_dom

x_test_s=mnist_test
y_test_s=mnist_test_label
y_test_dom_s=mnist_test_dom


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    x_train_dom=x_train_dom.reshape(x_train_dom.shape[0],3,img_rows,img_cols)
    x_test_s = x_test_s.reshape(x_test_s.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    x_train_dom=x_train_dom.reshape(x_train_dom.shape[0],img_rows,img_cols,3)
    input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train_dom = x_train_dom.astype('float32')
x_train /= 255
x_test /= 255
x_train_dom /= 255


# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
print('x_train shape:', x_train.shape)
print('x_train_dom shape:', x_train_dom.shape)
print(x_train.shape[0], 'train samples')
print(x_train_dom.shape[0], 'train_dom samples')
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

    clf_dom, _,pca = train_classifier(model, x_train_dom, y_train_dom)
    y_pred_dom = predict_classifier(model, clf_dom, x_test,pca)


    print(y_test.shape, y_pred.shape,y_pred_dom.shape)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_accuracy_dom=accuracy_score(y_test_dom,y_pred_dom)

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

            clf_dom, _,pca = train_classifier(model, x_train_dom[subsample_indices], y_train_dom[subsample_indices])
            # calculate the predictions on a different set
            y_pred_dom = predict_classifier(model, clf_dom, x_train_dom[subsample_indices_valid],pca)
            
            score_clas = accuracy_score(y_train[subsample_indices_valid], y_pred)
            score_dom = accuracy_score(y_train_dom[subsample_indices_valid], y_pred_dom)
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
    y_pred_s = predict_classifier(model, clf, x_test_s,pca)

    clf_dom,_,pca = train_classifier(model, x_train, y_train_dom)
    y_pred_dom = predict_classifier(model, clf_dom, x_test,pca)
    y_pred_dom_s = predict_classifier(model, clf_dom, x_test_s,pca)


    print(y_test.shape, y_pred.shape,y_pred_dom.shape)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_accuracy_dom = accuracy_score(y_test_dom, y_pred_dom)
    
    test_accuracy_s = accuracy_score(y_test_s, y_pred_s)
    test_accuracy_dom_s = accuracy_score(y_test_dom_s, y_pred_dom_s)
    


    print('Test accuracy target:', test_accuracy)
    print('Test accuracy target domain:', test_accuracy_dom)
    
    print('Test accuracy target:', test_accuracy_s)
    print('Test accuracy target domain:', test_accuracy_dom_s)



if __name__ == '__main__':
    main()
