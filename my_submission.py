# -*- coding: utf-8 -*-
'''
2018 IFN680 Assignment Two
group 025
Ellie Wang (n9913670); C.J. Ding (n10045091); Yixi Zhou (n9599291) 

Trains a Siamese MLP on pairs of digits from the MNIST dataset.

It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).

# References

- Dimensionality Reduction by Learning an Invariant Mapping
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

Gets to % test accuracy after 20 epochs.
Google GPU
'''

import numpy as np
import matplotlib.pyplot as plt
import random
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
from sklearn import model_selection
from tensorflow import keras


def euclidean_distance(vects):
    '''
    The function is to calculate the distance between the two embedding vectors.
    params: 
        vects: image information convert to high dimension vectors
    '''
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(sum_square)


def contrastive_loss(y_true, y_pred):
    '''
    The function is to calculate the contrastive loss.
    
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


def create_pairs(x, digit_indices, digits):
    '''This function is used to obtain the training pair,validation pair and testing pair
       parameters:
           x: The dataset, such as, x_train, x_test and x_validation
           digit_indices: np.array, the transformed list from the label list by using np.where
           digits: int, the digits in the dataset. 
           
       outputs:
           the output will be np.array format.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(digits)]) - 1
    for d in range(digits):
        for i in range(n):
            #positive pairs
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            
            #negative pairs
            dm = random.randrange(1, digits)
            dn = (d + dm) % digits
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            
            #add labels for each (positive and negative) pairs
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_complex_CNN_network(input_shape):
    '''
    params:
        input_shape: the input for the CNN network.
    output:
        model: the CNN network with multiple layers
    '''
    # build the Alexnet
    model = keras.models.Sequential()
     
    # Layer 1  convolutional layer
    # 16 kernals of size 5x5 to extract the features form image dataset
    model.add(keras.layers.Conv2D(16, (5, 5), input_shape = input_shape, padding='same')) 
    model.add(keras.layers.Activation('relu'))
    # choose highest value pooling with size 2x2
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        
    # Layer 2 convolutional layer
    # 32 kernals of size 3x3 to extract features from prevous layer 
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same'))
    model.add(keras.layers.Activation('relu'))  
    # select highest value in pooling with size 2x2
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        
    # Layer 3 convolutional layer
    #zero padding with size 1x1
    model.add(keras.layers.ZeroPadding2D((1,1)))
    # 128 kernals of size 1x1, case-senetive padding for select data features from prevous layer 
    model.add(keras.layers.Conv2D(128, (1, 1), padding='same'))
    model.add(keras.layers.Activation('relu'))
               
    # Layer 4  fully-connected layer
    # convert to 2D layer
    model.add(keras.layers.Flatten())
    # 64 kernels initial with glorot normal
    model.add(keras.layers.Dense(64, kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(keras.layers.Activation('relu'))
    #avoid overfitting to dropout 30% of data from prevous layer 
    model.add(keras.layers.Dropout(0.3)) 
       
    # Layer 5 fully-connected layer
    # initial with glorot normal by 64 kernels 
    model.add(keras.layers.Dense(64, kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(keras.layers.Activation('relu'))
    #avoid overfitting to dropout 40% of data from prevous layer 
    model.add(keras.layers.Dropout(0.4))
    
    # Layer 6 fully-connected layer
    # initial with glorot normal and use dense layer to represents for 128 output classes
    model.add(keras.layers.Dense(128, kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(keras.layers.Activation('relu'))
    
    return model

  
def create_base_network():
    '''
    Base network contain three Dense layers, one dropout layer and one flatten layer 
    output:
        model created by base network 
    '''
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    return model 

def compute_accuracy(y_true, y_pred):
    '''
    The function is to compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''
    The function is to compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def split_dataset():
    '''
    The function is to split set A as 80% for training (Among them, 80% for training, 20% for testing) and 20% for testing. 
    output:
        x_train: 80% of MNIST dataset 
        y_train:  20% of MNIST dataset
        x_test_1:  [0,1,8,9] dights test data  
        y_test_1:  [0,1,8,9] dights test data  
        x_test_2: 20% of [2-7] dights data for testing 
        y_test_2: 20% of [2,3,4,5,6,7] dights data for testing 
        x_test: numpy array which contain the x_tes1, and x_test2 data
        y_test: numpy array which contain the y_tes1, and y_test2 data
        x_validation: 20% of trainng dataset 
        y_validation: 20% of trainng dataset 
    '''
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # get all x_train and x_test data 
    x_all = np.append(x_train,x_test,axis=0)
    #get all y_train and y_test data
    y_all = np.append(y_train,y_test,axis=0)
    
    set_a =[2,3,4,5,6,7]
    set_b =[0,1,8,9]
    #create a filter to select data from [2,3,4,5,6,7] dights    
    a_filter = np.isin(y_all, set_a)
    # create a filter to select data from [0,1,8,9] dights 
    b_filter = np.isin(y_all, set_b)
    # get the x and y train data bt filter     
    x_train, y_train = x_all[a_filter], y_all[a_filter]
    x_test_1, y_test_1 = x_all[b_filter], y_all[b_filter]
    #split the whole dataset to 80% train dataset and 20% test dataset 
    x_train, x_test_2,  y_train, y_test_2=model_selection.train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    #split the train dataset to 80% train dataset and 20% validation dataset 
    x_train, x_validation, y_train, y_validation =model_selection.train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    #create numpy array to store all the data 
    x_test = np.append(x_test_1,x_test_2,axis=0)
    y_test = np.append(y_test_1,y_test_2,axis=0)
    
    return x_train, y_train, x_test_1, y_test_1, x_test_2, y_test_2, x_test, y_test,x_validation, y_validation


def run_base_network(x_train, y_train, x_test, y_test, epochs, train_sets, test_sets, x_validation, y_validation):    
    '''
   params:
       x_train: 80% of whole datasets which come from  [2,3,4,5,6,7] digts 
       y_train: 80% of whole datsets  whch come from [0,1,8,9] dights
       x_test:  test dataset which come from [0,1,8,9] dights
       y_test:  test dataset which come from [0,1,8,9] dights
       epochs:  the times to run whole neuroul network 
       train_sets:  [2,3,4,5,6,7 ] dights dataset 
       test_sets: three test pair of dataset which include first pair ([2,3,4,5,6,7]), second pair ([0-9]) and third pair ([0,1,8,9]) 
       x_validation: 20% of train datasets
       y_vadilation: 20% of train datasets
   output:
       tr_acc: train accuracy 
       te_acc: test accuracy
       result: modle created by base network     
    '''
    
    input_shape = (28,28,1) # height, width and depth of the input images
    x_train = x_train.reshape(x_train.shape[0], *input_shape) #reshape nparray
    x_test = x_test.reshape(x_test.shape[0], *input_shape) #reshape nparray
    x_validation = x_validation.reshape(x_validation.shape[0], *input_shape)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_validation = x_validation.astype('float32')
    x_train /= 255
    x_test /= 255
    x_validation /=255
    
    
    trainsets = len(train_sets)
    testsets = len(test_sets)
    # create training+test positive and negative pairs
    #digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
    digit_indices = [np.where(y_train == i)[0] for i in train_sets]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices, trainsets)
    
    #digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
    te_digit_indices = [np.where(y_test == i)[0] for i in test_sets]
    te_pairs, te_y = create_pairs(x_test, te_digit_indices, testsets)
   
    #validation pair 
    va_digit_indices = [np.where(y_validation == i)[0] for i in train_sets]
    va_pairs, va_y = create_pairs(x_validation, va_digit_indices, trainsets)
    # cnn network 
    run_base_network = create_base_network()
    
    #base network
    #run_network = create_complex_CNN_networ(input_shape)    
    
    input_a = keras.layers.Input(shape=(input_shape))
    input_b = keras.layers.Input(shape=(input_shape))
    
    
    # the weights of the network
    # will be shared across the two branches
    processed_a = run_base_network(input_a)
    processed_b = run_base_network(input_b)
    
    distance = keras.layers.Lambda(euclidean_distance)([processed_a, processed_b])

    model = keras.models.Model([input_a, input_b], distance)

    # train
    rms = keras.optimizers.RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
    result = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=96,
          epochs=epochs,
          validation_data=([va_pairs[:, 0], va_pairs[:, 1]], va_y))
          
  
    # compute final accuracy on training and test sets
    y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(tr_y, y_pred)
    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(te_y, y_pred)   

    return tr_acc, te_acc, result 

def run_complex_CNN_network(x_train, y_train, x_test, y_test, epochs, train_sets, test_sets, x_validation, y_validation):    
    '''
       params:
           x_train: 80% of whole datasets which come from  [2,3,4,5,6,7] digts 
           y_train: 80% of whole datsets  whch come from [0,1,8,9] dights
           x_test:  test dataset which come from [0,1,8,9] dights
           y_test:  test dataset which come from [0,1,8,9] dights
           epochs:  the times to run whole neuroul network 
           train_sets:  [2,3,4,5,6,7 ] dights dataset 
           test_sets: three test pair of dataset which include first pair ([2,3,4,5,6,7]), second pair ([0-9]) and third pair ([0,1,8,9]) 
           x_validation: 20% of train datasets
           y_vadilation: 20% of train datasets
       output:
           tr_acc: train accuracy 
           te_acc: test accuracy
           result: modle created by complex CNN network 
       
    '''   
    input_shape = (28,28,1) # height, width and depth of the input images
    x_train = x_train.reshape(x_train.shape[0], *input_shape) #reshape nparray
    x_test = x_test.reshape(x_test.shape[0], *input_shape) #reshape nparray
    x_validation = x_validation.reshape(x_validation.shape[0], *input_shape)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_validation = x_validation.astype('float32')
    x_train /= 255
    x_test /= 255
    x_validation /=255
    
    
    trainsets = len(train_sets)
    testsets = len(test_sets)
    # create training+test positive and negative pairs
    #digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
    digit_indices = [np.where(y_train == i)[0] for i in train_sets]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices, trainsets)
    
    #digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
    te_digit_indices = [np.where(y_test == i)[0] for i in test_sets]
    te_pairs, te_y = create_pairs(x_test, te_digit_indices, testsets)
   
    #validation pair 
    va_digit_indices = [np.where(y_validation == i)[0] for i in train_sets]
    va_pairs, va_y = create_pairs(x_validation, va_digit_indices, trainsets)
    # cnn network 
    run_complex_CNN_network = create_complex_CNN_network(input_shape)
    
    #base network
    #run_network = create_complex_CNN_networ(input_shape)    
    
    input_a = keras.layers.Input(shape=(input_shape))
    input_b = keras.layers.Input(shape=(input_shape))
    
    
    # the weights of the network
    # will be shared across the two branches
    processed_a = run_complex_CNN_network(input_a)
    processed_b = run_complex_CNN_network(input_b)
    
    distance = keras.layers.Lambda(euclidean_distance)([processed_a, processed_b])

    model = keras.models.Model([input_a, input_b], distance)

    # train
    rms = keras.optimizers.RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
    result = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=96,
          epochs=epochs,
          validation_data=([va_pairs[:, 0], va_pairs[:, 1]], va_y))
          
  
    # compute final accuracy on training and test sets
    y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(tr_y, y_pred)
    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(te_y, y_pred)   

    return tr_acc, te_acc, result 

def Draw_plot(result,epochs,name):
    '''
	The function is to draw the comparison of the training and validation loss.
    '''    
    print('\nThe Loss and Accuracy in {}:\n'.format(name))
    print([i for i in result.history.keys()])
       
    los=int(result.history['loss'][-1]*1000)/1000
    vlos=int(result.history['val_loss'][-1]*1000)/1000
    acc=int(result.history['accuracy'][-1]*1000)/1000
    vacc=int(result.history['val_accuracy'][-1]*1000)/1000
    #print(history.history['loss'])
    
    result.history['Epochs']=[i for i in range(epochs)]
    plt.plot('Epochs','loss',data=result.history)
    plt.plot('Epochs','val_loss',data=result.history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train','Validation'])
    plt.annotate('The Final Train loss: {}'.format(los),xy=(epochs-1,los))
    plt.annotate('The Final Validation loss: {}'.format(vlos),xy=(epochs-1,vlos))
    plt.xlim(0,epochs-1)
    plt.grid(True)
    plt.title('Siamese Network Loss (Sub-Network: {})'.format(name))
    plt.show()
    
    
    plt.plot('Epochs','accuracy',data=result.history)
    plt.plot('Epochs','val_accuracy',data=result.history)
    plt.legend(['Train','Validation'])
    plt.annotate('The Final Train Accuracy: {}'.format(acc),xy=(epochs-1,acc))
    plt.annotate('The Final Validation Accuracy: {}'.format(vacc),xy=(epochs-1,vacc))
    plt.xlim(0,epochs-1)
    plt.grid(True)
    plt.title('Siamese Network Accuacy (Sub-Network: {})'.format(name))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()


def dispaly_base_network():
    '''
	The function is to display the output of base network.
    '''
    
    train_sets = [2,3,4,5,6,7]
    epochs = 5
    x_train, y_train, x_test_1, y_test_1, x_test_2, y_test_2, x_test, y_test,x_validation, y_validation = split_dataset()
       
    print('-----------------------------Testing First Pair:------------------------------ ')
    test_sets = [2, 3, 4, 5, 6, 7]
    tr_acc, te_acc,information = run_base_network(x_train, y_train, x_test_2, y_test_2, epochs, train_sets, test_sets, x_validation,y_validation )
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

    
    print('-----------------------------Testing Second Union Pair:------------------------------ ')
    test_sets = [0, 1, 8, 9, 2, 3, 4, 5, 6, 7]
    tr_acc, te_acc, information = run_base_network(x_train, y_train, x_test, y_test, epochs, train_sets, test_sets, x_validation, y_validation)
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))     
    
    print('-----------------------------Testing Third Pair:------------------------------ ')
    test_sets = [0, 1, 8, 9]
    tr_acc, te_acc,information = run_base_network(x_train, y_train, x_test_1, y_test_1, epochs, train_sets, test_sets, x_validation, y_validation)
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))  
    
    Draw_plot(information, epochs, "Siamese network")
 
def dispaly_cnn_network():
   '''
	The function is to display the output of CNN network.
   '''      
   train_sets = [2,3,4,5,6,7]
   epochs = 5
   x_train, y_train, x_test_1, y_test_1, x_test_2, y_test_2, x_test, y_test,x_validation, y_validation = split_dataset()
       
   print('-----------------------------Testing First Pair:------------------------------ ')
   test_sets = [2, 3, 4, 5, 6, 7]
   tr_acc, te_acc,information = run_complex_CNN_network(x_train, y_train, x_test_2, y_test_2, epochs, train_sets, test_sets, x_validation,y_validation )
   print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
   print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    
    
   print('-----------------------------Testing Second Union Pair:------------------------------ ')
   test_sets = [0, 1, 8, 9, 2, 3, 4, 5, 6, 7]
   tr_acc, te_acc, information = run_complex_CNN_network(x_train, y_train, x_test, y_test, epochs, train_sets, test_sets, x_validation, y_validation)
   print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
   print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))     
    
   print('-----------------------------Testing Third Pair:------------------------------ ')
   test_sets = [0, 1, 8, 9]
   tr_acc, te_acc,information = run_complex_CNN_network(x_train, y_train, x_test_1, y_test_1, epochs, train_sets, test_sets, x_validation, y_validation)
   print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
   print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))  
    
   Draw_plot(information, epochs, "Siamese network")
        
def main():
    
   dispaly_base_network()
   dispaly_cnn_network() 
    
if __name__=='__main__':
       main()

