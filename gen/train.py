"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
import pandas 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Helper: Early stopping.
import numpy as np

def get_cifar10():
    """Retrieve the CIFAR dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = 64
    input_shape = (3072,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(50000, 3072)
    x_test = x_test.reshape(10000, 3072)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def get_mnist():
    """Retrieve the MNIST dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = 128
    input_shape = (784,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)


def get_water():
    nb_classes = 5
    batch_size = 2
    input_shape = (4,) 
    water=pandas.read_csv("water.csv")
    predictors=["MG","PH","NITRATE","BICARBONATE"]
    
    water1=water[predictors]
    water1=np.array(water1)
    np.random.shuffle(water1)
    water2=water["Water Quality"]
    labelencoder_y_1 = LabelEncoder()
    water2 = labelencoder_y_1.fit_transform(water2)
    x_train=water1[5:]
    y_train=water2[5:]
    x_test=water1[0:5]
    y_test=water2[0:5]
    
    #y_test = to_categorical(y_test, nb_classes)
   # y_train = to_categorical(y_train, nb_classes)
    
    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)
    



def compile_model(network, nb_classes, input_shape):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))

        model.add(Dropout(0.2))  # hard-coded dropout

    # Output layer.
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    return model

def train_and_score(network, dataset):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    if dataset == 'water':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_water()
    elif dataset == 'mnist':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_mnist()

    model = compile_model(network, nb_classes, input_shape)

    model.fit(x_train, y_train ,batch_size=batch_size,
              epochs=10,  # using early stopping, so no real limit
              verbose=0,
    validation_data=(x_test, y_test))
  
    score = model.evaluate(x_test, y_test, verbose=0)
    print(score[1])
    return score[1]  # 1 is accuracy. 0 is loss.
