#   Machine Learning Course
#   Shahrood University of Technology
#   Dr. Hossein Khosravi 1398


import keras
import tensorflow
print(keras.__version__)
print(tensorflow.__version__)

#Load MNIST dataset
"""
"""
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print( train_images.shape )
print (len(train_labels) )
print (train_labels, "data type:", train_labels.dtype)
print( test_images.shape )

#Create Network
from keras import models
from keras import layers

#	The sequential API allows to create models layer-by-layer for most problems. 
#It does not allow you to create models that share layers or have multiple inputs or outputs.

#	The functional API allows you to create models that have a lot more flexibility as you can 
# easily define models where layers connect to more than just the previous and next layers. 
# In fact, you can connect layers to (literally) any other layer. 
# As a result, creating complex networks such as siamese networks and residual networks become possible.
network = models.Sequential() 
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
# A loss function: the is how the network will be able to measure how good a job it is doing on its training data, and thus how it will be able to steer itself in the right direction.
# An optimizer: this is the mechanism through which the network will update itself based on the data it sees and its loss function.
# Metrics to monitor during training and testing. Here we will only care about accuracy (the fraction of the images that were correctly classified).
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

#Transform training/test data into a float32 array of shape (60000, 28 * 28) with values between 0 and 1.
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

#Convert labels into one-hot codes
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#Ready to train our network, which in Keras is done via a call to the fit method of the network: we "fit" the model to its training data.
network.fit(train_images, train_labels, epochs=5, batch_size=128)

#Evaluate on test dataset
print("\nComputing test accuracy")
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
