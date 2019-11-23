

"""
Fruit Recognition (6 Classes)
Author: Dr. Hossein Khosravi
Faculty of shahroodut.ac.ir
CEO at www.shahaab-co.ir
Date: 2019-Nov
"""


import Utils
import sys, os, signal
import argparse
import math
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #force tf to use only CPU
import tensorflow as tf
from tensorflow.python.framework import ops
import keras
from keras import models, layers, regularizers
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

print(keras.__version__)
print(tf.__version__)


#global variables and handlers (to stop training on user input CTRL+C)
network = models.Sequential() 
stop_training = False

def handler(signum, frame):
	print('Signal handler called with signal', signum)
	print('Training will finish after this epoch')
	global stop_training
	stop_training = True
	#raise OSError("Couldn't open device!")

signal.signal(signal.SIGINT, handler) # only in python version >= 3.2

class ManageTrainEvents(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.loss = []
		self.val_loss = []
		self.acc = []
		self.val_acc = []

	def on_batch_end(self, batch, logs={}):
		#self.losses.append(logs.get('loss'))
		global stop_training
		global network
		if(stop_training):
			network.stop_training = True
			
	def on_epoch_end(self, epoch, logs=None):
		self.loss.append(logs.get('loss'))
		self.val_loss.append(logs.get('val_loss'))
		self.acc.append(logs.get('acc'))
		self.val_acc.append(logs.get('val_acc'))
		Utils.plot_graphs(self.loss, self.val_loss, self.acc, self.val_acc)


def main(args):
	global network
	dataset = Utils.get_dataset('F:\Datasets\Fruits\Train') #Load dataset

	if args.validation_set_split_ratio>0.0:
		train_set, val_set = Utils.split_dataset(dataset, args.validation_set_split_ratio, args.min_nrof_val_images_per_class, 'SPLIT_IMAGES')
	else:
		train_set, val_set = dataset, []

	# Get a list of image paths and their labels
	train_img_list, label_list = Utils.get_image_paths_and_labels(train_set)
	assert len(train_img_list)>0, 'The training set should not be empty'

	val_img_list, val_label_list = Utils.get_image_paths_and_labels(val_set)

	#Utils.augment_images(train_img_list, 4) #it only must be called one time to generate several images from single image (don't forget to set validation_set_split_ratio = 0)
	
	train_images = Utils.load_images(train_img_list, True, False, args.resize_w ,args.crop_size, True)
	val_images = Utils.load_images(val_img_list, True, False, args.resize_w ,args.crop_size, True)

	use_augmentation = False;
	#cv2.imshow('img', val_images[1].astype(np.uint8))
	#cv2.waitKey(0)	
	#plt.matshow(val_images[1].astype(np.uint8))
	#plt.show()
	#Transform training/test data into a float32 array with values between 0 and 1.
	#train_images = train_images.astype('float32') / 255
	#val_images = val_images.astype('float32') / 255

	#Convert labels into one-hot codes
	train_labels = to_categorical(label_list)
	if (len(val_label_list) > 0):
		val_labels = to_categorical(val_label_list)


	#Create Network
	#	The sequential API allows to create models layer-by-layer for most problems. 
	#It does not allow you to create models that share layers or have multiple inputs or outputs.

	#	The functional API allows you to create models that have a lot more flexibility as you can 
	# easily define models where layers connect to more than just the previous and next layers. 
	# In fact, you can connect layers to (literally) any other layer. 
	# As a result, creating complex networks such as siamese networks and residual networks become possible.	
	
	network.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(args.crop_size, args.crop_size, 3)))
	#network.add(layers.MaxPooling2D((2, 2))) #256
	#network.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
	#network.add(layers.Dropout(0.25))
	#network.add(layers.MaxPooling2D((2, 2))) #128
	#network.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
	network.add(layers.MaxPooling2D((4, 4))) #64
	#network.add(layers.Dropout(0.5))
	network.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
	network.add(layers.MaxPooling2D((2, 2))) #32

	network.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
	network.add(layers.MaxPooling2D((2, 2))) #16

	#network.add(layers.GlobalAveragePooling2D())
	network.add(layers.Flatten())
	network.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
	#network.add(layers.Dropout(0.5))
	network.add(layers.Dense(6, activation='softmax'))
	optimizers = [keras.optimizers.RMSprop(lr=0.001, decay=1e-6), 
				  keras.optimizers.SGD(lr=0.001, momentum=0.5, decay=1e-6, nesterov=False),
				  #keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
				  keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)]
	
	print(network.summary())
	# Start running operations on the Graph.
	# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)

	#network.compile(optimizer=keras.optimizers.RMSprop(lr=0.001, decay=1e-6),
	#network.compile(optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.5, decay=1e-6, nesterov=False),
	#network.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
	#network.compile(optimizer=keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
					#loss='categorical_crossentropy',
					#metrics=['accuracy'])
	i = 0
	for opt in optimizers:
		network.reset_states()
		network.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])		
		#saves the model weights after each epoch if the validation loss decreased
		now = datetime.now() # current date and time
		i = i+1
		checkpointer = ModelCheckpoint(filepath='best_model_' + str(i) + '_' + now.strftime("%Y%m%d") + '.hdf5', verbose=1, save_best_only=True)
	
		manageTrainEvents = ManageTrainEvents()
		if(use_augmentation):
			train_datagen = ImageDataGenerator(
				#rescale=1./255,
				rotation_range=5,
				width_shift_range=0.1,
				height_shift_range=0.1,
				shear_range=0.,
				zoom_range=0.1,
				#brightness_range=[-0.1, 0.1],
				horizontal_flip=False,)

			train_generator = train_datagen.flow(
				train_images, train_labels,
				# All images will be resized to 150x150
				batch_size=args.batch_size)#, save_to_dir='E:/Dataset/FruitsTrainDataGen/', save_format='png')
			history = network.fit_generator(
			  train_generator,
			  steps_per_epoch = math.ceil(len(train_img_list)/args.batch_size),
			  epochs = args.max_nrof_epochs,
			  validation_data=[val_images, val_labels],
			  validation_steps = math.ceil(len(val_img_list)/args.batch_size), callbacks=[checkpointer, manageTrainEvents])
		else:
			history = network.fit(train_images, train_labels, validation_data=(val_images, val_labels), 
				epochs=args.max_nrof_epochs, batch_size=args.batch_size, callbacks=[checkpointer, manageTrainEvents])

		network.save('FruitRec_' + now.strftime("%Y%m%d-%H%M") + '.hdf5')
	#Plot loss and accuracy
		acc = history.history['accuracy']
		val_acc = history.history['val_accuracy']
		loss = history.history['loss']
		val_loss = history.history['val_loss']

		Utils.plot_graphs(loss, val_loss, acc, val_acc, True)

		#Evaluate on test dataset
		print("\nComputing test accuracy")
		test_loss, test_acc = network.evaluate(val_images, val_labels)
		print('test_acc:', test_acc)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation_set_split_ratio', type=float,
        help='The ratio of the total dataset to use for validation', default=0.1) #was 0
    
	#nrof: number of 
    parser.add_argument('--min_nrof_val_images_per_class', type=float,
        help='Classes with fewer images will be removed from the validation set', default=0)
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=50)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=64)
    parser.add_argument('--resize_w', type=int,
        help='Reduce width size in pixels.', default=400)
    parser.add_argument('--crop_size', type=int,
        help='Crop size (height, width) in pixels.', default=256)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.7)
   
   
    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
