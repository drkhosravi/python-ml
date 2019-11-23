"""
Fruit Recognition (6 Classes)
Author: Dr. Hossein Khosravi
Faculty of shahroodut.ac.ir
CEO at www.shahaab-co.ir
Date: 2019-Nov
"""

import keras
from keras import models, layers, regularizers
import Utils
from keras.utils import to_categorical

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    #force tf to use only CPU

model = models.load_model('D:/1.hdf5')
#model = models.load_model('D:/Projects/_Python/Fruit Detection/adam 64/model.hdf5')
print(model.summary())
model_best = models.load_model('D:/1.hdf5')
#model_best = models.load_model('D:/Projects/_Python/Fruit Detection/adam 64/model-best.hdf5')

img_path = 'F:/Datasets/Fruits/Train/Zardaloo/065.jpg'
Utils.CreateHeatMap(model_best, img_path)

test_set = Utils.get_dataset('F:\Datasets\Fruits\Test') #Load dataset

# Get a list of image paths and their labels
test_img_list, label_list = Utils.get_image_paths_and_labels(test_set)
assert len(test_img_list)>0, 'The test set should not be empty'

test_images = Utils.load_images(test_img_list, True, False, 400, 256, True)

test_labels = to_categorical(label_list)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
print('test_loss:', test_loss)

test_loss, test_acc = model_best.evaluate(test_images, test_labels)
print('test_acc best:', test_acc)
print('test_loss best:', test_loss)

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

Y_pred = model_best.predict(test_images)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(label_list, y_pred))
print('Classification Report')
names = ['Albaloo', 'Aloo', 'Holoo', 'Shalil', 'Sib', 'Zardaloo']
print(classification_report(label_list, y_pred, target_names=names, digits = 3))

