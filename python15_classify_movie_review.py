#   Deep Learning Course
#Shahrood University of Technology
#Dr. Hossein Khosravi 1397-98


import keras

#Load IMDB dataset

from keras.datasets import imdb
import numpy as np

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000) # ~17MB

#train_data is an array of lists with different sizes; e.g. first list contains 218 words and second contains 189 words,.. 5th contains 43...
print(train_data[5]) #0 is padding, 1 is start of sequence, 2 unknown (?), codes of real words start from 3

# word_index is a dictionary mapping words to an integer index
word_index = imdb.get_word_index()
print(word_index['the'])
# We reverse it, mapping integer indices to words
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# We decode the review; note that our indices were offset by 3
# because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[5]])
print(decoded_review)

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension), dtype=np.float32)
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results

# Our vectorized training data
# کافیست یک کلمه تنها یک بار در متن نظر آمده باشد که اندیس متناظر با آن 1 شود
# در این رویکرد، تعداد تکرار مهم نیست. فقط وجود یا عدم وجود کلمه مهم است
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)

print(x_train[0])

# Our vectorized labels
y_train = train_labels.astype('float32')
y_test = test_labels.astype('float32')

from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics


model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#model.compile(optimizer='rmsprop',
#              loss='binary_crossentropy',
#              metrics=['accuracy'])

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=10,
                    batch_size=512,
                    validation_data=(x_val, y_val))

import matplotlib.pyplot as plt

acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


plt.clf()   # clear figure
acc_values = history.history['binary_accuracy']
val_acc_values = history.history['val_binary_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

#plt.show()


sample1 = ["this", "is", "a", "great", "movie", "well", "good","fantastic"]
sample2 = ["this", "is", "awful", "bad", "terrible"]
sample1_i = [0]*len(sample1)
sample2_i = [0]*len(sample2)
i = 0
for s in sample1:
    sample1_i[i] = word_index[s]
    i += 1

i = 0
for s in sample2:
    sample2_i[i] = word_index[s]
    i += 1

s = [sample1_i, sample2_i]
s = vectorize_sequences(s)

p = model.predict(s)
print(p)
