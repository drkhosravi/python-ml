#import tensorflow.keras as keras
import keras
from keras import models, layers
#from keras.utils.np_utils import to_categorical

import  numpy as np
import  matplotlib.pyplot as plt

import  sklearn
from    sklearn import datasets, linear_model

def swish(x):
   beta = 1.0 #1, 1.5 or 2
   return beta * x * keras.backend.sigmoid(x)

# Build model
model = keras.models.Sequential()
model.add(keras.layers.Dense(10, input_shape=(2,), activation=swish))
#model.add(layers.Dense(5, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))
opt = keras.optimizers.RMSprop(lr=0.1, decay=0.02)
#opt = keras.optimizers.SGD(lr=0.5, momentum=0.0, decay=0.02, nesterov=False)
#opt = keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#opt = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train
#np.random.seed(0)
#X, y = sklearn.datasets.make_moons(200, noise=0.10)
#X, y = sklearn.datasets.make_circles(n_samples=200, noise=0.05, factor=0.7) #factor : 0 < double < 1 Scale factor between inner and outer circle.
#y_binary = to_categorical(y)
#model.fit(X, y_binary, nb_epoch=100)
history = model.fit(X, y, epochs=25)

# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
def plot_decision_boundary(model):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    step = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    # Predict the function value for the whole gid

    #.c_ : Translates slice objects to concatenation along the second axis:
        #np.c_[np.array([1,2,3]), np.array([4,5,6])] ==> 
        #array([[1, 4], [2, 5], [3, 6]])
    #ravel: Returns a contiguous flattened array:
        #x = np.array([[1, 2, 3], [4, 5, 6]])
        #np.ravel(x) ==> [1 2 3 4 5 6]
    points = np.c_[xx.ravel(), yy.ravel()]    
    Z = model.predict_classes(points)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    # f: filled (instead of lines)
    plt.contourf(xx, yy, Z, cmap='Spectral')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Spectral')
    
    plt.figure()
    Z_Soft = model.predict(points)
    Z_Soft = Z_Soft.reshape(xx.shape)
    plt.contourf(xx, yy, Z_Soft, cmap='Spectral')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Spectral')

def plot_history(history):
    plt.figure()
    acc = history.history['accuracy']
    loss = history.history['loss']

    epochs = range(1, len(acc) + 1)

    #ax1 = fig.add_subplot(211)
    #ax2 = fig.add_subplot(212)

    plt.plot(epochs, loss, 'red', label='Training loss')
    plt.plot(epochs, acc, 'blue', label='Training accuracy')
    plt.title('Training loss and accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.grid(True)


# Predict and plot
plot_decision_boundary(model)
plt.title("Decision Boundary")

plot_history(history)
plt.show()
