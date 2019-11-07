#   Machine Learning Course
#   Shahrood University of Technology
#   Dr. Hossein Khosravi 1398


import keras
from keras import losses, metrics
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

class ManageTrainEvents(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.loss = []
		self.val_loss = []
		self.acc = []
		self.val_acc = []
			
	def on_epoch_end(self, epoch, logs=None):
		global fig1, ax1, old_line
		Ylearn = model.predict(Xreal)
		#plt.cla() #clear current axis
		old_line.pop(0).remove() #only clear last series added
		old_line = ax1.plot(Xreal, Ylearn, label='Output of the Neural Net', linewidth=3.0, c='red')
		plt.pause(0.1)


def inputfunct(x):
    return 0.25*(np.sin(2*np.pi*x*x)+2.0)
	#return (63*np.power(x,5)-70*np.power(x,3) + 15*x)/1.0

#np.random.seed(5)
xmin = -1.0
xmax = 1.0
X = np.arange(xmin, xmax, (xmax-xmin)/200)
#X = np.random.sample([256])*(xmax-xmin) + xmin
Y = inputfunct(X) + 0.2*np.random.normal(0,0.2,len(X))

Xreal = np.arange(xmin, xmax, (xmax-xmin)/1000)
Yreal = inputfunct(Xreal)

### Model creation: adding layers and compilation
model = Sequential()
model.add(Dense(8, input_dim=1, activation='tanh'))
model.add(Dense(8, activation='tanh'))
#model.add(Dense(8, activation='tanh'))
model.add(Dense(1, activation='tanh'))#try linear

#opt = keras.optimizers.RMSprop(lr=0.1, decay=0.1) #lr = self.lr * (1. / (1. + self.decay * self.iterations))
#opt = keras.optimizers.SGD(lr=0.5, momentum=0.0, decay=0.1, nesterov=False)
opt = keras.optimizers.Adam(lr=0.2, beta_1=0.9, beta_2=0.999, decay=0, amsgrad=False)
#opt = keras.optimizers.Nadam(lr=0.2, beta_1=0.9, beta_2=0.999, schedule_decay=0.004)
model.compile(optimizer=opt, loss='mse', metrics=['mse'])

nepoch = 500
nbatch = 32
manageTrainEvents = ManageTrainEvents()

#plt.style.use('dark_background')
fig1, ax1 = plt.subplots(figsize=(10,6))
ax1.set_xticks(np.arange(xmin-0.01, xmax+0.01, (xmax-xmin)/20))
#ax1.set_yticks(np.arange(0, 1., 0.05))
ax1.set_xlim(xmin, xmax)
#ax1.set_ylim(0, 1)
#plt.rc('grid', linestyle="-.", color='black')
Ylearn = model.predict(Xreal)
ax1.grid(linestyle='--', linewidth=1)
ax1.plot(X,Y,'g.', label='Raw noisy input data')
ax1.plot(Xreal,Yreal, label='Actual function, not noisy', linewidth=3.0, c='black')
old_line = ax1.plot(Xreal, Ylearn, label='Output of the Neural Net', linewidth=3.0, c='red')
plt.legend()

model.fit(X, Y, epochs=nepoch, batch_size=nbatch, callbacks=[manageTrainEvents])

Ylearn = model.predict(Xreal)
### Make a nice graphic!
#ax1.plot(Xreal, Ylearn, label='Output of the Neural Net', linewidth=3.0, c='orange')
plt.pause(0)
#plt.savefig('neural-network-keras-function-interpolation.png')
