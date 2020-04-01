#Example of PyTorch Library

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

xmin = -1.0
xmax = 1.0

def main():
    """Main Function"""

    ############## Step 1: Create network model
    class Net(nn.Module):
        """Constructs neural network model."""
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = torch.nn.Linear(1, 8) #name fc1 is an arbitrary name
            self.fc2 = torch.nn.Linear(8, 8) #name fc1 is an arbitrary name
            self.fc3 = torch.nn.Linear(8, 1)

        def forward(self, x):
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = torch.tanh(self.fc3(x))
            return x


    net = Net()
    print(net)

    ############## Step 2: Visualize our data
    def inputfunct(x):
        return 0.25*(np.sin(2*np.pi*x*x)+2.0)
        #return np.sin(x) * np.power(x,3) + 3*x + np.random.rand(100)*0.8
        #return (63*np.power(x,5)-70*np.power(x,3) + 15*x)/8.0

    #np.random.seed(5)

    X = np.arange(xmin, xmax, (xmax-xmin)/200)
    #X = np.random.sample([256])*(xmax-xmin) + xmin
    Y = inputfunct(X) + 0.2*np.random.normal(0,0.2,len(X))

    fig1, ax1 = plt.subplots(figsize=(10,6))
    ax1.set_xticks(np.arange(xmin-0.01, xmax+0.01, (xmax-xmin)/20))
    #ax1.set_yticks(np.arange(0, 1., 0.05))
    ax1.set_xlim(xmin, xmax)

    ax1.set_ylim(Y.min()-0.05, Y.max()+0.05)
    ax1.grid(linestyle='--', linewidth=1)
    ax1.plot(X,Y,'g.', label='Raw noisy input data')

    x_real = np.arange(xmin, xmax, (xmax-xmin)/1000)
    y_real = inputfunct(x_real)
    ax1.plot(x_real,y_real, label='Actual function, not noisy', linewidth=3.0, c='black')

    #plt.scatter(X, Y)
    #plt.show()


    # convert numpy array to tensor in shape of input size
    X = torch.from_numpy(X.reshape(-1,1)).float()
    Y = torch.from_numpy(Y.reshape(-1,1)).float()
    #print(X, Y)

    prediction = net(X)
    old_line = ax1.plot(X, prediction.data.numpy(), label='Output of the Neural Net', linewidth=3.0, c='red')
    old_text = ax1.text(0.5, 0.2, 'Loss= --', fontdict={'size': 10, 'color':  'red'})
    plt.legend()

    ############## Step 3: Define Optimizer and Loss Function
    #optimizer = torch.optim.Adam(net.parameters(), lr=0.2, weight_decay=0.1, amsgrad=False) 
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.5, dampening=0.0, weight_decay=0.0) #dampening is for momentum
    loss_func = torch.nn.MSELoss()


    # Step 4: Training


    batch_size = 20
    for i in range(1000):
        for j in range(len(X)//batch_size):
            inputs = X[j*batch_size:(j+1)*batch_size]
            outputs = Y[j*batch_size:(j+1)*batch_size]
            prediction = net(inputs)
            loss = loss_func(prediction, outputs) 
            optimizer.zero_grad() # zero the gradient buffers
            loss.backward()
            optimizer.step() # Does the update

        if i % 10 == 0:
            # plot and show learning process
            prediction = net(X)
            loss = loss_func(prediction, Y) 
            print(i, "  loss = ", loss.data.numpy())
            old_line.pop(0).remove() #only clear last series added
            old_line = ax1.plot(X.data.numpy(), prediction.data.numpy(), label='Output of the Neural Net', linewidth=3.0, c='red')
            #plt.cla()
            #plt.scatter(X.data.numpy(), Y.data.numpy())
            #plt.plot(X.data.numpy(), prediction.data.numpy(), 'r-', lw=2)
            old_text.remove() #only clear last series added
            old_text = ax1.text(0.5, 0.2, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 10, 'color':  'red'})
            plt.pause(0.1)

    plt.show()


if __name__ == '__main__':
    main()    