import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import matplotlib.pyplot as plt

import numpy as np
import imageio
#ref: https://medium.com/@benjamin.phillips22/simple-regression-with-neural-networks-in-pytorch-313f06910379
def main():
    torch.manual_seed(1)    # reproducible

    #unsqueeze to make tensor from 1-D numpy array
    x = torch.unsqueeze(torch.linspace(-10, 10, 1000), dim=1)  # x data (tensor), shape=(1000, 1)
    y = torch.sin(x) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(1000, 1)

    # another way to define a network
    net = torch.nn.Sequential(
            torch.nn.Linear(1, 200),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(200, 100),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(100, 1),
        )

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

    BATCH_SIZE = 100
    EPOCH = 100

    torch_dataset = Data.TensorDataset(x, y)

    loader = Data.DataLoader(
        dataset=torch_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=1,)

    my_images = []
    fig, ax = plt.subplots(figsize=(10,6))

    ax.set_title('Regression Analysis', fontsize=35)
    ax.set_xlabel('Independent variable', fontsize=24)
    ax.set_ylabel('Dependent variable', fontsize=24)
    ax.set_xlim(-11.0, 13.0)
    ax.set_ylim(-1.1, 1.2)

    old_line_1 = ax.scatter(x.data.numpy(), y.data.numpy(), color = "blue")
    old_line_2 = ax.scatter(x.data.numpy(), y.data.numpy(), color = "green")
    old_text_1 = ax.text(8.8, 0, 'Epoch = --')
    old_text_2 = ax.text(8.8, -0.8, 'Loss = --')

    plt.savefig('curve_2.png')
    plt.pause(1)
    # start training
    for epoch in range(EPOCH):
        print("epoch ", epoch)
        for step, (b_x, b_y) in enumerate(loader): # for each training step
            
            prediction = net(b_x)     # input x and predict based on x

            loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

            if step == 1:
                # plot and show learning process
                #plt.cla()
                old_line_1.remove() #only clear last series added
                old_line_1 = ax.scatter(b_x.data.numpy(), b_y.data.numpy(), color = "blue", alpha=0.2)

                old_line_2.remove() #only clear last series added
                old_line_2 = ax.scatter(b_x.data.numpy(), prediction.data.numpy(), color='green', alpha=0.5)
                #ax.plot(b_x.data.numpy(), prediction.data.numpy(), color='green', alpha=0.5)
                old_text_1.remove()
                old_text_1 = ax.text(8.0, -0.8, 'Epoch = %d' % epoch,
                        fontdict={'size': 16, 'color':  'red'})

                old_text_2.remove()
                old_text_2 = ax.text(8.0, -0.95, 'Loss = %.4f' % loss.data.numpy(),
                        fontdict={'size': 16, 'color':  'red'})

                # Used to return the plot as an image array 
                # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
                plt.pause(0.1)
                fig.canvas.draw()       # draw the canvas, cache the renderer
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                my_images.append(image)

        


    # save images as a gif    
    imageio.mimsave('./curve_2_model_3_batch.gif', my_images, fps=12)


    fig, ax = plt.subplots(figsize=(10,6))
    plt.cla()
    ax.set_title('Regression Analysis - model 3, Batches', fontsize=35)
    ax.set_xlabel('Independent variable', fontsize=24)
    ax.set_ylabel('Dependent variable', fontsize=24)
    ax.set_xlim(-11.0, 13.0)
    ax.set_ylim(-1.1, 1.2)
    ax.scatter(x.data.numpy(), y.data.numpy(), color = "blue", alpha=0.2)
    prediction = net(x)     # input x and predict based on x
    ax.scatter(x.data.numpy(), prediction.data.numpy(), color='green', alpha=0.5)
    plt.savefig('curve_2_model_3_batches.png')
    plt.show()


if __name__ == '__main__'    :
    main()