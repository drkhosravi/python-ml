
# @Author  : play4fun
# @File    : opencv-with-tkinter.py

"""
https://www.pyimagesearch.com/2016/05/23/opencv-with-tkinter/

"""

# import the necessary packages
from tkinter import *
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog as tkFileDialog
import cv2


def select_image():
    # grab a reference to the image panels
    global panelA, panelB

    # open a file chooser dialog and allow the user to select an input
    # image
    path = tkFileDialog.askopenfilename()

    # ensure a file path was selected
    if len(path) > 0:
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        cv_img = cv2.imread(path)
        cv_img = cv2.resize(cv_img, (400, 300), interpolation = cv2.INTER_LINEAR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 100)

        #  represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        # convert the images to PIL format...
        pil_img = Image.fromarray(cv_img)
        pil_edged = Image.fromarray(edged)

        # ...and then to ImageTk format (to be shown using tkinter)
        image = ImageTk.PhotoImage(pil_img)
        edged = ImageTk.PhotoImage(pil_edged)

        # if the panels are None, initialize them
        if panelA is None or panelB is None:
            # the first panel will store our original image
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)

            # while the second panel will store the edge map
            panelB = Label(image=edged)
            panelB.image = edged
            panelB.pack(side="right", padx=10, pady=10)

        # otherwise, update the image panels
        else:
            # update the pannels
            panelA.configure(image=image)
            panelB.configure(image=edged)
            panelA.image = image
            panelB.image = edged


# initialize the window toolkit along with the two image panels
root = Tk()
panelA = None
panelB = None

# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn = Button(root, text="Select an image", command=select_image)

#expand − When set to true, widget expands to fill any space not otherwise used in widget's parent.
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

# kick off the GUI
root.mainloop()