"""
Example 13
"""
from tkinter import Tk, Button, Label, Entry
from tkinter import messagebox

def sample1():
    """
    Sample 1
    """

    window = Tk()
    window.title("Welcome to GUI in python using Tkinter")

    #Set Window Size
    window.geometry('500x400')

    #Create Label
    lbl = Label(window, text="Hello", font=("Calibri Bold", 15))
    #set its position on the form
    window.columnconfigure(0, minsize=100)
    lbl.grid(column=0, row=0)#for absolute position use: lbl.place(x=40, y=10)

    #Get Input Using Entry Class (Tkinter Textbox)
    txt = Entry(window, width=15) # try state='disabled'
    txt.grid(column=1, row=0) 
    txt.focus() #Set the Focus of the Entry Widget

    #Create a Button with handler
    def clicked():
        input_txt = txt.get()
        lbl.configure(text=input_txt)
        messagebox.showinfo('Input Text', input_txt)

    btn = Button(window, text='Click Here', bg="#ffffcc", fg="red", command=clicked)
    window.columnconfigure(1, minsize=100)
    btn.grid(column=2, row=0)



    window.mainloop()



sample1()
