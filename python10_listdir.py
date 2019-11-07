#work with folders and files
import os
from os import path, listdir, system  # to use path instead of os.path
my_path = "D:\\Projects\\AHFR 1.x\\CelebFace\\"

#List all files and directories in a path
for s in os.listdir(my_path):
    print(s)

print('\n------------------------------------------------------\n')
#List directories only
for s in listdir(my_path):
    if(path.isdir(path.join(my_path, s))):
        print(s)

print('\n------------------------------------------------------\n')

#prepare a dataset

classes = [s for s in listdir(my_path) \
                if path.isdir(path.join(my_path, s))]
classes.sort()
n = len(classes)

dataset = [0]*n #creates a list of size n
for i in range(n):
    class_name = classes[i]
    facedir = path.join(my_path, class_name)
    dataset[i] = [s for s in listdir(facedir) \
                if not path.isdir(path.join(facedir, s))]


print(dataset)                