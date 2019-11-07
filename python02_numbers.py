

for j in range(10):
    if(j == 5):
        continue
    #print(j)




x = int(1)
y = int(2.8)
z = int("3")
print(x)
print(y)
print(z)

print('**********\n')
x = float(1)
y = float(2.8)
z = float("3")
w = float("4.2")
print(x)
print(y)
print(z)
print(w)

print('**********\n')

import numpy as np
#numpy
x = np.array([1, 2, 2.5])
print(x)

y = x.astype(int)
print(y)

print('**********\n')

x = str("s1")
y = str(2)
z = str(3.0)
print(x)
print(y)
print(z)

print('**********\n')

#string manipulation

a = "Hello, World!"
print(a.lower())

a = "Hello, World!"
print(a.replace("World", "Python"))

print('**********\n')
a = "Hello, World, Hello, Python!"
b = a.split(",")
print(b)

