print('In the name of Allah');
print('Welcome to the world of Python');

i = 10
f = 3.14
s = "This is a string"
s2 = 'This is another string'

myList = [2, 3, 4, 5]
myTuple = ('Ali', 'Hassan', 4, 5)
myDic = {'Name' : 'Ali', 'Family' : 'Khosravi', 'Age': 38}
print(myList[0])
myList[0] = 'Jafar'
print(myList[0])

print(myDic['Name'])

print('i = {0}'.format(i))
print('f = {0}'.format(f))
print('s = {0}'.format(s))
print('s2 = {0}'.format(s2))
print('myList = {0}'.format(myList))
print('myTuple[0] = {0}, {1}'.format(myTuple[0], myTuple[1]))
print('myDic[\'Name\'] = {0}'.format(myDic['Name']))

print( type(i) ) 
print( type(f) ) 
print( type(s) ) 
print( type(myDic) )

x = y = z = 20
print("y = ", y)
y = 21;
print(y)

a, b, c = 12, 'test', 19.75
print(a)
print(b)
print(c)

print("5 / 3 = {0:.2f}".format(5/3))

v = (5 != 3)
print(v)
