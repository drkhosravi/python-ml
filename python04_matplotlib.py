import numpy as np
import matplotlib.pyplot as plt

plt.ion() #Interactive ON

x = np.arange(0, 7, 0.1);
#x2 = np.linspace(0, 7, 70);
y = np.sin(x)
plt.plot(x, y, label='Sin(x)')
plt.plot(x, np.cos(x), label='Cos(x)')
plt.legend()
plt.xlabel('x')
plt.ylabel('Cos(x)')
plt.title('Plot of Sin(x) & Cos(x) using MatPlotLib')
#Preceding 3 lines can also be written as:
#ax = plt.gca() #get current axes
#ax.set_xlabel('x') 
#ax.set_ylabel('Cos(x)')
#ax.set_title('Plot of Sin(x) using MatPlotLib')
plt.show()
#plt.pause(20)
#subplot
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.plot(x, np.sin(x), label='Sin Curve')
ax1.set_xlim(0, 7)
ax1.set_ylim(-1.1, 1.1)
ax1.grid(True)

ax2.plot(x, np.cos(x), label='Cos Curve')
ax2.set_xlim(0, 7)
ax2.set_ylim(-1.1, 1.1)
ax2.grid(True)
#ax1.axhline(0, color='black', lw=2)
plt.show()
plt.pause(2)
plt.cla()#clear axis
plt.pause(2)
ax2.plot(x, np.cos(x), label='Cos Curve')
plt.pause(10)
#More Examples:
#https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html
