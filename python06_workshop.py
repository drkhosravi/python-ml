#Sample 1 Plot
import numpy as np
import matplotlib.pyplot as plt

plt.ion() #Interactive ON

# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)

#create new figure
fig1, ax1 = plt.subplots()

# red dashes, blue squares and green triangles
ax1.plot(t, 30*t+10, 'orange', t, np.exp(t), 'r--', t, t**2, 'bs', t, t**3, 'g^')
#ax1.legend()
plt.show()
plt.pause(1)

#################################################################################
#Sample 2 - Stem Plot
x = np.linspace(0.1, 2 * np.pi, 41)
y = np.exp(np.sin(x))

fig2, ax2 = plt.subplots()

ax2.stem(x, y)
plt.show()
plt.pause(1)
#################################################################################
#Sample 3 - Histogram
# Fixing random state for reproducibility
np.random.seed(19680801)

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)
fig3, ax3 = plt.subplots()

# the histogram of the data
#bins: if an integer is given, bins + 1 bin edges are calculated and returned
#density: convert to Prob. Density
#alpha: transparency
n, bins, patches = plt.hist(x, 50, density=True, facecolor='g', alpha=0.75)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, '$\mu=100, \sigma=15$')
plt.xlim(40, 160)
plt.ylim(0, 0.03)
plt.grid(True)
plt.show()
plt.pause(1)


#################################################################################
#Sample 4 - Scatterplot

# Create data
N = 500
x = np.random.rand(N)
y = np.random.rand(N)
area = 20 #area of scattered circles

fig3, ax3 = plt.subplots()

# Plot
plt.scatter(x, y, s = area, c = 'red', alpha = 0.5)
plt.title('Scatter plot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.pause(1)

#################################################################################
#Sample 5 - Scatter plot with groups
# Create data
N = 60
g1 = (0.6 + 0.6 * np.random.rand(N), np.random.rand(N))
g2 = (0.4+0.3 * np.random.rand(N), 0.5*np.random.rand(N))
g3 = (0.3*np.random.rand(N),0.3*np.random.rand(N))

data = (g1, g2, g3)
colors = ("red", "green", "#00ffff")
groups = ("coffee", "tea", "water")

# Create plot
fig4, ax4 = plt.subplots()

for data, color, group in zip(data, colors, groups):
	x, y = data
	ax4.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

plt.title('Matplot scatter plot')
plt.legend(loc=2)
plt.show()
plt.pause(1)

#################################################################################
#Sample 6 - 3D plot
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused impor
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')
# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=False)
# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
plt.pause(100)