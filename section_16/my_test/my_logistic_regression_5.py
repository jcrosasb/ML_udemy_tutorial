import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
dataset = pd.read_csv('heights_weights_2.csv')
X = dataset.iloc[:, 1:].values  # Features
y = dataset.iloc[:, 0].values   # Labels

height = X[:,0]
weight = X[:,1]

heights_for_male = height[y == 1]
weights_for_male = weight[y == 1]
heights_for_female = height[y == 0]
weights_for_female = weight[y == 0]

a0 = 0.5336909798621929
a1 = -0.53170795  
a2 = 0.21172809

x = np.linspace(0,100,774)
y = (-a0/a2) - (a1/a2) * x


plt.scatter(heights_for_male, weights_for_male, color='blue', label='Male')
plt.scatter(heights_for_female, weights_for_female, color='red', label='Female')
plt.plot(x,y, color='black',linestyle='--')
plt.xlim(55,77.5)
plt.ylim(60,250)
ax = plt.gca()
bbox_props = dict(boxstyle="round,pad=0.3", fc="white", alpha=0.5)
textstr = f'$a_0={round(a0,3)}$, $a_1={round(a1,3)}$, $a_2={round(a2,3)}$'
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left', color='black', bbox=bbox_props)

ax = plt.gca()
bbox_props = dict(boxstyle="round,pad=0.3", fc="white", alpha=0.5)
textstr = r'$W=-\frac{a_0}{a_2}-\frac{a_1}{a_2}H$'
ax.text(0.95, 0.055, textstr, transform=ax.transAxes, fontsize=15, verticalalignment='bottom', horizontalalignment='right', color='black', bbox=bbox_props)
plt.xlabel('height')
plt.ylabel('weight')
plt.show()