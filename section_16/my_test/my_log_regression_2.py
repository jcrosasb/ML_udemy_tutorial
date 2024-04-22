import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('heights_2.csv')
y = dataset.iloc[:, :-1].values  # The gender (1 is male; 0 is female)
X = dataset.iloc[:, -1].values  # The height (in inches?)

male = pd.read_csv('male.csv')
y_male = male.iloc[:, :-1].values 
X_male = male.iloc[:, -1].values  

female = pd.read_csv('female.csv')
y_female = female.iloc[:, :-1].values 
X_female = female.iloc[:, -1].values

# Define the sigmoid function
def sigmoid_function(x, a, b, c):
    return c / (1 + np.exp(-a * (x - b)))

# Define the logit function from sigmoid
def logit_function(sigmoid):
    return np.log(sigmoid/(1-sigmoid))

# Define your parameters
a = 1.0  # Slope parameter
b = np.mean(X)  # Midpoint parameter
c = 1.0  # Scaling parameter

# Calculate probabilities using the sigmoid function
p = sigmoid_function(np.sort(X), a, b, c)

# Some straight line in logit space
a0, a1 = -65.5743332913415, 1.000000000000254
some_logit = a0 + a1 * np.sort(X)
#plt.plot(np.sort(X), some_logit)

# Sigmoid calculated from some_logit
my_sig = 1/(1+np.exp(-some_logit))

plt.subplot(1, 2, 1)
plt.scatter(X, y, edgecolors='black', facecolors='none')
#plt.plot(np.sort(X), p, color='red')
#plt.plot(np.sort(X), sigmoid_function(np.sort(X), a, b-5, c), color='green', linestyle='--')
plt.plot(np.sort(X), my_sig, color='red')
plt.scatter(np.sort(X), my_sig)
# plt.plot(np.sort(X), p2, color='green')
# plt.plot(np.sort(X), sp3, color='black')
plt.ylim(-0.1, 1.1)  
plt.yticks([0, 1], ['F', 'M'])
plt.xlabel('Height')
plt.ylabel('Probability ($p$)')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(np.sort(X), some_logit, color='red')
plt.scatter(np.sort(X), some_logit)
plt.xlabel('Height')
plt.ylabel(r'$log(\frac{p}{1-p})$')
plt.grid()


# plt.subplot(1, 2, 2)
# plt.scatter(np.sort(X), my_sig)
# #plt.plot(np.sort(X), p, color='red')
# plt.plot(np.sort(X), my_sig, color='red')
# # plt.plot(np.sort(X), p2, color='green')
# # plt.plot(np.sort(X), sp3, color='black')
# plt.ylim(-0.1, 1.1)  
# plt.yticks([0, 1], ['F', 'M'])
# plt.xlabel('Height')
# plt.ylabel('Probability ($p$)')
# plt.grid()

plt.show()


