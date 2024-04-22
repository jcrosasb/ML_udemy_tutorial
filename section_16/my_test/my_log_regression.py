import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('heights_2.csv')
y = dataset.iloc[:, :-1].values  # The gender (1 is male; 0 is female)
X = dataset.iloc[:, -1].values  # The height (in inches?)

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
#p = sigmoid(np.linspace(np.min(X), np.max(X), len(X)), a, b, c)
p = sigmoid_function(np.sort(X), a, b, c)
# p2 = sigmoid_function(np.sort(X), 0.5, b, c)
# p3 = sigmoid_function(np.sort(X), 1, b-3, c)

# Calculate logit
logit = logit_function(p)

plt.subplot(1, 2, 1)
plt.scatter(X, y)
plt.plot(np.sort(X), p, color='red')
# plt.plot(np.sort(X), p2, color='green')
# plt.plot(np.sort(X), sp3, color='black')
plt.ylim(-0.1, 1.1)  # Set the y-axis limits to include female and male labels
# Replace y-tick labels with 'Female' and 'Male'
plt.yticks([0, 1], ['F', 'M'])
plt.title('Height vs. Gender')
plt.xlabel('Height')
plt.ylabel('Probability ($p$)')
plt.grid()

plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
plt.plot(np.sort(X), logit, color='red')
#plt.scatter(np.sort(X), logit)
plt.xlabel('Height')
plt.ylabel(r'$log(\frac{p}{1-p})$')
# plt.xlim(0,75)
# plt.ylim(-100, 100)
plt.grid()

plt.show()

m = (np.max(logit)-np.min(logit))/(np.max(X) - np.min(X))

b = np.min(logit) - m * np.min(X)

print(b,m)

