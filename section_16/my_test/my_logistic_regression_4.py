import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
dataset = pd.read_csv('heights_weights_2.csv')
X = dataset.iloc[:, 1:].values  # Features
y = dataset.iloc[:, 0].values   # Labels

# Fit logistic regression model
clf = linear_model.LogisticRegression(C=1e40, solver='newton-cg')
fitted_model = clf.fit(X, y)

# Get coefficients and intercept
coef = fitted_model.coef_[0]  # Assuming only one feature, otherwise you can use fitted_model.coef_[0]
intercept = fitted_model.intercept_[0]

print(coef, intercept)

# Define the sigmoid function
def sigmoid(x1, x2):
    z = intercept + coef[0]*x1 + coef[1]*x2
    return 1 / (1 + np.exp(-z))

x1_values = X[:,0]
x1_min, x1_max = np.min(x1_values), np.max(x1_values)
x2_values = X[:,1]
x2_min, x2_max = np.min(x2_values), np.max(x2_values)


# Generate data for plotting
x1_values = np.linspace(x1_min, x1_max, 100)
x2_values = np.linspace(x2_min, x2_max, 100)
x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)
sigmoid_values = sigmoid(x1_grid, x2_grid)

# Plot the sigmoid function as a surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_grid, x2_grid, sigmoid_values, cmap='viridis')
ax.set_xlabel('height')
ax.set_ylabel('weight')
plt.show()

# Plot the sigmoid function as a contour plot
plt.contourf(x1_grid, x2_grid, sigmoid_values, levels=50, cmap='viridis')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Male')
plt.contour(x1_grid, x2_grid, sigmoid_values, levels=[0.5], colors='black')
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Female')
plt.xlabel('height')
plt.ylabel('weight')
plt.title('Sigmoid Function Contour Plot')
plt.colorbar(label='Sigmoid Value')
plt.xlim(55,77.5)
plt.ylim(60,250)
plt.show()

# # Get coefficients of the decision boundary
# coef = fitted_model.coef_[0]
# intercept = fitted_model.intercept_

# # Plot data points
# plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Male')
# plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Female')

# # Plot decision boundary (line)
# x_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
# y_values = -(coef[0] / coef[1]) * x_values - (intercept / coef[1])
# #plt.plot(x_values, y_values, color='black', linestyle='--')

# plt.xlabel('Height')
# plt.ylabel('Weight')
# plt.grid()
# plt.legend()
# plt.show()