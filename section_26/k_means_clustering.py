# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

#print(type(X))

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Assuming y_values is your list of data points
x = [1,2,3,4,5,6,7,8,9,10]
y = wcss  # Your data points here

# Calculate the rate of change of the slope
slope_change = np.diff(np.abs(np.diff(np.abs(y))))
best_k = np.argmax(slope_change) + 1

print(best_k)

# # Calculate the first derivative (slope)
# slope = first_diff_y / first_diff_x

# # Calculate the second differences of the slopes
# second_diff_slope = np.diff(slope)

# # Calculate the x points corresponding to the second finite difference
# second_finite_difference_x = x[1:-1]

# slope = [abs(wcss[i+1] - wcss[i]) for i in range(len(wcss)-1)]
# slope_change = [abs(slope[i+1] - slope[i]) for i in range(len(slope)-1)]
# print(slope)
# print(slope_change)

# k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# # Calculate the second derivative of the WCSS curve
# second_derivative = np.diff(np.diff(wcss))

# # Find the index of the maximum value in the second derivative array
# elbow_index = np.argmax(second_derivative) + 1

# # The optimal number of clusters (k) is the corresponding value in the k array
# optimal_k = k[elbow_index]

#print(optimal_k)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# # Training the K-Means model on the dataset
# kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
# y_kmeans = kmeans.fit_predict(X)

# # Visualising the clusters
# plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
# plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
# plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
# plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
# plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
# plt.title('Clusters of customers')
# plt.xlabel('Annual Income (k$)')
# plt.ylabel('Spending Score (1-100)')
# plt.legend()
# plt.show()