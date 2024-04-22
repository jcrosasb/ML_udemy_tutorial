import numpy as np
import matplotlib.pyplot as plt
from typing import Callable


class Plotter:

    def __init__(self, x: np.ndarray, y: np.ndarray, xlabel:list=None, ylabel:str=None):
        '''Plotter class. Use to visualize different types of regression
           Attributes:
                * x: (Numpy array) independent variables (usually x_test). If x has more than 2 columns,
                     it will plot only the first two.
                * y: (Numpy array) dependent variables (usually x_test).
                * predict: predict method from class
                * xlabel: (List) Labels for independent variables axis. If len(xlabel) > 2, only the 
                          first two will be usedthe. Default is None.
                * ylabel: (String) Label for the dependent variable axis'''
        self.x = x
        self.y = y
        self.xlabel = xlabel
        self.ylabel = ylabel


    def _indep_variables(self):
        '''Method that returns the independent variables for plotting'''
        if self.x.shape[1] == 1:
            return self.x[:, 0]
        elif self.x.shape[1] == 2:
            return self.x[:, 0], self.x[:, 1]
        else:
            raise ValueError('Cannot plot more than 2 independent variables')
        

    def _split_by_target(self):
        '''Method to split independent variables into values of target variable'''        
        # Create variables to store split data for each unique value
        split_x = {}
        for i, value in enumerate(np.unique(self.y)):
            split_x[f'x1_{i}'] = self.x[self.y == value, 0]  
            split_x[f'x2_{i}'] = self.x[self.y == value, 1]  
        return split_x
        

    def classifier(self, predictor: Callable, legends: list=[0, 1]):
        '''Method to plot classifier. NOTE: for 'logistic', target variable must be either 0 or 1
           Parameters:
                * type: (String) type of lassifier
                * predictor: (Method) predict method from LogRegression class
                * legends: (List) list with labels for data. Default is set to [0, 1]'''
        
        # Separate independent variables into x1 and x2
        x1, x2 = self._indep_variables()
        

        # Find min and max values
        x1_min, x1_max = np.min(x1), np.max(x1)
        x2_min, x2_max = np.min(x2), np.max(x2)

        # print(x1_min, x1_max)
        # print(x2_min, x2_max)

        # Create grid
        x1_grid, x2_grid = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                             np.arange(x2_min, x2_max, 0.1))
        
        # Predict probabilities for each point in the meshgrid
        Z = predictor(np.c_[x1_grid.ravel(), x2_grid.ravel()])

        # Reshape the predictions to match the meshgrid shape
        Z = Z.reshape(x1_grid.shape)

        # Plot the contours
        plt.contourf(x1_grid, x2_grid, Z, alpha=1, cmap='RdBu')

        x1_0 = x1[self.y == 0]  # x1 when target variable is 0
        x1_1 = x1[self.y == 1]  # x1 when target variable is 1

        x2_0 = x2[self.y == 0]  # x2 when target variable is 0
        x2_1 = x2[self.y == 1]  # x2 when target variable is 1

        # Plot the data points
        plt.scatter(x1_0, x2_0, color='red', edgecolors='black', label=legends[0], alpha=0.3)
        plt.scatter(x1_1, x2_1, color='dodgerblue', edgecolors='black', label=legends[1], alpha=0.3)



        # # Split indep. variable sinto target values
        # targets = self._split_by_target()
        # values = list(targets.values())
        
        # # Unique values of y (for coloring)
        # unique_values = np.unique(self.y)

        # # Create a colormap with as many colors as there are unique values in self.y
        # cmap = plt.get_cmap('viridis', len(np.unique(self.y)))

        # # Iterate over pairs of values and plot them
        # for i in range(0, len(values), 2):
        #     x1_values = values[i]    # First values in pair
        #     x2_values = values[i+1]  # Second values in pair

        #     # Plot the pair of values with colors based on target values
        #     for x1, x2, target in zip(x1_values, x2_values, self.y):
        #         color = cmap((target - np.min(self.y)) / (np.max(self.y) - np.min(self.y)))
        #         plt.scatter(x1, x2, c=color, edgecolors='black', alpha=0.7)
                
        # # Iterate over pairs of values and plot them
        # j = 0
        # for i in range(0, len(values), 2):
        #     x1_values = values[i]    # First values in pair
        #     x2_values = values[i+1]  # Second values in pair

        #     # Plot the pair of values with colors based on target values
        #     plt.scatter(x1_values, x2_values, c=self.y[self.y == unique_values[j]], cmap='viridis', edgecolors='black', label=legends[j], alpha=1)
        #     j += 1

            # # Create an array containing normalized values for each point
            # c_values = np.full(len(x1_values), unique_values[j])
                
            # # Plot the pair of values with colors based on c_values
            # plt.scatter(x1_values, x2_values, c=c_values, cmap='viridis', edgecolors='black', label=legends[j], alpha=1)
            # j += 1

        # plt.xlabel(self.xlabel[0])
        # plt.ylabel(self.xlabel[1])
        # # plt.xlim(21.263878, 36.217764)
        # # plt.ylim(103.772366, 141.303764)
        # # plt.xlim(np.minimum(x1_0, x1_1),np.maximum(x1_0, x1_1))
        # # plt.ylim(np.minimum(x2_0, x2_1),np.maximum(x2_0, x2_1))
        # plt.xlim(np.min(x1), np.max(x1))
        # plt.ylim(np.min(x2), np.max(x2))
        # plt.xlim(np.min(x1_grid), np.max(x1_grid))
        # plt.ylim(np.min(x2_grid), np.max(x2_grid))
        plt.legend(frameon=True, facecolor='white', framealpha=1)
        plt.show()

        # elif type == 'knn' or type == 'svm' or type == 'ksvm':
        #     plt.contourf(x1_grid, x2_grid, Z, alpha=1)
        #     # for class_label in np.unique(self.y):
        #     #     plt.scatter(X[y == class_label, 0], X[y == class_label, 1], label=f'{class_label}', cmap='viridis', edgecolors='black')
        #     plt.scatter(x1, x2, c=self.y, cmap='viridis', edgecolors='black', label=legends, alpha=0.3)
            



       
        
    # def logistic(self, predictor: Callable, legends: list=[0, 1]):
    #     '''Method to plot logistic regression. NOTE: target variable must be either 0 or 1
    #        Parameters:
    #             * predictor: (Method) predict method from LogRegression class
    #             * legends: (List) list with labels for data. Default is set to [0, 1]'''
        
    #     # Separate independent variables into x1 and x2
    #     x1, x2 = self._indep_variables()

    #     # Find min and max values
    #     x1_min, x1_max = np.min(x1), np.max(x1)
    #     x2_min, x2_max = np.min(x2), np.max(x2)

    #     # Create grid
    #     x1_grid, x2_grid = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
    #                          np.arange(x2_min, x2_max, 0.1))
        
    #     # Predict probabilities for each point in the meshgrid
    #     Z = predictor(np.c_[x1_grid.ravel(), x2_grid.ravel()])

    #     # Reshape the predictions to match the meshgrid shape
    #     Z = Z.reshape(x1_grid.shape)

    #     x1_0 = x1[self.y == 0]  # x1 when target variable is 0
    #     x1_1 = x1[self.y == 1]  # x1 when target variable is 1

    #     x2_0 = x2[self.y == 0]  # x2 when target variable is 0
    #     x2_1 = x2[self.y == 1]  # x2 when target variable is 1

    #     # Plot the contours
    #     plt.contourf(x1_grid, x2_grid, Z, alpha=1, cmap='coolwarm')

    #     # Plot the data points
    #     plt.scatter(x1_0, x2_0, color='red', edgecolors='black', label=legends[0], alpha=0.3)
    #     plt.scatter(x1_1, x2_1, color='dodgerblue', edgecolors='black', label=legends[1], alpha=0.3)

    #     # Add labels and legend
    #     plt.xlabel(self.xlabel[0])
    #     plt.ylabel(self.xlabel[1])
    #     plt.xlim(x1_min, x1_max)
    #     plt.ylim(x2_min, x2_max)
    #     plt.legend(frameon=True, facecolor='white', framealpha=1)
    #     plt.show()

  
    def clustering(self, y_kmeans):

        plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
        plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
        plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
        plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
        plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
#        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
#        plt.title('Clusters of customers')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.legend()
        plt.show()




    def knn(self):
        
        x1, x2 = self._indep_variables()

        

        