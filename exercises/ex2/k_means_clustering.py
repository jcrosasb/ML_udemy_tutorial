import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
from preprocessing import PreprocessingPipe
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


class KMeansClustering:
    

    def __init__(self, data:np.ndarray, k:int=None) -> None:
        self.data = data
        if k == None:
            self.k = self._best_k()
        else:
            self.k = k
        self._train_model()


    def _wcss(self, k_max=30):
        wcss = []
        for i in range(1, k_max+1):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(self.data)
            wcss.append(kmeans.inertia_)
        return wcss


    def _best_k(self, k_max=30):
        '''Method to find the optimal number of clusters using the elbow method'''
        wcss = self._wcss()
        slope_change = np.diff(np.abs(np.diff(np.abs(wcss))))  # rate of change of the slope   
        return np.argmax(slope_change) + 1


    def _train_model(self):
        self.trained_model = KMeans(n_clusters = self.k, init=np.array([[20,100],[30,125], [40,150]  ]), random_state = 42, n_init=1, max_iter=1)
        self.y_kmeans = self.trained_model.fit_predict(self.data)


    def plot_clusters(self, xlabel=None, ylabel=None):
        colors = list(mcolors.TABLEAU_COLORS.keys())[:self.k]
        for i in range(self.k):
            plt.scatter(self.data[self.y_kmeans == i, 0], self.data[self.y_kmeans == i, 1], c=colors[i], label=f'Cluster {i+1}')
        plt.scatter(self.trained_model.cluster_centers_[:, 0], self.trained_model.cluster_centers_[:, 1], c = 'yellow', label = 'Centroids')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.axis('equal')
        plt.title(f'$k={self.k}$')
        plt.show()


    def plot_elbow(self, k_max=10):
        wcss = self._wcss(k_max=k_max)
        plt.plot(range(1, k_max+1), wcss)
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()



if __name__ == '__main__':

    # dataset = pd.read_csv('data/Mall_Customers.csv')
    # X = dataset.iloc[:, [3, 4]].values

    # dataset = pd.read_csv('data/data_csl.csv')
    # X = dataset[['Temperature_C','Vibration_mm_s']].sample(n=100).values

    file = 'data/data_csl.csv'
    dep = 'Status'
    indep = ['Temperature_C','Pressure_psi']

    categorical_columns = ['Status']
    numerical_columns = ['Temperature_C','Pressure_psi','Vibration_mm_s','Power_Consumption_kW',
                                        'Oil_Level','Error_Code','Production_Rate_units_min','Motor_Speed_RPM',
                                        'Ambient_Temperature_C','Ambient_Humidity_percent','Part_Count','Voltage_V']
                    
    prep_pipe = PreprocessingPipe(filepath=file,
                                  num_columns=numerical_columns,
                                  cat_columns=categorical_columns)
                    
    prep_pipe.add_numerical_step(name='Imputer', action=SimpleImputer(missing_values=np.nan, strategy='mean'))
    #prep_pipe.add_numerical_step(name='Scaling', action=StandardScaler())
    prep_pipe.add_categorical_step(name='Encoder', action=OrdinalEncoder())
    prep_df = prep_pipe.rearrange_pipe(names=['Timestamp', 'Machine_ID'] + numerical_columns+categorical_columns, remove_nan=[dep] + indep)

    cluster = KMeansClustering(data=prep_df.sample(n=1000, random_state=42)[['Temperature_C','Pressure_psi']].values,k=3)

#    cluster.plot_elbow()



    cluster.plot_clusters(xlabel='Annual Income (k$)', ylabel='Spending Score (1-100)')