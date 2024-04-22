import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from preprocessing import PreprocessingPipe
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score, max_error


class ModelEvaluator:

    def __init__(self, x, y, y_pred) -> None:
        self.x = x
        self.y = y
        self.y_pred = y_pred
    
    
    def evaluate(self):
        mse = mean_squared_error(self.y, self.y_pred)
        r2 = r2_score(self.y, self.y_pred)
        max_e = max_error(self.y, self.y_pred)
        print(f'Mean Squared Error = {mse}\n'
              f'R^2 = {r2}\n'
              f'Max Error = {max_e}')
        return mse, r2, max_e
    

    def plot_residuals(self, dep_variable):
        plt.scatter(self.y, abs(self.y_pred - self.y))
        plt.xlabel(f'Actual-{dep_variable}')
        plt.ylabel('Residual')

        plt.show()



    

