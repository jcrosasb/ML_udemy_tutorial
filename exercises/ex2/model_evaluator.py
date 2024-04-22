import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, roc_curve, confusion_matrix, log_loss, mean_squared_error, r2_score, max_error


class ModelEvaluator:

    def __init__(self, x: np.ndarray, y: np.ndarray, y_pred: np.array) -> None:
        '''Initializer of ModelEvaluator
           Attributes:
                * x: (Numpy array) dependent variable from testing set
                * y: (Numpy array) independent variable(s) from testing set
                * y_pred: (Numpy array) testing set predictions'''
        self.x = x
        self.y = y
        self.y_pred = y_pred
    
    
    def metrics(self, type:str, show:bool=False):
        '''Method that provides evaluation metrics for different regressions and classifiirs
           Parameters:
                * type: (String) 'regressor' if simple or multiple linear regression, polynomial regression, 
                        support vector regression, decision tree regression or random forest regression.
                        'classifier' if logistic regression, k-nearest neighbor, support vector machine classifier, 
                        kernel support vector machine classifier, naive Bayes classifier, decision tree classifier or
                        random forest classifier.
                * show: (Boolean) if True, method will print all metrics. Deafult is False
           Returns:
                * Tuple with all metrics'.  
                  If 'regressor': mean square error, R2 and max error.
                  If 'classifier': accuracy, precision, recall, f1, confusion matrix, logloss.'''
        if type == 'classifier':
            accuracy = accuracy_score(self.y, self.y_pred)  
            precision = precision_score(self.y, self.y_pred)
            recall = recall_score(self.y, self.y_pred)
            f1 = f1_score(self.y, self.y_pred)
            cm = confusion_matrix(self.y, self.y_pred)
            logloss = log_loss(self.y, self.y_pred)
            if show:
                print(f'Accuracy = {accuracy}\n'
                      f'Precision = {precision}\n'
                      f'Recall = {recall}\n'
                      f'F1 = {f1}\n'
                      f'Confusion Matrix= {cm}\n'
                      f'Log Loss = {logloss}')
            return accuracy, precision, recall, f1, cm, logloss
        elif type == 'regressor':
            mse = mean_squared_error(self.y, self.y_pred)
            r2 = r2_score(self.y, self.y_pred)
            max_e = max_error(self.y, self.y_pred)
            if show:
                print(f'Mean Squared Error = {mse}\n'
                    f'R^2 = {r2}\n'
                    f'Max Error = {max_e}')
            return mse, r2, max_e
        raise ValueError("Invalid type for metrics. Must be either 'regressor' or 'classifier'")
    

    def plot_residuals(self, dep_variable):
        plt.scatter(self.y, abs(self.y_pred - self.y))
        plt.xlabel(f'Actual-{dep_variable}')
        plt.ylabel('Residual')

        plt.show()



    

