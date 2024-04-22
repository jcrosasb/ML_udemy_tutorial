import numpy as np
import pandas as pd
from plotter import Plotter
from sklearn.impute import SimpleImputer
from model_evaluator import ModelEvaluator
from preprocessing import PreprocessingPipe
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MinMaxScaler, PolynomialFeatures


class PolynomialRegression:
    '''Polynomial regression class'''

    def __init__(self, data, independent:str, dependent:str, degree:int, size:int=0.2, seed:int=42):
        '''Initializer of PolynomialRegression object.
           Attributes and parameters:
                * data: (Pandas dataframe) the dataframe to be read
                * independent: (String) name of independent variable
                * dependent: (String) name of dependent variable
                * degree: (integer) degree of polynomial to be fitted
                * size: (Integer) the proportion of the file that will 
                        be used for testing
                * seed: (Integer) the randome seed used to shuffle data
           Additional attributes:
                * X_train, y_train: (Numpy arrays) arrays containing data 
                                    to be trainned
                * X_test, y_test: (Numpy arrays) arrays containing data 
                                    to be tested
                * y_pred: (Numpy array) predicted values for X_test
                * poly_trained_model: PolynomialFeatures object
                * trained_model: LinearRegression object'''
        self.data = data
        self.independent = self.data[independent].values
        self.dependent = self.data[dependent].values
        self.degree = degree
        self.y_pred = None
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self._split_dataset_for_training(size=size, seed=seed)
        self._train_model(deg=degree)


    def _split_dataset_for_training(self, size:int, seed:int):
        '''Method to split datatset into training and testing. The method is ran
           automatically when a PolynomialRegression object is created.
           Parameters:
                * size: (Integer) the proportion of the file that will 
                        be used for testing
                * seed: (Integer) the randome seed used to shuffle data
           Returns:
                * X_train, X_test, y_train, y_test'''
        return train_test_split(self.independent, 
                                self.dependent, 
                                test_size=size, 
                                random_state=seed)
        

    def _train_model(self, deg:int):
        '''Method to train model using method 'fit'. The method is ran
           automatically when a PolynomialRegression object is created.'''
        self.poly_trained_model = PolynomialFeatures(degree=deg)
        self.trained_model = LinearRegression()
        self.trained_model.fit(self.poly_trained_model.fit_transform(self.X_train), self.y_train)
        self.y_pred = self.predict(self.X_test)


    def get_coefficients(self):
        '''Method to get the coefficients of the Polynomial regression
           Returns:
                * (List) coefficient of Polynomial regression'''
        return self.trained_model.coef_


    def get_intercept(self):
        '''Method to get the intercept of the linear regression
           Returns:
                * (list) intercept of Polynomial regression'''
        return self.trained_model.intercept_


    def predict(self, values:list):
        '''Method to get values predicted by Polynomial regression
           Parameters:
                * values: (list) the values on which the
                          prediction will be performed
           Returns:
                * (List) predicted values'''
        return self.trained_model.predict(self.poly_trained_model.transform(values))

    def evaluate(self):
        '''Method to evaluate Polynomial regression by calculating mean 
           squared error and R-square over the 'test' dataset
           Returns:
                * (Float) mean squared error and R-square'''
        y_pred = self.predict(self.X_test)    
        return mean_squared_error(self.y_test, y_pred), r2_score(self.y_test, y_pred)


if __name__ == '__main__':

    file = 'data/data_reg.csv'
    dep = 'Temperature_C'
    indep = ['Oil_Level']

    categorical_columns = ['Status']
    numerical_columns = ['Temperature_C','Pressure_psi','Vibration_mm_s','Power_Consumption_kW',
                                        'Oil_Level','Error_Code','Production_Rate_units_min','Motor_Speed_RPM',
                                        'Ambient_Temperature_C','Ambient_Humidity_percent','Part_Count','Voltage_V']
                    
    prep_pipe = PreprocessingPipe(filepath=file,
                                                num_columns=numerical_columns,
                                                cat_columns=categorical_columns)
                    
    prep_pipe.add_numerical_step(name='Imputer', action=SimpleImputer(missing_values=np.nan, strategy='mean'))
    prep_pipe.add_numerical_step(name='Scaling', action=StandardScaler())
    prep_pipe.add_categorical_step(name='Encoder', action=OrdinalEncoder())
    prep_df = prep_pipe.rearrange_pipe(names=['Timestamp', 'Machine_ID'] + numerical_columns+categorical_columns, remove_nan=[dep] + indep)

    # Polynomial regression =============================================
    pr = PolynomialRegression(data=prep_df.sample(n=1000, random_state=42), 
                              independent=indep,
                              dependent=dep,
                              degree=2)

    # plot = Plotter(x=pr.X_test, 
    #                y=pr.y_test, 
    #                xlabel=indep, 
    #                ylabel=dep,
    #                predict=pr.predict,
    #                evaluate=pr.evaluate)
    

    eval = ModelEvaluator(x=pr.X_test, y=pr.y_test, y_pred=pr.y_pred)
    eval.evaluate()
    eval.plot_residuals(dep_variable=dep)












    # plt.scatter(pr.X_test, pr.y_test, color='red')
    # plt.plot(np.sort(pr.X_test, axis=0), pr.predict(np.sort(pr.X_test, axis=0)), color='blue')
    # plt.grid()
    # plt.title(f'Polynomial LR: {indep} vs {dep}')
    # plt.xlabel(f'{indep}')
    # plt.ylabel(f'{dep}')
    # plt.text(np.percentile(pr.X_test, 50), np.percentile(pr.X_test, 25), \
    #              f'$MSE$={round(pr.evaluate()[0],6)}, $R^2$={round(pr.evaluate()[1],6)}', \
    #              bbox=dict(facecolor='white', alpha=0.5))
    # plt.show()


    # dataset = pd.read_csv('data/Position_Salaries.csv')

    # # Polynomial regression =============================================
    # pr = PolynomialRegression(data=dataset, 
    #                                  independent='Level',
    #                                  dependent='Salary',
    #                                  degree=4)
    
    # print(pr.get_intercept())
    # print(pr.get_coefficients())

    # print(pr.X_train)
    # plt.scatter(pr.X_train, pr.y_train, color='red')
    # plt.plot(np.sort(pr.X_train, axis=0), pr.predict(np.sort(pr.X_train, axis=0)), color='blue')
    # plt.grid()
    # plt.title(f'Simple LR: {indep} vs {dep}')
    # plt.xlabel(f'{indep}')
    # plt.ylabel(f'{dep}')
    # plt.text(np.percentile(pr.X_test, 50), np.percentile(pr.X_test, 25), \
    #              f'$MSE$={round(pr.evaluate()[0],6)}, $R^2$={round(pr.evaluate()[1],6)}', \
    #              bbox=dict(facecolor='white', alpha=0.5))
    # plt.show()