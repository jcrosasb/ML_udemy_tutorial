
import numpy as np
from plotter import Plotter
from sklearn.impute import SimpleImputer
from model_evaluator import ModelEvaluator
from preprocessing import PreprocessingPipe
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

class DecisionTreeRegression:

    def __init__(self, data, independent:str, dependent:str, \
                 parameters:dict={'random_state':0, 'min_samples_split':20}, size:int=0.2, seed:int=42):
        '''Initializer of DecisionTreeRegression object.
           Attributes and parameters:
                * data: (Pandas dataframe) the sample dataframe to be read
                * independent: (String) name of independent variable
                * dependent: (String) name of dependent variable
                * parameters: (Dict) decision tree parameters. Default values are 
                              random_state and min_samples_split. See DecisionTreeRegressor
                              documentation for more information.
                * size: (Integer) the proportion of the file that will 
                        be used for testing
                * seed: (Integer) the randome seed used to shuffle data
           Additional attributes:
                * X_train, y_train: (Numpy arrays) arrays containing data 
                                    to be trainned
                * X_test, y_test: (Numpy arrays) arrays containing data 
                                    to be tested
                * y_pred: (Numpy array) predicted values for X_test
                * trained_model: LinearRegression object'''
        self.data = data
        self.independent = self.data[independent].values
        self.dependent = self.data[dependent].values
        self.parameters = parameters
        self.y_pred = None
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self._split_dataset_for_training(size=size, seed=seed)
        self._train_model()


    def _split_dataset_for_training(self, size:int, seed: int):
        '''Method to split datatset into training and testing. The method is ran
           automatically when a DecisionTreeRegression object is created.
           Parameters:
                * size: (Integer) the proportion of the file that will 
                        be used for testing
                * seed: (Integer) the random seed used to shuffle data
           Returns:
                * X_train, X_test, y_train, y_test'''
        return train_test_split(self.independent, 
                                self.dependent, 
                                test_size=size, 
                                random_state=seed)


    def _train_model(self):
        '''Method to train model using method 'fit'. The method is ran
           automatically when a DecisionTreeRegression object is created.'''
        self.trained_model = DecisionTreeRegressor(**self.parameters)
        self.trained_model.fit(self.X_train, self.y_train)
        self.y_pred = self.predict(self.X_test)


    def predict(self, values):
        '''Method to get values predicted by the regression
           Parameters:
                * values: (list) the values on which the
                          prediction will be performed
           Returns:
                * (List) predicted values'''
        return self.trained_model.predict(values)


    def evaluate(self):
        '''Method to evaluate regression by calculating mean 
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
    prep_pipe.add_categorical_step(name='Encoder', action=OrdinalEncoder())
    prep_df = prep_pipe.rearrange_pipe(names=['Timestamp', 'Machine_ID'] + numerical_columns+categorical_columns, remove_nan=[dep] + indep)

    dtr = DecisionTreeRegression(data=prep_df.sample(n=1000, random_state=42),
                                 independent=indep,
                                 dependent=dep)
    
    
    # plot = Plotter(x=dtr.X_test, 
    #                y=dtr.y_test, 
    #                xlabel=indep,
    #                ylabel=dep,
    #                predict=dtr.predict,
    #                evaluate=dtr.evaluate)

    eval = ModelEvaluator(x=dtr.X_test, y=dtr.y_test, y_pred=dtr.y_pred)
    eval.evaluate()
    eval.plot_residuals(dep_variable=dep)