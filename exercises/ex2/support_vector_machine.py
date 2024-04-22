import numpy as np
import pandas as pd
from plotter import Plotter
from model_evaluator import ModelEvaluator
from preprocessing import PreprocessingPipe
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


class SvmClassifier:

    def __init__(self, data, independent, dependent, parameters={'random_state': 0}, \
                 size:float=0.2, seed:int=42) -> None:
        '''Initializer of SvmClassifier object.
           Attributes and parameters:
                * data: (Pandas dataframe) the sample dataframe to be read
                * independent: (String) name of independent variable
                * dependent: (String) name of dependent variable
                * parameters: (Dict) SvmClassifier parameters. Default values is
                              random_state=0. See SvmClassifier documentation for more information.
                * size: (Integer) the proportion of the file that will 
                        be used for testing
                * seed: (Integer) the randome seed used to shuffle data
           Additional attributes:
                * X_train, y_train: (Numpy arrays) arrays containing data 
                                    to be trainned
                * X_test, y_test: (Numpy arrays) arrays containing data 
                                    to be tested
                * y_pred: (Numpy array) predicted values for X_test
                * trained_model: LogRegression object'''
        self.data = data
        self.independent = self.data[independent].values
        self.dependent = self.data[dependent].values
        self.parameters = parameters
        self.y_pred = None
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self._split_dataset_for_training(size=size, seed=seed)
        self._train_model()
        
    
    def _split_dataset_for_training(self, size:float, seed:int):
        '''Method to split datatset into training and testing. The method is ran
           automatically when a SvmClassifier object is created.
           Parameters:
                * size: (Integer) the proportion of the file that will 
                        be used for testing
                * seed: (Integer) the random seed used to shuffle data
           Returns:
                * X_train, X_test, y_train, y_test'''
        return train_test_split(self.independent, self.dependent, test_size=size, random_state=seed)


    def _train_model(self):
        '''Method to train model using method 'fit'. The method is ran
           automatically when a SvmClassifier object is created.'''
        self.trained_model = SVC(kernel='linear', **self.parameters)
        self.trained_model.fit(self.X_train, self.y_train)
        self.y_pred = self.predict(self.X_test)


    def predict(self, values):
        return self.trained_model.predict(values)


if __name__ == '__main__':

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
#    prep_pipe.add_numerical_step(name='Scaling', action=StandardScaler())
    prep_pipe.add_categorical_step(name='Encoder', action=OrdinalEncoder())
    prep_df = prep_pipe.rearrange_pipe(names=['Timestamp', 'Machine_ID'] + numerical_columns+categorical_columns, remove_nan=[dep] + indep)

    # Instantiate SvmClassifier 
    classifier = SvmClassifier(data=prep_df.sample(n=10000, random_state=42),
                                   independent=indep,
                                   dependent=dep)
    
    # Instantiate Plotter
    plot = Plotter(x=classifier.X_test, 
                   y=classifier.y_test, 
                   xlabel=indep,
                   ylabel=dep).classifier(predictor=classifier.predict, legends=['Normal', 'Fail'])

    eval = ModelEvaluator(x=classifier.X_test, 
                          y=classifier.y_test,
                          y_pred=classifier.y_pred).metrics(type='classifier', show=True)

