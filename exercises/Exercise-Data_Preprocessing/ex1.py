import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler


class PreprocessingPipe:

    def __init__(self, filepath: str, num_columns:list, cat_columns:list):
        '''Initializer for PreprocessingPipe object
           Parameters:
                * filepath: (String) the path to the file that will be preprocessed
                * num_column: (List) list with the name of the numerical columns
                * cat_column: (List) list with the name of the categorical columns
           Additional attributes:
                * numerical_steps: (List) list with the numerical steps that will be 
                                   perfomed in the preprocessing
                * categorical_steps: (List) list with the categorical steps that will 
                                   be perfomed in the preprocessing'''
        self.file = pd.read_csv(filepath)
        self.num_columns = num_columns
        self.cat_columns = cat_columns
        self.numerical_steps = []
        self.categorical_steps = []


    def add_numerical_step(self, name: str, action):
        '''Method to add a numerical step.
           Parameters:
                * name: (String) a name for the preprocessing step.
                        e.g: 'Imputer'
                * action: (callable object) preprocessing operation 
                        that will be applied. e.g. 'SimpleImputer' '''
        self.numerical_steps.append((name, action))


    def add_categorical_step(self, name: str, action):
        '''Method to add a categorical step.
           Parameters:
                * name: (String) a name for the preprocessing step.
                        e.g: 'Imputer'
                * action: (callable object) preprocessing operation 
                        that will be applied. e.g. 'SimpleImputer' '''
        self.categorical_steps.append((name, action))


    def pipe(self, filename=None):
        '''Method to pipe datafile through and preprocess the data. Data is transformed according
           to the steps given in 'add_numerical_step' and 'add_categorical_step'. Any columns that 
           were not included in categorical or numerical columns will be left unchanged and will 
           take the first columns of resulting dataframe. The order of the rest of the columns 
           (named feature_1, feature_2, etc) is as in self.num_columns + self.cat_columns
           Parameters:
                * filename: (String) if given, the resulting dataframe is saved in a file 
                            named 'filename.csv'. Default is set to None
           Returns:
                * Pandas Dataframe with the transformed data'''
        # Create individual pipelines for numerical and categorical data
        num_pipe = Pipeline(steps=self.numerical_steps)
        cat_pipe = Pipeline(steps=self.categorical_steps)

        # Transformation object of the pipelines
        ct = ColumnTransformer(transformers=[('numeric', num_pipe, self.num_columns),
                                             ('categorical', cat_pipe, self.cat_columns)],
                               remainder='passthrough')
        
        # Apply transformations
        transformed_df = pd.DataFrame(ct.fit_transform(self.file))

        # Number of non feature columns (i.e. not in numerical and categorical columns)
        num_non_feature = self.file.shape[1] - len(self.num_columns) - len(self.cat_columns)

        # Get remainder columns and set names
        rem_columns = transformed_df.iloc[:, -num_non_feature:]    
        rem_columns.columns = [col for col in self.file.columns if col not in self.num_columns + self.cat_columns]

        # Get feature columns and set names
        feature_columns = transformed_df.iloc[:, :-num_non_feature]
        feature_columns.columns = [f'feature_{i+1}' for i in range(len(feature_columns.columns))]

        # Put remainder columns first
        transformed_df = pd.concat([rem_columns, feature_columns], axis=1)

        # Save file if 'filename' is provided
        if filename:
            transformed_df.to_csv(filename + '.csv', index=False) 
        return transformed_df
        

if __name__ == '__main__':

    categorical_columns = ['Status']

    numerical_columns = ['Temperature_C','Pressure_psi','Vibration_mm_s','Power_Consumption_kW',
                         'Oil_Level','Error_Code','Production_Rate_units_min','Motor_Speed_RPM',
                         'Ambient_Temperature_C','Ambient_Humidity_percent','Part_Count','Voltage_V']
    
    file = 'data/data_pp.csv'

    prep_pipe = PreprocessingPipe(filepath=file,
                                  num_columns=numerical_columns,
                                  cat_columns=categorical_columns)


    # First pipeline ==========================================================================
    prep_pipe = PreprocessingPipe(filepath='data/data_pp.csv', 
                                  num_columns=numerical_columns, 
                                  cat_columns=categorical_columns)
    
    prep_pipe.add_numerical_step(name='Imputer', action=SimpleImputer(missing_values=np.nan, strategy='median'))
    prep_pipe.add_numerical_step(name='Scaling', action=StandardScaler())

    prep_pipe.add_categorical_step(name='Encoder', action=OrdinalEncoder())

    prep_pipe.pipe(filename='data_median_std_ordinal')


    # Second pipeline ==========================================================================
    prep_pipe_2 = PreprocessingPipe(filepath='data/data_pp.csv', 
                                  num_columns=numerical_columns, 
                                  cat_columns=categorical_columns)
    
    prep_pipe_2.add_numerical_step(name='Imputer', action=SimpleImputer(missing_values=np.nan, strategy='median'))
    prep_pipe_2.add_numerical_step(name='Scaling', action=StandardScaler())

    prep_pipe_2.add_categorical_step(name='Encoder', action=OneHotEncoder())

    prep_pipe_2.pipe(filename='data_median_std_onehot')


    # Third pipeline ==========================================================================
    prep_pipe_3 = PreprocessingPipe(filepath='data/data_pp.csv', 
                                  num_columns=numerical_columns, 
                                  cat_columns=categorical_columns)
    
    prep_pipe_3.add_numerical_step(name='Imputer', action=SimpleImputer(missing_values=np.nan, strategy='mean'))
    prep_pipe_3.add_numerical_step(name='Scaling', action=MinMaxScaler())

    prep_pipe_3.add_categorical_step(name='Encoder', action=OneHotEncoder())

    prep_pipe_3.pipe(filename='data_median_norm_onehot')








    # df = pd.read_csv('data/data_pp.csv')

    # categorical_columns = ['Status']

    # numerical_columns = ['Temperature_C','Pressure_psi','Vibration_mm_s','Power_Consumption_kW',
    #                      'Oil_Level','Error_Code','Production_Rate_units_min','Motor_Speed_RPM',
    #                      'Ambient_Temperature_C','Ambient_Humidity_percent','Part_Count','Voltage_V']
    
    # numerical_steps = [('Imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
    #                    ('Scaling', StandardScaler())]
    

    # categorical_steps = [('Encoder', OrdinalEncoder())]

    # numeric_pipeline = Pipeline(steps=numerical_steps)

    # categorical_pipeline = Pipeline(steps=categorical_steps)

    # ct = ColumnTransformer(transformers=[('numeric', numeric_pipeline, numerical_columns),
    #                                      ('categorical', categorical_pipeline, categorical_columns)],
    #                        remainder='passthrough')
    

    # df_transformed = pd.DataFrame(ct.fit_transform(df))


    # print(df_transformed)