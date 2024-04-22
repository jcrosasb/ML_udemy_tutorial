import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, LabelEncoder


class PreprocessingPipe:

    def __init__(self, filepath: str, num_columns:list, cat_columns:list):
        '''Initializer for PreprocessingPipe object
           Parameters:
                * filepath: (String) the path to the file that will be preprocessed
                * num_column: (List) list with the name of the numerical columns
                * cat_column: (List) list with the name of the categorical columns
           Additional attributes:
                * removed_indices: indices of NaN.
                * numerical_steps: (List) list with the numerical steps that will be 
                                   perfomed in the preprocessing
                * categorical_steps: (List) list with the categorical steps that will 
                                   be perfomed in the preprocessing'''
        self.file = pd.read_csv(filepath)
        self.num_columns = num_columns
        self.cat_columns = cat_columns
        self.removed_indices = None
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


    def pipe(self, remove_nan=None):
        '''Method to pipe datafile through and preprocess the data. Data is transformed according
           to the steps given in 'add_numerical_step' and 'add_categorical_step'. The order of the
           columns is as in num_columns + cat_columns + (remaining columns in original order)
           Parameters:
                * remove_nan: (List) names of columns. If in any of those columns a row 
                              with a NaN is found, the entire row of the dataframe is deleted.
           Returns:
                * Pandas Dataframe with the transformed data'''       
        # Remove nan's
        if remove_nan:
#            self.removed_indices = self.file.index[self.file[remove_nan].isna().any(axis=1)]
            self.file.dropna(subset=remove_nan, inplace=True)

        # Create individual pipelines for numerical and categorical data
        num_pipe = Pipeline(steps=self.numerical_steps)
        cat_pipe = Pipeline(steps=self.categorical_steps)

        # Transformation object of the pipelines
        # ct = ColumnTransformer(transformers=[('numeric', num_pipe, self.num_columns),
        #                                      ('categorical', cat_pipe, self.cat_columns)])
        ct = ColumnTransformer(transformers=[('numeric', num_pipe, self.num_columns),
                                             ('categorical', cat_pipe, self.cat_columns)])
     
        # # Apply transformations and return
        return pd.DataFrame(ct.fit_transform(self.file))
        
    
    def rearrange_pipe(self, names=None, filename=None, remove_nan=None):
        '''Method to rearange pipe dataframe and save it. The initial order of the columns from pipe()
           is as in num_columns + cat_columns + (remaining columns in original order)
            Parameters:
                * names: (List) list of names of columns. Default is set to None, in which case
                         the columns are numbered
                * filename: (String) if given, the resulting dataframe is saved in a file 
                            named 'filename.csv'. Default is set to None
                * remove_nan: (List) list with names of columns. If in any of those columns a row 
                              with a NaN is found, the entire row of the dataframe is deleted.
            Returns:
                * Pandas Dataframe with the rearrange transformed data'''
        
        # Call transformed dataframe
        if remove_nan:
            df = self.pipe(remove_nan=remove_nan)
        else:
            df = self.pipe()

        # Dataframe of non-features
        non_feature_df = self.file[[col for col in self.file.columns if col not in self.num_columns + self.cat_columns]]

        # Put together the non-features with the features
        df = pd.concat([non_feature_df.reset_index(drop=True),df.reset_index(drop=True)], axis=1)

        # Rename columns if names list is provided
        if names:
            df.columns = names

        # Save file is filename is provided        
        if filename:
            df.to_csv(filename, index=False) 

        return df



        

if __name__ == '__main__':

    categorical_columns = ['Status']

    numerical_columns = ['Temperature_C','Pressure_psi','Vibration_mm_s','Power_Consumption_kW',
                         'Oil_Level','Error_Code','Production_Rate_units_min','Motor_Speed_RPM',
                         'Ambient_Temperature_C','Ambient_Humidity_percent','Part_Count','Voltage_V']
    
    file = 'data/data_csl.csv'

    prep_pipe = PreprocessingPipe(filepath=file,
                                  num_columns=numerical_columns,
                                  cat_columns=categorical_columns)

    prep_pipe.add_numerical_step(name='Imputer', action=SimpleImputer(missing_values=np.nan, strategy='median'))
    prep_pipe.add_numerical_step(name='Scaling', action=MinMaxScaler())

    prep_pipe.add_categorical_step(name='Encoder', action=OrdinalEncoder())
   
    df = prep_pipe.rearrange_pipe(names=['Timestamp', 'Machine_ID'] + numerical_columns+categorical_columns,
                                  remove_nan=['Temperature_C','Pressure_psi'])

    print(df)


    # df2 = pd.read_csv('data/data_reg.csv')
    # columns_to_check = ['Temperature_C','Pressure_psi']
    # rows_with_nan = df2.index[df2[columns_to_check].isna().any(axis=1)]
    # print(rows_with_nan)