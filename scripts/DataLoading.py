import pandas as pd
import numpy as np
import os
from IPython.display import display
#from sklearn.impute import KNNImputer

class data_loading:
    def __init__(self, df_path, output_folder):
        """
        Initialise the EDA procecssor class with the necessary parameters.

        Args:
            df_path (str): The path to the DataFrame.
            output_folder (str): The path to the output folder
        """
        self.df_path = df_path
        self.new_df = None #Initialise new_df attribute
        self.output_folder = output_folder if output_folder else os.path.join(os.getcwd(),
                                                                            'data')
    def load_df (self):
        """
        Loads the dataframe from the specified path, performs basic cleaning and displays info.
        
        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        
        #load data
        try:
            raw_df = pd.read_csv(self.df_path, engine='python', sep='|')
            print('File read successfully!')
        except Exception as e:
            print(f'An error occurred: {e}')
            print('Please inspect the file to determine the correct delimiter.')
            return None #if loading fails
        
        #capitalise the first letter of each col names if it is small letter
        raw_df.columns = [col.capitalize() if col[0].islower() else col for col in raw_df.columns]
        print('\nColoumns are capitalised.')
        
        #replace empty indexes with nan ('.infer_objects(copy=False)' is added b/c .replace is depretiaitng)
        new_df = raw_df.replace('  ', np.nan).infer_objects(copy=False)
        print(f'\nEmpty indexes are filled with "NAN" values.')
        
        #change dtype to to datetime
        col_list = ['TransactionMonth','RegistrationYear', 'VehicleIntroDate']
        for col in col_list:
            try:
                new_df[col] = pd.to_datetime(new_df[col], format='ISO8601', errors='coerce')
            except ValueError:
                print(f'\nConversion failed, {col} column is not datetime type.')
        
        #remove columns with empty values greater than 60
        col= [col for col in new_df.columns if new_df[col].isnull().sum ()/ len(new_df) * 100 >= 60]
        new_df=new_df.drop(columns=col)
        print(f'\nColoumns {col} are dropped.')
        
        """
        #impute columns with empty values between 30 and 60
        impute_col= [col for col in new_df.columns if new_df[col].isnull().sum ()/ len(new_df) * 100 >= 30 and
                raw_df[col].isnull().sum() / len(raw_df) * 100 < 60]
        numerical_impute_cols= new_df[impute_col].select_dtypes(include=np.number).columns
        non_numerical_impute_cols= new_df[impute_col].select_dtypes(exclude=np.number).columns
        
        ##impute numerical columns with missing values
        if impute_col and numerical_impute_cols:
            #Apply KNN Imputer
            imputer = KNNImputer(n_neighbors=2)
            new_df[numerical_impute_cols]= imputer.fit_transform(new_df[numerical_impute_cols])
            print(f'\nNumerical coloumns {numerical_impute_cols} are imputed.')
        else:
            pass

        ##impute non-numerical columns with missing values
        if impute_col and non_numerical_impute_cols:
            for col in non_numerical_impute_cols:
                #calculate category probabilities
                category_counts = new_df[col].value_counts(normalize=True)
                #impute missing values based on probabilities
                new_df[col] = new_df[col].apply(lambda x: np.random.choice(category_counts.index,
                                                            p=category_counts.values) if pd.isna(x) else x)
            
            print(f'\nNon-Numerical Coloumns {non_numerical_impute_cols} are imputed.')
        else:
            pass
        """

        #remove index if column empty values is less than 30
        remove_col= [col for col in new_df.columns if new_df[col].isnull().sum ()/ len(new_df) * 100 < 30]
        if remove_col:
            new_df=new_df.dropna()
            print(f'\nIndexes with empty values are dropped.')
        
        #convert dtypes   
        ##to float
        col_list = ['CapitalOutstanding']
        for col in col_list:
            try:
                new_df[col] = new_df[col].str.replace(',', '.').astype(float)
            except ValueError:
                print(f'\nConversion for {col} failed.')
                pass
        
        ##to int
        col_list = ['Mmcode','Cubiccapacity','Kilowatts','NumberOfDoors', 'Cylinders']
        for col in col_list:
            try:
                new_df[col] = new_df[col].astype(int)
            except ValueError:
                print(f'\nConversion for {col} failed.')
                pass
        
        #sort values in Dataframe in ascending order
        col = 'TransactionMonth'
        new_df = new_df.sort_values(by=col)
        new_df.reset_index(drop=True, inplace=True)
        print(f'\nDataFrame is sorted by {col} column.')

        #___________________________________#
        print('\nDataFrame is preprocessed.')
        #___________________________________#
        
        #save processed new_df
        ##create output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        df_name = os.path.join(self.output_folder, 'processed_insurance_data.csv')
        
        ##calculate the relative path
        current_directory = os.getcwd()
        relative_path = os.path.relpath(df_name, current_directory)
        
        ##save processed data to CSV
        new_df.to_csv(df_name, index=False)
        print(f'\nPrrocessed DataFrame Saved to: {relative_path}')
        
        #assign processed new_df to self.new_df
        self.new_df = new_df
        
        print('\nDataFrame Head:')
        out_head=self.new_df.head()
        display (out_head)
        
        print('\nDataFrame Info:')
        self.new_df.info()
        
        print('\nDataFrame Shape:')
        display(self.new_df.shape)
        
        print('\nDataFrame Description:')
        cols_to_describe = ['SumInsured','CalculatedPremiumPerTerm','TotalPremium','TotalClaims']
        display(self.new_df[cols_to_describe].describe())
        
        return new_df
