import pandas as pd
import numpy as np
import os
from IPython.display import display
import ipywidgets as widgets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class EDA_procecssor:
    def __init__(self, df_path, output_folder, plot_folder):
        """
        Initialise the EDA procecssor class with the necessary parameters.

        Args:
            df_path (str): The path to the DataFrame.
            output_folder (str): The path to the output folder
            plot_folder (str): The folder to save the plot.
        """
        self.df_path = df_path
        self.df = None #Initialise df attribute
        self.output_folder = output_folder if output_folder else os.path.join(os.getcwd(),
                                                                            'data')
        self.plot_folder = plot_folder if plot_folder else os.path.join(os.getcwd(),
                                                                            'plot image')
    def save_plot(self, plot_folder, plot_name, plot_path):
        """
        Saves the current matplotlib plot to a specified location.

        Args:
            plot_folder (str): The folder to save the plot.
            plot_name (str): The name of the plot file.
            plot_path (str): The full path to save the plot.
        """
        
        #create the directory if it doesn't exist
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        
        #save plot
        plt.savefig(plot_path)

        #calculate the relative path
        current_directory = os.getcwd()
        relative_plot_path = os.path.relpath(plot_path, current_directory)

        #display message
        print(f'\nPlot is saved to {relative_plot_path}.\n') 

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
        #print('\nColoumns are capitalised.')

        #replace empty indexes with nan ('.infer_objects(copy=False)' is added b/c .replace is depretiaitng)
        df = raw_df.replace('  ', np.nan).infer_objects(copy=False)
        #print(f'\nEmpty indexes are filled with "NAN" values.')

        #sort values in Dataframe in ascending order
        #change dtype to to datetime
        col_list = ['TransactionMonth','RegistrationYear', 'VehicleIntroDate']
        for col in col_list:
            if df[col].dtype != 'datetime64[ns]':
                try:
                    df[col] = pd.to_datetime(df[col], format='ISO8601', errors='coerce')
                except ValueError:
                    print(f'\nConversion failed, {col} column is not datetime type.')
                    pass
        col = 'TransactionMonth'
        df = df.sort_values(by=col).reset_index(drop=True)
        #print(f'\nDataFrame is sorted by {col} column.')
        
        #remove columns with empty values greater than 60
        col= [col for col in df.columns if df[col].isnull().sum ()/ len(df) * 100 >= 60]
        df=df.drop(columns=col)
        #print(f'\nColoumns {col} are dropped.')

        #impute columns with empty values between 30 and 60
        impute_col= [col for col in df.columns if df[col].isnull().sum ()/ len(df) * 100 >= 30 and
                raw_df[col].isnull().sum() / len(raw_df) * 100 < 60]
        numerical_impute_cols= df[impute_col].select_dtypes(include=np.number).columns
        non_numerical_impute_cols= df[impute_col].select_dtypes(exclude=np.number).columns

        ##impute numerical columns with missing values
        if impute_col and numerical_impute_cols:
            #Apply KNN Imputer
            imputer = KNNImputer(n_neighbors=2)
            df[numerical_impute_cols]= imputer.fit_transform(df[numerical_impute_cols])
            print(f'\nNumerical coloumns {numerical_impute_cols} are imputed.')
        else:
            pass

        ##impute non-numerical columns with missing values
        if impute_col and non_numerical_impute_cols:
            for col in non_numerical_impute_cols:
                #calculate category probabilities
                category_counts = df[col].value_counts(normalize=True)
                #impute missing values based on probabilities
                df[col] = df[col].apply(lambda x: np.random.choice(category_counts.index,
                                                            p=category_counts.values) if pd.isna(x) else x)

            print(f'\nNon-Numerical Coloumns {non_numerical_impute_cols} are imputed.')
        else:
            pass

        #remove index if column empty values is less than 30
        remove_col= [col for col in df.columns if df[col].isnull().sum ()/ len(df) * 100 < 30]
        if remove_col:
            df=df.dropna()
            #print(f'\nIndexes with empty values are dropped.')

        #convert dtypes   
        ##to float
        col_list = ['CapitalOutstanding']
        for col in col_list:
            try:
                df[col] = df[col].str.replace(',', '.').astype(float)
            except ValueError:
                print(f'\nConversion for {col} failed.')
                pass
        
        ##to int
        col_list = ['Mmcode','Cubiccapacity','Kilowatts','NumberOfDoors', 'Cylinders']
        for col in col_list:
            try:
                df[col] = df[col].astype(int)
            except ValueError:
                print(f'\nConversion for {col} failed.')
                pass
        
        #___________________________________#
        print('\nDataFrame is preprocessed.')
        #___________________________________#

        #save processed df
        ##create output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        df_name = os.path.join(self.output_folder, 'processed_insurance_data.csv')

        ##calculate the relative path
        current_directory = os.getcwd()
        relative_path = os.path.relpath(df_name, current_directory)

        ##save processed data to CSV
        df.to_csv(df_name, index=False)
        print(f'\nPrrocessed DataFrame Saved to: {relative_path}')
        
        #assign processed df to self.df
        self.df = df

        #create separate output areas 
        out_head = widgets.Output()
        with out_head:
            print('DataFrame Head:')
            print(self.df.head())

        out_info = widgets.Output()
        with out_info:
            print('DataFrame Info:')
            self.df.info()

        out_shape = widgets.Output()
        with out_shape:
            print('DataFrame Shape:')
            display(self.df.shape)

        out_desc = widgets.Output()
        with out_desc:
            print('DataFrame Description:')
            cols_to_describe = ['SumInsured','CalculatedPremiumPerTerm',
                                'TotalPremium','TotalClaims']
            display(self.df[cols_to_describe].describe())
        
        #display all outputs in sequence
        display(out_head, out_info, out_shape, out_desc)

        return self.df
    
    def calculate_loss(self):
        """
        Calculates and prints the overall loss ratio.
        """
    
        if self.df is not None:
            #----calculate overall loss ratio----#
            ##create separate output areas
            out_overall_loss = widgets.Output()
            with out_overall_loss:
                overall_loss_ratio = self.df['TotalClaims'].sum() / self.df['TotalPremium'].sum()
                print(f'Overall Loss Ratio: {overall_loss_ratio:.2f}')
            
            #----loss ratio by Province----#            
            ##create separate output areas
            out_province_loss = widgets.Output()
            with out_province_loss:
                loss_ratio_by_province = self.df.groupby('Province').agg(total_claims=('TotalClaims', 'sum'),
                                                                        total_premium=('TotalPremium', 'sum'))
                loss_ratio_by_province['LossRatio'] = loss_ratio_by_province['total_claims'] / loss_ratio_by_province['total_premium']
                print('\nLoss Ratio by Province:')
                display(loss_ratio_by_province[['LossRatio']])

            #---loss ratio by VehicleType----#            
            ##create separate output areas
            out_vehicle_loss = widgets.Output()
            with out_vehicle_loss:
                loss_ratio_by_vehicle_type = self.df.groupby('VehicleType').agg(total_claims=('TotalClaims', 'sum'),
                                                                                total_premium=('TotalPremium', 'sum'))
                loss_ratio_by_vehicle_type['LossRatio'] = loss_ratio_by_vehicle_type['total_claims'] / loss_ratio_by_vehicle_type['total_premium']
                print('\nLoss Ratio by Vehicle Type:')
                display(loss_ratio_by_vehicle_type[['LossRatio']])

            #----loss ratio by Gender----#
            #create separate output areas
            out_gender_loss = widgets.Output()
            with out_gender_loss:
                loss_ratio_by_gender = self.df.groupby('Gender').agg(total_claims=('TotalClaims', 'sum'),
                                                                    total_premium=('TotalPremium', 'sum'))
                loss_ratio_by_gender['LossRatio'] = loss_ratio_by_gender['total_claims'] / loss_ratio_by_gender['total_premium']

                print('\nLoss Ratio by Gender:')
                display(loss_ratio_by_gender[['LossRatio']])

            #display all outputs in sequence
            display(out_overall_loss, out_province_loss, 
                    out_vehicle_loss, out_gender_loss)

        else:
            print('DataFrame is not loaded. Please run load_df first.')

    def temporal_trends (self, plot_folder):
        """
        Calculate and visualise claim frequency and severity per month. 
        """
        if self.df is not None:
            if 'TransactionMonth' in self.df.columns:

                #analyse claim frequency per month
                self.df['TransactionMonth'] = self.df['TransactionMonth'].dt.to_period('M')
                claim_frequency = self.df.groupby('TransactionMonth').size()
                print('Claim Frequency per month (head):')
                display(claim_frequency.head(10))

                #analyse claim severity per month
                claim_severity = self.df.groupby('TransactionMonth')['TotalClaims'].mean()
                print('\nClaim Severity per month (head):')
                display(claim_severity.head(10))

                #visualise temporal trends
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                claim_frequency.plot()
                plt.title('Claim Frequency per Month') 
                plt.xlabel('Month') 
                plt.ylabel('Number of Claims')
                plt.grid()

                plt.subplot(1, 2, 2)
                claim_severity.plot()
                plt.title('Claim Severity per Month')
                plt.xlabel('Month') 
                plt.ylabel('Average Claim Amount')
                plt.grid() 
                plt.tight_layout()

                #select plot directory and plot name to save plot
                plot_name = 'Claim Frequency and Claim Severity per Month.png'
                plot_path = os.path.join(plot_folder, plot_name)
                self.save_plot (plot_folder, plot_name, plot_path)

                #show plot
                plt.show()
                #close plot to free up space
                plt.close()
            else:
                print("\nNo 'TransactionMonth' column found. Please inspect data for a suitable date column.")
        else:
            print('DataFrame is not loaded. Please run load_df first.')
    
    def make_model_claim (self):

        if self.df is not None:
            #----average claim amount by Vehicle Make----#
            avg_claim_by_make = self.df.groupby('Make')['TotalClaims'].mean().sort_values(ascending=False)
            print('Average Claim Amount by Vehicle Make (Top 10):')
            display(avg_claim_by_make.head(10))
            
            #----average claim amount by Vehicle Model----#
            avg_claim_by_model = self.df.groupby('Model')['TotalClaims'].mean().sort_values(ascending=False)
            print('\nAverage Claim Amount by Vehicle Model (Top 10):')
            display(avg_claim_by_model.head(10))
            
        else:
            print('DataFrame is not loaded. Please run load_df first.')
    
    def univariate_analysis_visualiser(self):
        """
        Performs univariate analysis by plotting histograms for numerical columns
        and bar charts for categorical columns.
        """
        if self.df is not None:
            numerical_cols = self.df.select_dtypes(include=np.number).columns
            categorical_cols = self.df.select_dtypes(include=['object','bool']).columns
            #exclude 'Make' and 'Model' from categorical columns
            categorical_cols = [col for col in categorical_cols if col not in ['Make', 'Model']] 

            #plot histograms for numerical columns
            print('Plotting Histograms for Numerical Columns:')
            for col in numerical_cols:
                plt.figure(figsize=(10, 5))
                sns.histplot(data=self.df, x=col, color='red', kde=True)
                plt.title(f'Distribution of {col}')
                plt.xlabel(col, labelpad=15) #adds distance between lable ticks
                plt.ylabel('Frequency')
                plt.grid()
                plt.show()

            #plot bar charts for categorical columns
            print('\nPlotting Bar Charts for Categorical Columns with more than one unique values:')
            for col in categorical_cols:
                #filter columns more than single unique values and plot
                if self.df[col].nunique() >1: 
                    plt.figure(figsize=(12, 15))
                    ax = self.df[col].value_counts().plot(kind='bar', 
                                                        color=sns.color_palette('viridis', 
                                                                                len(self.df[col].unique()))) 
                    plt.title(f'Distribution of {col}')
                    plt.xlabel(col, labelpad=15) #adds distance between lable ticks
                    plt.ylabel('Count')
                    plt.grid()    
                    #rotate x-ticks based on its the number of unique values
                    if len(self.df[col].unique()) >= 12:
                        plt.xticks(rotation=90, ha='right')
                    elif len(self.df[col].unique()) >= 4:
                        plt.xticks(rotation=45, ha='right')
                    else:
                        plt.xticks(rotation=0)
                    #add value labels on top of each bar
                    for container in ax.containers:
                        ax.bar_label(container, rotation=90, padding=3)

                    plt.tight_layout()
                    plt.show()
        else:
            print('DataFrame is not loaded. Please run load_df first.')

    def bivariate_analysis_and_visualiser(self):
        """
        Explores relationships between monthly changes in TotalPremium and TotalClaims
        as a function of PostalCode.
        """
        if self.df is not None:
            if 'TransactionMonth' in self.df.columns and 'PostalCode' in self.df.columns:
                #extract month from col
                self.df['Month'] = self.df['TransactionMonth']

                #aggregate data by Month and PostalCode
                monthly_postalcode_agg = self.df.groupby(['Month', 'PostalCode']).agg(
                    MonthlyTotalPremium=('TotalPremium', 'sum'),
                    MonthlyTotalClaims=('TotalClaims', 'sum')).reset_index()

                print('Monthly aggregated data by PostalCode (head):')
                display(monthly_postalcode_agg.head())

                #calculate monthly changes
                monthly_postalcode_agg['PremiumChange'] = monthly_postalcode_agg.groupby(
                    'PostalCode')['MonthlyTotalPremium'].diff()
                monthly_postalcode_agg['ClaimsChange'] = monthly_postalcode_agg.groupby(
                    'PostalCode')['MonthlyTotalClaims'].diff()

                #drop the first month for each postalcode b/c diff will be NaN
                monthly_postalcode_agg.dropna(subset=['PremiumChange', 'ClaimsChange'], inplace=True)

                print('\nMonthly changes by PostalCode (head):')
                display(monthly_postalcode_agg.head())

                #scatter plot of ClaimsChange vs PremiumChange, colored by PostalCode
                plt.figure(figsize=(10, 7))
                sns.scatterplot(data=monthly_postalcode_agg, 
                                x='PremiumChange', y='ClaimsChange', 
                                hue='PostalCode', alpha=0.6)
                plt.title('Monthly Claims Change vs Monthly Premium Change by PostalCode')
                plt.xlabel('Monthly Premium Change')
                plt.ylabel('Monthly Claims Change')
                plt.legend(title='PostalCode', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True)
                plt.tight_layout()
                plt.show()

                #correlation matrix for monthly changes (can be less meaningful with many PostalCodes)
                print('\nCorrelation Matrix for Monthly Changes (all PostalCodes):')
                correlation_matrix = monthly_postalcode_agg[['PremiumChange', 'ClaimsChange']].corr()
                display(correlation_matrix)

                #heatmap for the overall correlation matrix
                plt.figure(figsize=(6, 5))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
                plt.title('Correlation Heatmap for Monthly Premium and Claims Change')
                plt.show()

            else:
                print("\nRequired columns ('TransactionMonth', 'PostalCode', 'TotalPremium', 'TotalClaims') not found in DataFrame.")
        else:
            print('\nDataFrame is not loaded. Please run load_df first.')


    def geographic_trends_analysis_and_visualiser(self, plot_folder):
        """
        Compares trends in insurance cover type, premium, auto make, etc. over provinces.
        """
        if self.df is not None:
            if 'Province' in self.df.columns:

                #compare Cover Type distribution by Province
                print('\nDistribution of Cover Type by Province:')
                plt.figure(figsize=(12, 6))
                sns.countplot(data=self.df, x='Province')
                plt.title('Distribution of Insurance Cover Type by Province')
                plt.xlabel('Province')
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.grid()
                plt.tight_layout()

                #select plot directory and plot name to save plot
                plot_name = 'Distribution of Insurance Cover Type by Province.png'
                plot_path = os.path.join(plot_folder, plot_name)
                self.save_plot (plot_folder, plot_name, plot_path)

                #show plot
                plt.show()
                #close plot to free up space
                plt.close()

                #compare average premium by Province
                print('\nAverage TotalPremium by Province:')
                avg_premium_by_province = self.df.groupby('Province')['TotalPremium'].mean().sort_values(ascending=False)
                display(avg_premium_by_province)

                plt.figure(figsize=(10, 6))
                avg_premium_by_province.plot(kind='bar')
                plt.title('Average TotalPremium by Province')
                plt.xlabel('Province')
                plt.ylabel('Average Premium')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.grid()
                plt.show()

                #compare distribution of top auto makes by Province
                print('\nDistribution of Top Auto Makes by Province:(head)')
                top_makes = self.df['Make'].value_counts().head(10).index 

                #filter the DataFrame to include only the top makes
                df_top_makes = self.df[self.df['Make'].isin(top_makes)]

                plt.figure(figsize=(14, 8))
                sns.countplot(data=df_top_makes, x='Province', hue='Make')
                plt.title('Distribution of Top Auto Makes by Province')
                plt.xlabel('Province')
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.legend(title='Make', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid()
                plt.tight_layout()
                plt.show()


            else:
                print("The 'Province' column is not found in the DataFrame.")
        else:
            print('DataFrame is not loaded. Please run load_df first.')
    
    def outlier_detection_boxplots(self,plot_folder):
        """
        Uses box plots to detect outliers in numerical data.
        """

        if self.df is not None:
            numerical_cols = self.df.select_dtypes(include=np.number).columns

            print('\nPlotting Box Plots for Outlier Detection in Numerical Columns:')
            for col in numerical_cols:
                plt.figure(figsize=(8, 6))
                sns.boxplot(y=self.df[col], orientation='vertical')
                plt.title(f'Box Plot of {col} for Outlier Detection')
                plt.ylabel(col)
                plt.grid(True)

                if col == 'TotalPremium':
                    #select plot directory and plot name to save plot
                    plot_name = f'Box Plot of {col} for Outlier Detection.png'
                    plot_path = os.path.join(plot_folder, plot_name)
                    self.save_plot (plot_folder, plot_name, plot_path)

                #show plot
                plt.show()
                #close plot to free up space
                plt.close()
        else:
            print('DataFrame is not loaded. Please run load_df first.')

