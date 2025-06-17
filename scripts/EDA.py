import pandas as pd
import numpy as np
import os
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

class EDA_processor:
    def __init__(self, processed_df_path=None, plot_folder=None, output_folder=None):
        """
        Initialise the EDA procecssor class with the necessary parameters.
        
        Args:
            processed_df_path (str): The path to the processed DataFrame.
            plot_folder (str): The folder to save plots.
            output_folder (str): The path to the output folder
        """
        self.df = None #Initalises df to None
        self.processed_df_path = processed_df_path
        self.plot_folder = plot_folder if plot_folder else os.path.join(os.getcwd(),
                                                                        'plot images')
        self.output_folder = output_folder if output_folder else os.path.join(os.getcwd(),
                                                                            'data')
        
        #create plot folder if it doesnt exist                                                              
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)
        
        #load df if df path is provided
        if self.processed_df_path:
            self.load_processed_df_and_impute()
    
    def load_processed_df_and_impute(self):
        """
        Loads the processed DataFrame from the specified path.
        Assumes the file is in CSV format.
        """
        #calculate the relative path
        relative_processed_df_path = os.path.relpath(self.processed_df_path, os.getcwd())
        
        if self.processed_df_path and os.path.exists(self.processed_df_path):
            try:
                self.df = pd.read_csv(self.processed_df_path)
                print(f'DataFrame loaded successfully from {relative_processed_df_path}')
                
                #perform imputation after loading data
                numerical_cols = self.df.select_dtypes(include=np.number).columns
                for col in numerical_cols:
                    non_zero_values = self.df[self.df[col] != 0][col]
                    if not non_zero_values.empty: #ensure non-zero values exist before computing median
                        median_non_zero_claims = non_zero_values.median()
                        self.df[col] = self.df[col].replace(0, median_non_zero_claims)
                        #print (f'\nNumerical columns with zeros are imputed.')
                    else:
                        print(f'\nDataFrame contains only zeros; skipping imputation.')

                #save imputed df
                ##create output folder if it doesn't exist
                if not os.path.exists(self.output_folder):
                    os.makedirs(self.output_folder)

                df_name = os.path.join(self.output_folder, 'imputed_and_processed_insurance_data.csv')

                ##calculate the relative path
                current_directory = os.getcwd()
                relative_path = os.path.relpath(df_name, current_directory)

                ##save processed data to CSV
                self.df.to_csv(df_name, index=False)
                print(f'\nImputed DataFrame Saved to: {relative_path}')

                print('\nDataFrame Head:')
                out_head=self.df.head()
                display (out_head)
            
            except Exception as e:
                print(f'Error loading DataFrame from {relative_processed_df_path}: {e}')
                self.df = None 
        elif self.processed_df_path:
            print(f'\nError: File not found at {relative_processed_df_path}')
            self.df = None
        else:
            print('\nNo DataFrame path provided during initialization.')
            self.df = None
        return self.df
    
    def save_plot(self, plot_name):
        """
        Saves the current matplotlib plot to the designated plot folder.
        
        Args:
            plot_name (str): The name of the plot file (including extension, e.g., '.png').
        """
        
        #create the directory if it doesn't exist
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)
        
        plot_path = os.path.join(self.plot_folder, plot_name)
        #calculate the relative path
        relative_plot_path = os.path.relpath(plot_path, os.getcwd())
        
        try:
            plt.savefig(plot_path)
            print(f'\nPlot saved to {relative_plot_path}')
        except Exception as e:
            print(f'\nError saving plot: {e}')
    
    def calculate_loss(self):
        """
        Calculates and prints the overall loss ratio.
        """
    
        if self.df is not None:
            #----calculate overall loss ratio----#
            overall_loss_ratio = self.df['TotalClaims'].sum() / self.df['TotalPremium'].sum()
            print(f'Overall Loss Ratio: {overall_loss_ratio:.2f}')
            
            #----loss ratio by Province----#            
            loss_ratio_by_province = self.df.groupby('Province').agg(total_claims=('TotalClaims', 'sum'),
                                                                        total_premium=('TotalPremium', 'sum'))
            loss_ratio_by_province['LossRatio'] = loss_ratio_by_province['total_claims'] / loss_ratio_by_province['total_premium']
            print('\nLoss Ratio by Province:')
            display(loss_ratio_by_province[['LossRatio']])
            
            #---loss ratio by VehicleType----#            
            loss_ratio_by_vehicle_type = self.df.groupby('VehicleType').agg(total_claims=('TotalClaims', 'sum'),
                                                                                total_premium=('TotalPremium', 'sum'))
            loss_ratio_by_vehicle_type['LossRatio'] = loss_ratio_by_vehicle_type['total_claims'] / loss_ratio_by_vehicle_type['total_premium']
            print('\nLoss Ratio by Vehicle Type:')
            display(loss_ratio_by_vehicle_type[['LossRatio']])
            
            #----loss ratio by Gender----#
            loss_ratio_by_gender = self.df.groupby('Gender').agg(total_claims=('TotalClaims', 'sum'),
                                                                    total_premium=('TotalPremium', 'sum'))
            loss_ratio_by_gender['LossRatio'] = loss_ratio_by_gender['total_claims'] / loss_ratio_by_gender['total_premium']
            print('\nLoss Ratio by Gender:')
            display(loss_ratio_by_gender[['LossRatio']])
        
        else:
            print('DataFrame is not loaded. Please run load_df first.')
    
    def temporal_trends (self):
        """
        Calculate and visualise claim frequency and severity per month. 
        """
        if self.df is not None:
            if 'TransactionMonth' in self.df.columns:
                
                #analyse claim frequency per month
                #self.df['TransactionMonth'] = self.df['TransactionMonth'].dt.to_period('M')
                claim_frequency = self.df.groupby('TransactionMonth').size()
                print('Claim Frequency per month (head):')
                display(claim_frequency.head(10))
                
                #analyse claim severity per month
                claim_severity = self.df.groupby('TransactionMonth')['TotalClaims'].mean()
                print('\nClaim Severity per month (head):')
                display(claim_severity.head(10))
                
                #visualise temporal trends
                plt.figure(figsize=(15, 10))
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
                plot_name = '01 Claim Frequency and Claim Severity per Month.png'
                self.save_plot (plot_name)
                
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
            #convert the 'TransactionMonth' column and 'RegistrationYear' again to datetime
            self.df['TransactionMonth'] = pd.to_datetime(self.df['TransactionMonth']
                                                        , format='ISO8601', errors='coerce')
            self.df['RegistrationYear'] = pd.to_datetime(self.df['RegistrationYear']
                                                        , format='ISO8601', errors='coerce')
            numerical_cols = self.df.select_dtypes(include=np.number).columns
            categorical_cols = self.df.select_dtypes(include=['object','bool']).columns
            categorical_cols = [col for col in categorical_cols if col not in ['Make', 'Model']]
            
            #plot histograms for numerical columns
            print('Plotting Histograms for Numerical Columns:')
            for col in numerical_cols:
                plt.figure(figsize=(10, 10))
                sns.histplot(data=self.df, x=col, color='red', kde=True)
                plt.title(f'Distribution of {col} Column')
                plt.xlabel(col, labelpad=5) #adds distance between lable ticks
                plt.ylabel('Frequency')
                plt.grid()
                plt.tight_layout()
                
                #select plot directory and plot name to save plot
                plot_name = f'02 Distribution of {col} Column.png'
                self.save_plot (plot_name)
                
                #show plot
                plt.show()
                #close plot to free up space
                plt.close()
            
            #plot bar charts for categorical columns
            print('\nPlotting Bar Charts for Categorical Columns with more than one unique values:')
            for col in categorical_cols:
                #filter columns more than single unique values and plot
                if self.df[col].nunique() >1:
                    if self.df[col].nunique() <=3: 
                        plt.figure(figsize=(10, 10))
                        ax = self.df[col].value_counts().plot(kind='bar',
                                                                color=sns.color_palette('viridis',
                                                                                        len(self.df[col].unique()))) 
                        plt.title(f'Distribution of {col} Column')
                        plt.xlabel(col, labelpad=5) #adds distance between lable ticks
                        plt.xticks(rotation=0, ha='center')
                        plt.ylabel('Count')
                        plt.grid()    
                        for container in ax.containers:
                            ax.bar_label(container, rotation=0, padding=3)
                        plt.tight_layout()
                        
                        #select plot directory and plot name to save plot
                        plot_name = f'03 Distribution of {col} Column.png'
                        self.save_plot (plot_name)
                        
                        #show plot
                        plt.show()
                        #close plot to free up space
                        plt.close()
                    
                    elif self.df[col].nunique() >3 and self.df[col].nunique() <=12: 
                        plt.figure(figsize=(10, 10))
                        ax = self.df[col].value_counts().plot(kind='bar',color=sns.color_palette('viridis',
                                                                                                len(self.df[col].unique()))) 
                        plt.title(f'Distribution of {col} Column')
                        plt.xlabel(col)
                        plt.xticks(rotation=45, ha='right')
                        plt.ylabel('Count')
                        plt.grid()    
                        for container in ax.containers:
                            ax.bar_label(container, rotation=0, padding=3)
                        plt.tight_layout()
                        
                        #select plot directory and plot name to save plot
                        plot_name = f'03 Distribution of {col} Column.png'
                        self.save_plot (plot_name)
                        
                        #show plot
                        plt.show()
                        #close plot to free up space
                        plt.close()
                    
                    else:
                        plt.figure(figsize=(10, 10))
                        ax = self.df[col].value_counts().plot(kind='bar',color=sns.color_palette('viridis',
                                                                                                len(self.df[col].unique()))) 
                        plt.title(f'Distribution of {col} Column')
                        plt.xlabel(col)
                        plt.xticks(rotation=90, ha='right')
                        plt.ylabel('Count')
                        plt.grid()    
                        for container in ax.containers:
                            ax.bar_label(container, rotation=90, padding=3)
                        plt.tight_layout()
                        
                        #select plot directory and plot name to save plot
                        plot_name = f'03 Distribution of {col} Column.png'
                        self.save_plot (plot_name)
                        
                        #show plot
                        plt.show()
                        #close plot to free up space
                        plt.close()
        
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
                plt.figure(figsize=(10, 10))
                sns.scatterplot(data=monthly_postalcode_agg, 
                                x='PremiumChange', y='ClaimsChange', 
                                hue='PostalCode', alpha=0.6)
                plt.title('Monthly Claims Change vs Monthly Premium Change by PostalCode')
                plt.xlabel('Monthly Premium Change')
                plt.ylabel('Monthly Claims Change')
                plt.legend(title='PostalCode', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True)
                plt.tight_layout()
                
                #select plot directory and plot name to save plot
                plot_name = '04 Monthly Claims Change vs Monthly Premium Change by PostalCode.png'
                self.save_plot (plot_name)
                
                #show plot
                plt.show()
                #close plot to free up space
                plt.close()
                
                #correlation matrix for monthly changes (can be less meaningful with many PostalCodes)
                print('\nCorrelation Matrix for Monthly Changes (all PostalCodes):')
                correlation_matrix = monthly_postalcode_agg[['PremiumChange', 'ClaimsChange']].corr()
                display(correlation_matrix)
                
                #heatmap for the overall correlation matrix
                plt.figure(figsize=(5, 5))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
                plt.title('Correlation Heatmap for Monthly Premium and Claims Change')
                #plt.tight_layout()
                
                #select plot directory and plot name to save plot
                plot_name = '05 Correlation Heatmap for Monthly Premium and Claims Change.png'
                self.save_plot (plot_name)
                
                #show plot
                plt.show()
                #close plot to free up space
                plt.close()            
            
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
                plt.figure(figsize=(10, 10))
                sns.countplot(data=self.df, x='Province', hue='Province')
                plt.title('Distribution of Insurance Cover Type by Province')
                plt.xlabel('Province')
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.grid()
                plt.tight_layout()
                
                #select plot directory and plot name to save plot
                plot_name = '06 Distribution of Insurance Cover Type by Province.png'
                self.save_plot (plot_name)
                
                #show plot
                plt.show()
                #close plot to free up space
                plt.close()    
                
                #compare average premium by Province
                print('\nAverage TotalPremium by Province:')
                avg_premium_by_province = self.df.groupby('Province')['TotalPremium'].mean().sort_values(ascending=False)
                display(avg_premium_by_province)
                
                plt.figure(figsize=(10, 10))
                avg_premium_by_province.plot(kind='bar')
                avg_premium_by_province.plot(kind='bar', 
                                                        color=sns.color_palette('viridis', 
                                                                                len(avg_premium_by_province.unique()))) 
                plt.title('Average TotalPremium by Province')
                plt.xlabel('Province')
                plt.ylabel('Average Premium')
                plt.xticks(rotation=45, ha='right')
                plt.grid()
                plt.tight_layout()
                
                #select plot directory and plot name to save plot
                plot_name = '07 Average TotalPremium by Province.png'
                self.save_plot (plot_name)
                
                #show plot
                plt.show()
                #close plot to free up space
                plt.close()    
                
                #compare distribution of top auto makes by Province
                print('\nDistribution of Top Auto Makes by Province:(head)')
                top_makes = self.df['Make'].value_counts().head(10).index 
                
                #filter the DataFrame to include only the top makes
                df_top_makes = self.df[self.df['Make'].isin(top_makes)]
                
                plt.figure(figsize=(10, 10))
                sns.countplot(data=df_top_makes, x='Province', hue='Make')
                plt.title('Distribution of Top Auto Makes by Province')
                plt.xlabel('Province')
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.legend(title='Make', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid()
                plt.tight_layout()
                
                #select plot directory and plot name to save plot
                plot_name = '08 Distribution of Top Auto Makes by Province.png'
                self.save_plot (plot_name)
                
                #show plot
                plt.show()
                #close plot to free up space
                plt.close()    
            
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
                plt.figure(figsize=(5, 5))
                sns.boxplot(y=self.df[col], orientation='vertical')
                plt.title(f'Box Plot of {col} for Outlier Detection')
                plt.ylabel(col)
                plt.grid()
                #plt.tight_layout()
                
                #select plot directory and plot name to save plot
                plot_name = f'09 Box Plot of {col} for Outlier Detection.png'
                self.save_plot (plot_name)
                
                #show plot
                plt.show()
                #close plot to free up space
                plt.close()    
        
        else:
            print('DataFrame is not loaded. Please run load_df first.')