import pandas as pd
import numpy as np
import os
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind, f_oneway

class ABHypothesis_testing:
    def __init__(self, df_path=None, output_folder=None, plot_folder=None):
        """
        Initialise the A/B hypothesis testing processor with the necessary parameters.
        
        Args:
            df_path (str): The path to the processed CSV DataFrame.
            output_folder (str): The path to the output folder
            plot_folder (str): The folder to save plots.
        """
        self.df = None  #initialise df to None
        self.df_path = df_path
        self.output_folder = output_folder if output_folder else os.path.join(os.getcwd(), 'data')
        self.plot_folder = plot_folder if plot_folder else os.path.join(os.getcwd(), '/plot images/hypothesis')
    
        #create plot folder if it does not exist                                                              
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)
        
        #load df if a DataFrame path is provided
        if self.df_path:
            self.load_processed_df()
    
    def load_processed_df(self):
        """
        Loads the processed DataFrame from the specified path.
        Assumes the file is in CSV format.
        """
        relative_processed_df_path = os.path.relpath(self.df_path, os.getcwd())
        
        if self.df_path and os.path.exists(self.df_path):
            try:
                self.df = pd.read_csv(self.df_path)
                print(f'DataFrame loaded successfully from {relative_processed_df_path}')
            except Exception as e:
                print(f'Error loading DataFrame from {relative_processed_df_path}: {e}')
                self.df = None 
        elif self.df_path:
            print(f'Error: File not found at {relative_processed_df_path}')
            self.df = None
        else:
            print('No DataFrame path provided during initialisation.')
            self.df = None
                
    def save_plot(self, plot_name):
        """
        Saves the current plot to the designated plot folder.

        Args:
            plot_name (str): The name of the plot file.
        """
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)

        plot_path = os.path.join(self.plot_folder, plot_name)
        relative_plot_path = os.path.relpath(plot_path, os.getcwd())

        try:
            plt.savefig(plot_path)
            print(f'\nPlot saved to {relative_plot_path}')
        except Exception as e:
            print(f'\nError saving plot: {e}')

    def derived_metrics(self):
        """
        Derive key performance metrics for hypothesis testing.

        - HasClaim: Flag set to 1 if TotalClaims > 0, else 0.
        - ClaimSeverity: TotalClaims value for policies with claims, else NaN.
        - Margin: Difference between TotalPremium and TotalClaims.
        """
        if self.df is not None:
            #create binary flag for claim occurrence
            self.df['HasClaim'] = (self.df['TotalClaims'] > 0).astype(int)

            #derive ClaimSeverity only for policies with claims
            self.df['ClaimSeverity'] = self.df.apply(
                lambda row: row['TotalClaims'] if row['HasClaim'] == 1 else np.nan, axis=1)
            
            self.df['Margin'] = self.df['TotalPremium'] - self.df['TotalClaims']
            print("\nDerived metrics: 'HasClaim', 'ClaimSeverity', and 'Margin' have been added to the DataFrame.")

            #save new df with new derived metrics
            ##create output folder if it doesn't exist
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)

            df_name = os.path.join(self.output_folder, 'processed_data_with_metrics.csv')

            ##calculate the relative path
            current_directory = os.getcwd()
            relative_path = os.path.relpath(df_name, current_directory)

            ##save processed data to CSV
            self.df.to_csv(df_name, index=False)
            print(f'\nNew DataFrame Saved to: {relative_path}')

            print('\nDataFrame Head:')
            out_head=self.df.head()
            display (out_head)        
        
        else:
            print('DataFrame not loaded yet.')
        
        return self.df
    
    def impute_process_save (self):
                
        #perform imputation on approporaite numerical columns                
        if self.df is not None:
            impute_cols = ['Cubiccapacity', 'Kilowatts', 'CapitalOutstanding', 
                            'SumInsured', 'CalculatedPremiumPerTerm', 
                            'TotalPremium', 'TotalClaims']
            for col in impute_cols:
                non_zero_values = self.df[self.df[col] != 0][col]
                if not non_zero_values.empty: #ensure non-zero values exist before computing median
                        median_non_zero_value = non_zero_values.median()
                        self.df[col] = self.df[col].replace(0, median_non_zero_value)
                else:
                    print(f'\nColumn {col} contains only zeros; skipping imputation.')

            #improve 'Gender' column based on 'Title' column
            if 'Gender' in self.df.columns and 'Title' in self.df.columns:
                self.df.loc[self.df['Title'] == 'Mr', 'Gender'] = 'Male'
                self.df.loc[self.df['Title'] == 'Mrs', 'Gender'] = 'Female'
                self.df.loc[self.df['Title'] == 'Ms', 'Gender'] = 'Female'
                self.df.loc[self.df['Title'] == 'Miss', 'Gender'] = 'Female'
                self.df.drop(self.df.loc[self.df['Gender'] == 'Not Specified'].index, inplace=True)
            else:
                ('\n Caution: "Gender" column data not improved. Investigate further.')


            #save imputed and transformed df
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

        else:
            print('DataFrame is not loaded. Please run load_processed_df first.')

    
    def plot_distribution(self, column, hue=None, plot_type='box'):
        """
        Plots the distribution of a given column.

        Args:
            column (str): The column to plot.
            hue (str, optional): Column to use for colouring the plot. Defaults to None.
            plot_type (str, optional): Type of plot ('box' or 'bar'). Defaults to 'bar'.
            bins (int, optional): Number of bins for histogram. Defaults to 30.
        """
        if self.df is not None:
            plt.figure(figsize=(10, 5))
            if plot_type == 'bar':
                contingency = pd.crosstab(self.df[column], self.df['HasClaim'])
                contingency.plot(kind='bar', stacked=True, colormap='coolwarm')
                plt.title(f'Claim Frequency by {column}')
                save_title = f'Claim Frequency by {column}.png'
            elif plot_type == 'box':
                if hue:
                    sns.boxplot(data=self.df, x=hue, y=column)
                else:
                    None
                plt.title(f'Claim Severity by {hue}')
                save_title = f'Claim Severity by {hue}.png'
            else:
                print(f"Invalid plot_type: {plot_type}. Choose between 'bar', and 'box'.")
                plt.close()
                return

            plt.xlabel(column)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Number of Claims' if plot_type == "bar" else "Total Claims Amount")

            plt.grid()
            plt.tight_layout()
            self.save_plot(save_title)
            plt.show()
        else:
            print('DataFrame is not loaded. Cannot plot distribution.')    
    
    def risk_differences_across_provinces(self):
        """
        Tests the hypothesis:
            H₀: There are no risk differences across provinces.         
        This method performs:
        - A Chi-Square test for differences in claim frequency (HasClaim) across 'Province'.
        - An ANOVA test (using TotalClaims as proxy for claim severity) 
            among provinces for those with claims.
        """

        if self.df is not None:           
            print('Analysing risk differences across Provinces...')

            #chi-Square test for claim frequency by Province.
            contingency = pd.crosstab(self.df['Province'], self.df['HasClaim'])
            chi2, p_val_chi2, dof, expected = chi2_contingency(contingency)
            print('Province Claim Frequency - Chi-square test p-value:', p_val_chi2)

            #interpret p value
            alpha = 0.05
            if p_val_chi2 < alpha:
                print('Reject null hypothesis: Significant risk differences in claim frequency across provinces.\n')
            else:
                print('Fail to reject null hypothesis: No significant risk differences in claim frequency across provinces.\n')

            
            #ANOVA test for claim severity by Province.
            df_claims = self.df[self.df['HasClaim'] == 1]
            groups = [group['TotalClaims'].values for name, group in df_claims.groupby('Province')]
            if len(groups) > 1:
                anova_result = f_oneway(*groups)
                p_val_anova = anova_result.pvalue
                print('Province Claim Severity - ANOVA test p-value:', anova_result.pvalue)

                #interpret p value
                if p_val_anova < alpha:
                    print('Reject null hypothesis: Significant risk differences in claim severity across provinces.\n')
                else:
                    print('Fail to reject null hypothesis: No significant risk differences in claim severity across provinces.\n')

            else:
                print('Not enough groups for ANOVA test on Province Claim Severity.')
            
            #plot
            self.plot_distribution(column='Province', plot_type='bar')
            self.plot_distribution('TotalClaims', hue='Province', plot_type='box')

        else:    
            print('DataFrame is not loaded. Please run load_processed_df first.')
    
    def risk_differences_across_postalcodes(self):
        """
        Tests the hypothesis:
            H₀: There are no risk differences between postal codes.
            
        This method performs:
        - A Chi-Square test for differences in claim frequency across 'PostalCode'.
        - An ANOVA test for claim severity differences among postal codes.
        """
        if self.df is not None:
        
            #chi-Square test for claim frequency by PostalCode.
            contingency = pd.crosstab(self.df['PostalCode'], self.df['HasClaim'])
            chi2, p_val_chi2, dof, expected = chi2_contingency(contingency)
            print('PostalCode Claim Frequency - Chi-square test p-value:', p_val_chi2)

            #interpret p value
            alpha = 0.05
            if p_val_chi2 < alpha:
                print('Reject null hypothesis: Significant risk differences in claim frequency across postal codes.\n')
            else:
                print('Fail to reject null hypothesis: No significant risk differences in claim frequency across postal codes.\n')

            
            #ANOVA test for claim severity by PostalCode.
            df_claims = self.df[self.df['HasClaim'] == 1]
            groups = [group['TotalClaims'].values for name, group in df_claims.groupby('PostalCode')]
            if len(groups) > 1:
                anova_result = f_oneway(*groups)
                p_val_anova = anova_result.pvalue
                print("PostalCode Claim Severity - ANOVA test p-value:", p_val_anova)

                #interpret p value
                alpha = 0.05
                if p_val_anova < alpha:
                    print('Reject null hypothesis: Significant risk differences in claim severity across postal codes.\n')
                else:
                    print('Fail to reject null hypothesis: No significant risk differences in claim severity across postal codes.\n')

            else:
                print('Not enough groups for ANOVA test on PostalCode Claim Severity.')
            
            #plot
            #self.plot_distribution(column='PostalCode', plot_type='bar')
            #self.plot_distribution('TotalClaims', hue='PostalCode', plot_type='box')
        
        else:    
            print('DataFrame is not loaded. Please run load_processed_df first.')
        
    def margin_differences_across_postalcodes(self):
        """
        Tests the hypothesis:
            H₀: There are no significant margin (profit) differences between postal codes.
            
        This method performs:
        - An ANOVA test for differences in margin (TotalPremium - TotalClaims) across 'PostalCode'.
        """
        if self.df is not None:
        
            groups = []  #initialise an empty list to store the results
            for name, group in self.df.groupby('PostalCode'):
                margin_values = group['Margin'].values  #get the 'Margin' values 
                groups.append(margin_values)  
                
            if len(groups) > 1:
                anova_margin = f_oneway(*groups)
                p_val_anova = anova_margin.pvalue
                print('PostalCode Margin - ANOVA test p-value:', p_val_anova)

                #interpret p value
                alpha = 0.05
                if p_val_anova < alpha:
                    print('Reject null hypothesis: Significant margin difference across postal codes.\n')
                else:
                    print('Fail to reject null hypothesis: Significant margin difference across postal codes.\n')
                
            else:
                print('Not enough groups for ANOVA test on PostalCode Margin.')
        
        else:    
            print('DataFrame is not loaded. Please run load_processed_df first.')

    def risk_differences_between_genders(self):
        """
        Tests the hypothesis:
            H₀: There are no significant risk differences between Women and Men.
            
        This method performs:
        - A Chi-Square test for differences in claim frequency (HasClaim) by 'Gender'.
        - A T-test for differences in claim severity (TotalClaims for policies with claims) by 'Gender'.
        """

        if self.df is not None:
            print('Analysing risk differences between Men and Women ...\n')

            #chi-Square test for claim frequency differences by Gender.
            contingency = pd.crosstab(self.df['Gender'], self.df['HasClaim'])
            chi2, p_val_chi2, dof, expected = chi2_contingency(contingency)
            print('Gender Claim Frequency - Chi-square test p-value:', p_val_chi2)

            #interpret p value
            alpha = 0.05
            if p_val_chi2 < alpha:
                print('Reject null hypothesis: Significant risk differences in claim frequency between men and women.\n')
            else:
                print('Fail to reject null hypothesis: No significant risk differences in claim frequency between men and women.\n')
            
            #t-test for claim severity differences by Gender.
            df_claims = self.df[self.df['HasClaim'] == 1]
            severity_male = df_claims[df_claims['Gender'] == 'Male']['TotalClaims']
            severity_female = df_claims[df_claims['Gender'] == 'Female']['TotalClaims']
            if len(severity_male) > 0 and len(severity_female) > 0:
                ttest_result = ttest_ind(severity_male, severity_female, equal_var=False)
                p_val_ttest = ttest_result.pvalue
                print('Gender Claim Severity - T-test p-value:', p_val_ttest)

                #interpret p value
                alpha = 0.05
                if p_val_ttest < alpha:
                    print('Reject null hypothesis: Significant risk differences in claim severity between men and women.\n')
                else:
                    print('Fail to reject null hypothesis: No significant risk differences in claim severity between men and women.\n')

            else:
                print('Not enough data for T-test on Gender Claim Severity.')

            #plot
            self.plot_distribution(column='Gender', plot_type='bar')
            self.plot_distribution('TotalClaims', hue='Gender', plot_type='box')

        else:    
            print('DataFrame is not loaded. Please run load_processed_df first.')
