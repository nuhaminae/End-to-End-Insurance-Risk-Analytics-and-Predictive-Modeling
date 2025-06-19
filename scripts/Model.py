import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

import shap
import lime
import lime.lime_tabular

class RegressionModelEvaluator:
    """
    A class to build, train, and evaluate various regression models using a pipeline.
    Includes data loading, processing, outlier handling, and plotting predictions,
    as well as SHAP and LIME interpretation.
    """

    def __init__(self, file_path, target_column):
        """
        Initialises the RegressionModelEvaluator.

        Args:
            file_path (str): The path to the CSV data file.
            target_column (str): The name of the target column.
        """
        self.file_path = file_path
        self.target_column = target_column
        self.data = None
        self.feature_names = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.preprocessor = None

        self._load_and_process_data()
        self._define_preprocessor()

    def _load_and_process_data(self):
        """
        Loads data, performs filtering, feature engineering, and outlier removal.
        """
        #Load the dataset
        dafr = pd.read_csv(self.file_path)
        dafr = dafr.dropna().copy()

        #Filter for policies with a claim and valid gender
        dafr = dafr[(dafr['HasClaim'] == 1) & (dafr['Gender'] != 'Not specified')].copy()

        #Feature Engineering (+avoid division by zero)
        dafr['ClaimRatio'] = dafr['TotalClaims'] / (dafr['SumInsured'] + 1)
        dafr['PowerPerCapacity'] = dafr['Kilowatts'] / (dafr['Cubiccapacity'] + 1)

        df = dafr[['VehicleType', 'Province', 'Gender', 'PowerPerCapacity', 'Cubiccapacity',
                    'Make', 'Model', 'TrackingDevice', 'CapitalOutstanding', 'ClaimRatio', self.target_column, 'CalculatedPremiumPerTerm']].copy()

        #Function to remove outliers using Z-score (from your notebook)
        def remove_outliers(df, column):
            data = df[column]
            z_scores = np.abs((data - data.mean()) / data.std())
            threshold = 1
            df_out = df[(z_scores <= threshold)]
            return df_out

        columns_to_process = ['Cubiccapacity', 'PowerPerCapacity',
                            'CapitalOutstanding', 'ClaimRatio',
                            self.target_column, 'CalculatedPremiumPerTerm']

        df_out = df.copy()  #Start with a copy of the original dataframe

        for col in columns_to_process:
            if col in df_out.columns: #Ensure column exists before processing
                df_out = remove_outliers(df_out, col)

        self.data = df_out
        self.feature_names = [col for col in self.data.columns if col != self.target_column]
        self.X = self.data[self.feature_names]
        self.y = self.data[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        print("Data loaded and processed successfully.")
        print("\nTraining data shape:", self.X_train.shape)
        print("Test data shape:", self.X_test.shape)


    def _define_preprocessor(self):
        """
        Defines the preprocessing pipeline based on data types.
        """

        if self.X_train is None:
            print("Data not loaded or processed. Cannot define preprocessor.")
            return

        numeric_features = self.X_train.select_dtypes(include=np.number).columns.tolist()
        categorical_features = self.X_train.select_dtypes(include='object').columns.tolist()

        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown="ignore"))])

        self.preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    def add_model(self, model_name, model_instance):
        """
        Adds a regression model to the evaluator.

        Args:
            model_name (str): A name for the model (e.g., 'Linear Regression').
            model_instance: An instance of a scikit-learn compatible regression model.
        """

        if self.preprocessor is None:
            print("Preprocessor not defined. Cannot add model.")
            return

        self.models[model_name] = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', model_instance)
        ])

    def train_models(self):
        """
        Trains all added models.
        """

        if not self.models:
            print("No models added to train.")
            return

        for name, pipeline in self.models.items():
            print(f"\nTraining {name}...")
            pipeline.fit(self.X_train, self.y_train)
            print(f"{name} trained.")

    def evaluate_models(self):
        """
        Evaluates all trained models and returns their metrics.
        """

        if not self.models:
            print("No models added to evaluate.")
            return

        results = {}
        for name, pipeline in self.models.items():
            print(f"\n Evaluating {name}...")
            y_pred = pipeline.predict(self.X_test)
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            results[name] = {'MSE': mse, 'R-squared': r2}
            print(f"{name} - MSE: {mse:.4f}, R-squared: {r2:.4f}")
        return results

    def plot_shap_summary(self, model_name):
        """
        Generates a SHAP summary plot for a specified model.

        Args:
            model_name (str): The name of the trained model.
        """
        
        if model_name not in self.models:
            print(f"Model '{model_name}' not found.")
            return

        print(f"\n--- Generating SHAP Summary Plot for {model_name} ---")
        pipeline = self.models[model_name]
        regressor = pipeline.named_steps['regressor']
        preprocessor = pipeline.named_steps['preprocessor']

        try:
            #Get the processed test data
            X_test_processed = preprocessor.transform(self.X_test)

            #Handle different model types for SHAP
            if isinstance(regressor, (LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor)):
                #SHAP explainer for tree-based models and linear models
                #For LinearRegression, we need to handle the preprocessed data and the model's predict method
                if isinstance(regressor, LinearRegression):
                    explainer = shap.LinearExplainer(regressor, X_test_processed)
                else:
                    explainer = shap.TreeExplainer(regressor)

                shap_values = explainer.shap_values(X_test_processed)

                #Get feature names after preprocessing
                if isinstance(preprocessor, ColumnTransformer):
                    #Get feature names from the preprocessor
                    ohe_feature_names = preprocessor.transformers_[1][1].named_steps['encoder'].get_feature_names_out(self.X.select_dtypes(include='object').columns.tolist())
                    numeric_feature_names = self.X.select_dtypes(include=np.number).columns.tolist()
                    all_feature_names = list(numeric_feature_names) + list(ohe_feature_names)

                    #If shap_values is a list (for multi-output models, which is not the case here),
                    #take the first element.
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]

                    #Ensure the number of SHAP values matches the number of feature names
                    if shap_values.shape[1] != len(all_feature_names):
                        print("Warning: Number of SHAP values does not match the number of feature names after preprocessing.")
                        print(f"SHAP values columns: {shap_values.shape[1]}")
                        print(f"Feature names: {len(all_feature_names)}")
                        #Attempt to align based on known structures if possible, or skip plotting
                        print("Skipping SHAP plot due to feature mismatch.")
                        return

                    shap.summary_plot(shap_values, X_test_processed, feature_names=all_feature_names)
                else:
                    #If preprocessor is not ColumnTransformer (shouldn't happen with your current code)
                    shap.summary_plot(shap_values, X_test_processed)

            else:
                print(f"SHAP is not directly supported for {type(regressor).__name__}. Consider using KernelExplainer or explainers for specific libraries.")
        except Exception as e:
            print(f"Could not generate SHAP plot for {model_name}: {e}")


    def plot_lime_explanation(self, model_name, instance_index=0):
        """
        Generates a LIME explanation for a specific instance in the test set.

        Args:
            model_name (str): The name of the trained model.
            instance_index (int): The index of the instance in the test set to explain.
        """
        if model_name not in self.models:
            print(f"Model '{model_name}' not found.")
            return
        if instance_index >= len(self.X_test):
            print(f"Instance index {instance_index} is out of bounds for the test set (size {len(self.X_test)}).")
            return

        print(f"\n--- Generating LIME Explanation for {model_name} (Instance Index {instance_index}) ---")
        pipeline = self.models[model_name]
        regressor = pipeline.named_steps['regressor']
        preprocessor = pipeline.named_steps['preprocessor']

        try:
            #Create a LIME explainer
            #Need to provide training data (can be a subset) and feature names
            #Also need to handle categorical features correctly for LIME
            categorical_features_indices = [self.X_train.columns.get_loc(col) for col in self.X_train.select_dtypes(include='object').columns.tolist()]
            feature_names = self.X_train.columns.tolist()

            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.array(self.X_train),
                feature_names=feature_names,
                class_names=[self.target_column], #LIME expects class names, but we can use the target column name for regression
                mode='regression',
                categorical_features=categorical_features_indices,
                verbose=False
            )

            #Get the instance to explain
            instance = self.X_test.iloc[instance_index]

            #Get the explanation
            explanation = explainer.explain_instance(
                data_row=instance,
                predict_fn=pipeline.predict, #Use the pipeline's predict method
                num_features=5 #Number of features to show in the explanation
            )

            #Display the explanation
            explanation.show_in_notebook(show_table=True, show_all=False)

        except Exception as e:
            print(f"Could not generate LIME explanation for {model_name}: {e}")