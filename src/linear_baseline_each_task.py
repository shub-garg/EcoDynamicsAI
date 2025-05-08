import io
from pprint import pprint
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer, MinMaxScaler
from sklearn.compose import ColumnTransformer
from tabulate import tabulate

# If you need python-pptx, you can install it in your local environment
# !pip install python-pptx

#############################
# Step 1: Load the Datasets #
#############################

# Replace 'ecological_data.xlsx' with the path to your ecological dataset
df_ecological = pd.read_excel('Ecological_characterization_Hanna_Markle.xlsx')
print("\n--- First Few Rows of the Ecological Dataset ---\n")
print(df_ecological.head())

# Exploratory Data Analysis function
def explore_data(data):
    """Perform exploratory data analysis on the dataset."""
    print("\n--- Dataset Overview ---\n")
    print(data.info())
    print("\n--- Summary Statistics ---\n")
    print(data.describe())

# Perform EDA
explore_data(df_ecological)

# Read the OTUs associated with pathogens
df_gyrb = pd.read_excel('gyrb_fragment_dataset1.xlsx')  # Replace with correct filename/path
print("\n--- First Few Rows of the Pathogen-Associated OTUs ---\n")
print(df_gyrb.head())

# Read the relative abundance info
df_otu_abundance = pd.read_excel('Leaf_and_Root_OTU_Relative_Abundance.xlsx')  # Replace with correct filename/path
print("\n--- First Few Rows of the OTU Abundance Data ---\n")
print(df_otu_abundance.head())

#################################
# Additional Data Processing    #
#################################
# You can now continue with any preprocessing, merging, modeling, etc.

# Example: Print shapes to verify successful loading
print("\nDataset shapes:")
print("Ecological data shape:", df_ecological.shape)
# print("Pathogen OTUs data shape:", df_gyrb.shape)
print("OTU Abundance data shape:", df_otu_abundance.shape)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer, MinMaxScaler
from sklearn.compose import ColumnTransformer
from tabulate import tabulate
# import shap  # Ensure SHAP is installed (pip install shap)
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

class EcodynamicsAI:
    def __init__(self, data_path=""):
        """Initialize the framework with the dataset path."""
        self.data_path = data_path
        self.data = None
        self.target = None
        self.features = None
        self.otu_data = None
    def remove_low_importance_features(self, X, importance_df, threshold=0.01):
        """Remove features with low importance scores."""
        low_importance_features = importance_df[importance_df['Importance'] < threshold]['Feature'].tolist()
        print(f"Removing {len(low_importance_features)} features with importance < {threshold}.")
        return X.drop(columns=low_importance_features, errors='ignore')

    def remove_dominant_features(self, X, importance_df, dominance_threshold=0.1):
        """Remove features that dominate with very high importance scores."""
        dominant_features = importance_df[importance_df['Importance'] > dominance_threshold]['Feature'].tolist()
        print(f"Removing {len(dominant_features)} dominant features with importance > {dominance_threshold}.")
        return X.drop(columns=dominant_features, errors='ignore')

    def remove_top_k_dominant_features(self, X, importance_df, k=1):
        """Remove the top k dominant features with the highest importance."""
        top_features = importance_df.head(k)['Feature'].tolist()
        print(f"Removing top {k} dominant features: {top_features}")
        return X.drop(columns=top_features, errors='ignore')

    def select_top_k_features(self, X, importance_df, k=20):
        """Select the top k most important features and return the filtered dataset."""
        top_k_features = importance_df.head(k)['Feature'].tolist()
        print(f"Selecting top {k} features: {top_k_features}")
        return X[top_k_features]
    def load_data(self):
        """Load and preview the dataset."""
        self.data = df_ecological
        if self.data is None:
            raise ValueError("Dataset not loaded. Please check the data path.")
        if self.data.index.name != 'population' and 'population' in self.data.columns:
            self.data.set_index('population', inplace=True)
        # Replace '.' with NaN (the FutureWarning can be silenced by setting an option if desired)
        self.data = self.data.replace('.', np.nan)
        print(f"Dataset Loaded Successfully. {self.data.shape}")

    def load_otu_data(self):
        """Load, preprocess, and CLR-transform OTU relative abundance data."""
        self.otu_data = df_otu_abundance
        if self.otu_data is None:
            raise ValueError("Dataset otu data not loaded. Please check the data path.")
        if self.otu_data.index.name != 'population' and 'population' in self.otu_data.columns:
            self.otu_data.set_index('population', inplace=True)
        self.otu_data = self.otu_data.replace('.', np.nan)
        print(f"OTU Data Loaded Successfully., {self.otu_data.shape}")

        leaf_otus = [col for col in self.otu_data.columns if col.startswith('leaf_Otu')]
        root_otus = [col for col in self.otu_data.columns if col.startswith('root_Otu')]
        print(f"Number of leaf_OTUs in OTU data: {len(leaf_otus)}")
        print(f"Number of root_OTUs in OTU data: {len(root_otus)}")

        def clr_transformation(df, pseudocount=1e-6):
            """Perform CLR transformation on a DataFrame."""
            if (df <= 0).any().any():
                print(f"Warning: Some values are <= 0. Adding pseudocount {pseudocount}.")
                df = df + pseudocount  # Avoid log(0) issues
            geometric_mean = df.apply(lambda row: np.exp(np.log(row).mean()), axis=1)
            clr_df = df.apply(lambda row: np.log(row / geometric_mean[row.name]), axis=1)
            return clr_df

        self.otu_data[leaf_otus] = clr_transformation(self.otu_data[leaf_otus])
        self.otu_data[root_otus] = clr_transformation(self.otu_data[root_otus])

    def categorize_features(self):
        """Categorize features into predefined groups."""
        self.feature_categories = {
            'environmental': [
                'latitude', 'longitude', 'Elevation', 'MAT', 'MCMT', 'PPT_wt', 'PPT_sp', 'PPT_sm', 'PPT_at',
                'Nitrogen', 'CN', 'pH', 'Phosphore', 'Calcium', 'Magnesium', 'Sodium', 'Potassium', 'Iron',
                'Aluminium', 'WHC', 'OC', 'SOM', 'Manganese'
            ],
            'leaf_otus': [col for col in self.data.columns if col.startswith('leaf_Otu')],
            'root_otus': [col for col in self.data.columns if col.startswith('root_Otu')],
            'plant_otus': [col for col in self.data.columns if col.startswith('plant_OTU')],
            'specific_metrics': [
                'richness_microbiota_leaf', 'Shannon_microbiota_leaf',
                'richness_microbiota_root', 'Shannon_microbiota_root',
                'richness_pathobiota_leaf', 'Shannon_pathobiota_leaf',
                'richness_pathobiota_root', 'Shannon_pathobiota_root'
            ],
            'plant_metrics': [
                'plant_richness', 'plant_Shannon'
            ]
        }
        for category, features in self.feature_categories.items():
            print(f"Category '{category}' has {len(features)} features.")
        total_features = sum(len(features) for features in self.feature_categories.values())
        print(f"Total features: {total_features}")
        print(f"Number of data points: {len(self.data)}")

    def filter_columns_of_interest(self):
        """
        Retain only columns of interest.
        """
        columns_of_interest = (
            self.feature_categories['environmental'] +
            self.feature_categories['leaf_otus'] +
            self.feature_categories['root_otus'] +
            self.feature_categories['specific_metrics'] +
            self.feature_categories['plant_otus'] +
            self.feature_categories['plant_metrics']
        )
        retained_columns = [col for col in columns_of_interest if col in self.data.columns]
        discarded_columns = [col for col in self.data.columns if col not in retained_columns]
        print(f"Retaining {len(retained_columns)} columns.")
        print(f"Discarding {len(discarded_columns)} columns.")
        print("\n--- Discarded Columns ---")
        print(discarded_columns)
        self.data = self.data[retained_columns]

    def validate_and_assign_otu_data(self):
        """Align populations and assign OTU data."""
        if self.otu_data is not None:
            common_populations = self.data.index.intersection(self.otu_data.index)
            if len(common_populations) == 0:
                raise ValueError("No matching populations found between data and OTU data.")
            self.data = self.data.loc[common_populations]
            self.otu_data = self.otu_data.loc[common_populations]
            print(f"Data dimensions after aligning populations: {self.data.shape}")
            leaf_otus = [col for col in self.otu_data.columns if col.startswith('leaf_Otu')]
            root_otus = [col for col in self.otu_data.columns if col.startswith('root_Otu')]
            for col in leaf_otus + root_otus:
                if col in self.otu_data.columns:
                    self.data[col] = self.otu_data[col]
            print(f"Data dimensions after assigning OTU data: {self.data.shape}")

    def select_target_and_features(self, target_column, feature_categories=None):
        """Select target and features."""
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")
        target = target_column
        if feature_categories:
            selected_features = []
            for category in feature_categories:
                if category in self.feature_categories:
                    selected_features.extend(self.feature_categories[category])
                else:
                    raise ValueError(f"Feature category '{category}' not defined.")
            features = selected_features
        else:
            features = [col for col in self.data.columns if col != target]
        print(f"Target column set to: {target}")
        print(f"Feature columns selected: {features}")
        return target, features

    def impute_or_drop(self, X, y, strategy='impute'):
        """Handle missing values."""
        valid_rows = ~y.isna()
        X = X[valid_rows]
        y = y[valid_rows]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_test = self.scale_data(X_train, X_test)
        if strategy == 'impute':
            imputer = KNNImputer(n_neighbors=5)
            X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
            X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
            print(f"Training data shape after imputation: {X_train_imputed.shape}")
            print(f"Test data shape after imputation: {X_test_imputed.shape}")
            return X_train_imputed, X_test_imputed, y_train, y_test
        elif strategy == 'drop':
            full_data = pd.concat([X, y], axis=1).dropna()
            X_cleaned = full_data.iloc[:, :-1]
            y_cleaned = full_data.iloc[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)
            print(f"Training data shape after dropping missing values: {X_train.shape}")
            print(f"Test data shape after dropping missing values: {X_test.shape}")
            return X_train, X_test, y_train, y_test
        else:
            raise ValueError("Invalid strategy. Use 'impute' or 'drop'.")

    def scale_data(self, X_train, X_test):
        """Scale data using StandardScaler."""
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        print(f"Training data scaled shape: {X_train_scaled.shape}")
        print(f"Test data scaled shape: {X_test_scaled.shape}")
        return X_train_scaled, X_test_scaled

    def add_polynomial_features(self, X, degree=2, interaction_only=False, include_bias=False):
        """Generate polynomial features."""
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
        X_poly = poly.fit_transform(X)
        poly_feature_names = poly.get_feature_names_out(X.columns)
        return pd.DataFrame(X_poly, columns=poly_feature_names, index=X.index)

    def optimize_xgb_params(self, X, y):
        """Optimize XGBoost parameters using GridSearchCV."""
        from sklearn.model_selection import GridSearchCV
        import xgboost as xgb
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
        param_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.05, 0.1],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'reg_alpha': [0.1, 0.5, 1.0],
            'reg_lambda': [0.5, 1.0, 2.0]
        }
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                                   scoring='neg_root_mean_squared_error',
                                   cv=3, verbose=1)
        grid_search.fit(X, y)
        print("Best parameters found: ", grid_search.best_params_)
        print("Best RMSE: ", -grid_search.best_score_)
        return grid_search.best_estimator_


    def analyze_feature_importance(self, model, features):
        """Analyze feature importance and identify dominant features."""
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        print("\n--- Feature Importances ---\n")
        print(importance_df)

        # Plot the top 10 most important features
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df.head(20), x='Importance', y='Feature')
        plt.title('Top 20 Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.show()

        return importance_df

    def assess_rmse_quality(self, desc, y_train, y_test, y_train_pred, y_test_pred):
        """Assess the quality of RMSE for training and test sets."""
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        std_test = np.std(y_test)
        print(f"\n--- RMSE Assessment {desc} ---")
        print(f"RMSE (Train): {rmse_train}")
        print(f"RMSE (Test): {rmse_test}")
        print(f"Standard Deviation of Test Target: {std_test}")
        if rmse_test >= std_test:
            print("Test RMSE ≥ SD: Model performs no better than predicting the mean (R² ≤ 0).")
        elif rmse_test <= 0.25 * std_test:
            print("RMSE ≤ 0.25×SD: Exceptional performance (R² ≥ ~0.9).")
        elif rmse_test <= 0.5 * std_test:
            print("RMSE ≤ 0.5×SD: Strong performance (R² ~≥ 0.7).")
        else:
            print("RMSE < SD: Model is useful but has room for improvement.")
        if rmse_test <= 1.15 * rmse_train:
            print("Test RMSE ≤ 1.15× Training RMSE: Model generalizes well.")
        else:
            print("Test RMSE > 1.15× Training RMSE: Model might be overfitting.")
        if abs(rmse_train - rmse_test) / rmse_train < 0.1:
            print("Test RMSE ≈ Training RMSE: Model is underfitting or data has low noise.")
        if not hasattr(self, 'rmse_results'):
            self.rmse_results = {}
        self.rmse_results[desc] = {
            'RMSE Train': rmse_train,
            'RMSE Test': rmse_test,
            'Test Target Std': std_test
        }
        return {
            'RMSE Train': rmse_train,
            'RMSE Test': rmse_test,
            'Test Target Std': std_test
        }

    def train_linear_regression(self, tasks):
        """Train a basic Linear Regression model for each task without hyperparameter optimization."""
        self.results_linear = {}
        for task_name, task in tasks.items():
            print(f"\n--- Training Linear Regression for {task_name} ---")
            target = task['target']
            features = task['features']
            missing_features = [f for f in features if f not in self.data.columns]
            if target not in self.data.columns or missing_features:
                print(f"Skipping {task_name}: Missing features or target. Missing: {missing_features}")
                continue

            X = self.data[features]
            y = self.data[target]
            # Drop rows with missing values
            mask = y.notna()
            X = X.loc[mask]
            y = y.loc[mask]
            X_train, X_test, y_train, y_test = self.impute_or_drop(X, y, strategy='impute')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
            rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
            std_test = np.std(y_test)
            correlation, _ = pearsonr(y_test, y_test_pred)

            print(f"RMSE (Train): {rmse_train:.3f}")
            print(f"RMSE (Test): {rmse_test:.3f}")

            self.results_linear[task_name] = {
                'RMSE (Train)': rmse_train,
                'RMSE (Test)': rmse_test,
                'Test Target Std': std_test,
                'Correlation': correlation
            }
        return self.results_linear

    def display_linear_results(self):
        """Display Linear Regression results in a table."""
        headers = ["Task", "LR Train RMSE", "LR Test RMSE", "LR Corr", "Test Target Std"]
        table_data = []
        for task_name, metrics in self.results_linear.items():
            table_data.append([
                task_name,
                f"{metrics['Test Target Std']:.3f}",
                f"{metrics['RMSE (Train)']:.3f}",
                f"{metrics['RMSE (Test)']:.3f}",
                f"{metrics['Correlation']:.3f}"
            ])
        print("\nLinear Regression Results:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))



    def train_multiple_models_linear(self, tasks):
        """
        Train a plain LinearRegression model for each task,
        reporting test RMSE, Pearson’s r, and test-target standard deviation.
        """
        self.results_linear = {}
        for task_name, task in tasks.items():
            print(f"\n--- Training Linear Regression for {task_name} ---")
            target = task['target']
            features = task['features']

            # Check that features/target exist
            missing = [f for f in features + [target] if f not in self.data.columns]
            if missing:
                print(f"Skipping {task_name}: missing columns {missing}")
                continue

            # Prepare data
            X = self.data[features]
            y = self.data[target]

            # impute/drop + scale as before
            X_train, X_test, y_train, y_test = self.impute_or_drop(X, y, strategy='impute')

            # fit linear regression
            lr = LinearRegression()
            lr.fit(X_train, y_train)

            # predict on test set
            y_pred = lr.predict(X_test)

            # compute metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            std_test = np.std(y_test)
            try:
                corr, _ = pearsonr(y_test, y_pred)
            except Exception:
                corr = np.nan

            print(f"Linear RMSE (Test): {rmse:.3f}")
            print(f"Test Target Std: {std_test:.3f}")
            print(f"Pearson r: {corr:.3f}")

            # store results
            self.results_linear[task_name] = {
                'Linear RMSE (Test)': rmse,
                'Test Target Std': std_test,
                'Linear Corr': corr
            }

        return self.results_linear


# -------------------------------
# USAGE EXAMPLE:
# -------------------------------

# Create the framework object and load data
framework = EcodynamicsAI()
framework.load_data()
framework.load_otu_data()
framework.categorize_features()
framework.filter_columns_of_interest()
framework.validate_and_assign_otu_data()

# Define tasks (as before)
tasks = {
    "Microbiota Richness (Root) with Plant OTUs": {
        "target": "richness_microbiota_root",
        "features": framework.feature_categories['environmental'] + framework.feature_categories['plant_otus']
    },
    "Microbiota Richness (Leaf) with Plant OTUs": {
        "target": "richness_microbiota_leaf",
        "features": framework.feature_categories['environmental'] + framework.feature_categories['plant_otus']
    },
    "Microbiota Richness (Root) with Plant Metrics": {
        "target": "richness_microbiota_root",
        "features": framework.feature_categories['environmental'] + framework.feature_categories['plant_metrics']
    },
    "Microbiota Richness (Leaf) with Plant Metrics": {
        "target": "richness_microbiota_leaf",
        "features": framework.feature_categories['environmental'] + framework.feature_categories['plant_metrics']
    },
    "Microbiota Shannon (Root) with Plant OTUs": {
        "target": "Shannon_microbiota_root",
        "features": framework.feature_categories['environmental'] + framework.feature_categories['plant_otus']
    },
    "Microbiota Shannon (Leaf) with Plant OTUs": {
        "target": "Shannon_microbiota_leaf",
        "features": framework.feature_categories['environmental'] + framework.feature_categories['plant_otus']
    },
    "Microbiota Shannon (Root) with Plant Metrics": {
        "target": "Shannon_microbiota_root",
        "features": framework.feature_categories['environmental'] + framework.feature_categories['plant_metrics']
    },
    "Microbiota Shannon (Leaf) with Plant Metrics": {
        "target": "Shannon_microbiota_leaf",
        "features": framework.feature_categories['environmental'] + framework.feature_categories['plant_metrics']
    },
    "Pathobiota Richness (Root) with Plant OTUs": {
        "target": "richness_pathobiota_root",
        "features": framework.feature_categories['environmental'] + framework.feature_categories['plant_otus']
    },
    "Pathobiota Richness (Leaf) with Plant OTUs": {
        "target": "richness_pathobiota_leaf",
        "features": framework.feature_categories['environmental'] + framework.feature_categories['plant_otus']
    },
    "Pathobiota Richness (Root) with Plant Metrics": {
        "target": "richness_pathobiota_root",
        "features": framework.feature_categories['environmental'] + framework.feature_categories['plant_metrics']
    },
    "Pathobiota Richness (Leaf) with Plant Metrics": {
        "target": "richness_pathobiota_leaf",
        "features": framework.feature_categories['environmental'] + framework.feature_categories['plant_metrics']
    },
    "Pathobiota Shannon (Root) with Plant OTUs": {
        "target": "Shannon_pathobiota_root",
        "features": framework.feature_categories['environmental'] + framework.feature_categories['plant_otus']
    },
    "Pathobiota Shannon (Leaf) with Plant OTUs": {
        "target": "Shannon_pathobiota_leaf",
        "features": framework.feature_categories['environmental'] + framework.feature_categories['plant_otus']
    },
    "Pathobiota Shannon (Root) with Plant Metrics": {
        "target": "Shannon_pathobiota_root",
        "features": framework.feature_categories['environmental'] + framework.feature_categories['plant_metrics']
    },
    "Pathobiota Shannon (Leaf) with Plant Metrics": {
        "target": "Shannon_pathobiota_leaf",
        "features": framework.feature_categories['environmental'] + framework.feature_categories['plant_metrics']
    }
}

linear_results = framework.train_multiple_models_linear(tasks)

print("\nPlain Linear Regression Results:")
for name, m in linear_results.items():
    print(f"{name}: RMSE={m['Linear RMSE (Test)']:.3f}, Std={m['Test Target Std']:.3f}, r={m['Linear Corr']:.3f}")
