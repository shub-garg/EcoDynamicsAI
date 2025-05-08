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
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import networkx as nx
from scipy.stats import pearsonr
from sklearn.compose import ColumnTransformer
from tabulate import tabulate
from sklearn.decomposition import PCA

# (Optional) Uncomment the following line if you want to use SHAP for further analysis:
# import shap

# NOTE: Ensure that df_ecological and df_otu_abundance are defined in your environment.

class EcodynamicsAI:
    def __init__(self, data_path=""):
        """Initialize the framework with the dataset path."""
        self.data_path = data_path
        self.data = None
        self.target = None
        self.features = None
        self.otu_data = None
        self.results = {}
        self.results_xgb = {}
        self.results_xgb_poly = {}
        self.rmse_results = {}
        self.results_stack = {}

    def remove_low_importance_features(self, X, importance_df, threshold=0.01):
        low_importance_features = importance_df[importance_df['Importance'] < threshold]['Feature'].tolist()
        print(f"Removing {len(low_importance_features)} features with importance < {threshold}.")
        return X.drop(columns=low_importance_features, errors='ignore')

    def remove_dominant_features(self, X, importance_df, dominance_threshold=0.1):
        dominant_features = importance_df[importance_df['Importance'] > dominance_threshold]['Feature'].tolist()
        print(f"Removing {len(dominant_features)} dominant features with importance > {dominance_threshold}.")
        return X.drop(columns=dominant_features, errors='ignore')

    def remove_top_k_dominant_features(self, X, importance_df, k=1):
        top_features = importance_df.head(k)['Feature'].tolist()
        print(f"Removing top {k} dominant features: {top_features}")
        return X.drop(columns=top_features, errors='ignore')

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

    def select_top_k_features(self, X, importance_df, k=20):
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
        # Replace '.' with NaN
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
            if (df <= 0).any().any():
                print(f"Warning: Some values are <= 0. Adding pseudocount {pseudocount}.")
                df = df + pseudocount
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
        """Retain only columns of interest."""
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
        valid_rows = ~y.isna()
        X = X[valid_rows]
        y = y[valid_rows]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Here we use robust scaling by specifying scaling_method='robust'
        X_train, X_test = self.scale_data(X_train, X_test, scaling_method='robust')
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

    def scale_data(self, X_train, X_test, scaling_method='standard'):
        if scaling_method == 'robust':
            scaler = RobustScaler()
            print("Using RobustScaler for scaling.")
        else:
            scaler = StandardScaler()
            print("Using StandardScaler for scaling.")
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        print(f"Training data scaled shape: {X_train_scaled.shape}")
        print(f"Test data scaled shape: {X_test_scaled.shape}")
        return X_train_scaled, X_test_scaled

    def add_polynomial_features(self, X, degree=2, interaction_only=False, include_bias=False):
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
        X_poly = poly.fit_transform(X)
        poly_feature_names = poly.get_feature_names_out(X.columns)
        return pd.DataFrame(X_poly, columns=poly_feature_names, index=X.index)

    def optimize_xgb_params(self, X, y):
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

    def train_xgb_with_early_stopping(self, X, y, early_stopping_rounds=20, optimize_params=False):
        import xgboost as xgb
        from sklearn.metrics import mean_squared_error
        mask = y.notna()
        X = X.loc[mask]
        y = y.loc[mask]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        dtrain = xgb.DMatrix(X_train_sub, label=y_train_sub)
        dval   = xgb.DMatrix(X_val, label=y_val)
        dtest  = xgb.DMatrix(X_test, label=y_test)
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 3,
            'learning_rate': 0.05,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 1.0,
            'reg_alpha': 0.5,
            'reg_lambda': 1.0,
            'seed': 42
        }
        if optimize_params:
            best_model = self.optimize_xgb_params(X_train, y_train)
            params.update(best_model.get_xgb_params())
            print("Using optimized parameters: ", params)
        num_rounds = 500
        watchlist = [(dtrain, 'train'), (dval, 'eval')]
        print("Starting training with early stopping...")
        model = xgb.train(params, dtrain, num_rounds, watchlist,
                          early_stopping_rounds=early_stopping_rounds, verbose_eval=10)
        y_pred_train = model.predict(xgb.DMatrix(X_train))
        y_pred_test  = model.predict(dtest)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test  = np.sqrt(mean_squared_error(y_test, y_pred_test))
        print("Training RMSE: {:.3f}".format(rmse_train))
        print("Test RMSE: {:.3f}".format(rmse_test))
        return model, X_test, y_test, rmse_train, rmse_test

    def train_xgb_poly_pca(self, X, y, early_stopping_rounds=20, poly_degree=2, n_components=50, optimize_params=False):
        import xgboost as xgb
        from sklearn.metrics import mean_squared_error
        # Identify environmental features (from our feature categories)
        env_features = [feat for feat in X.columns if feat in self.feature_categories['environmental']]
        non_env_features = [feat for feat in X.columns if feat not in env_features]
        if env_features:
            print(f"Applying polynomial transformation (degree={poly_degree}) to environmental features: {env_features}")
            X_env_poly = self.add_polynomial_features(X[env_features], degree=poly_degree)
            print(f"Environmental features expanded to {X_env_poly.shape[1]} dimensions.")
            print(f"Applying PCA to reduce to {n_components} components.")
            pca = PCA(n_components=n_components, random_state=42)
            X_env_reduced = pd.DataFrame(pca.fit_transform(X_env_poly),
                                         index=X.index,
                                         columns=[f'PC{i+1}' for i in range(n_components)])
            if non_env_features:
                X = pd.concat([X_env_reduced, X[non_env_features]], axis=1)
            else:
                X = X_env_reduced
            print(f"Combined feature matrix shape after poly & PCA: {X.shape}")
        else:
            print("No environmental features found; skipping polynomial expansion and PCA.")
        mask = y.notna()
        X = X.loc[mask]
        y = y.loc[mask]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        dtrain = xgb.DMatrix(X_train_sub, label=y_train_sub)
        dval   = xgb.DMatrix(X_val, label=y_val)
        dtest  = xgb.DMatrix(X_test, label=y_test)
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 3,
            'learning_rate': 0.05,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 1.0,
            'reg_alpha': 0.5,
            'reg_lambda': 1.0,
            'seed': 42
        }
        if optimize_params:
            best_model = self.optimize_xgb_params(X_train, y_train)
            params.update(best_model.get_xgb_params())
            print("Using optimized parameters: ", params)
        num_rounds = 500
        watchlist = [(dtrain, 'train'), (dval, 'eval')]
        print("Starting training with early stopping (poly + PCA pipeline)...")
        model = xgb.train(params, dtrain, num_rounds, watchlist,
                          early_stopping_rounds=early_stopping_rounds, verbose_eval=10)
        y_pred_train = model.predict(xgb.DMatrix(X_train))
        y_pred_test  = model.predict(dtest)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test  = np.sqrt(mean_squared_error(y_test, y_pred_test))
        print("Training RMSE (poly+PCA): {:.3f}".format(rmse_train))
        print("Test RMSE (poly+PCA): {:.3f}".format(rmse_test))
        return model, X_test, y_test, rmse_train, rmse_test

    def train_multiple_models(self, tasks, top_k=None):
        top_k = 20
        self.results = {}
        for task_name, task in tasks.items():
            print(f"\n--- Training Model for {task_name} ---")
            target = task['target']
            features = task['features']
            missing_features = [f for f in features if f not in self.data.columns]
            if target not in self.data.columns or missing_features:
                print(f"Skipping {task_name}: Missing features or target. Missing: {missing_features}")
                continue
            X = self.data[features]
            y = self.data[target]
            X_train, X_test, y_train, y_test = self.impute_or_drop(X, y)
            model = RandomForestRegressor(
                n_estimators=400,
                random_state=42,
                max_features='sqrt',
                oob_score=True,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=3
            )
            model.fit(X_train, y_train)
            importance_df = self.analyze_feature_importance(model, features)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
            rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
            correlation, _ = pearsonr(y_test, y_test_pred)
            std_test = np.std(y_test)
            self.assess_rmse_quality("Original " + task_name, y_train, y_test, y_train_pred, y_test_pred)
            if top_k is not None:
                X_train_top_k = self.select_top_k_features(X_train, importance_df, k=top_k)
                X_test_top_k = self.select_top_k_features(X_test, importance_df, k=top_k)
                top_k_model = RandomForestRegressor(
                    n_estimators=400,
                    random_state=42,
                    max_features='sqrt',
                    oob_score=True,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=3
                )
                top_k_model.fit(X_train_top_k, y_train)
                y_train_pred_top_k = top_k_model.predict(X_train_top_k)
                y_test_pred_top_k = top_k_model.predict(X_test_top_k)
                rmse_train_top_k = np.sqrt(mean_squared_error(y_train, y_train_pred_top_k))
                rmse_test_top_k = np.sqrt(mean_squared_error(y_test, y_test_pred_top_k))
                self.assess_rmse_quality("TOP-K " + task_name, y_train, y_test, y_train_pred_top_k, y_test_pred_top_k)
            else:
                rmse_train_top_k = rmse_test_top_k = None
            alphas = [1e-2, 1e-1, 1, 10, 100]
            linear_model = RidgeCV(alphas=alphas, store_cv_values=True)
            linear_model.fit(X_train, y_train)
            y_train_pred_linear = linear_model.predict(X_train)
            y_test_pred_linear = linear_model.predict(X_test)
            rmse_train_linear = np.sqrt(mean_squared_error(y_train, y_train_pred_linear))
            rmse_test_linear = np.sqrt(mean_squared_error(y_test, y_test_pred_linear))
            self.results[task_name] = {
                'RMSE (Train)': rmse_train,
                'RMSE (Test)': rmse_test,
                'Test Target Std': std_test,
                'Correlation': correlation,
                'Linear RMSE (Train)': rmse_train_linear,
                'Linear RMSE (Test)': rmse_test_linear,
                'Top-k RMSE (Train)': rmse_train_top_k,
                'Top-k RMSE (Test)': rmse_test_top_k
            }
        return self.results

    def train_multiple_models_xgb(self, tasks, early_stopping_rounds=20, optimize_params=False):
        import xgboost as xgb
        from sklearn.metrics import mean_squared_error
        self.results_xgb = {}
        for task_name, task in tasks.items():
            print(f"\n--- Training XGBoost for {task_name} ---")
            target = task['target']
            features = task['features']
            missing_features = [f for f in features if f not in self.data.columns]
            if target not in self.data.columns or missing_features:
                print(f"Skipping {task_name}: Missing features or target. Missing: {missing_features}")
                continue
            X = self.data[features]
            y = self.data[target]
            print(f"Training XGBoost for task: {task_name}")
            model, X_test_inner, y_test_inner, rmse_train, rmse_test = self.train_xgb_with_early_stopping(
                X, y, early_stopping_rounds=early_stopping_rounds, optimize_params=optimize_params
            )
            dtest_inner = xgb.DMatrix(X_test_inner)
            y_pred_inner = model.predict(dtest_inner)
            try:
                correlation, _ = pearsonr(y_test_inner, y_pred_inner)
            except Exception as e:
                correlation = np.nan
            std_test = np.std(y_test_inner)
            self.results_xgb[task_name] = {
                'RMSE (Train)': rmse_train,
                'RMSE (Test)': rmse_test,
                'Test Target Std': std_test,
                'Correlation': correlation
            }
        return self.results_xgb

    def train_multiple_models_xgb_poly_pca(self, tasks, early_stopping_rounds=20, poly_degree=2, n_components=50, optimize_params=False):
        import xgboost as xgb
        from sklearn.metrics import mean_squared_error
        self.results_xgb_poly = {}
        for task_name, task in tasks.items():
            print(f"\n--- Training XGBoost (poly+PCA) for {task_name} ---")
            target = task['target']
            features = task['features']
            missing_features = [f for f in features if f not in self.data.columns]
            if target not in self.data.columns or missing_features:
                print(f"Skipping {task_name}: Missing features or target. Missing: {missing_features}")
                continue
            X = self.data[features]
            y = self.data[target]
            model, X_test_inner, y_test_inner, rmse_train, rmse_test = self.train_xgb_poly_pca(
                X, y, early_stopping_rounds=early_stopping_rounds, poly_degree=poly_degree, n_components=n_components, optimize_params=optimize_params
            )
            import xgboost as xgb
            dtest_inner = xgb.DMatrix(X_test_inner)
            y_pred_inner = model.predict(dtest_inner)
            try:
                correlation, _ = pearsonr(y_test_inner, y_pred_inner)
            except Exception as e:
                correlation = np.nan
            std_test = np.std(y_test_inner)
            self.results_xgb_poly[task_name] = {
                'RMSE (Train)': rmse_train,
                'RMSE (Test)': rmse_test,
                'Test Target Std': std_test,
                'Correlation': correlation
            }
        return self.results_xgb_poly

    def train_stacking_model(self, X, y, cv=5):
        from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
        from xgboost import XGBRegressor
        from sklearn.linear_model import Ridge
        base_models = [
            RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
            ExtraTreesRegressor(n_estimators=200, max_depth=8, random_state=42),
            XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1)
        ]
        meta_learner = Ridge(alpha=1.0)
        stack = StackingCVRegressor(regressors=base_models,
                                    meta_regressor=meta_learner,
                                    cv=cv,
                                    use_features_in_secondary=True,
                                    random_state=42)
        stack.fit(X, y)
        return stack

    def train_multiple_models_stacking(self, tasks, cv=5):
        self.results_stack = {}
        for task_name, task in tasks.items():
            print(f"\n--- Training Stacking Ensemble for {task_name} ---")
            target = task['target']
            features = task['features']
            missing_features = [f for f in features if f not in self.data.columns]
            if target not in self.data.columns or missing_features:
                print(f"Skipping {task_name}: Missing features or target. Missing: {missing_features}")
                continue
            X = self.data[features]
            y = self.data[target]
            X_train, X_test, y_train, y_test = self.impute_or_drop(X, y)
            stack_model = self.train_stacking_model(X_train, y_train, cv=cv)
            y_train_pred = stack_model.predict(X_train)
            y_test_pred = stack_model.predict(X_test)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
            rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
            try:
                correlation, _ = pearsonr(y_test, y_test_pred)
            except Exception as e:
                correlation = np.nan
            std_test = np.std(y_test)
            print(f"Stacking Model for {task_name} - Training RMSE: {rmse_train:.3f}, Test RMSE: {rmse_test:.3f}, Test Std: {std_test:.3f}, Corr: {correlation:.3f}")
            self.results_stack[task_name] = {
                'RMSE (Train)': rmse_train,
                'RMSE (Test)': rmse_test,
                'Test Target Std': std_test,
                'Correlation': correlation
            }
        return self.results_stack

    def display_combined_results(self):
        combined = []
        headers = ["Task", "RF Train RMSE", "RF Test RMSE", "RF Corr",
                   "XGB Train RMSE", "XGB Test RMSE", "XGB Std", "XGB Corr",
                   "XGB+Poly+PCA Train RMSE", "XGB+Poly+PCA Test RMSE", "XGB+Poly+PCA Std", "XGB+Poly+PCA Corr",
                   "Stacking Train RMSE", "Stacking Test RMSE", "Stacking Corr"]
        tasks_all = set(self.results.keys()).union(self.results_xgb.keys()).union(self.results_xgb_poly.keys()).union(self.results_stack.keys())
        for task in tasks_all:
            rf = self.results.get(task, {})
            xgb_res = self.results_xgb.get(task, {})
            xgb_poly = self.results_xgb_poly.get(task, {})
            stack = self.results_stack.get(task, {})
            combined.append([
                task,
                f"{rf.get('RMSE (Train)', np.nan):.3f}",
                f"{rf.get('RMSE (Test)', np.nan):.3f}",
                f"{rf.get('Correlation', np.nan):.3f}",
                f"{xgb_res.get('RMSE (Train)', np.nan):.3f}",
                f"{xgb_res.get('RMSE (Test)', np.nan):.3f}",
                f"{xgb_res.get('Test Target Std', np.nan):.3f}",
                f"{xgb_res.get('Correlation', np.nan):.3f}",
                f"{xgb_poly.get('RMSE (Train)', np.nan):.3f}",
                f"{xgb_poly.get('RMSE (Test)', np.nan):.3f}",
                f"{xgb_poly.get('Test Target Std', np.nan):.3f}",
                f"{xgb_poly.get('Correlation', np.nan):.3f}",
                f"{stack.get('RMSE (Train)', np.nan):.3f}",
                f"{stack.get('RMSE (Test)', np.nan):.3f}",
                f"{stack.get('Correlation', np.nan):.3f}"
            ])
        print("\nCombined Comparison (RF vs. XGB vs. XGB with Poly+PCA vs. Stacking):")
        print(tabulate(combined, headers=headers, tablefmt="grid"))

    def print_results_table(self):
        table_data = []
        headers = ["Task", "RMSE (Train)", "RMSE (Test)", "Linear RMSE (Train)", "Linear RMSE (Test)",
                   "Test Target Std", "Top-k RMSE (Train)", "Top-k RMSE (Test)"]
        for task_name, metrics in self.results.items():
            table_data.append([
                task_name,
                f"{metrics['RMSE (Train)']:.3f}",
                f"{metrics['RMSE (Test)']:.3f}",
                f"{metrics.get('Linear RMSE (Train)', np.nan):.3f}",
                f"{metrics.get('Linear RMSE (Test)', np.nan):.3f}",
                f"{metrics['Test Target Std']:.3f}",
                f"{metrics.get('Top-k RMSE (Train)', np.nan):.3f}" if metrics.get('Top-k RMSE (Train)') is not None else "N/A",
                f"{metrics.get('Top-k RMSE (Test)', np.nan):.3f}" if metrics.get('Top-k RMSE (Test)') is not None else "N/A"
            ])
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    def train_multiple_models_xgb_with_top_features(self, tasks, top_k=20, early_stopping_rounds=20, optimize_params=False):
        import xgboost as xgb
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import train_test_split
        results = {}
        
        for task_name, task in tasks.items():
            print(f"\n--- Training XGBoost (with top features) for {task_name} ---")
            target = task['target']
            features = task['features']
            missing_features = [f for f in features if f not in self.data.columns]
            if target not in self.data.columns or missing_features:
                print(f"Skipping {task_name}: Missing features or target. Missing: {missing_features}")
                continue
            
            # Extract data and impute/scale
            X = self.data[features]
            y = self.data[target]
            X_train, X_test, y_train, y_test = self.impute_or_drop(X, y)
            
            # Further split training set for early stopping validation:
            X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            
            # Prepare DMatrices for full model
            dtrain = xgb.DMatrix(X_train_sub, label=y_train_sub)
            dval   = xgb.DMatrix(X_val, label=y_val)
            dtest  = xgb.DMatrix(X_test, label=y_test)
            
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 3,
                'learning_rate': 0.05,
                'min_child_weight': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 1.0,
                'reg_alpha': 0.5,
                'reg_lambda': 1.0,
                'seed': 42
            }
            if optimize_params:
                best_model = self.optimize_xgb_params(X_train, y_train)
                params.update(best_model.get_xgb_params())
                print("Using optimized parameters: ", params)
            
            num_rounds = 500
            watchlist = [(dtrain, 'train'), (dval, 'eval')]
            print("Training full XGBoost model on all features...")
            model_full = xgb.train(params, dtrain, num_rounds, watchlist,
                                   early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
            
            # Predict and compute RMSE for full model
            y_train_pred_full = model_full.predict(xgb.DMatrix(X_train))
            y_test_pred_full  = model_full.predict(dtest)
            rmse_train_full = np.sqrt(mean_squared_error(y_train, y_train_pred_full))
            rmse_test_full  = np.sqrt(mean_squared_error(y_test, y_test_pred_full))
            
            # --- Extract Feature Importance from XGBoost ---
            importance_dict = model_full.get_score(importance_type='gain')
            importance_df = pd.DataFrame({
                'Feature': list(importance_dict.keys()),
                'Importance': list(importance_dict.values())
            }).sort_values(by='Importance', ascending=False)
            print("\nXGBoost Feature Importances:")
            print(importance_df)
            
            # Select the top-k features (make sure these exist in the training data)
            top_features = importance_df['Feature'].head(top_k).tolist()
            print(f"\nSelected top {top_k} features: {top_features}")
            
            # Subset the (already imputed/scaled) training and test sets:
            X_train_top = X_train[top_features]
            X_test_top  = X_test[top_features]
            
            # For the top features model, split training data for early stopping:
            X_train_sub_top, X_val_top, y_train_sub_top, y_val_top = train_test_split(X_train_top, y_train, test_size=0.2, random_state=42)
            dtrain_top = xgb.DMatrix(X_train_sub_top, label=y_train_sub_top)
            dval_top   = xgb.DMatrix(X_val_top, label=y_val_top)
            dtest_top  = xgb.DMatrix(X_test_top, label=y_test)
            
            print("Training XGBoost model with top features...")
            model_top = xgb.train(params, dtrain_top, num_rounds, [(dtrain_top, 'train'), (dval_top, 'eval')],
                                  early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
            
            y_train_pred_top = model_top.predict(xgb.DMatrix(X_train_top))
            y_test_pred_top  = model_top.predict(dtest_top)
            rmse_train_top = np.sqrt(mean_squared_error(y_train, y_train_pred_top))
            rmse_test_top  = np.sqrt(mean_squared_error(y_test, y_test_pred_top))
            
            results[task_name] = {
                'All Features': {
                    'RMSE Train': rmse_train_full,
                    'RMSE Test': rmse_test_full
                },
                'Top-k Features': {
                    'RMSE Train': rmse_train_top,
                    'RMSE Test': rmse_test_top
                }
            }
            
            print(f"\nTask: {task_name}")
            print("Full Feature Model - RMSE Train: {:.3f}, RMSE Test: {:.3f}".format(rmse_train_full, rmse_test_full))
            print("Top-k Feature Model  - RMSE Train: {:.3f}, RMSE Test: {:.3f}".format(rmse_train_top, rmse_test_top))
            
        print("\n--- Combined XGBoost with Top Features Results ---")
        for task, metrics in results.items():
            print(f"Task: {task}")
            print("  All Features:", metrics['All Features'])
            print("  Top-k Features:", metrics['Top-k Features'])
        self.results_xgb_top = results
        return results

    def display_final_table(self):
        combined = []
        headers = ["Task", "RF Train RMSE", "RF Test RMSE", "RF Test Std", "XGB Top Train RMSE", "XGB Top Test RMSE", "XGB Top Test Std"]
        # Only include tasks that are present in both RF results and XGB Top Features results
        tasks_all = set(self.results.keys()).intersection(set(self.results_xgb_top.keys()))
        for task in tasks_all:
            rf = self.results.get(task, {})
            xgb_top = self.results_xgb_top.get(task, {})
            combined.append([
                task,
                f"{rf.get('RMSE (Train)', np.nan):.3f}",
                f"{rf.get('RMSE (Test)', np.nan):.3f}",
                f"{rf.get('Test Target Std', np.nan):.3f}",
                f"{xgb_top.get('RMSE (Train)', np.nan):.3f}",
                f"{xgb_top.get('RMSE (Test)', np.nan):.3f}",
                f"{xgb_top.get('Test Target Std', np.nan):.3f}"
            ])
        print("\nFinal Results (Random Forest vs. XGBoost Top Features):")
        print(tabulate(combined, headers=headers, tablefmt="grid"))




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

# Define tasks
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

rf_results = framework.train_multiple_models(tasks)
print(rf_results)
# Train using XGBoost with early stopping (optimized)
# xgb_results = framework.train_multiple_models_xgb(tasks, early_stopping_rounds=20, optimize_params=True)
# Train using XGBoost with poly+PCA (optimized)
xgb_poly_results = framework.train_multiple_models_xgb_poly_pca(tasks, early_stopping_rounds=20, poly_degree=2, n_components=50, optimize_params=True)
print(xgb_poly_results)
# Train stacked ensemble for all tasks
stack_results = framework.train_multiple_models_stacking(tasks, cv=5)
print(stack_results)
xgb_top_features_results = framework.train_multiple_models_xgb_with_top_features(tasks, top_k=20, early_stopping_rounds=20, optimize_params=True)

print(xgb_top_features_results)
# Display the combined comparison table including stacking
framework.display_final_table()

# (Optional) Print presentation table for RF results
framework.print_results_table()
