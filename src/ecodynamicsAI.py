# ecodynamics_ai.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import seaborn as sns
import xgboost as xgb

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

# Optional: from tabulate import tabulate for nice table printing

class EcodynamicsAI:
    """
    Main framework for the Ecodynamics AI pipeline:
    - Data loading and alignment
    - OTU data processing (CLR transformation)
    - Feature categorization
    - Model training (RF, XGBoost) with optional polynomial + PCA transformations
    - Feature importance analysis
    """
    def __init__(self):
        self.data = None
        self.otu_data = None
        self.feature_categories = {}
        self.results = {}
        self.results_xgb = {}
        self.results_xgb_poly = {}
        self.rmse_results = {}

    def load_data(self, ecological_path):
        """Load the ecological/environmental dataset from a local file."""
        print(f"Loading ecological data from: {ecological_path}")
        self.data = pd.read_excel(ecological_path)
        # If your dataset uses 'population' as an index:
        if self.data.index.name != 'population' and 'population' in self.data.columns:
            self.data.set_index('population', inplace=True)
        self.data = self.data.replace('.', np.nan)
        print(f"Dataset Loaded Successfully. Shape: {self.data.shape}")

    def load_otu_data(self, otu_path):
        """Load the OTU relative abundance data and perform CLR transformation."""
        print(f"Loading OTU data from: {otu_path}")
        self.otu_data = pd.read_excel(otu_path)
        # If your dataset uses 'population' as an index:
        if self.otu_data.index.name != 'population' and 'population' in self.otu_data.columns:
            self.otu_data.set_index('population', inplace=True)
        self.otu_data = self.otu_data.replace('.', np.nan)

        leaf_otus = [col for col in self.otu_data.columns if col.startswith('leaf_Otu')]
        root_otus = [col for col in self.otu_data.columns if col.startswith('root_Otu')]
        print(f"Number of leaf_OTUs in OTU data: {len(leaf_otus)}")
        print(f"Number of root_OTUs in OTU data: {len(root_otus)}")

        def clr_transformation(df, pseudocount=1e-6):
            if (df <= 0).any().any():
                print(f"Warning: Some values are <= 0. Adding pseudocount {pseudocount} to avoid log(0).")
                df = df + pseudocount
            geometric_mean = df.apply(lambda row: np.exp(np.log(row).mean()), axis=1)
            clr_df = df.apply(lambda row: np.log(row / geometric_mean[row.name]), axis=1)
            return clr_df

        # Apply CLR transformation separately to leaf and root OTUs
        self.otu_data[leaf_otus] = clr_transformation(self.otu_data[leaf_otus])
        self.otu_data[root_otus] = clr_transformation(self.otu_data[root_otus])
        print(f"OTU Data Loaded and CLR-transformed. Shape: {self.otu_data.shape}")

    def validate_and_assign_otu_data(self):
        """Align rows by population and update self.data with OTU columns."""
        if self.data is None:
            raise ValueError("Ecological dataset has not been loaded.")
        if self.otu_data is None:
            raise ValueError("OTU dataset has not been loaded.")

        common_populations = self.data.index.intersection(self.otu_data.index)
        if len(common_populations) == 0:
            raise ValueError("No matching populations found between data and OTU data.")

        self.data = self.data.loc[common_populations]
        self.otu_data = self.otu_data.loc[common_populations]
        print(f"Data shape after aligning populations: {self.data.shape}")

        leaf_otus = [col for col in self.otu_data.columns if col.startswith('leaf_Otu')]
        root_otus = [col for col in self.otu_data.columns if col.startswith('root_Otu')]

        for col in leaf_otus + root_otus:
            if col in self.otu_data.columns:
                self.data[col] = self.otu_data[col]
        print(f"Data shape after assigning OTU data: {self.data.shape}")

    def categorize_features(self):
        """Define feature category groups and store them in self.feature_categories."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Example categorization – adjust as needed
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

        # Just a quick summary
        for category, features in self.feature_categories.items():
            print(f"Category '{category}' has {len(features)} features.")

    def filter_columns_of_interest(self):
        """Retain only the columns from the defined feature categories."""
        if not self.feature_categories:
            raise ValueError("Feature categories not set. Call categorize_features() first.")

        columns_of_interest = (
            self.feature_categories['environmental'] +
            self.feature_categories['leaf_otus'] +
            self.feature_categories['root_otus'] +
            self.feature_categories['plant_otus'] +
            self.feature_categories['specific_metrics'] +
            self.feature_categories['plant_metrics']
        )
        # Retain only those that actually exist in self.data
        retained_columns = [col for col in columns_of_interest if col in self.data.columns]
        discarded_columns = [col for col in self.data.columns if col not in retained_columns]
        print(f"Retaining {len(retained_columns)} columns.")
        print(f"Discarding {len(discarded_columns)} columns:", discarded_columns)

        self.data = self.data[retained_columns]

    def impute_or_drop(self, X, y, strategy='impute'):
        """Handle missing values in features or target."""
        valid_rows = ~y.isna()
        X = X[valid_rows]
        y = y[valid_rows]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled, X_test_scaled = self.scale_data(X_train, X_test)

        if strategy == 'impute':
            imputer = KNNImputer(n_neighbors=5)
            X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_scaled), columns=X_train.columns)
            X_test_imputed = pd.DataFrame(imputer.transform(X_test_scaled), columns=X_test.columns)
            return X_train_imputed, X_test_imputed, y_train, y_test
        elif strategy == 'drop':
            full_data = pd.concat([X, y], axis=1).dropna()
            X_cleaned = full_data.iloc[:, :-1]
            y_cleaned = full_data.iloc[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        else:
            raise ValueError("Invalid strategy. Use 'impute' or 'drop'.")

    def scale_data(self, X_train, X_test):
        """Scale data using StandardScaler."""
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        return X_train_scaled, X_test_scaled

    def analyze_feature_importance(self, model, features):
        """Compute and optionally display feature importances."""
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)
        print("\n--- Feature Importances ---")
        print(importance_df.head(20).to_string(index=False))

        # Quick plot of top-20
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df.head(20), x='Importance', y='Feature')
        plt.title('Top 20 Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

        return importance_df

    def analyze_feature_signs(self, model, X, features):
        """
        Analyze feature directionality using SHAP.
        Computes the Pearson correlation between each feature's values and its SHAP values.
        Returns a DataFrame with the sign correlation for each feature.
        """
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        feature_signs = {}
        for i, feature in enumerate(features):
            # Correlation between raw feature and SHAP value
            corr = np.corrcoef(X[feature], shap_values[:, i])[0, 1]
            feature_signs[feature] = corr
        feature_signs_df = pd.DataFrame({
            'Feature': list(feature_signs.keys()),
            'Sign Correlation': list(feature_signs.values())
        }).sort_values(by='Sign Correlation', ascending=False)
        print("\n--- Feature Sign Analysis (SHAP Correlations) ---")
        print(feature_signs_df)
        return feature_signs_df

    def assess_rmse_quality(self, desc, y_train, y_test, y_train_pred, y_test_pred):
        """Utility for evaluating and printing RMSE comparisons."""
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        std_test = np.std(y_test)
        print(f"\n--- RMSE Assessment {desc} ---")
        print(f"RMSE (Train): {rmse_train}")
        print(f"RMSE (Test): {rmse_test}")
        print(f"Standard Deviation of Test Target: {std_test}")

        # Simple heuristics
        if rmse_test >= std_test:
            print("Test RMSE ≥ SD: Model performs ~ no better than the mean.")
        elif rmse_test <= 0.25 * std_test:
            print("RMSE ≤ 0.25×SD: Exceptional performance (R² ≥ ~0.9).")
        elif rmse_test <= 0.5 * std_test:
            print("RMSE ≤ 0.5×SD: Strong performance (R² ~≥ 0.7).")
        else:
            print("RMSE < SD: Model is useful but has room for improvement.")

        if rmse_test <= 1.15 * rmse_train:
            print("Test RMSE ≤ 1.15× Train RMSE: Model generalizes well.")
        else:
            print("Test RMSE > 1.15× Train RMSE: Potential overfitting.")

        if abs(rmse_train - rmse_test) / rmse_train < 0.1:
            print("Test RMSE ≈ Train RMSE: Model might be underfitting or data is low-noise.")

        self.rmse_results[desc] = {
            'RMSE Train': rmse_train,
            'RMSE Test': rmse_test,
            'Test Target Std': std_test
        }
        return self.rmse_results[desc]

    def train_multiple_models(self, tasks, top_k=20):
        """
        Trains a RandomForestRegressor for each task.
        tasks: dictionary of form:
            {
                "Task Name": {
                    "target": "target_column_name",
                    "features": [list_of_feature_names]
                },
                ...
            }
        """
        self.results = {}
        for task_name, task in tasks.items():
            print(f"\n=== Training RandomForest Model for {task_name} ===")
            target_col = task['target']
            feat_cols = task['features']

            missing_features = [f for f in feat_cols if f not in self.data.columns]
            if target_col not in self.data.columns or missing_features:
                print(f"Skipping {task_name}. Missing target or features: {missing_features}")
                continue

            X = self.data[feat_cols]
            y = self.data[target_col]

            # Impute or drop
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
            importance_df = self.analyze_feature_importance(model, feat_cols)
            _ = self.analyze_feature_signs(model, X_test, feat_cols)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            correlation, _ = pearsonr(y_test, y_test_pred) if len(y_test) > 1 else (np.nan, None)
            self.assess_rmse_quality(f"Original {task_name}", y_train, y_test, y_train_pred, y_test_pred)

            # Optionally: train again with only top_k features
            top_k_features = importance_df.head(top_k)['Feature'].tolist()
            X_train_top = X_train[top_k_features]
            X_test_top = X_test[top_k_features]

            top_k_model = RandomForestRegressor(
                n_estimators=400,
                random_state=42,
                max_features='sqrt',
                oob_score=True,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=3
            )
            top_k_model.fit(X_train_top, y_train)
            y_train_pred_top = top_k_model.predict(X_train_top)
            y_test_pred_top = top_k_model.predict(X_test_top)
            self.assess_rmse_quality(f"TOP-K {task_name}", y_train, y_test, y_train_pred_top, y_test_pred_top)

            # Also compare to a baseline linear model (RidgeCV)
            alphas = [1e-2, 1e-1, 1, 10, 100]
            linear_model = RidgeCV(alphas=alphas, store_cv_values=True)
            linear_model.fit(X_train, y_train)
            y_train_lin = linear_model.predict(X_train)
            y_test_lin = linear_model.predict(X_test)

            self.results[task_name] = {
                'RF Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'RF Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'Correlation': correlation,
                'Linear Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_lin)),
                'Linear Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_lin))
            }
        return self.results

    def optimize_xgb_params(self, X, y):
        """Optimize XGBoost parameters using GridSearchCV."""
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
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='neg_root_mean_squared_error',
            cv=3,
            verbose=1
        )
        grid_search.fit(X, y)
        print("Best parameters found:", grid_search.best_params_)
        print("Best RMSE:", -grid_search.best_score_)
        return grid_search.best_estimator_

    def train_xgb_with_early_stopping(self, X, y, early_stopping_rounds=20, optimize_params=False):
        """Train an XGBoost model with early stopping, optionally optimizing params."""
        # Clean out missing target
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
            # Optimize parameters on the full training set
            best_model = self.optimize_xgb_params(X_train, y_train)
            params.update(best_model.get_xgb_params())
            print("Using optimized parameters:", params)

        num_rounds = 500
        watchlist = [(dtrain, 'train'), (dval, 'eval')]
        print("Starting XGBoost training with early stopping...")
        model = xgb.train(
            params, dtrain, num_rounds,
            watchlist,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=10
        )

        y_pred_train = model.predict(xgb.DMatrix(X_train))
        y_pred_test  = model.predict(dtest)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test  = np.sqrt(mean_squared_error(y_test, y_pred_test))
        print(f"Training RMSE: {rmse_train:.3f}")
        print(f"Test RMSE: {rmse_test:.3f}")

        return model, X_test, y_test, rmse_train, rmse_test

    def train_multiple_models_xgb(self, tasks, early_stopping_rounds=20, optimize_params=False):
        """Train XGBoost models with early stopping for each task."""
        self.results_xgb = {}
        for task_name, task in tasks.items():
            print(f"\n=== Training XGBoost for task: {task_name} ===")
            target_col = task['target']
            feat_cols = task['features']

            missing_features = [f for f in feat_cols if f not in self.data.columns]
            if target_col not in self.data.columns or missing_features:
                print(f"Skipping {task_name}. Missing features or target. Missing: {missing_features}")
                continue

            X = self.data[feat_cols]
            y = self.data[target_col]

            model, X_test_inner, y_test_inner, rmse_train, rmse_test = self.train_xgb_with_early_stopping(
                X, y,
                early_stopping_rounds=early_stopping_rounds,
                optimize_params=optimize_params
            )
            y_pred_inner = model.predict(xgb.DMatrix(X_test_inner))
            correlation, _ = pearsonr(y_test_inner, y_pred_inner) if len(y_test_inner) > 1 else (np.nan, None)
            std_test = np.std(y_test_inner)

            self.results_xgb[task_name] = {
                'RMSE (Train)': rmse_train,
                'RMSE (Test)': rmse_test,
                'Test Target Std': std_test,
                'Correlation': correlation
            }
        return self.results_xgb

    def add_polynomial_features(self, X, degree=2, interaction_only=False, include_bias=False):
        """Generate polynomial features on a dataframe X."""
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
        X_poly = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out(X.columns)
        return pd.DataFrame(X_poly, columns=feature_names, index=X.index)

    def train_xgb_poly_pca(self, X, y, early_stopping_rounds=20, poly_degree=2, n_components=50, optimize_params=False):
        """
        1) Apply polynomial expansion to environmental features,
        2) Apply PCA,
        3) Train XGBoost with early stopping.
        """
        env_features = [feat for feat in X.columns if feat in self.feature_categories.get('environmental', [])]
        non_env_features = [feat for feat in X.columns if feat not in env_features]

        # Polynomial + PCA only on environmental features
        if env_features:
            print(f"Applying polynomial transformation (degree={poly_degree}) to environmental features.")
            X_env_poly = self.add_polynomial_features(X[env_features], degree=poly_degree)
            print(f"Environmental features expanded to {X_env_poly.shape[1]} dims.")

            pca = PCA(n_components=n_components, random_state=42)
            X_env_reduced = pd.DataFrame(
                pca.fit_transform(X_env_poly),
                index=X.index,
                columns=[f'PC{i+1}' for i in range(n_components)]
            )
            # Re-combine with the non-environmental features
            if non_env_features:
                X_transformed = pd.concat([X_env_reduced, X[non_env_features]], axis=1)
            else:
                X_transformed = X_env_reduced
            print(f"Feature matrix shape after poly+PCA: {X_transformed.shape}")
        else:
            print("No environmental features found; skipping polynomial+PCA step.")
            X_transformed = X

        # Clean out missing target
        mask = y.notna()
        X_transformed = X_transformed.loc[mask]
        y = y.loc[mask]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
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
            print("Using optimized parameters:", params)

        num_rounds = 500
        watchlist = [(dtrain, 'train'), (dval, 'eval')]
        print("Starting XGBoost training (poly+PCA) with early stopping...")
        model = xgb.train(
            params,
            dtrain,
            num_rounds,
            watchlist,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=10
        )

        y_pred_train = model.predict(xgb.DMatrix(X_train))
        y_pred_test  = model.predict(dtest)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test  = np.sqrt(mean_squared_error(y_test, y_pred_test))
        print(f"Training RMSE (poly+PCA): {rmse_train:.3f}")
        print(f"Test RMSE (poly+PCA): {rmse_test:.3f}")

        return model, X_test, y_test, rmse_train, rmse_test

    def train_multiple_models_xgb_poly_pca(self, tasks, early_stopping_rounds=20, poly_degree=2, n_components=50, optimize_params=False):
        """
        For each task, apply polynomial+PCA transformations on environmental features,
        then train XGBoost with early stopping.
        """
        self.results_xgb_poly = {}
        for task_name, task in tasks.items():
            print(f"\n=== XGBoost (poly+PCA) for task: {task_name} ===")
            target_col = task['target']
            feat_cols = task['features']

            missing_features = [f for f in feat_cols if f not in self.data.columns]
            if target_col not in self.data.columns or missing_features:
                print(f"Skipping {task_name}. Missing features or target. Missing: {missing_features}")
                continue

            X = self.data[feat_cols]
            y = self.data[target_col]

            model, X_test_inner, y_test_inner, rmse_train, rmse_test = self.train_xgb_poly_pca(
                X,
                y,
                early_stopping_rounds=early_stopping_rounds,
                poly_degree=poly_degree,
                n_components=n_components,
                optimize_params=optimize_params
            )
            dtest_inner = xgb.DMatrix(X_test_inner)
            y_pred_inner = model.predict(dtest_inner)
            correlation, _ = pearsonr(y_test_inner, y_pred_inner) if len(y_test_inner) > 1 else (np.nan, None)
            std_test = np.std(y_test_inner)

            self.results_xgb_poly[task_name] = {
                'RMSE (Train)': rmse_train,
                'RMSE (Test)': rmse_test,
                'Test Target Std': std_test,
                'Correlation': correlation
            }
        return self.results_xgb_poly

    def display_combined_results(self):
        """
        Display a table comparing:
         - RF results (self.results)
         - XGB results (self.results_xgb)
         - XGB (poly+PCA) results (self.results_xgb_poly)
        """
        # If you want to use tabulate:
        # from tabulate import tabulate
        headers = [
            "Task",
            "RF Train RMSE", "RF Test RMSE", "RF Corr",
            "XGB Train RMSE", "XGB Test RMSE", "XGB Std", "XGB Corr",
            "XGB+Poly Train RMSE", "XGB+Poly Test RMSE", "XGB+Poly Std", "XGB+Poly Corr"
        ]
        tasks_all = set(self.results.keys()).union(self.results_xgb.keys()).union(self.results_xgb_poly.keys())
        combined = []
        for task in tasks_all:
            rf = self.results.get(task, {})
            xg = self.results_xgb.get(task, {})
            xp = self.results_xgb_poly.get(task, {})
            combined.append([
                task,
                f"{rf.get('RF Train RMSE', np.nan):.3f}",
                f"{rf.get('RF Test RMSE', np.nan):.3f}",
                f"{rf.get('Correlation', np.nan):.3f}",
                f"{xg.get('RMSE (Train)', np.nan):.3f}",
                f"{xg.get('RMSE (Test)', np.nan):.3f}",
                f"{xg.get('Test Target Std', np.nan):.3f}",
                f"{xg.get('Correlation', np.nan):.3f}",
                f"{xp.get('RMSE (Train)', np.nan):.3f}",
                f"{xp.get('RMSE (Test)', np.nan):.3f}",
                f"{xp.get('Test Target Std', np.nan):.3f}",
                f"{xp.get('Correlation', np.nan):.3f}",
            ])

        # Print in a simple table format
        print("\n=== Combined Results (RF vs. XGB vs. XGB+Poly/PCA) ===")
        max_width = max(len(row[0]) for row in combined) + 2
        print(" | ".join(headers))
        print("-" * 100)
        for row in combined:
            print(" | ".join(str(x).rjust(max_width) for x in row))
