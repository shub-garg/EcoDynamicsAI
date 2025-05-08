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


"""
Conditional Independence Testing using Hardcoded XGBoost Parameters
with Recursive Forward Feature Selection (Gain One Feature at a Time) that avoids data leakage by using a holdout set,
handles missing data plus robust scaling.
Also, a quadratic expansion (degree-2) of the top 15 most important features (from the full-set model) is added
to the candidate features before forward selection.
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from math import sqrt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from scipy.stats import ttest_rel
import xgboost as xgb

# ---------------------------------------------------------------------
# 1) TASKS DEFINITION
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 2) HELPER: Preprocess and Split (Impute/Drop + Scaling)
# ---------------------------------------------------------------------
def preprocess_data_for_task(full_df, features, target, strategy='impute'):
    valid_idx = full_df[target].notna()
    df_clean = full_df[valid_idx].copy()

    train_df, holdout_df = train_test_split(df_clean, test_size=0.2, random_state=42)
    train_df = train_df.reset_index(drop=True)
    holdout_df = holdout_df.reset_index(drop=True)

    X_train = train_df[features].copy()
    y_train = train_df[target].copy()

    X_holdout = holdout_df[features].copy()
    y_holdout = holdout_df[target].copy()

    if strategy == 'drop':
        train_valid = X_train.notna().all(axis=1)
        X_train = X_train[train_valid].copy()
        y_train = y_train[train_valid].copy()

        holdout_valid = X_holdout.notna().all(axis=1)
        X_holdout = X_holdout[holdout_valid].copy()
        y_holdout = y_holdout[holdout_valid].copy()
    elif strategy == 'impute':
        imputer = KNNImputer(n_neighbors=5)
        X_train_array = imputer.fit_transform(X_train)
        X_holdout_array = imputer.transform(X_holdout)
        X_train = pd.DataFrame(X_train_array, columns=X_train.columns, index=X_train.index)
        X_holdout = pd.DataFrame(X_holdout_array, columns=X_holdout.columns, index=X_holdout.index)
    else:
        raise ValueError("Invalid strategy. Must be 'impute' or 'drop'.")

    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_holdout_scaled = pd.DataFrame(
        scaler.transform(X_holdout),
        columns=X_holdout.columns,
        index=X_holdout.index
    )
    return X_train_scaled, y_train, X_holdout_scaled, y_holdout

# ---------------------------------------------------------------------
# 3) XGBoost Training with Early Stopping (Hardcoded Params)
# ---------------------------------------------------------------------
def xgb_train_early_stopping(X, y, params, early_stopping_rounds=20, num_rounds=500, random_state=42):
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns.tolist())
    dval   = xgb.DMatrix(X_val, label=y_val, feature_names=X_val.columns.tolist())
    
    watchlist = [(dtrain, 'train'), (dval, 'eval')]
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_rounds,
        evals=watchlist,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False
    )

    y_pred_train = model.predict(xgb.DMatrix(X_train, feature_names=X_train.columns.tolist()))
    y_pred_val   = model.predict(xgb.DMatrix(X_val, feature_names=X_val.columns.tolist()))
    
    rmse_train = sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_val   = sqrt(mean_squared_error(y_val, y_pred_val))
    
    return model, rmse_train, rmse_val

# ---------------------------------------------------------------------
# 4) Nested CV: Compute Mean CV RMSE (Modified to Unpack Three Values)
# ---------------------------------------------------------------------
def compute_cv_rmse(X, y, params, features, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_list = []
    for train_idx, val_idx in kf.split(X):
        X_tr = X.iloc[train_idx][features]
        y_tr = y.iloc[train_idx]
        # Unpack three returned values; we only need the validation RMSE
        _, _, rmse_val = xgb_train_early_stopping(X_tr, y_tr, params)
        rmse_list.append(rmse_val)
    return np.mean(rmse_list)

# ---------------------------------------------------------------------
# 5) Forward Feature Selection (Gain One Feature at a Time)
# ---------------------------------------------------------------------
def forward_feature_selection(X_train, y_train, params, candidate_features=None, selected_features=None, n_splits=5, rmse_history=None):
    if selected_features is None:
        selected_features = []
    if candidate_features is None:
        candidate_features = X_train.columns.tolist()
    if rmse_history is None:
        rmse_history = []

    if not selected_features:
        baseline_rmse = np.inf
    else:
        baseline_rmse = compute_cv_rmse(X_train, y_train, params, selected_features, n_splits)
    
    print(f"[ForwardFS] Current features ({len(selected_features)}): {selected_features}")
    print(f"[ForwardFS] Baseline CV RMSE = {baseline_rmse:.3f}")
    rmse_history.append({'features': selected_features.copy(), 'rmse': baseline_rmse})

    best_candidate = None
    best_rmse = baseline_rmse

    for feat in candidate_features:
        if feat in selected_features:
            continue
        candidate_set = selected_features + [feat]
        candidate_rmse = compute_cv_rmse(X_train, y_train, params, candidate_set, n_splits)
        print(f"Candidate add {feat}: CV RMSE = {candidate_rmse:.3f} (Baseline = {baseline_rmse:.3f})")
        if candidate_rmse < best_rmse - 1e-6:
            best_candidate = feat
            best_rmse = candidate_rmse

    if best_candidate is not None:
        print(f"Improvement found: adding '{best_candidate}' => new CV RMSE = {best_rmse:.3f}")
        selected_features.append(best_candidate)
        return forward_feature_selection(X_train, y_train, params,
                                         candidate_features=candidate_features,
                                         selected_features=selected_features,
                                         n_splits=n_splits,
                                         rmse_history=rmse_history)
    else:
        print("No further improvement found.")
        return selected_features, rmse_history

# ---------------------------------------------------------------------
# 6) Final Model Evaluation on the Holdout Test Set
# ---------------------------------------------------------------------
def final_model_evaluation(X_train, y_train, X_holdout, y_holdout, params, early_stopping_rounds=25, num_rounds=500):
    model, rmse_train, _ = xgb_train_early_stopping(X_train, y_train, params, early_stopping_rounds, num_rounds)
    d_holdout = xgb.DMatrix(X_holdout, feature_names=X_holdout.columns.tolist())
    y_pred_holdout = model.predict(d_holdout)
    rmse_holdout = sqrt(mean_squared_error(y_holdout, y_pred_holdout))
    r2_holdout   = r2_score(y_holdout, y_pred_holdout)
    test_std     = np.std(y_holdout)
    return model, rmse_train, rmse_holdout, r2_holdout, test_std

# ---------------------------------------------------------------------
# 7) Helper: Quadratic Expansion of Top Features
# ---------------------------------------------------------------------
def add_quadratic_expansion(X, top_features, degree=2):
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)
    X_sub = X[top_features]
    poly_array = poly.fit_transform(X_sub)
    poly_feature_names = poly.get_feature_names_out(top_features)
    powers = poly.powers_
    # Keep only features with total degree equal to 2
    quad_indices = [i for i, power in enumerate(powers) if sum(power) == 2]
    quad_feature_names = [poly_feature_names[i] for i in quad_indices]
    quad_features = poly_array[:, quad_indices]
    quad_df = pd.DataFrame(quad_features, columns=quad_feature_names, index=X.index)
    return quad_df


def perform_forward_feature_selection(framework, tasks, strategy='impute'):
    results = {}
    seeds = [42, 1337, 8675309]
    
    for task_name, task_info in tasks.items():
        print(f"\n=== Processing Task: {task_name} ===")
        target = task_info['target']
        features = task_info['features']

        task_results = {}

        for seed in seeds:
            print(f"\n[Seed {seed}] Starting run...")
            # Use seed-specific params
            # Hardcoded params for the task
            hardcoded_params = {
                "Microbiota Richness (Root) with Plant OTUs": {
                    'colsample_bytree': 0.7, 'learning_rate': 0.01, 'max_depth': 5,
                    'min_child_weight': 1, 'reg_alpha': 0.5, 'reg_lambda': 0.5,
                    'subsample': 0.8, 'objective': 'reg:squarederror', 'seed': seed
                },
                "Microbiota Richness (Leaf) with Plant OTUs": {
                    'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 5,
                    'min_child_weight': 1, 'reg_alpha': 1.0, 'reg_lambda': 2.0,
                    'subsample': 0.7, 'objective': 'reg:squarederror', 'seed': seed
                },
                "Microbiota Richness (Root) with Plant Metrics": {
                    'colsample_bytree': 0.9, 'learning_rate': 0.1, 'max_depth': 3,
                    'min_child_weight': 1, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
                    'subsample': 0.9, 'objective': 'reg:squarederror', 'seed': seed
                },
                "Microbiota Richness (Leaf) with Plant Metrics": {
                    'colsample_bytree': 0.7, 'learning_rate': 0.01, 'max_depth': 4,
                    'min_child_weight': 1, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
                    'subsample': 0.8, 'objective': 'reg:squarederror', 'seed': seed
                },
                "Microbiota Shannon (Root) with Plant OTUs": {
                    'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 5,
                    'min_child_weight': 5, 'reg_alpha': 1.0, 'reg_lambda': 0.5,
                    'subsample': 0.9, 'objective': 'reg:squarederror', 'seed': seed
                },
                "Microbiota Shannon (Leaf) with Plant OTUs": {
                    'colsample_bytree': 0.7, 'learning_rate': 0.01, 'max_depth': 5,
                    'min_child_weight': 3, 'reg_alpha': 0.1, 'reg_lambda': 0.5,
                    'subsample': 0.8, 'objective': 'reg:squarederror', 'seed': seed
                },
                "Microbiota Shannon (Root) with Plant Metrics": {
                    'colsample_bytree': 0.9, 'learning_rate': 0.01, 'max_depth': 4,
                    'min_child_weight': 1, 'reg_alpha': 1.0, 'reg_lambda': 0.5,
                    'subsample': 0.9, 'objective': 'reg:squarederror', 'seed': seed
                },
                "Microbiota Shannon (Leaf) with Plant Metrics": {
                    'colsample_bytree': 0.7, 'learning_rate': 0.05, 'max_depth': 3,
                    'min_child_weight': 3, 'reg_alpha': 0.1, 'reg_lambda': 0.5,
                    'subsample': 0.7, 'objective': 'reg:squarederror', 'seed': seed
                },
                "Pathobiota Richness (Root) with Plant OTUs": {
                    'colsample_bytree': 0.9, 'learning_rate': 0.01, 'max_depth': 5,
                    'min_child_weight': 1, 'reg_alpha': 0.5, 'reg_lambda': 2.0,
                    'subsample': 0.9, 'objective': 'reg:squarederror', 'seed': seed
                },
                "Pathobiota Richness (Leaf) with Plant OTUs": {
                    'colsample_bytree': 0.7, 'learning_rate': 0.01, 'max_depth': 5,
                    'min_child_weight': 1, 'reg_alpha': 0.5, 'reg_lambda': 2.0,
                    'subsample': 0.7, 'objective': 'reg:squarederror', 'seed': seed
                },
                "Pathobiota Richness (Root) with Plant Metrics": {
                    'colsample_bytree': 0.9, 'learning_rate': 0.05, 'max_depth': 5,
                    'min_child_weight': 5, 'reg_alpha': 1.0, 'reg_lambda': 2.0,
                    'subsample': 0.9, 'objective': 'reg:squarederror', 'seed': seed
                },
                "Pathobiota Richness (Leaf) with Plant Metrics": {
                    'colsample_bytree': 0.7, 'learning_rate': 0.01, 'max_depth': 5,
                    'min_child_weight': 1, 'reg_alpha': 1.0, 'reg_lambda': 2.0,
                    'subsample': 0.7, 'objective': 'reg:squarederror', 'seed': seed
                },
                "Pathobiota Shannon (Root) with Plant OTUs": {
                    'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 4,
                    'min_child_weight': 5, 'reg_alpha': 1.0, 'reg_lambda': 1.0,
                    'subsample': 0.9, 'objective': 'reg:squarederror', 'seed': seed
                },
                "Pathobiota Shannon (Leaf) with Plant OTUs": {
                    'colsample_bytree': 0.9, 'learning_rate': 0.01, 'max_depth': 3,
                    'min_child_weight': 1, 'reg_alpha': 1.0, 'reg_lambda': 2.0,
                    'subsample': 0.7, 'objective': 'reg:squarederror', 'seed': seed
                },
                "Pathobiota Shannon (Root) with Plant Metrics": {
                    'colsample_bytree': 0.7, 'learning_rate': 0.01, 'max_depth': 5,
                    'min_child_weight': 1, 'reg_alpha': 1.0, 'reg_lambda': 2.0,
                    'subsample': 0.7, 'objective': 'reg:squarederror', 'seed': seed
                },
                "Pathobiota Shannon (Leaf) with Plant Metrics": {
                    'colsample_bytree': 0.7, 'learning_rate': 0.01, 'max_depth': 4,
                    'min_child_weight': 5, 'reg_alpha': 0.1, 'reg_lambda': 2.0,
                    'subsample': 0.7, 'objective': 'reg:squarederror', 'seed': seed
                }
            }
            params = hardcoded_params.get(task_name, {
                'objective': 'reg:squarederror',
                'max_depth': 3,
                'learning_rate': 0.05,
                'min_child_weight': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 1.0,
                'reg_alpha': 0.5,
                'reg_lambda': 1.0,
                'seed': seed
            })

            # Preprocess
            X_train_full, y_train_full, X_holdout, y_holdout = preprocess_data_for_task(
                framework.data, features, target, strategy=strategy
            )

            if len(X_train_full) == 0 or len(X_holdout) == 0:
                print(f"[WARNING] No valid data left for {task_name} with seed {seed}. Skipping...")
                continue

            # Top features for quadratic expansion
            model_full, _, _ = xgb_train_early_stopping(X_train_full, y_train_full, params, early_stopping_rounds=20, random_state=seed)
            importance_dict = model_full.get_score(importance_type='gain')
            importance_df = pd.DataFrame(list(importance_dict.items()), columns=['Feature', 'Importance'])
            importance_df = importance_df.sort_values("Importance", ascending=False)
            top10 = importance_df.head(15)['Feature'].tolist()
            print(f"Top 10 features for quadratic expansion: {top10}")

            # Quadratic expansion
            poly_df_train = add_quadratic_expansion(X_train_full, top10, degree=2)
            poly_df_holdout = add_quadratic_expansion(X_holdout, top10, degree=2)
            X_train_aug = pd.concat([X_train_full, poly_df_train], axis=1)
            X_holdout_aug = pd.concat([X_holdout, poly_df_holdout], axis=1)

            candidate_features = X_train_aug.columns.tolist()

            # Forward feature selection
            best_features, cv_history = forward_feature_selection(
                X_train_aug, y_train_full, params,
                candidate_features=candidate_features,
                selected_features=[],
                n_splits=5
            )
            print(f"[Seed {seed}] Selected Features: {best_features}")
            print(f"[Seed {seed}] CV RMSE History: {cv_history}")

            # Final evaluation
            final_model, final_train_rmse, final_test_rmse, final_r2, final_test_std = final_model_evaluation(
                X_train_aug[best_features], y_train_full,
                X_holdout_aug[best_features], y_holdout,
                params,
                early_stopping_rounds=25,
                num_rounds=500
            )

            print(f"[Seed {seed}] Final Model Performance:")
            print(f"  Train RMSE: {final_train_rmse:.3f}")
            print(f"  Test RMSE : {final_test_rmse:.3f}")
            print(f"  R2        : {final_r2:.3f}")
            print(f"  Test Std  : {final_test_std:.3f}")

            # Save results
            task_results[seed] = {
                "Selected_Features": best_features,
                "CV_RMSE_History": cv_history,
                "Final_Train_RMSE": final_train_rmse,
                "Final_Test_RMSE": final_test_rmse,
                "Final_R2": final_r2,
                "Final_Test_Std": final_test_std
            }

        results[task_name] = task_results

    return results

from ecodynamics_multimodel_framework import EcodynamicsAI

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
results = perform_forward_feature_selection(framework, tasks, strategy='impute')
print(results)
