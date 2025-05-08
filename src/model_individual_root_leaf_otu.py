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

import pandas as pd

# 1. Load your existing dataframes (assuming they’re already in memory)
# df_ecological  # your original ecological DataFrame
# df_otu_abundance  # your second DataFrame with only OTU columns

# 2. Make an intermediate copy
df_intermediate = df_ecological.copy()

# 3. Align on 'population' and overwrite all leaf_/root_ columns
df_intermediate.set_index('population', inplace=True)
otu = df_otu_abundance.set_index('population')

# Identify the OTU columns present in otu
otu_cols = [c for c in otu.columns if c.startswith(('leaf_Otu','root_Otu'))]

# Overwrite in one go
df_intermediate.update(otu[otu_cols])

# 4. Drop the unwanted columns
to_drop = [
    # microbial richness / diversity / PCoA metrics
    'richness_microbiota_leaf','Shannon_microbiota_leaf',
    'PCOA1_microbiota_leaf','PCOA2_microbiota_leaf',
    'richness_pathobiota_leaf','Shannon_pathobiota_leaf',
    'PCOA1_pathobiota_leaf','PCOA2_pathobiota_leaf',
    'richness_microbiota_root','Shannon_microbiota_root',
    'PCOA1_microbiota_root','PCOA2_microbiota_root',
    'richness_pathobiota_root','Shannon_pathobiota_root',
    'PCOA2_pathobiota_root',
    # plant PCoA
    'plant_pcoa1','plant_pcoa2','plant_pcoa3'
]
df_intermediate.drop(columns=[c for c in to_drop if c in df_intermediate.columns],
                     inplace=True)

# If you need to restore population as a column
df_intermediate.reset_index(inplace=True)

# df_intermediate now holds your updated data
print(df_intermediate.shape)
print(df_intermediate.columns.tolist())

# Lead and Root OTUs

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler

# 1. Start from the merged & filtered intermediate you already have:
#    (i.e. df_intermediate from the previous snippet)
df_clean = df_intermediate.copy()

# 2. Replace any stray “.” placeholders with actual NaN
df_clean.replace('.', np.nan, inplace=True)

# 3. Drop columns that are nearly empty
#    (here: > 80% missing; you can adjust the threshold)
missing_frac = df_clean.isna().mean()
cols_to_drop = missing_frac[missing_frac > 0.8].index.tolist()
df_clean.drop(columns=cols_to_drop, inplace=True)
print(f"Dropped {len(cols_to_drop)} columns with >80% missing.")

# 4. (Optional) Drop any rows that have no data at all
df_clean.dropna(how='all', inplace=True)

# 5. Impute remaining missing values via KNN
imputer = KNNImputer(n_neighbors=5)
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
print("Imputed missing numeric values with KNN.")

# 6. Scale all numeric features with a RobustScaler
scaler = RobustScaler()
df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])
print("Applied Robust scaling to numeric features.")

# 7. (Optional) If you want to keep a copy before modeling
df_model_ready = df_clean.copy()

# 8. Inspect
print("Final shape:", df_model_ready.shape)
print("Columns remaining:", df_model_ready.columns.tolist())
print(df_model_ready)

# Save the final cleaned dataset
df_model_ready.to_csv('df_model_ready.csv', index=False)

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error

# 1) Load your cleaned, model-ready data
df = pd.read_csv('df_model_ready.csv')
if 'population' in df.columns:
    df.drop(columns=['population'], inplace=True)

# 2) Prepare output directories
os.makedirs('models2/full',  exist_ok=True)
os.makedirs('models2/top20', exist_ok=True)

# 3) Identify all OTU columns
leaf_targets = [c for c in df.columns if c.startswith('leaf_Otu')]
root_targets = [c for c in df.columns if c.startswith('root_Otu')]
all_targets  = leaf_targets + root_targets

# 4) Storage for results
records = []

# 5) Define hyperparameter grid
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'reg_alpha': [0.1, 0.5, 1.0],
    'reg_lambda': [0.5, 1.0, 2.0],
    'gamma': [1.0],
    'objective': ['reg:squarederror'],
    'seed': [42]
}

# 6) Loop through each target
for target in all_targets:
    print(f"\n>> Processing target: {target}")

    # a) Prepare X, y
    y = df[target].copy()
    X = df.drop(columns=[target])

    # b) Drop the opposite OTU
    if target.startswith('leaf_Otu'):
        pair = target.replace('leaf_Otu', 'root_Otu')
    else:
        pair = target.replace('root_Otu', 'leaf_Otu')

    dropped_pair = ''
    if pair in X.columns:
        X.drop(columns=[pair], inplace=True)
        dropped_pair = pair
        print(f"   Dropped paired column: {pair}")

    # c) Filter non-missing y
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]

    # d) Train/validation/test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )

    # e) DMatrix construction
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val, label=y_val)
    dtest  = xgb.DMatrix(X_test, label=y_test)

    # f) Hyperparameter tuning for full model
    best_rmse_full = float('inf')
    best_model_full = None
    best_params_full = None

    for params in ParameterGrid(param_grid):
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dtrain, 'train'), (dval, 'valid')],
            early_stopping_rounds=20,
            verbose_eval=False
        )
        preds = model.predict(dtest)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        if rmse < best_rmse_full:
            best_rmse_full = rmse
            best_model_full = model
            best_params_full = params

    std_test = np.std(y_test)

    # g) Extract top-20 features from best full model
    imp_dict = best_model_full.get_score(importance_type='gain')
    imp_df = (
        pd.DataFrame.from_dict(imp_dict, orient='index', columns=['gain'])
          .reset_index().rename(columns={'index':'feature'})
          .sort_values('gain', ascending=False)
    )
    raw_top20 = imp_df['feature'].head(20).tolist()
    top20 = [f for f in raw_top20 if f in X_train.columns]

    # h) Retrain model on top-20 if possible
    rmse_top = np.nan
    top20_path = ''
    if top20:
        X_train20 = X_train[top20]
        X_val20   = X_val[top20]
        X_test20  = X_test[top20]

        dtrain20 = xgb.DMatrix(X_train20, label=y_train)
        dval20   = xgb.DMatrix(X_val20, label=y_val)
        dtest20  = xgb.DMatrix(X_test20, label=y_test)

        top20_model = xgb.train(
            best_params_full,
            dtrain20,
            num_boost_round=500,
            evals=[(dtrain20, 'train'), (dval20, 'valid')],
            early_stopping_rounds=20,
            verbose_eval=False
        )
        y_pred_top = top20_model.predict(dtest20)
        rmse_top = np.sqrt(mean_squared_error(y_test, y_pred_top))
        top20_path = f"models2/top20/xgb_top20_{target}.model"
        top20_model.save_model(top20_path)

    # i) Save best full model
    full_path = f"models2/full/xgb_full_{target}.model"
    best_model_full.save_model(full_path)

    # j) Record results
    records.append({
        'target':           target,
        'dropped_pair':     dropped_pair,
        'rmse_full_test':   best_rmse_full,
        'rmse_top20_test':  rmse_top,
        'test_std':         std_test,
        'top20_feats':      ';'.join(top20),
        'model_full_path':  full_path,
        'model_top20_path': top20_path,
        'best_params':      str(best_params_full)
    })

# 7) Save summary
results_df = pd.DataFrame(records)
results_df.to_csv('models2/model_results.csv', index=False)
print("\n✅ All done! Models saved in 'models1/', summary in 'models2/model_results.csv'")

import pandas as pd

# 1) Load the results
df = pd.read_csv('models2/model_results.csv')

# 2) Compute full‐model averages
mean_full_rmse = df['rmse_full_test'].mean()
mean_full_std  = df['test_std'].mean()

# 3) Restrict to targets where a top20 model was actually trained
top20_df = df[df['rmse_top20_test'].notna()]

# 4) Compute top20‐model averages
mean_top20_rmse = top20_df['rmse_top20_test'].mean()
mean_top20_std  = top20_df['test_std'].mean()

# 5) Print or collect into a small DataFrame
summary = pd.DataFrame({
    'Model':         ['Full', 'Top20'],
    'Mean RMSE':     [mean_full_rmse,   mean_top20_rmse],
    'Mean Test STD': [mean_full_std,     mean_top20_std]
})

print(summary)
