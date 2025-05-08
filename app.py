import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os
from sklearn.metrics import mean_squared_error

# Define paths for model and result files
MODEL_DIR_FULL = 'models2/full/'
MODEL_DIR_TOP20 = 'models2/top20/'
RESULTS_FILE = 'models2/model_results.csv'

# Load model results
@st.cache_data
def load_results():
    if os.path.exists(RESULTS_FILE):
        return pd.read_csv(RESULTS_FILE)
    else:
        st.error("Model results file not found. Ensure models are trained and saved.")
        return pd.DataFrame()

results_df = load_results()

# Sidebar
st.sidebar.title("OTU Prediction App")

uploaded_file = st.sidebar.file_uploader("Upload CSV for Prediction", type=["csv"])
target = st.sidebar.selectbox("Select Target Variable", results_df['target'].unique() if not results_df.empty else [])

if uploaded_file and target:
    st.write("### Uploaded Data")
    input_data = pd.read_csv(uploaded_file)
    st.write(input_data.head())

    # Determine the paired column
    pair = target.replace('leaf_Otu', 'root_Otu') if target.startswith('leaf_Otu') else target.replace('root_Otu', 'leaf_Otu')

    # Drop the paired column if present
    if pair in input_data.columns:
        input_data.drop(columns=[pair], inplace=True)
        st.info(f"Dropped paired column: {pair}")

    # Extract actual values for RMSE calculation (if present)
    actual_values = input_data[target].copy() if target in input_data.columns else None
    has_actuals = actual_values is not None

    # Load the best model for the target
    full_model_path = os.path.join(MODEL_DIR_FULL, f"xgb_full_{target}.model")
    top20_model_path = os.path.join(MODEL_DIR_TOP20, f"xgb_top20_{target}.model")

    # Prepare data for prediction
    X = input_data.drop(columns=[target], errors='ignore')

    # Check for missing values
    if X.isnull().sum().sum() > 0:
        st.warning("Uploaded data contains missing values. Predictions may be affected.")

    # Predictions using Full Model
    if os.path.exists(full_model_path):
        model = xgb.Booster()
        model.load_model(full_model_path)
        dmatrix = xgb.DMatrix(X)
        preds_full = model.predict(dmatrix)

        # Calculate RMSE for Full Model
        rmse_full = np.sqrt(mean_squared_error(actual_values, preds_full)) if has_actuals else "N/A"

        # Output Full Model Predictions and RMSE
        output_full = pd.DataFrame({
            f"Predicted_{target}_Full": preds_full
        })
        if has_actuals:
            output_full['Actual'] = actual_values
        st.write("### Full Model Predictions")
        st.write(output_full)
        st.write(f"RMSE (Full Model): {rmse_full}")
    else:
        st.warning("Full model not found for the selected target.")

    # Predictions using Top-20 Model
    if os.path.exists(top20_model_path):
        model_top20 = xgb.Booster()
        model_top20.load_model(top20_model_path)

        # Extract top-20 features from the results dataframe
        top20_feats = results_df[results_df['target'] == target]['top20_feats'].values[0].split(';')

        if all(f in X.columns for f in top20_feats):
            dmatrix_top20 = xgb.DMatrix(X[top20_feats])
            preds_top20 = model_top20.predict(dmatrix_top20)

            # Calculate RMSE for Top-20 Model
            rmse_top20 = np.sqrt(mean_squared_error(actual_values, preds_top20)) if has_actuals else "N/A"

            # Output Top-20 Model Predictions and RMSE
            output_top20 = pd.DataFrame({
                f"Predicted_{target}_Top20": preds_top20
            })
            if has_actuals:
                output_top20['Actual'] = actual_values
            st.write("### Top-20 Model Predictions")
            st.write(output_top20)
            st.write(f"RMSE (Top-20 Model): {rmse_top20}")
        else:
            st.warning("Not all top-20 features are present in the uploaded dataset.")
else:
    st.write("Upload a CSV and select a target to start prediction.")
