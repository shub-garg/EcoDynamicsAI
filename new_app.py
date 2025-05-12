import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics import mean_squared_error

# --- Paths ---
MODEL_DIR_FULL   = 'models2/full/'
MODEL_DIR_TOP20  = 'models2/top20/'
RESULTS_FILE     = 'models2/model_results.csv'
JSON_FILE        = 'visualization/feature_importance.json'
TEMPLATE_HTML    = 'graph_template.html'
SAMPLE_FILE      = 'sample.csv'
MAPPING_FILE     = '16S_dataSet1_41396_2018_152_MOESM2_ESM.xlsx'

# --- Load mapping Excel and build legend df ---
# assumes sheet has columns: '16S (0.03)', 'gyrB(0.020)', 'Taxonomy', 'Family'
mapping_df = pd.read_excel(MAPPING_FILE)
# melt so each OTU ends up in one column
legend_df = (
    mapping_df
      .melt(
         id_vars=['Taxonomy','Family'],
         value_vars=['16S (0.03)','gyrB(0.020)'],
         var_name='Marker',
         value_name='Raw_OTU'
      )
      # normalize OTU IDs: strip leading zeros and uppercase
      .assign(
         OTU_ID=lambda d: d['Raw_OTU']
                         .str.replace(r'^Otu0*', 'Otu', regex=True)
      )
      [['OTU_ID','Marker','Taxonomy','Family']]
      .dropna(subset=['OTU_ID'])
      .drop_duplicates()
      .reset_index(drop=True)
)

# --- Load results_df ---
def load_results():
    if os.path.exists(RESULTS_FILE):
        return pd.read_csv(RESULTS_FILE)
    st.error("Model results file not found.")
    return pd.DataFrame()

results_df = load_results()

# --- Layout ---
st.title("OTU Prediction App")

# Display model and dataset information
st.markdown("**Model:** XGBoost with Early Stopping (Hyperparameter Optimized)")
st.markdown("**Dataset:** South-France Dataset")

# Sidebar
st.sidebar.title("EcoDynamicsAI")

# OTU Legend pop-up
with st.sidebar.expander("🔍 OTU Legend"):
    st.markdown(
        "This table shows, for each OTU (16S or gyrB), its taxonomic classification."
    )
    st.dataframe(legend_df)

# Sample format pop-up
if st.sidebar.button("View Sample File Format"):
    st.sidebar.markdown("### Sample File Format")
    st.sidebar.write(pd.read_csv(SAMPLE_FILE).head())
    st.sidebar.info("Ensure the data has appropriate column names matching the target variable and features.")

# File upload & target selection
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
target = st.sidebar.selectbox(
    "Select Target",
    options=results_df['target'].unique() if not results_df.empty else []
)

# Main: show D3 graph button
if st.button("Show Abiotic-Biotic Interaction Network"):
    if os.path.exists(JSON_FILE) and os.path.exists(TEMPLATE_HTML):
        data = json.load(open(JSON_FILE, 'r'))
        tpl  = open(TEMPLATE_HTML, 'r').read()
        html = tpl.replace("__DATA__", json.dumps(data))
        components.html(html, height=700, width=700, scrolling=True)
    else:
        st.error("Missing `feature_importance.json` or `graph_template.html`.")

# As soon as a target is selected, show top-20 feature chart
if target:
    full_path = os.path.join(MODEL_DIR_FULL, f"xgb_full_{target}.model")
    if os.path.exists(full_path):
        m_full = xgb.Booster(); m_full.load_model(full_path)
        imp_full = m_full.get_score(importance_type='gain')
        df_full = (
            pd.DataFrame.from_dict(imp_full, orient='index', columns=['gain'])
              .reset_index().rename(columns={'index':'feature'})
              .sort_values('gain', ascending=False)
              .head(20)
        )
        st.subheader("Top 20 Features by Gain")
        st.bar_chart(df_full.set_index('feature')['gain'])
    else:
        st.info("Full model not available for this target.")

# Upload + prediction logic
if uploaded_file and target:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # drop paired OTU column
    pair = (target.replace('leaf_Otu', 'root_Otu')
            if target.startswith('leaf_Otu')
            else target.replace('root_Otu','leaf_Otu'))
    if pair in df.columns:
        df.drop(columns=[pair], inplace=True)
        st.info(f"Dropped `{pair}`")

    actual = df[target] if target in df.columns else None
    X = df.drop(columns=[target], errors='ignore')

    if X.isnull().any().any():
        st.warning("Missing values in features.")

    # Full model prediction
    full_path = os.path.join(MODEL_DIR_FULL, f"xgb_full_{target}.model")
    if os.path.exists(full_path):
        m = xgb.Booster(); m.load_model(full_path)
        preds = m.predict(xgb.DMatrix(X))
        rmse  = (np.sqrt(mean_squared_error(actual, preds))
                 if actual is not None else "N/A")
        out = pd.DataFrame({f"Pred_Full": preds})
        if actual is not None: out["Actual"] = actual.values
        st.subheader("Full Model")
        st.dataframe(out)
        st.markdown(f"**RMSE (Full):** {rmse}")
    else:
        st.warning("Full model missing.")

    # Top-20 model prediction
    top20_path = os.path.join(MODEL_DIR_TOP20, f"xgb_top20_{target}.model")
    if os.path.exists(top20_path):
        feats = (results_df.loc[results_df.target==target, 'top20_feats']
                 .iat[0].split(';'))
        if all(f in X.columns for f in feats):
            m2 = xgb.Booster(); m2.load_model(top20_path)
            preds2 = m2.predict(xgb.DMatrix(X[feats]))
            rmse2  = (np.sqrt(mean_squared_error(actual, preds2))
                      if actual is not None else "N/A")
            out2 = pd.DataFrame({f"Pred_Top20": preds2})
            if actual is not None: out2["Actual"] = actual.values
            st.subheader("Top-20 Model")
            st.dataframe(out2)
            st.markdown(f"**RMSE (Top-20):** {rmse2}")
        else:
            st.warning("Not all 20 features in upload.")
    else:
        st.warning("Top-20 model missing.")
else:
    st.info("Upload data & select a target to predict.")
