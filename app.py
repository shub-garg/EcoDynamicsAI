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
RESULTS_FILE1     = 'models2/model_results.csv'
JSON_FILE        = 'visualization/feature_importance.json'
TEMPLATE_HTML    = 'graph_template.html'
SAMPLE_FILE      = 'sample.csv'
MAPPING_FILE     = '16S_dataSet1_41396_2018_152_MOESM2_ESM.xlsx'

# --- Load mapping Excel and build legend df ---
# assumes sheet has columns: '16S (0.03)', 'gyrB(0.020)', 'Taxonomy', 'Family'
mapping_df = pd.read_excel(MAPPING_FILE)

# melt so each OTU ends up in one column
legend_long = (
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
      .dropna(subset=['OTU_ID'])
      .drop_duplicates(subset=['OTU_ID','Marker'])
)

# give gyrB entries higher priority
priority = {'gyrB(0.020)': 0, '16S (0.03)': 1}
legend_long['priority'] = legend_long['Marker'].map(priority)

# pick one row per OTU, preferring gyrB
legend_df = (
    legend_long
      .sort_values(['OTU_ID','priority'])
      .drop_duplicates(subset=['OTU_ID'], keep='first')
      [['OTU_ID','Taxonomy','Family']]
      .reset_index(drop=True)
)
def main_app_page():

    # --- Load results_df ---
    def load_results():
        if os.path.exists(RESULTS_FILE1):
            return pd.read_csv(RESULTS_FILE1)
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


RESULTS_FILE = 'data/results.csv'

# Load results data
def load_results_otu():
    if os.path.exists(RESULTS_FILE):
        return pd.read_csv(RESULTS_FILE)
    st.error("Results file not found.")
    return pd.DataFrame()

# Load data into session state if not present
if 'results' not in st.session_state:
    st.session_state['results'] = load_results_otu()

results = st.session_state['results']

# Top 20 OTU Features Page
def top20_otu_page():
    st.title("Top 20 Features - Best Models using Plant OTUs")
    tasks_of_interest = [
        "Microbiota Richness (Root) with Plant OTUs",
        "Microbiota Richness (Leaf) with Plant OTUs",
        "Microbiota Shannon (Root) with Plant OTUs",
        "Microbiota Shannon (Leaf) with Plant OTUs",
        "Pathobiota Richness (Root) with Plant OTUs",
        "Pathobiota Richness (Leaf) with Plant OTUs",
        "Pathobiota Shannon (Root) with Plant OTUs",
        "Pathobiota Shannon (Leaf) with Plant OTUs"
    ]

    selected_task = st.sidebar.selectbox("Select a Task", tasks_of_interest)

    if selected_task:
        task_data = results[results['task'] == selected_task]
        if not task_data.empty:
            top20_str = task_data.iloc[0]['top20_features']
            top20_pairs = [item.split(':') for item in top20_str.split(';')]
            top20_df = pd.DataFrame(top20_pairs, columns=['Feature', 'Gain']).astype({'Gain': 'float'})
            st.subheader(f"Top 20 Features for {selected_task}")
            st.bar_chart(top20_df.set_index('Feature')['Gain'])
        else:
            st.warning(f"No data available for {selected_task}")
    else:
        st.info("Select a task to view the top 20 features.")


PLANT_MAPPING_FILE = 'data/___OTU_ID_Species_Family_mapping.xlsx'

def plant_otu_legend_page():
    st.title("Plant-OTU Legend")
    st.markdown("This table maps each plant OTU ID to its species and family.")
    # read the new mapping file
    plant_df = pd.read_excel(PLANT_MAPPING_FILE)
    st.dataframe(plant_df)

st.sidebar.title("Navigation")
page_selection = st.sidebar.radio(
    "Select Page",
    ["Main App", "Top 20 OTU Features", "Plant-OTU Legend"]
)

if page_selection == "Top 20 OTU Features":
    top20_otu_page()
elif page_selection == "Plant-OTU Legend":
    plant_otu_legend_page()
else:
    main_app_page()