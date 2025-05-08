import os
import pandas as pd
import numpy as np
import xgboost as xgb

# Define paths
MODEL_DIR_FULL = '../models2/full'
MODEL_DIR_TOP20 = '../models2/top20'
OUTPUT_DIR = '../feature_importance'

# Create output directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to extract feature importance and save as CSV
def extract_feature_importance(model_path, target_name):
    model = xgb.Booster()
    model.load_model(model_path)
    importance_dict = model.get_score(importance_type='gain')
    
    # Convert to DataFrame and sort by gain
    imp_df = pd.DataFrame.from_dict(importance_dict, orient='index', columns=['gain'])
    imp_df = imp_df.reset_index().rename(columns={'index': 'feature'})
    imp_df = imp_df.sort_values('gain', ascending=False)

    # Save as CSV
    output_path = os.path.join(OUTPUT_DIR, f"{target_name}.csv")
    imp_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

# Process Full Models
for model_file in os.listdir(MODEL_DIR_FULL):
    if model_file.endswith(".model"):
        target_name = model_file.replace("xgb_full_", "").replace(".model", "")
        model_path = os.path.join(MODEL_DIR_FULL, model_file)
        extract_feature_importance(model_path, target_name)

# Process Top-20 Models
# for model_file in os.listdir(MODEL_DIR_TOP20):
#     if model_file.endswith(".model"):
#         target_name = model_file.replace("xgb_top20_", "").replace(".model", "")
#         model_path = os.path.join(MODEL_DIR_TOP20, model_file)
#         extract_feature_importance(model_path, target_name)

print("\n✅ Feature importance extraction completed for all models.")
