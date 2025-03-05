# main.py

import os
from ecodynamicsAI import EcodynamicsAI
from tasks import define_tasks

def run_pipeline():
    # 1. Initialize EcodynamicsAI instance
    framework = EcodynamicsAI()

    # 2. Load ecological data and OTU data from local files
    ecological_file = os.path.join("../data/Ecological_characterization_Hanna_Markle.xlsx")
    otu_file = os.path.join("../data/Leaf_and_Root_OTU_Relative_Abundance.xlsx")

    framework.load_data(ecological_file)
    framework.load_otu_data(otu_file)
    
    # If you need gyrB data, load similarly:
    # gyrb_file = os.path.join("data", "df_gyrb.xlsx")
    # df_gyrb = pd.read_excel(gyrb_file)
    # (handle gyrB as needed)

    # 3. Categorize features, filter columns, and align data with OTUs
    framework.categorize_features()
    framework.filter_columns_of_interest()
    framework.validate_and_assign_otu_data()
    feature_categories = framework.feature_categories

    # 4. Generate tasks by calling define_tasks(...)
    tasks = define_tasks(feature_categories)

    # 4. Train RandomForest-based models
    rf_results = framework.train_multiple_models(tasks, top_k=20)

    # 5. Train XGBoost-based models
    xgb_results = framework.train_multiple_models_xgb(tasks, early_stopping_rounds=20, optimize_params=True)

    # 6. Train XGBoost with polynomial and PCA transformations
    xgb_poly_results = framework.train_multiple_models_xgb_poly_pca(
        tasks,
        early_stopping_rounds=20,
        poly_degree=2,
        n_components=50,
        optimize_params=True
    )

    # 7. Display a combined results table
    framework.display_combined_results()

if __name__ == "__main__":
    run_pipeline()
