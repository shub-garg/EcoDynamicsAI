import os
import pandas as pd
import json

# Paths
FEATURE_IMPORTANCE_DIR = '../feature_importance'
OUTPUT_PATH = '../visualization/feature_importance.json'
os.makedirs('../visualization', exist_ok=True)

# Initialize data structures
nodes = set()
links = []

# Iterate through each CSV file in the directory
for csv_file in os.listdir(FEATURE_IMPORTANCE_DIR):
    if csv_file.endswith(".csv"):
        target_name = csv_file.replace(".csv", "")
        csv_path = os.path.join(FEATURE_IMPORTANCE_DIR, csv_file)

        # Read CSV
        imp_df = pd.read_csv(csv_path)

        # Check for required columns
        if "feature" in imp_df.columns and "gain" in imp_df.columns:
            
            # Add the target as a node
            nodes.add(target_name)

            # Sort by gain and select top 10 features
            imp_df = imp_df.sort_values("gain", ascending=False).head(5)

            # Process each feature
            for _, row in imp_df.iterrows():
                source = row['feature']
                gain = round(row['gain'], 2)  # Limit to 2 decimal places

                # Ensure the source node exists in the nodes set
                nodes.add(source)

                # Create link
                links.append({
                    "source": source,
                    "target": target_name,
                    "gain": gain
                })

# Convert nodes set to a list of dictionaries
nodes_list = [{"id": node} for node in nodes]

# Construct the final JSON structure
graph_data = {
    "nodes": nodes_list,
    "links": links
}

# Save JSON
with open(OUTPUT_PATH, 'w') as json_file:
    json.dump(graph_data, json_file, indent=4)

print(f"✅ JSON saved to {OUTPUT_PATH}")
