# EcoDynamicsAI

EcoDynamicsAI is a comprehensive framework designed to model and analyze plant-pathogen-microbe interactions using advanced data preprocessing, feature selection, and machine learning approaches. It leverages methods like Recursive Feature Selection, Conditional Independence Testing, Polynomial Expansions, and Ensemble Learning to predict microbial richness and diversity in plant-pathogen systems.


<img width="1512" alt="image" src="https://github.com/user-attachments/assets/d742f2c3-69a3-4f78-9f61-1f8c8b5bcdc4" />

---

## 📁 Directory Structure

```
├── data
│   ├── Ecological_characterization_Handbook.csv
│   ├── Leaf_and_Root_OTU_Relative_Abundance.csv
│   ├── df_model_ready.csv
│   └── gyrb_fragment_dataset1.xlsx
│
├── models2
│   ├── full
│   ├── top20
│   └── model_results.csv
│
├── src
│   ├── GOFO_poly.py
│   ├── GOFO_poly_3_seeds.py
│   ├── LOFO_normal.py
│   ├── LOFO_poly.py
│   ├── Normal_conditional_independence.py
│   ├── ecodynamics_multimodel_framework.py
│   ├── linear_baseline_each_task.py
│   └── model_individual_root_leaf_otu.py
│
├── EcoDynamicsAI.ipynb
├── LICENSE
├── ecodynamicsAI_description.pptx
└── requirements.txt
└── app.py
└── test.csv
└── test1.csv
```

---

## 🚀 Key Features

1. **Data Preprocessing:** Handles missing data via KNN Imputation, applies Robust Scaling, and CLR transformation for compositional data.
2. **Feature Selection:** Implements Top-k, LOFO (Leave-One-Feature-Out), GOFO (Gain-One-Feature-At-a-Time) and polynomial expansions.
3. **Modeling Methods:** XGBoost, Random Forest, and Ensemble Stacking.
4. **Conditional Independence Testing:** Assess feature relevance and potential confounding using early-stopping XGBoost models.
5. **Polynomial Expansion:** Captures non-linear interactions among top-ranked features.
6. **OTU-Level Modeling:** Predicts individual OTU abundance separately for leaf and root compartments.
7. **PowerPoint Report Generation:** Auto-generates PowerPoint summaries of key findings.
8. **Streamlit Web App:** User-friendly interface for running predictions using pre-trained models.

---

## ✅ Installation

1. Clone the repository:
```bash
git clone https://github.com/shub-garg/EcoDynamicsAI.git
cd EcoDynamicsAI
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

---

## 📦 Data Preparation

Place all relevant datasets under the `data/` directory. Ensure the filenames match those specified in the `src/` scripts.

---

## 🛠️ Usage

### 1. Normal Conditional Independence Testing:
```bash
python src/Normal_conditional_independence.py
```

### 2. LOFO Feature Selection:
```bash
python src/LOFO_normal.py
```

### 3. GOFO with Polynomial Expansion:
```bash
python src/GOFO_poly.py
```

### 4. OTU-Level Modeling:
```bash
python src/model_individual_root_leaf_otu.py
```

---

## 🛠️ Usage

##$ Running the Streamlit App:

- Ensure the `models2/` directory contains the pre-trained models and `model_results.csv`.
- Run the Streamlit app:

```bash
streamlit run app.py
```

- The app will be accessible at: `http://localhost:8501`

### 2. Deploying the App on Streamlit Cloud:

- The app is already deployed at [EcoDynamicsAI Streamlit App](https://ecodynamicsai-wupsitbyy78rvvhl7aapkt.streamlit.app).

- To deploy updates:
  - Push changes to the GitHub repository.
  - The Streamlit Cloud will automatically redeploy the app.

### 3. Using the Web App:

- Upload a CSV file containing the relevant features (test1.csv and test.csv for example).
- Select the target OTU variable for prediction.
- View the predictions and RMSE for both Full and Top-20 models.

**Expected Columns:**
- Environmental features (e.g., `latitude`, `longitude`, `MAT`, etc.)
- OTU features (e.g., `leaf_Otu0000001`, `root_Otu0000001`, etc.)



## 📈 Results and Analysis

- Model results and evaluation metrics are stored in `models2/full/` and `models2/top20/`.
- A consolidated summary table is located at `models2/model_results.csv`.
- Detailed analysis and methodology are presented in `ecodynamicsAI_description.pptx`.

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
