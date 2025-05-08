# EcoDynamicsAI

EcoDynamicsAI is a comprehensive framework designed to model and analyze plant-pathogen-microbe interactions using advanced data preprocessing, feature selection, and machine learning approaches. It leverages methods like Recursive Feature Selection, Conditional Independence Testing, Polynomial Expansions, and Ensemble Learning to predict microbial richness and diversity in plant-pathogen systems.

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

---

## ✅ Installation

1. Clone the repository:
```bash
git clone https://github.com/your_username/EcoDynamicsAI.git
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

## 📈 Results and Analysis

- Model results and evaluation metrics are stored in `models2/full/` and `models2/top20/`.
- A consolidated summary table is located at `models2/model_results.csv`.
- Detailed analysis and methodology are presented in `ecodynamicsAI_description.pptx`.

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
