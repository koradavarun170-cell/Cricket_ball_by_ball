# 🏏 IPL Ball-by-Ball Boundary Prediction (Virat Kohli Edition)

An end-to-end Machine Learning project and interactive live delivery simulator predicting ball-by-ball boundary occurrences (fours and sixes) for **Virat Kohli** in the Indian Premier League (IPL) up to 2023.

---

## 📁 Repository Structure

```
IPL/
├── app.py                             # 🌟 Interactive Streamlit Web Simulator App
├── presentation.html                  # 🖥️ Interactive Presentation Slide Deck (Keynote style)
├── PROJECT_JOURNEY.md                 # 📄 Complete In-Depth Documentation & Research Story
│
├── 01_initial_exploration.ipynb       # Step 1: Initial exploration & baseline Random Forest + ROS/RUS
├── 02_model_benchmarking.ipynb        # Step 2: Feature engineering, Classifiers (LogReg, DT, RF, XGB) & Resampling (SMOTE)
├── 03_deep_learning_mlp.ipynb         # Step 3: PyTorch Deep Learning MLP architecture
├── 04_ensemble_model.ipynb            # Step 4: The Final Hybrid Probability Ensemble (Best F1: 0.3934)
│
├── src/                               # Clean shared utilities
│   ├── bowler_mapping.py              # Centralized 296-bowler mapping (pace vs spin)
│   ├── data_loader.py                 # Data loading & filtering helpers
│   ├── feature_engineering.py         # Momentum & context rolling feature definitions
│   ├── preprocessor.py                # StandardScaler + OneHotEncoder ColumnTransformers
│   ├── metrics.py                     # Evaluation metrics & threshold tuning utilities
│   └── train_and_save_model.py        # Model serialization script for production serving
│
├── models/                            # Serialized production model artifacts
│   ├── preprocessor.joblib            # Fitted ColumnTransformer
│   ├── logistic_model.joblib          # Trained BorderlineSMOTE Logistic Regression
│   ├── catboost_model.cbm             # Trained Tuned CatBoost model
│   └── config.json                    # Blending weights & optimal threshold configuration
│
├── charts/                            # High-resolution presentation visuals
│   ├── 01_class_imbalance.png
│   ├── 02_model_evolution.png
│   ├── 03_precision_recall_tradeoff.png
│   ├── 04_ensemble_confusion_matrix.png
│   └── 05_catboost_feature_importance.png
│
├── archive_legacy/                    # Archived original exploratory files
│   ├── Untitled.ipynb
│   ├── logistic.ipynb
│   ├── IPL-dl.ipynb
│   └── Ensembleipl.ipynb
│
├── requirements.txt                   # Dependency specifications
├── IPL.csv                            # Full historical dataset (~107 MB)
├── IPL_bowler_categories.csv          # Bowler category dataset
└── New_Ipl.csv                        # Filtered Virat Kohli records
```

---

## ⚡ Quickstart: Launch the Interactive App

Run the live delivery boundary simulator locally:

```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser to test different match scenarios, progressive strike rates, boundary drought counters, and bowler matchups!

---

## 🏆 Project Best Model Highlights

* **Model**: Hybrid Probability Ensemble
  $$P_{\text{ensemble}} = 0.30 \times P_{\text{Logistic (BorderlineSMOTE)}} + 0.70 \times P_{\text{CatBoost}}$$
* **Decision Threshold**: **0.550** (optimized via 2D grid search)
* **Performance Metrics**:
  * **Overall Accuracy**: **76.51%**
  * **Boundary Recall**: **52.17%** (catches more than half of all boundaries)
  * **Precision**: **31.58%** (>2.1x better than random baseline)
  * **F1 Score**: **0.3934** (+77.2% improvement over baseline)
  * **ROC-AUC**: **0.6681** | **PR-AUC**: **0.3030**
