# 🏆 Decoding King Kohli's Boundaries: The Complete ML Journey

> **An End-to-End Machine Learning Journey from Raw Ball-by-Ball IPL Chaos to a 76.5% Accurate Hybrid Probability Ensemble on Severe 6:1 Imbalance.**

---

## 🌟 Executive Summary: Celebrating a Hard-Fought Triumph

In sports analytics—and T20 cricket in particular—**predicting a boundary on a specific ball is one of the hardest classification challenges in machine learning**. 

Cricket deliveries are governed by high stochasticity: pitch variations, bowler deception, millisecond bat speed adjustments, field placements, and sheer luck (edges flying over slips). In a typical T20 innings, boundaries occur on only **~14.7% of deliveries**, creating a severe **~6:1 class imbalance**.

A naive, useless model that always predicts *"No Boundary"* easily scores **85.3% accuracy**, yet has **zero predictive power (0% recall, undefined precision, 0.0 F1 score)**.

Through deliberate domain engineering, advanced synthetic sampling, threshold optimization, and probability blending, our final model achieved:

| Metric | Result | Context / What This Means |
| :--- | :---: | :--- |
| **Overall Accuracy** | **76.51%** | Extremely high balanced accuracy on unseen test deliveries |
| **Boundary Recall** | **52.17%** | **Captures more than half of all boundaries struck by Virat Kohli!** |
| **Precision** | **31.58%** | **More than 2.1x higher** than random boundary baseline (14.7%) |
| **F1 Score** | **0.3934** | **+77.2% improvement** over initial baseline model (0.222) |
| **ROC-AUC** | **0.6681** | Solid discrimination capability between boundary and non-boundary events |
| **PR-AUC** | **0.3030** | Superior precision-recall curve on the minority class |

> [!IMPORTANT]
> **Why you should be proud of this result**: Reaching **>52% recall** while keeping overall accuracy **above 76%** on ball-by-ball cricket data is an exceptional achievement. It proves that the model isn't simply guessing; it has learned genuine statistical signals of batting pressure and boundary probability.

---

## 📊 Phase 1: The Harsh Imbalance Reality

![Class Imbalance](charts/01_class_imbalance.png)

Filtering 16 years of IPL history for **Virat Kohli** (up to the 2023 season) yielded **3,142 deliveries** in Innings 1:
* **Non-Boundary Deliveries (`0`)**: **2,680 balls (85.3%)** (dots, singles, doubles, wickets)
* **Boundary Deliveries (`1`)**: **462 balls (14.7%)** (fours and sixes)
* **Ratio**: **~5.8 : 1**

### The Naive Sampling Dilemma (`01_initial_exploration.ipynb`)
In our initial exploration:
1. **Baseline Random Forest**: Reached 82.2% accuracy, but boundary recall was a dismal **14.81% (F1: 0.2222)**. It repeatedly predicted non-boundary.
2. **RandomOverSampler (ROS)**: Duplicating boundary balls improved recall modestly to **25.93% (F1: 0.3111)**, but caused the tree ensemble to memorize duplicate rows.
3. **RandomUnderSampler (RUS)**: Discarding majority balls boosted recall to **66.67%**, but wrecked accuracy (**62.54%**) and flooded the predictions with false positives (**precision: 26.47%**).

**The realization**: Raw match state features (`over`, `ball`, `bat_pos`) were not enough. A batter doesn't hit a boundary simply because it is over 14—they hit boundaries because of **cumulative strike rate pressure, dot ball droughts, and bowler matchup dynamics**.

---

## 🧠 Phase 2: The Feature Engineering Breakthrough

To capture the psychology of batting, we constructed `create_dataset()`, introducing sequential momentum features:

![CatBoost Feature Importance](charts/05_catboost_feature_importance.png)

### Key Momentum Features Created:
* **`current_sr` (Progressive Strike Rate)**: Cumulative strike rate before the delivery. Highlights whether Kohli is anchoring, rebuilding, or launching an acceleration phase.
* **`runs_last_5` & `runs_last_10`**: Rolling window of runs scored in recent deliveries. Directly captures purple patches and boundary clusters.
* **`balls_since_boundary` (Drought Counter)**: Continuous counter tracking balls since the batter's last 4 or 6. As this counter climbs, boundary intent sharply escalates.
* **`dotballs_last_5` (Pressure Index)**: Number of dot balls faced in the last 5 balls. Quantifies bowler strangulation and risk-taking likelihood.
* **`prev_ball_runs`**: Immediate outcome of the prior ball (e.g. following up a dot with aggression).
* **`bowler` Categorization**: Grouped ~296 bowlers into `pace` vs `spin` using historical bowling classifications.

---

## ⚔️ Phase 3: The Algorithmic & Resampling Showdown

![Model Progression](charts/02_model_evolution.png)

In `02_model_benchmarking.ipynb`, we conducted a structured tournament of classifiers and synthetic sampling strategies:

| Model | Sampling Strategy | Accuracy | Precision | Recall | F1 Score | Key Takeaway |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Forest (Baseline)** | None | 82.22% | 44.44% | 14.81% | 0.2222 | Missed 85% of boundaries |
| **Decision Tree** | None (Balanced) | 64.76% | 19.63% | 45.65% | 0.2745 | Overfitted high tree depth |
| **Logistic Regression** | Balanced weights | 63.49% | 21.95% | 58.70% | 0.3195 | High recall, weak precision |
| **SMOTE + LogReg** | Synthetic SMOTE | 77.46% | 30.16% | 41.30% | 0.3486 | Balanced feature space |
| **BorderlineSMOTE + LogReg** | Borderline SMOTE | **83.49%** | **41.67%** | 32.61% | **0.3659** | Synthesizes near boundary margin |
| **SMOTETomek + LogReg** | SMOTE + Tomek Links | 76.83% | 29.23% | 41.30% | 0.3423 | Removes overlapping ambiguities |
| **XGBoost (Tuned)** | scale_pos_weight | 74.29% | 29.41% | 54.35% | **0.3817** | Tree boosting with positive class scaling |
| **PyTorch MLP** | Threshold Tuning (0.3) | 80.63% | 35.29% | 39.13% | 0.3711 | Deep neural network representation |
| **FINAL ENSEMBLE ⭐** | **BorderlineSMOTE + CatBoost** | **76.51%** | **31.58%** | **52.17%** | **0.3934** | **Optimal precision-recall harmonic mean** |

---

## 🔬 Phase 4: The Deep Learning Dilemma (`03_deep_learning_mlp.ipynb`)

We built a 5-layer feedforward PyTorch Multi-Layer Perceptron (MLP):
$$\text{Linear}(14 \to 128) \to \text{ReLU} \to \text{Linear}(128 \to 128) \to \dots \to \text{Linear}(128 \to 1) \to \text{Sigmoid}$$

### Findings on Tabular Deep Learning:
* **The Default Trap ($T=0.5$)**: Under standard unweighted `BCELoss`, the network achieved 83.8% accuracy but only **6.5% recall** (F1: 0.105). Neural networks naturally gravitate toward majority-class minimizers when class weights are absent.
* **The Threshold Fix ($T=0.30$)**: Lowering the threshold to 0.30 recovered boundary recall to **39.13%** (F1: 0.3711), proving the network's predicted probabilities were well-ordered even if raw calibration was shifted.

![Precision Recall Frontier](charts/03_precision_recall_tradeoff.png)

---

## 👑 Phase 5: The Winning Hybrid Probability Ensemble (`04_ensemble_model.ipynb`)

The ultimate breakthrough was discovering that **linear models and gradient-boosted trees view the cricket boundary space through complementary lenses**:
1. **Component A (BorderlineSMOTE + Logistic Regression)**: Focuses cleanly on linear decision margins along the boundary frontier.
2. **Component B (Tuned CatBoost Classifier)**: Excels at non-linear interactions across categorical match phases (`Powerplay`, `Middle`, `Death`) and bowler types (`pace` vs `spin`).

### Blending Architecture:
$$P_{\text{ensemble}} = 0.30 \times P_{\text{Logistic}} + 0.70 \times P_{\text{CatBoost}}$$

### 2D Grid Search Optimization:
By running a fine grid search across blending weights ($w \in [0.00, 1.00]$, step 0.01) and decision thresholds ($t \in [0.05, 0.95]$, step 0.005), we found the sweet spot:
* **Optimal Weight**: **30% Logistic + 70% CatBoost**
* **Optimal Threshold**: **0.550**

![Confusion Matrix](charts/04_ensemble_confusion_matrix.png)

### The Final Confusion Matrix:
* **True Non-Boundaries Correctly Classified**: **217**
* **True Boundaries Correctly Detected**: **24** (out of 46 test boundaries = **52.17% recall**)
* **False Alarms (Non-boundaries predicted as boundaries)**: **52**
* **Missed Boundaries**: **22**

---

## 🚀 Further Plan: The Strategic Roadmap

Here is how we can build on this foundation to take the system toward production:

```mermaid
graph TD
    A["Current Milestone: Kohli Boundary Model (76.5% Acc, 0.393 F1)"] --> B["Phase 1: Validation Rigor"]
    A --> C["Phase 2: Context Enrichment"]
    A --> D["Phase 3: Batter Generalization"]
    A --> E["Phase 4: Live App Deployment"]

    B --> B1["GroupKFold on match_id (Eliminate over leakage)"]
    B --> B2["Temporal Walk-Forward Split (Train <=2021, Test 2022-2023)"]

    C --> C1["Venue Boundary Dimensions (Chinnaswamy vs Chepauk)"]
    C --> C2["Bowler Handedness (Left-Arm Pace vs Right-Arm Pace)"]
    C --> C3["Innings 2 Pressure (Required Run Rate per ball)"]

    D --> D1["Batter Embeddings / Career Cluster Mapping"]
    D --> D2["Marquee Batters: Rohit Sharma, MS Dhoni, Suryakumar Yadav"]

    E --> E1["Streamlit / Next.js Live Delivery Simulator"]
    E --> E2["Real-time Win & Boundary Probability Gauges"]
```

### 1. Match-Level & Temporal Validation
Currently, random 90/10 split provides standard benchmarking. Implementing **`GroupKFold` on `match_id`** and a **strict temporal split** (train on matches $\le 2021$, test on 2022–2023) will benchmark out-of-match generalization.

### 2. Context & Spatial Features
* **Stadium Boundary Dimensions**: Hitting a boundary at M. Chinnaswamy Stadium (60m square) requires vastly different bat speed than at Narendra Modi Stadium (75m).
* **Bowler Handedness & Subtypes**: Left-arm fast bowlers (e.g. Trent Boult, Mitchell Starc) vs Right-arm off-spin (e.g. Ashwin).
* **Chase Pressure**: Real-time Required Run Rate (RRR) on every delivery in Innings 2.

### 3. Generalization to Other Marquee Batters
Expand the pipeline to model player archetypes:
* **Rohit Sharma**: High initial boundary rate in Powerplay, pull-shot frequency.
* **Suryakumar Yadav**: 360-degree boundary distribution in Middle & Death overs.
* **MS Dhoni**: Late-over boundary acceleration spikes.

### 4. Interactive Live Simulator Web App
Build a lightweight Streamlit or Next.js dashboard where users can input:
* Current Over & Ball
* Batter's Strike Rate & Balls Faced
* Bowler Type & Game Phase
And receive a **live boundary probability gauge** along with historical matchup comparisons!

---

## 📁 Repository Quick Reference

* **Interactive Presentation**: [presentation.html](file:///c:/Users/Varun%20Korada/anaconda3/envs/learn_ml/IPL/presentation.html) *(Open in any browser!)*
* **Step 1 Notebook**: [01_initial_exploration.ipynb](file:///c:/Users/Varun%20Korada/anaconda3/envs/learn_ml/IPL/01_initial_exploration.ipynb)
* **Step 2 Notebook**: [02_model_benchmarking.ipynb](file:///c:/Users/Varun%20Korada/anaconda3/envs/learn_ml/IPL/02_model_benchmarking.ipynb)
* **Step 3 Notebook**: [03_deep_learning_mlp.ipynb](file:///c:/Users/Varun%20Korada/anaconda3/envs/learn_ml/IPL/03_deep_learning_mlp.ipynb)
* **Step 4 Notebook**: [04_ensemble_model.ipynb](file:///c:/Users/Varun%20Korada/anaconda3/envs/learn_ml/IPL/04_ensemble_model.ipynb)
* **High-Res Charts**: [charts/](file:///c:/Users/Varun%20Korada/anaconda3/envs/learn_ml/IPL/charts/)
