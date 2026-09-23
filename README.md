# Automated Home Loan Approval System

# **Project Overview**

**Objective:** Build and compare interpretable machine learning models to predict home loan approval (`Loan_Status: Y/N`) using historical applicant data.

**Dataset:** 592 loan applications with 13 features (demographics, financials, credit history, property details).

**Models:** CART (Decision Tree) & Logistic Regression.

**Performance:** Both models achieved **83.15% test accuracy** with strong generalization.

**Business Value:** Provides a transparent, data-driven framework for automating loan screening while maintaining regulatory compliance through model interpretability.

---

# **End-to-End Thought Process & Methodology**

### **1. Problem Scoping & Exploratory Data Analysis**

- **Question:** Can we reliably predict loan approval using structured applicant data?
- **Initial Checks:** Loaded the dataset, and identified 13 columns. `Loan_Status` is binary.
- **Missing Values:** Found gaps in `Gender`, `Married`, `Dependents`, `Self_Employed`, `Loan_Amount_Term`, and `Credit_Score`. Decided to impute rather than drop rows to preserve sample size (592 → 592).
- **Distribution Insight:** Target is imbalanced (~70% Approved, ~30% Rejected). Noted that `ApplicantIncome` is heavily right-skewed and `Credit_Score` is binary (0/1).

### **2. Data Cleaning & Feature Engineering**

- **Irrelevant Feature Removal:** Dropped `Loan_ID` as it is of zero predictive power.
- **Imputation Strategy:**
    - Categorical → Mode imputation (preserves majority class distribution)
    - Numerical → Median imputation (robust to income outliers like $81,000)
- **Encoding:** Applied `LabelEncoder` to all categorical features (`Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `Property_Area`, `Loan_Status`). Chose label encoding over one-hot due to low cardinality and tree/linear model compatibility.
- **Train/Test Split:** 70/30 stratified split (`random_state=42`) to ensure consistent evaluation and prevent data leakage.

<img width="237" height="260" alt="image" src="https://github.com/user-attachments/assets/30a8f714-c096-4303-be4d-20bca9a4082a" />
<img width="262" height="244" alt="image" src="https://github.com/user-attachments/assets/12197b11-1018-4cba-a4e3-5b5777a8d41e" />


### **3. Model Selection & Training**

- **CART (Decision Tree):**
    - Set `max_depth=4` intentionally to limit complexity, improve interpretability, and prevent overfitting.
    - Resulted in a compact 12-leaf tree.
- **Logistic Regression:**
    - Chosen for its probabilistic outputs, linear decision boundary, and highly interpretable coefficients.
    - Set `max_iter=1000` to ensure convergence.
- **Why These Two?** They represent two ends of the ML spectrum: non-linear rule-based logic (CART) vs. linear probabilistic modeling (LogReg). Both are highly valued in regulated industries like banking.

<img width="1570" height="812" alt="image" src="https://github.com/user-attachments/assets/641a52cd-d9c2-4c87-9066-498c9cd1c75f" />

### **4. Evaluation & Business Interpretation**

- **Accuracy:** Both models hit **83.15%** on the test set.
- **Coefficient Analysis (LogReg):** `Credit_Score` had the highest positive coefficient (3.07), confirming it as the strongest approval driver. `Married` and `Gender` showed mild positive influence. `Education` and `Property_Area` had negative weights, hinting at nuanced risk factors.
- **Overfitting Check:** Training vs. Testing accuracies were nearly identical. The shallow tree depth and linear model structure naturally regularized the learning process. No overfitting observed.

<img width="489" height="64" alt="image" src="https://github.com/user-attachments/assets/b9c9c5a2-5803-434f-b8ba-888f33400885" />
<img width="277" height="377" alt="image" src="https://github.com/user-attachments/assets/1dc028d4-ce2a-4b92-8c3a-6f15dc536c08" />

### **5. Strategic Recommendations & Next Steps**

- Recommended **Logistic Regression for pilot deployment** due to regulatory transparency (coefficients map directly to business logic).
- Data enrichment (debt-to-income, employment tenure), class-imbalance handling (SMOTE/class weights), and continuous model monitoring for concept drift.

---

# **Key Results & Business Insights**

| Metric | CART | Logistic Regression |
| --- | --- | --- |
| Test Accuracy | 83.15% | 83.15% |
| Model Complexity | 12 leaf nodes (max_depth=4) | 11 coefficients |
| Interpretability | High (visual decision paths) | Very High (transparent coefficients) |
| Top Predictor | `Credit_Score` | `Credit_Score` |

**Key Business Insight:** Creditworthiness outweighs raw income in this dataset. A strong credit history can compensate for moderate income, while high income alone doesn't guarantee approval if credit or demographic risk factors are present.

---

# **Key Takeaways & Skills Matrix**

### **Technical Skills Applied & Refined**

- **End-to-End ML Pipeline:** Data ingestion → Exploratory Data Analysis→ Cleaning → Encoding → Train/Test split → Modeling → Evaluation → Interpretation.
- **Feature Engineering & Imputation:** Strategic handling of missing values (median vs. mode), outlier awareness, and categorical encoding.
- **Model Training & Tuning:** Decision Tree pruning (`max_depth`), Logistic Regression convergence management, cross-validation awareness.
- **Performance Evaluation:** Accuracy tracking, coefficient analysis, overfitting diagnostics, and generalization validation.
- **Data Visualization:** Decision tree plotting, coefficient ranking, distribution analysis.

### **Analytical & Strategic Skills Developed**

- **Problem Scoping:** Translated a business question  into a structured classification task.
- **Bias & Imbalance Awareness:** Recognized 70/30 class distribution and proposed mitigation strategies for production readiness.
- **Model Interpretability for Compliance:** Prioritized transparent models to meet banking regulatory standards.
- **MLOps Mindset:** Identified next steps for production: data enrichment, concept drift monitoring, and continuous retraining pipelines.
