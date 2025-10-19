# Airline Passenger Satisfaction Prediction Project

## 1. Project Summary

This project aims to develop and evaluate machine learning models to predict passenger satisfaction (satisfied vs. dissatisfied) using survey data collected by an airline company. The project encompasses Exploratory Data Analysis (EDA), data preprocessing, training various classification models, and hyperparameter optimization.

## 2. Dataset

The dataset used consists of airline passenger survey data featuring 26 features and one target variable (`satisfaction`).

### Target Variable

* `satisfaction`: Passenger satisfaction (`satisfied` / `neutral or dissatisfied`). (Binary Classification: 1 / 0)

### Key Features (Examples)

* `Age`, `Gender`, `Customer Type`
* `Type of Travel`, `Class`
* Evaluation scores for various services (Inflight Entertainment, Ease of Online Check-in, etc.)
* `Flight Distance`, `Departure Delay in Minutes`, `Arrival Delay in Minutes`

## 3. Methodology and Workflow

1.  **Data Cleaning and EDA:** Identification of missing values (`Arrival Delay in Minutes`) and appropriate imputation (mean imputation).
2.  **Preprocessing:**
    * **One-Hot Encoding** for categorical variables (`OneHotEncoder`).
    * **Scaling** for numerical variables (`StandardScaler`).
3.  **Model Training:** Training and comparison of various models, including Logistic Regression and Ensemble methods (Random Forest, Gradient Boosting).
4.  **Hyperparameter Optimization:** Using `RandomizedSearchCV` to fine-tune the best-performing models to mitigate overfitting and boost performance.
5.  **Final Model Output:** Saving the final model, including all preprocessing steps, as a Scikit-learn **Pipeline**.

## 4. Model Comparison and Results

Models were evaluated using the **ROC AUC (Area Under the Receiver Operating Characteristic Curve)** metric, commonly used in classification problems.

| Model | Test Accuracy | Test ROC AUC Score | Overfitting Risk |
| :--- | :--- | :--- | :--- |
| **Random Forest (Tuned)** | $\mathbf{\approx 0.9660}$ | $\mathbf{\approx 0.9960}$ | Low/Controlled |
| LightGBM (Baseline) | $\approx 0.9637$ | $\approx 0.9950$ | Low |
| XGBoost (Baseline) | $\approx 0.9635$ | $\approx 0.9950$ | Moderate |
| Gradient Boosting (Tuned) | $\approx 0.9590$ | $\approx 0.9600$ | Low |
| Logistic Regression | $\approx 0.8717$ | $\approx 0.9250$ | None |

### Best Model: Tuned Random Forest Classifier

The tuned **Random Forest** model delivered the highest performance, achieving an outstanding **0.9960 Test ROC AUC** score.

## 5. Project Outputs and Usage

The final project output is a saved Scikit-learn **Pipeline** object that contains both the entire preprocessing chain and the final classifier.

### Saved Model File

* `best_satisfaction_model_pipeline.pkl`

### Code to Load and Use the Model

Use the following code block to load the saved model and make predictions on new, un-preprocessed data. The Pipeline automatically handles all necessary preprocessing steps.

```python
import joblib
import pandas as pd

# 1. Load the Model
loaded_pipeline = joblib.load('best_satisfaction_model_pipeline.pkl')

# 2. Load New Data (Example: The raw DataFrame)
# new_data = pd.read_csv("new_passenger_data.csv")

# 3. Make Predictions
# The 'loaded_pipeline' automatically processes the raw data and makes predictions.
# y_pred_proba = loaded_pipeline.predict_proba(new_data)[:, 1]
# y_predictions = loaded_pipeline.predict(new_data)

# print("Predictions successfully generated.")
