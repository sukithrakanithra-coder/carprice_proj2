# Used Vehicle Price Prediction & Flask Web App

## Project Overview
This project predicts used car selling prices using historical vehicle data from CarDekho. It features an end-to-end machine learning pipeline covering Exploratory Data Analysis (EDA), feature engineering, model training/hyperparameter tuning, and a interactive Flask web application for real-time price estimation.

## Dataset Information
- **Source:** CarDekho Dataset (`cardekho_dataset.csv`).
- **Numerical Features:** `vehicle_age`, `km_driven`, `mileage`, `engine`, `max_power`, `seats`, `selling_price`.
- **Categorical Features:** `car_name`, `brand`, `model`, `seller_type`, `fuel_type`, `transmission_type`.
- **Preprocessing:** 
  - Checked and removed missing values.
  - Performed One-Hot Encoding (`pd.get_dummies`) with `drop_first=True` on categorical variables (`fuel_type`, `transmission_type`, `seller_type`).
  - Evaluated feature importance using `ExtraTreesRegressor`.

## Model Implementation
- **Algorithms Evaluated:** Linear Regression, Ridge Regression, Lasso Regression, Support Vector Regressor (SVR), Decision Tree Regressor, Random Forest Regressor, and Extra Trees Regressor.
- **Model Selection & Tuning:** Hyperparameter optimization was performed on the **Random Forest Regressor** using `RandomizedSearchCV` across parameters such as `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features`.
- **Data Split:** 80% Training set, 20% Testing set (`random_state=3`).

## Results & Performance Insights
- **Best Model:** Tuned Random Forest Regressor (`best_random_forest_model.pkl`).
- **Evaluation Metrics:** Evaluated models using Root Mean Squared Error (RMSE) and $R^2$ Score to achieve the highest predictive performance.
- **Key Findings:** `max_power`, `vehicle_age`, and `engine` size were identified among the strongest drivers of vehicle selling price.

## How to Setup and Run

### 1. Requirements
Ensure Python 3.8+ is installed along with the following packages:
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn flask
## Visualizations & Web App Preview

### Exploratory Data Analysis & Model Outputs
![EDA Chart 1](Screenshot%20\(20\).png)
![EDA Chart 2](Screenshot%20\(21\).png)

### Web Application Interface
![Flask App Interface](Screenshot%20\(22\).png)
