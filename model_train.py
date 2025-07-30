# model_train.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib


df = pd.read_csv("employee_salary_dataset_600.csv")


X = df.drop("Salary", axis=1)
y = df["Salary"]


cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()


preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', SimpleImputer(strategy='mean'), num_cols)
])


model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model.fit(X_train, y_train)


joblib.dump(model, "salary_pipeline.pkl")


preds = model.predict(X_test)
print("✅ Model Trained")
print("MSE:", mean_squared_error(y_test, preds))
print("R² Score:", r2_score(y_test, preds))
