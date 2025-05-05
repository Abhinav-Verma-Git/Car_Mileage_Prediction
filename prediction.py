import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

# Load data
data = pd.read_csv("mpg.csv")
print(data.head())
print(data.info())

# Visualization 
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

# Features and Target
features = data[["Horsepower", "Weight"]]
target = data["MPG"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Customized XGBoost Model
param_grid = {
    'n_estimators': [50, 75, 98, 120],
    'max_depth': [2, 3, 4],
    'learning_rate': [0.05, 0.11]
}

model= XGBRegressor()
gridsearch=GridSearchCV(estimator=model,param_grid=param_grid,cv=5,scoring='neg_mean_squared_error',verbose=1)
gridsearch.fit(x_train,y_train)
print("Best Hyperparameters:", gridsearch.best_params_)
y_pred = gridsearch.best_estimator_.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("RMSE:",np.sqrt(mse))
print("r2_score:",r2_score(y_test,y_pred))

# Example Prediction
new_pred=gridsearch.predict([[100,2000]])[0]
print("Predicted Average MPG:",round(new_pred,2))

# Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.title('Actual vs Predicted MPG')
plt.grid(True)
plt.show()

joblib.dump(gridsearch,"model.pkl")





