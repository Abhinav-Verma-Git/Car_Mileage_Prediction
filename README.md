<h1>🚗 MPG Prediction using XGBoost</h1>

<p>This project builds and evaluates a machine learning model to predict a car's <strong>Miles Per Gallon (MPG)</strong> using key features like <em>Horsepower</em> and <em>Weight</em>. The model is trained using <strong>XGBoost Regressor</strong> with hyperparameter tuning via <code>GridSearchCV</code>.</p>

<hr>

<h2>📁 Dataset</h2>
<ul>
  <li><strong>File:</strong> <code>mpg.csv</code></li>
  <li><strong>Target variable:</strong> MPG (Miles Per Gallon)</li>
  <li><strong>Features used:</strong> Horsepower, Weight</li>
</ul>

<hr>

<h2>🛠️ Libraries Used</h2>
<ul>
  <li><code>numpy</code></li>
  <li><code>pandas</code></li>
  <li><code>matplotlib</code>, <code>seaborn</code> for data visualization</li>
  <li><code>scikit-learn</code> for model evaluation and cross-validation</li>
  <li><code>xgboost</code> for regression model</li>
</ul>

<hr>

<h2>📊 Steps Performed</h2>
<ol>
  <li>Load and inspect the dataset</li>
  <li>Visualize feature correlations using a heatmap</li>
  <li>Split the dataset into training and testing sets (80/20)</li>
  <li>Use GridSearchCV to find the best XGBoost hyperparameters</li>
  <li>Evaluate the model using MSE, RMSE, and R² Score</li>
  <li>Make a sample prediction</li>
  <li>Plot Actual vs Predicted MPG values</li>
</ol>

<hr>

<h2>🔍 Hyperparameter Grid</h2>
<pre>
n_estimators: [50, 75, 98, 120]
max_depth: [2, 3, 4]
learning_rate: [0.05, 0.11]
</pre>

<hr>

<h2>📈 Evaluation Metrics</h2>
<ul>
  <li><strong>Mean Squared Error (MSE)</strong></li>
  <li><strong>Root Mean Squared Error (RMSE)</strong></li>
  <li><strong>R² Score</strong></li>
</ul>

<hr>

<h2>✅ Sample Prediction</h2>
<p>Predicting MPG for a car with:</p>
<ul>
  <li><strong>Horsepower:</strong> 100</li>
  <li><strong>Weight:</strong> 2000 lbs</li>
</ul>
<p><strong>Predicted MPG:</strong> ~ (Output varies based on model)</p>

<hr>

<h2>📷 Visual Output</h2>
<p>Scatter plot comparing actual vs predicted MPG values with a red reference line for perfect prediction.</p>

<hr>

<h2>📌 How to Run</h2>
<pre>
   Ensure all dependencies are installed:
  
1. pip install numpy pandas matplotlib seaborn scikit-learn xgboost
2. Place <code>mpg.csv</code> in the same directory as your script.
3. Run the Python script:
   python3 prediction.py
</pre>

<hr>

<h2>📬 Contact</h2>
<p>For queries, feel free to reach out to the project maintainer.</p>

</body>
</html>
