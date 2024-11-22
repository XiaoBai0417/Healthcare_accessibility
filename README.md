# An unexpectedly large population faces barriers to accessible healthcare in China
This project automates the process of generating travel-time data and training machine learning models to analyze relationships between geospatial features and predicted outcomes. The workflow involves the following main components:

1. API URL Generation and Data Collection
The generate_urls function creates API URLs for driving directions using raster-based population data and healthcare facility locations from a shapefile. It calculates the nearest facility for each valid raster cell and generates URLs for route calculations.
The post_urls function sends requests to the generated URLs, retrieves travel-time data (distance and duration), and saves the results for further analysis.
2. Machine Learning Model Training
The ml_model function trains a machine learning model (XGBoost) to predict outcomes based on features extracted from the data. It includes:
Hyperparameter Tuning: Uses GridSearchCV to find the best parameters for the XGBoost model.
Model Evaluation: Calculates evaluation metrics (MSE, RMSE, MAE, R²) for both training and test datasets.
Visualization:
Compares training and testing performance using bar plots.
Creates scatter plots for true vs. predicted values with regression lines and R² annotations.
SHAP Analysis: Explains feature importance using SHAP (SHapley Additive exPlanations) and visualizes the results in a summary plot.
3. Visualization Outputs
The project generates several key plots:
Metrics Comparison: A bar chart comparing training and testing performance metrics.
True vs. Predicted: A scatter plot comparing true values and predictions for both training and testing sets.
SHAP Summary: A SHAP feature importance plot, showing how each feature contributes to the predictions.
4. Output Files
Travel-time data is saved to output_urls.txt and processed results to post_output_urls.txt.
Model and evaluation outputs:
best_xgb_model.pkl: The saved machine learning model.
Visualizations: PDF and PNG files for metrics comparison, true vs. predicted values, and SHAP feature importance.
