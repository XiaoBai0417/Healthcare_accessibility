# An unexpectedly large population faces barriers to accessible healthcare in China
This project automates the process of generating travel-time data from geospatial datasets and training machine learning models to analyze relationships between geographic features and predicted outcomes.

Key Features
1. API URL Generation and Data Collection
generate_urls:
Generates driving direction API URLs based on population raster data and healthcare facility locations.
Calculates the nearest healthcare facility for each valid raster cell.
post_urls:
Sends requests to the generated URLs using the AMap API.
Extracts travel-time data (distance and duration) and saves the results for further analysis.
2. Machine Learning Model Training and Analysis
ml_model:
Trains an XGBoost regression model to predict outcomes based on geospatial features.
Includes hyperparameter tuning using GridSearchCV.
Evaluates model performance using metrics (MSE, RMSE, MAE, R²) for training and testing sets.
Visualizes results with:
Bar plots comparing model performance on training and testing data.
Scatter plots of true vs. predicted values with regression lines and R² annotations.
SHAP (SHapley Additive exPlanations) feature importance plots to explain model predictions.
3. Visualization Outputs
The following visualizations are generated:

Metrics Comparison: A bar chart comparing MSE, RMSE, MAE, and R² for training and testing sets.
True vs. Predicted: A scatter plot showing true values vs. predictions with regression lines.
SHAP Summary: A plot highlighting feature contributions to the model's predictions.
4. Outputs
Travel-Time Data:
API URLs: output_urls.txt.
Processed results: post_output_urls.txt.
Machine Learning Outputs:
Trained model: best_xgb_model.pkl.
Visualizations: Saved as both PDF and PNG files.
