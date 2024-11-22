from osgeo import gdal, ogr
import math
import json
import requests
import time

def generate_urls(tif_path, shp_path, out_txt_urls):
    """
    Generate API URLs for driving directions between population grid centers and healthcare facilities.

    Parameters:
    tif_path (str): Path to the population raster file.
    shp_path (str): Path to the healthcare facility shapefile.
    out_txt_urls (str): Output file for storing generated API URLs.
    """

    # Open the TIFF file
    dataset = gdal.Open(tif_path)
    geotransform = dataset.GetGeoTransform()  # Get geotransformation information
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()  # Read pixel values
    rows, cols = array.shape

    # Open the shapefile
    shapefile = ogr.Open(shp_path)
    layer = shapefile.GetLayer()

    # Function: Convert pixel indices to geographic coordinates
    def pixel_to_coords(row, col, geotransform):
        x = geotransform[0] + col * geotransform[1] + row * geotransform[2]
        y = geotransform[3] + col * geotransform[4] + row * geotransform[5]
        return x, y

    # Function: Calculate Euclidean distance between two points
    def calculate_distance(lon1, lat1, lon2, lat2):
        return math.sqrt((lon1 - lon2) ** 2 + (lat1 - lat2) ** 2)

    # Loop through each pixel in the raster
    with open(out_txt_urls, "w") as f:
        for row in range(rows):
            for col in range(cols):
                # Skip invalid pixels
                if array[row, col] <= 0:
                    continue
                if array[row, col] > 10000000:  
                    continue

                # Get pixel coordinates
                lon, lat = pixel_to_coords(row, col, geotransform)

                # Find the nearest healthcare facility
                nearest_distance = float("inf")
                nearest_point = None
                for feature in layer:
                    geom = feature.GetGeometryRef()
                    point_lon, point_lat = geom.GetX(), geom.GetY()
                    distance = calculate_distance(lon, lat, point_lon, point_lat)
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_point = (point_lon, point_lat)
                layer.ResetReading()  # Reset the layer for the next iteration

                # Construct the API URL
                if nearest_point:
                    dest_lon, dest_lat = nearest_point
                    api_url = f"https://restapi.amap.com/v3/direction/driving?origin={lon},{lat}&destination={dest_lon},{dest_lat}&key=0d1daa20ab3f09e97b2d705ac17a98f9"
                    f.write(api_url + "\n")

def post_urls(url_txt_filename, out_txt_filename):
    """
    Send API requests for the generated URLs and extract route information.

    Parameters:
    url_txt_filename (str): Input file containing API URLs.
    out_txt_filename (str): Output file for storing the results.
    """

    # Read URLs from the input file
    links = []
    with open(url_txt_filename, 'r', encoding='utf-8') as file:
        for line in file:
            links.append(line.strip())

    i = 0
    results = []
    for link in links:
        try:
            # Send the GET request
            response = requests.get(link, timeout=10)  # Set timeout to 10 seconds
            data = json.loads(response.text)
            status = data.get("status")
            if int(status) == 1:
                paths = data.get("route", {}).get("paths", [])
                if paths:
                    distance = paths[0].get("distance")
                    duration = paths[0].get("duration")
                    out_result = distance + "," + duration + "," + link
                    results.append(out_result + "\n")
                    i += 1

                    if i % 10 == 0:
                        print(f"Processed {i} URLs.")
            else:
                print("No paths found in the response.")

        except requests.RequestException as e:
            print(f"Request exception: {link}, Error: {e}")
        time.sleep(0.2)  # Add delay to avoid API rate limits

    # Write results to the output file
    with open(out_txt_filename, "w", encoding="utf-8") as file:
        file.writelines(results)

# File paths
tif_path = "./pop_filename.tif"
shp_path = "./healthcare_pois.shp"
out_txt_urls = "./output_urls.txt"
url_txt_filename = "./output_urls.txt"
out_txt_filename = "./post_output_urls.txt"

# Generate API URLs
generate_urls(tif_path, shp_path, out_txt_urls)

# Post URLs to API and extract results
post_urls(url_txt_filename, out_txt_filename)


def ml_model(filename):
    from sklearn import metrics
    import numpy as np
    import matplotlib.pyplot as plt
    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import pandas as pd
    import joblib
    import seaborn as sns
    import shap

    # Set global plot style
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False

    # Read the dataset
    df = pd.read_csv('./driver_analyse.csv')

    # Define target variable and features
    y = df.iloc[:, 2:3]  # Target variable (third column)
    X = df.iloc[:, 3:]   # Features (columns from the fourth onward)

    # Split dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define hyperparameter grid for XGBoost
    param_grid = {
        'n_estimators': np.arange(10, 201, 10).tolist(),   # Number of trees
        'max_depth': [3, 5, 7, 9, 11],                    # Maximum tree depth
        'learning_rate': [0.01, 0.05, 0.1, 0.15],         # Learning rate
        'subsample': [0.7, 0.8, 0.9, 1.0],                # Subsample ratio of training instances
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],         # Subsample ratio of features for tree
        'reg_alpha': [0, 0.1, 1, 10],                     # L1 regularization term
        'reg_lambda': [0.1, 1, 10, 100]                   # L2 regularization term
    }

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                            cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_
    joblib.dump(model, r'./best_xgb_model.pkl')
    model = joblib.load(r'./best_xgb_model.pkl')

    # Predict on training and testing sets
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate metrics
    def calculate_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return mse, rmse, mae, r2

    # Metrics for training and testing sets
    train_metrics = calculate_metrics(y_train, y_pred_train)
    test_metrics = calculate_metrics(y_test, y_pred_test)

    # Bar plot for metrics comparison
    def plot_metrics_comparison(train_metrics, test_metrics):
        metrics_labels = ['MSE', 'RMSE', 'MAE', 'R-squared']
        x = np.arange(len(metrics_labels))  # x-axis positions
        width = 0.35  # Bar width

        fig, ax = plt.subplots()

        # Training and testing bars
        bars_train = ax.bar(x - width/2, train_metrics, width, label='Train')
        bars_test = ax.bar(x + width/2, test_metrics, width, label='Test')

        # Add labels, title, and legend
        ax.set_ylabel('Scores')
        ax.set_title('Comparison of Train and Test Set Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_labels)
        ax.legend()

        # Annotate bars
        def annotate_bars(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}', 
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

        annotate_bars(bars_train)
        annotate_bars(bars_test)

        fig.tight_layout()
        plt.savefig("Comparison_of_Train_and_Test_Set_Metrics.pdf", format='pdf', bbox_inches='tight')
        plt.show()

    plot_metrics_comparison(train_metrics, test_metrics)

    # Scatter plot for true vs. predicted values
    def plot_true_vs_predicted(y_train, y_pred_train, y_test, y_pred_test):
        data_train = pd.DataFrame({'True': y_train.to_numpy().flatten(), 'Predicted': y_pred_train, 'Dataset': 'Train'})
        data_test = pd.DataFrame({'True': y_test.to_numpy().flatten(), 'Predicted': y_pred_test, 'Dataset': 'Test'})
        data = pd.concat([data_train, data_test])

        # Filter extreme values for better visualization
        data = data[(data['True'] <= 100) & (data['Predicted'] <= 100)]

        # Define a custom color palette
        palette = {'Train': '#b4d4e1', 'Test': '#f4ba8a'}

        plt.figure(figsize=(8, 6), dpi=1200)
        g = sns.JointGrid(data=data, x="True", y="Predicted", hue="Dataset", height=10, palette=palette)
        g.plot_joint(sns.scatterplot, alpha=0.5)

        # Regression lines
        sns.regplot(data=data_train, x="True", y="Predicted", scatter=False, ax=g.ax_joint, color='#b4d4e1', label='Train Regression Line')
        sns.regplot(data=data_test, x="True", y="Predicted", scatter=False, ax=g.ax_joint, color='#f4ba8a', label='Test Regression Line')

        # Marginal histograms
        g.plot_marginals(sns.histplot, kde=False, element='bars', multiple='stack', alpha=0.5)

        # Add R^2 text
        ax = g.ax_joint
        ax.text(0.95, 0.1, f'Train $R^2$ = {train_metrics[3]:.3f}', transform=ax.transAxes, fontsize=12,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
        ax.text(0.95, 0.05, f'Test $R^2$ = {test_metrics[3]:.3f}', transform=ax.transAxes, fontsize=12,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

        # Add x=y line
        ax.plot([data['True'].min(), data['True'].max()], [data['True'].min(), data['True'].max()], c="black", alpha=0.5, linestyle='--', label='x=y')
        ax.legend()
        plt.savefig("True_vs_Predicted.pdf", format='pdf', bbox_inches='tight')
        plt.show()

    plot_true_vs_predicted(y_train, y_pred_train, y_test, y_pred_test)

    # SHAP analysis
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    # SHAP summary plot
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("SHAP_Feature_Importance.pdf", format='pdf', bbox_inches='tight')
    plt.show()

ml_model("path_to_dataset.csv")

