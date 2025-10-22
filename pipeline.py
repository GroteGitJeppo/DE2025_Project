import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
import seaborn as sns
# from xgboost import XGBRegressor

def FinalModel(file_path: str) -> np.array:

    """
    file_path: Path to the csv file containing the California Housing dataset.
    
    Returns a numpy array containing the predictions.
    """

    # Import the Dataset
    df = pd.read_csv(file_path)
    df.rename(columns={
    'Median_House_Value': 'house_value',
    'Median_Income': 'income',
    'Median_Age': 'age',
    'Tot_Rooms': 'rooms',
    'Tot_Bedrooms': 'bedrooms',
    'Population': 'population',
    'Households': 'households',
    'Latitude': 'lat',
    'Longitude': 'long',
    'Distance_to_coast': 'coast',
    'Distance_to_LA': 'los_angeles',
    'Distance_to_SanDiego': 'san_diego',
    'Distance_to_SanJose': 'san_jose',
    'Distance_to_SanFrancisco': 'san_francisco'
    }, inplace=True)

    # Create New Features
    # 1. rooms per household
    df['rooms_per_household'] = df['rooms'] / df['households']

    # 2. bedrooms per room
    df['bedrooms_per_room'] = df['bedrooms'] / df['rooms']

    # 3. population per household
    df['population_per_household'] = df['population'] / df['households']

    # 4. distance to nearest major city
    df['distance_to_nearest_city'] = df[['los_angeles', 'san_diego', 'san_jose', 'san_francisco']].min(axis=1)

    # 5. income per household
    df['income_per_household'] = (df['income'] * 10000) / df['households']

    # 6. proximity to coastal areas (is_coastal binary feature)
    df['is_coastal'] = df['coast'].apply(lambda x: 1 if x < 50000 else 0)

    # 7. house age category (binning age)
    df['house_age_category'] = pd.cut(df['age'], bins=[0, 10, 30, 60], labels=['new', 'moderate', 'old'])

    # 8. weighted distance to cities
    df['weighted_distance_to_cities'] = (df['los_angeles'] * 0.4 + df['san_diego'] * 0.3 + df['san_jose'] * 0.2 + df['san_francisco'] * 0.1)

    # 9. geographic clusters using latitude and longitude (KMeans clustering)
    kmeans = KMeans(n_clusters=5)
    df['geo_cluster'] = kmeans.fit_predict(df[['lat', 'long']])

    # 10. urban vs rural (is_urban binary feature)
    df['is_urban'] = df['population'].apply(lambda x: 1 if x > 1500 else 0)

    # 11. income disparity index
    df['income_disparity'] = df['income'] - df['income'].mean()

    # 12. house age per income
    df['house_age_per_income'] = df['age'] / df['income']

    # 13. distance to city per population ratio
    df['distance_to_city_per_population'] = df['distance_to_nearest_city'] / df['population']


    # List of relevant columns for capping
    columns_to_cap = [
        'rooms', 
        'bedrooms', 
        'population', 
        'households', 
        'coast', 
        'rooms_per_household', 
        'bedrooms_per_room', 
        'population_per_household', 
        'distance_to_nearest_city', 
        'income_per_household', 
        'distance_to_city_per_population'
    ]

    # Define the lower and upper percentiles for capping
    lower_percentile = 0.01  # 1st percentile
    upper_percentile = 0.99  # 99th percentile

    # Loop through each column and apply capping
    for col in columns_to_cap:
        # Calculate the lower and upper bounds
        lower_limit = df[col].quantile(lower_percentile)
        upper_limit = df[col].quantile(upper_percentile)
        
        # Apply capping to the column
        df[col] = np.where(df[col] < lower_limit, lower_limit, df[col])
        df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])


    # List of categorical variables
    categorical_variables = [
        'is_coastal',
        'house_age_category',
        'geo_cluster',
        'is_urban'
    ]

    # Perform one-hot encoding using pandas' get_dummies function
    df_encoded = pd.get_dummies(df, columns=categorical_variables, drop_first=True)

    # List of continuous variables to scale or normalize
    continuous_variables = [
        'rooms', 
        'bedrooms', 
        'population', 
        'households', 
        'coast', 
        'rooms_per_household', 
        'bedrooms_per_room', 
        'population_per_household', 
        'distance_to_nearest_city', 
        'income_per_household', 
        'distance_to_city_per_population'
    ]

    # Initialize the scalers
    scaler = StandardScaler() 

    # Apply scaling to continuous variables
    df_encoded[continuous_variables] = scaler.fit_transform(df_encoded[continuous_variables])

    # Turn df_encoded into csv and read it back to ensure all transformations are applied correctly
    df_encoded.to_csv('California_Houses_Processed.csv', index=False)


    # X = df_encoded.drop('house_value', axis=1)
    # y = df_encoded['house_value']

    # # Train-test split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Extracting Top 15 Features Based on Importance

    # # Train XGBoost Model to get feature importance
    # xgb_model = XGBRegressor(n_estimators=100, random_state=42)
    # xgb_model.fit(X_train, y_train)
    # xgb_importances = xgb_model.feature_importances_
    # xgb_indices = np.argsort(xgb_importances)[::-1][:15]  # Top 15 features for XGBoost
    # top_15_features_xgb = X_train.columns[xgb_indices]

    # # Train-Test Split with Top 15 Features

    # X_train_xgb = X_train[top_15_features_xgb]
    # X_test_xgb = X_test[top_15_features_xgb]

    # # Grid Search for XGBoost with Top 15 Features

    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [3, 5, 7],
    #     'learning_rate': [0.01, 0.1, 0.3],
    #     'subsample': [0.8, 1]
    # }

    # grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
    # grid_search.fit(X_train_xgb, y_train)

    # # Best XGBoost model after grid search
    # best_xgb_model = grid_search.best_estimator_

    # # XGBoost with top 15 features
    # y_pred_xgb = best_xgb_model.predict(X_test_xgb)

    # return y_pred_xgb

print(FinalModel('California_Houses.csv'))
