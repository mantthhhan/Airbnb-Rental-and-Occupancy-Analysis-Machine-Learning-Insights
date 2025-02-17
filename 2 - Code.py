# Section 1: Data Loading and Preparation
import pandas as pd

# Load the Airbnb dataset from Kaggle
airbnb_data = pd.read_csv('AB_NYC_2019.csv')

# Check for missing values and anomalies in the data
print(airbnb_data.isnull().sum())

# Filter out any irrelevant columns or rows that are not needed for the analysis
airbnb_data = airbnb_data.drop(['id', 'name', 'host_id', 'host_name', 'last_review'], axis=1)

# Convert data types to appropriate formats
airbnb_data['price'] = airbnb_data['price'].replace('[\$,]', '', regex=True).astype(float)

# Section 2: Exploratory Data Analysis
import seaborn as sns
import matplotlib.pyplot as plt

# Create visualizations to explore the data and gain insights into the factors that affect Airbnb rental prices and occupancy rates in New York City
sns.scatterplot(x='longitude', y='latitude', data=airbnb_data, hue='neighbourhood_group')
plt.show()

# Use descriptive statistics and summary metrics to gain a general understanding of the data
print(airbnb_data.describe())

# Identify any patterns or trends in the data
sns.boxplot(x='neighbourhood_group', y='price', data=airbnb_data)
plt.show()

# Conduct hypothesis testing to identify significant differences between different groups or categories
from scipy.stats import ttest_ind

manhattan_data = airbnb_data.loc[airbnb_data['neighbourhood_group'] == 'Manhattan']
brooklyn_data = airbnb_data.loc[airbnb_data['neighbourhood_group'] == 'Brooklyn']

ttest_result = ttest_ind(manhattan_data['price'], brooklyn_data['price'])
print(ttest_result)

# Section 3: Data Preprocessing and Cleaning
# Handle missing values by imputing or dropping them as appropriate
airbnb_data = airbnb_data.dropna()

# Correct any errors or inconsistencies in the data
airbnb_data['neighbourhood_group'] = airbnb_data['neighbourhood_group'].replace({'Brookyn': 'Brooklyn'})

# Normalize or standardize the data as appropriate
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
airbnb_data[['latitude', 'longitude', 'price']] = scaler.fit_transform(airbnb_data[['latitude', 'longitude', 'price']])

# Address any outliers or extreme values in the data
q1 = airbnb_data['price'].quantile(0.25)
q3 = airbnb_data['price'].quantile(0.75)
iqr = q3 - q1
airbnb_data = airbnb_data[(airbnb_data['price'] >= q1 - 1.5*iqr) & (airbnb_data['price'] <= q3 + 1.5*iqr)]

# Section 4: Feature Engineering and Selection
# Create new features or variables that may be useful for the analysis
airbnb_data['availability_ratio'] = airbnb_data['availability_365'] / 365

# Select the most relevant features or variables for the predictive model
X = airbnb_data[['neighbourhood_group', 'room_type', 'latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_ratio']]
y = airbnb_data['price']

# Use feature scaling or selection techniques as appropriate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression

# One-hot encode categorical variables
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['neighbourhood_group', 'room_type'])], remainder='passthrough')
X = ct.fit_transform(X)

# Feature scaling using standardization
scaler = StandardScaler()
X[:, 6:] = scaler.fit_transform(X[:, 6:])

# Select the k best features using F-test
selector = SelectKBest(score_func=f_regression, k=5)
X = selector.fit_transform(X, y)

# Print the selected feature names
selected_features = ct.get_feature_names(['neighbourhood_group', 'room_type']) + ['latitude', 'longitude', 'minimum_nights', 'availability_ratio']
mask = selector.get_support()  # get the mask of selected features
selected_features = [feature for feature, boolean in zip(selected_features, mask) if boolean]
print('Selected Features:', selected_features)

# Section 5: Predictive Modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_predictions = lr.predict(X_test)
lr_r2 = r2_score(y_test, lr_predictions)
lr_rmse = mean_squared_error(y_test, lr_predictions, squared=False)

# Train a decision tree regression model
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
dt_predictions = dt.predict(X_test)
dt_r2 = r2_score(y_test, dt_predictions)
dt_rmse = mean_squared_error(y_test, dt_predictions, squared=False)

# Train a random forest regression model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)
rf_r2 = r2_score(y_test, rf_predictions)
rf_rmse = mean_squared_error(y_test, rf_predictions, squared=False)

# Print the evaluation metrics
print('Linear Regression - R^2:', lr_r2, 'RMSE:', lr_rmse)
print('Decision Tree Regression - R^2:', dt_r2, 'RMSE:', dt_rmse)
print('Random Forest Regression - R^2:', rf_r2, 'RMSE:', rf_rmse)

# Section 6: Data Visualization and Dashboard Creation
import plotly.express as px

# Create visualizations that illustrate the findings and insights gained from the analysis
fig = px.scatter_mapbox(airbnb_data, lat='latitude', lon='longitude', color='neighbourhood_group', size='price', size_max=15, zoom=10, mapbox_style='carto-positron')
fig.show()

# Use interactive charts and graphs if possible to enable users to explore the data and gain a better understanding of Airbnb rental prices and occupancy rates in New York City
fig = px.scatter(airbnb_data, x='price', y='availability_365', color='neighbourhood_group', hover_data=['room_type', 'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count', 'availability_ratio'])
fig.show()

# Create a dashboard to summarize the findings and insights gained from the analysis
import dash
import dash_html_components as html
import dash_core_components as dcc

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Airbnb Rental Prices and Occupancy Rates in New York City'),
    
    html.Div(children='''
        Predictive models using machine learning algorithms
    '''),

    dcc.Graph(
        id='scatter-map',
        figure=fig
    ),
    
    dcc.Graph(
        id='scatter',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)

# Section 7: Report Generation and Conclusion
# Summarize the findings and insights gained from the analysis in a report
# Provide recommendations for future research and areas for further exploration
# Discuss the limitations of the analysis and possible sources of error in the data

