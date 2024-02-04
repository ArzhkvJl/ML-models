import pandas as pd
import pyarrow
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Path of the file to read
iowa_file_path = 'train.csv'

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(iowa_file_path)
# Print summary statistics
home_data.describe()
# What is the average lot size (rounded to nearest integer)?
avg_lot_size = math.ceil(home_data.describe()['LotArea']['mean'])
print(avg_lot_size)

# As of today, how old is the newest home (current year - the date in which it was built)
newest_home_age = 2024 - home_data.describe()['YearBuilt']['max']
print(newest_home_age)
# print a list of the columns
cols = home_data.columns
print(cols)
# Create the list of features
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath',
                 'BedroomAbvGr', 'TotRmsAbvGrd']
# Select data corresponding to features in feature_names
X = home_data[feature_names]
# target variable, which corresponds to the sales price
y = home_data['SalePrice']
# For model reproducibility, set a numeric value for random_state
# when specifying the model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit the model
iowa_model.fit(X, y)
# Make predictions
predictions = iowa_model.predict(X)
print(predictions)

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
# Fit iowa_model with the training data.
iowa_model.fit(train_X, train_y)

# Predict with all validation observations
val_predictions = iowa_model.predict(val_X)
# check error
val_mae = mean_absolute_error(val_y, val_predictions)
print(val_mae)
