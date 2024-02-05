import pandas as pd
import pyarrow
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae


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
# a loop that tries the following values for max_leaf_nodes
# from a set of possible values.
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
errors = []
for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    errors.append(my_mae)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" % (max_leaf_nodes, my_mae))
min_error = errors[0]
# Store the best value of max_leaf_nodes
best_tree_size = candidate_max_leaf_nodes[0]
for i in range (len(candidate_max_leaf_nodes)):
    if errors[i] < min_error:
        min_error = errors[i]
        best_tree_size = candidate_max_leaf_nodes[i]
print(best_tree_size)
# Fill in argument to make optimal size and uncomment
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)

# fit the final model and uncomment the next two lines
final_model.fit(X, y)

# Define the model using random forest. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)

# fit your model
rf_model.fit(train_X,train_y)

# Calculate the mean absolute error of your Random Forest model on the validation data
rf_val_mae = mean_absolute_error(val_y, rf_model.predict(val_X))

print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))

