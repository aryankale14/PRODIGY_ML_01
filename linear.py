import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load train and test datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Select relevant features for training and testing
X_train = train_data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']]
y_train = train_data['SalePrice']
X_test = test_data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']]

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the training set
y_train_pred = model.predict(X_train)

# Calculate Mean Squared Error (MSE) on the training set
mse = mean_squared_error(y_train, y_train_pred)
r2 = r2_score(y_train, y_train_pred)

print(f'Mean Squared Error on training set: {mse}')
print(f'R^2 Score on training set: {r2}')

# Make predictions on the test set
test_data['SalePrice'] = model.predict(X_test)

# Create a new file to store predicted sales with their ID
submission = test_data[['Id', 'SalePrice']]
submission.to_csv('submission.csv', index=False)

print("Predictions saved to submission.csv")

