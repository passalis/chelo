from sklearn.linear_model import Ridge
from chelo import DatasetRegistry
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score


def load_data(use_future_weather=False):

    features = [#'temperature', 'radiation_direct_horizontal', 'radiation_diffuse_horizontal',
                'solar_generation_actual']

    # Load and prepare the dataset
    train_dataset = DatasetRegistry.get_dataset("OPSDPVDataset", end_date='2018-12-31 00:00:00',
                                                selected_features=features,
                                                use_future_weather=use_future_weather)
    train_dataset.load_data()

    # Extract features (X) and target (y)
    X_train, y_train = train_dataset.to_numpy()

    # Flatten the past window
    X_train = X_train.reshape(X_train.shape[0], -1)
    print(X_train.shape)
    # Do the same for the test split
    test_dataset = DatasetRegistry.get_dataset("OPSDPVDataset", start_date='2019-1-1 00:00:00',
                                               selected_features=features,
                                               use_future_weather=use_future_weather)
    test_dataset.load_data()
    X_test, y_test = test_dataset.to_numpy()
    X_test = X_test.reshape(X_test.shape[0], -1)
    scaler_features = MinMaxScaler()
    scaler_targets = MinMaxScaler()

    X_train = scaler_features.fit_transform(X_train)
    X_test = scaler_features.fit_transform(X_test)
    y_train = scaler_targets.fit_transform(y_train)
    y_test = scaler_targets.fit_transform(y_test)

    return X_train, y_train, X_test, y_test


print("Without future weather features:")

X_train, y_train, X_test, y_test = load_data()

# Train the linear regression model
model = Ridge(alpha=10000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display evaluation metrics
print(f"Mean Squared Error: {mse:.7f}")
print(f"R² Score: {r2:.2f}")

# Repeat using weather forecasts/true weather

print("With future weather features:")
X_train, y_train, X_test, y_test = load_data(use_future_weather=True)

# Train the linear regression model
model = Ridge(alpha=10000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display evaluation metrics
print(f"Mean Squared Error: {mse:.7f}")
print(f"R² Score: {r2:.2f}")
