from chelo import DatasetRegistry
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load and prepare the dataset
dataset = DatasetRegistry.get_dataset("WineQualityDataset", wine_type="white")
dataset.load_data()
dataset.select_features(['fixed acidity', 'volatile acidity'])

# Extract features (X) and target (y)
X, y = dataset.to_numpy()
y = y.reshape(-1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the regressor
regressor = LinearRegression()

# Train the regressor
regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
