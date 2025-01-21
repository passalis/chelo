from chelo import DatasetRegistry
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load and prepare the dataset
dataset = DatasetRegistry.get_dataset("CoalFiredPlantDataset")
dataset.load_data()

print("Available features: ", dataset.list_features())
print("Available targets: ", dataset.list_targets())

# Select the first 12 features
dataset.select_features(dataset.list_features()[:12])

# Extract features (X) and target (y)
X, y = dataset.to_numpy()
y = y.reshape(-1)

# By default, boiler efficiency is only used
print("Selected features: ", dataset.selected_features())
print("Selected targets: ", dataset.selected_targets())
print("Features and target shape: ", X.shape, y.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the regressor
regressor = LinearRegression()

# Train the regressor
regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = regressor.predict(X_test)
print(f"R2 score: {r2_score(y_test, y_pred):.5f}")
