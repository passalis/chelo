from chelo import DatasetRegistry
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load and prepare the dataset
dataset = DatasetRegistry.get_dataset("BCFactorDataset")
dataset.load_data()

# Display available features and targets
print("Available features:", dataset.list_features())
print("Available targets:", dataset.list_targets())

# Extract features (X) and target (y)
X, y = dataset.to_numpy()
y = y.ravel()  # Ensure y is a 1D array

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the KNN regressor
n_neighbors = 5  # You can parameterize this for experimentation
knn = KNeighborsRegressor(n_neighbors=n_neighbors)
knn.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display evaluation metrics
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
