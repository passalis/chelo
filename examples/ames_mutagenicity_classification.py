from chelo import DatasetRegistry
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load and prepare the dataset
dataset = DatasetRegistry.get_dataset("AmesMutagenicityDataset")
dataset.load_data()

print("Available features: ", dataset.list_features())
print("Available targets: ", dataset.list_targets())

# Extract features (X) and target (y)
X, y = dataset.to_numpy()
y = y.reshape(-1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {100*accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

