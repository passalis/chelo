from sklearn.neighbors import KNeighborsRegressor
from chelo import DatasetRegistry
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import datamol as dm
import pandas as pd
import numpy as np

# Load and prepare the dataset
dataset = DatasetRegistry.get_dataset(
    "VaporPressureDataset",
    selected_targets=["p_vap"],
    selected_features=["SMILES", "T"]
)
dataset.load_data()

# Extract features (X) and target (y)
X, y = dataset.to_numpy()
y = y.ravel()  # Ensure y is a 1D array

# Extract temperature and SMILES for processing
T = X[:, 1].astype(float)  # Ensure temperature is numeric
df = pd.DataFrame(X[:, 0], columns=["SMILES"])
df["Mol"] = df["SMILES"].apply(dm.to_mol)

# Generate fingerprints from SMILES
X_fingerprints = np.vstack(
    df["Mol"].apply(lambda mol: dm.to_fp(mol, fp_type="topological")).tolist()
)

# Combine temperature and fingerprints into a single feature matrix
X_combined = np.hstack([T.reshape(-1, 1), X_fingerprints])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the KNN regressor
n_neighbors = 5  # Number of neighbors for KNN
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
