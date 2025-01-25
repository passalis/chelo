from sklearn.neighbors import KNeighborsRegressor
from chelo import DatasetRegistry
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import datamol as dm
import pandas as pd
import numpy as np

# Load and prepare the dataset
dataset = DatasetRegistry.get_dataset("VaporPressureDataset", selected_targets=['p_vap'],
                                      selected_features=['SMILES', 'T'])
dataset.load_data()


# Extract features (X) and target (y)
X, y = dataset.to_numpy()
y = y.reshape(-1)

# Extract temperature
T  = X[:, 1]
# Before fitting a model, we need to extract a fingerprint from the SMILES representation
df = pd.DataFrame(X[:, 0], columns=['SMILES'])
df['Mol'] = df['SMILES'].apply(dm.to_mol)
X_fingerprints = np.vstack(df['Mol'].apply(lambda x:dm.to_fp(x, fp_type='topological')).tolist())
# Stack temperature and fingerprints
X = np.zeros((X.shape[0], X_fingerprints.shape[1] + 1))
X[:, 1:] = X_fingerprints
X[:, 0] = T

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the KNN regressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")


