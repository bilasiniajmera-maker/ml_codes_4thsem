import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio = None

    def fit(self, X):
        # 1. Mean Centering
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 2. Covariance Matrix
        cov = np.cov(X_centered.T)

        # 3. Eigenvalues & Eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # 4. Sort by Eigenvalues (Descending)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # 5. Store results
        self.components = eigenvectors[0:self.n_components]
        
        # Calculate how much "information" we kept
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio = [ev / total_var for ev in eigenvalues[:self.n_components]]

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T)

# --- [User Input Section] ---
print("--- Interactive PCA Tool ---")
data_list = []
print("Enter your data points (e.g., '2.5, 3.1, 0.5'). One row at a time.")
print("Type 'done' to finish.")

while True:
    row_input = input("Row: ").lower()
    if row_input == 'done': break
    try:
        data_list.append([float(i) for i in row_input.split(',')])
    except:
        print("Invalid format. Use numbers separated by commas.")

X = np.array(data_list)

if X.shape[1] < 2:
    print("PCA requires at least 2 features (columns) to work!")
else:
    k = int(input(f"Reduce to how many dimensions? (Current: {X.shape[1]}): "))
    
    pca = PCA(n_components=k)
    pca.fit(X)
    X_reduced = pca.transform(X)

    print("\n" + "="*30)
    print("PCA RESULTS")
    print(f"Original Shape: {X.shape}")
    print(f"Reduced Shape:  {X_reduced.shape}")
    print(f"Information Retained: {sum(pca.explained_variance_ratio)*100:.2f}%")
    print("="*30)

    # 6. Plotting (only if reduced to 2D)
    if k == 2:
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c='crimson', edgecolor='white')
        plt.title("2D Projection of your Data")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid