class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param # Regularization strength
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        # Convert labels to -1 and 1 (SVM requirement)
        y_transformed = [1 if val > 0 else -1 for val in y]
        n_samples = len(X)
        n_features = len(X[0])

        # Initialize weights and bias
        self.w = [0.0] * n_features
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Condition: y_i * (w * x_i - b) >= 1
                linear_output = sum(self.w[j] * x_i[j] for j in range(n_features)) - self.b
                condition = y_transformed[idx] * linear_output >= 1

                if condition:
                    # Only apply regularization gradient
                    for j in range(n_features):
                        self.w[j] -= self.lr * (2 * self.lambda_param * self.w[j])
                else:
                    # Apply regularization + hinge loss gradient
                    for j in range(n_features):
                        self.w[j] -= self.lr * (2 * self.lambda_param * self.w[j] - x_i[j] * y_transformed[idx])
                    self.b -= self.lr * y_transformed[idx]

    def predict(self, X):
        predictions = []
        for x_i in X:
            approx = sum(self.w[j] * x_i[j] for j in range(len(self.w))) - self.b
            predictions.append(1 if approx >= 0 else 0)
        return predictions

# --- User Input & Testing Section ---
print("--- Manual Linear SVM ---")
X_train = []
print("Enter coordinates (e.g., '1, 2'). Type 'done' to finish.")
while True:
    val = input("Point: ").lower()
    if val == 'done': break
    X_train.append([float(i) for i in val.split(',')])

print("\nEnter labels (0 or 1) for each point, separated by commas:")
y_train = [int(i) for i in input("Labels: ").split(',')]

# Training
model = LinearSVM()
model.fit(X_train, y_train)
print("\nSVM Trained! Weights:", model.w, "Bias:", model.b)

# Prediction
while True:
    test_val = input("\nEnter point to predict (or 'exit'): ").lower()
    if test_val == 'exit': break
    test_point = [[float(i) for i in test_val.split(',')]]
    print(f"Result: {model.predict(test_point)[0]}")