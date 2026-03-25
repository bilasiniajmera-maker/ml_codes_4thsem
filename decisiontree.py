import math

# --- [Logic for Entropy and Tree Building] ---
def entropy(labels):
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    ent = 0
    for count in counts.values():
        p = count / len(labels)
        ent -= p * math.log2(p)
    return ent

class Node:
    def __init__(self, feature=None, children=None, is_leaf=False, prediction=None):
        self.feature = feature    
        self.children = children or {} 
        self.is_leaf = is_leaf
        self.prediction = prediction

class CategoricalDecisionTree:
    def fit(self, X, y, feat_indices):
        if len(set(y)) == 1:
            return Node(is_leaf=True, prediction=y[0])
        if not feat_indices:
            return Node(is_leaf=True, prediction=max(set(y), key=y.count))

        best_gain, best_feat = -1, None
        current_ent = entropy(y)

        for feat_idx in feat_indices:
            values = set([row[feat_idx] for row in X])
            weighted_ent = 0
            for val in values:
                subset_y = [y[i] for i, row in enumerate(X) if row[feat_idx] == val]
                weighted_ent += (len(subset_y) / len(y)) * entropy(subset_y)
            
            gain = current_ent - weighted_ent
            if gain > best_gain:
                best_gain, best_feat = gain, feat_idx

        node = Node(feature=best_feat)
        remaining_feats = [f for f in feat_indices if f != best_feat]
        
        feat_values = set([row[best_feat] for row in X])
        for val in feat_values:
            indices = [i for i, row in enumerate(X) if row[best_feat] == val]
            node.children[val] = self.fit([X[i] for i in indices], [y[i] for i in indices], remaining_feats)
        return node

    def predict(self, x, node):
        if node.is_leaf: return node.prediction
        val = x[node.feature]
        return self.predict(x, node.children.get(val, Node(is_leaf=True, prediction="Unknown")))

# --- [User Input Section] ---
print("--- Categorical Decision Tree System ---")
X_train = []

# 1. Collect Data
print("Enter row data (e.g., 'Sunny, Weak'). Type 'done' to stop.")
while True:
    raw = input("Row: ").strip()
    if raw.lower() == 'done': break
    X_train.append([item.strip() for item in raw.split(',')])

print("\nEnter the labels for those rows (e.g., 'No, No, Yes, Yes'):")
y_train = [label.strip() for label in input("Labels: ").split(',')]

# 2. Train
tree_model = CategoricalDecisionTree()
root_node = tree_model.fit(X_train, y_train, list(range(len(X_train[0]))))
print("\nModel Trained!")

# 3. Predict
while True:
    test_raw = input("\nEnter features to predict (e.g., 'Overcast, Strong') or 'exit': ").strip()
    if test_raw.lower() == 'exit': break
    test_row = [item.strip() for item in test_raw.split(',')]
    print(f"Result: {tree_model.predict(test_row, root_node)}")