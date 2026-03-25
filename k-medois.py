import math
import random

def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

class KMedoids:
    def __init__(self, k=2, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.medoids = []
        self.labels = []

    def _get_total_cost(self, X, medoids):
        # Total sum of distances from each point to its closest medoid
        total_cost = 0
        for x in X:
            total_cost += min(euclidean_distance(x, m) for m in medoids)
        return total_cost

    def fit(self, X):
        # 1. Initialize medoids randomly from the data points
        self.medoids = random.sample(X, self.k)
        
        for _ in range(self.max_iters):
            best_medoids = list(self.medoids)
            current_total_cost = self._get_total_cost(X, self.medoids)

            # 2. Try swapping medoids with non-medoid points
            for i in range(self.k):
                for x in X:
                    if x in self.medoids: continue
                    
                    # Create a temporary set of medoids with the swap
                    temp_medoids = list(self.medoids)
                    temp_medoids[i] = x
                    temp_cost = self._get_total_cost(X, temp_medoids)

                    # If the swap is cheaper, keep it
                    if temp_cost < current_total_cost:
                        current_total_cost = temp_cost
                        best_medoids = temp_medoids
            
            # Check if we actually improved anything
            if best_medoids == self.medoids:
                break
            self.medoids = best_medoids

        # Final Assignment
        self.labels = []
        for x in X:
            distances = [euclidean_distance(x, m) for m in self.medoids]
            self.labels.append(distances.index(min(distances)))
        
        return self.labels

# --- User Input Section ---
print("--- K-Medoids Clustering ---")
X_data = []
print("Enter points (e.g., '1,2'). Type 'done' when finished.")
while True:
    entry = input("Point: ").lower()
    if entry == 'done': break
    X_data.append([float(i) for i in entry.split(',')])

k_val = int(input("Enter K (number of clusters): "))

model = KMedoids(k=k_val)
assignments = model.fit(X_data)

# Print Results
print("\n" + "="*30)
print("FINAL CLUSTERS (Medoids are actual data points)")
for i in range(k_val):
    print(f"\nCluster {i} (Medoid: {model.medoids[i]}):")
    for idx, cluster_id in enumerate(assignments):
        if cluster_id == i:
            print(f"  -> {X_data[idx]}")