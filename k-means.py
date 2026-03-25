import math
import random

def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = []
        self.labels = [] # Stores which cluster each point belongs to

    def fit(self, X):
        # 1. Initialize centroids randomly
        self.centroids = random.sample(X, self.k)

        for _ in range(self.max_iters):
            # 2. Assignment Phase
            clusters = [[] for _ in range(self.k)]
            current_labels = []
            
            for x in X:
                distances = [euclidean_distance(x, c) for c in self.centroids]
                closest_idx = distances.index(min(distances))
                clusters[closest_idx].append(x)
                current_labels.append(closest_idx)

            prev_centroids = list(self.centroids)

            # 3. Update Phase (Moving Centroids to the Mean)
            for i in range(self.k):
                if not clusters[i]: continue
                
                new_centroid = []
                for dim in range(len(X[0])):
                    dim_mean = sum(point[dim] for point in clusters[i]) / len(clusters[i])
                    new_centroid.append(dim_mean)
                self.centroids[i] = new_centroid

            self.labels = current_labels
            
            # 4. Check for Convergence
            if prev_centroids == self.centroids:
                break
        
        return self.labels

# --- User Input & Results ---
print("--- K-Means: Point-to-Cluster Mapper ---")
X_train = []
print("Enter coordinates (e.g., '10, 20'). Type 'done' to finish.")

while True:
    val = input("Point: ").lower()
    if val == 'done': break
    try:
        X_train.append([float(i) for i in val.split(',')])
    except ValueError:
        print("Invalid format. Use numbers separated by a comma.")

if len(X_train) < 2:
    print("Need more data points!")
else:
    k_input = int(input("Enter number of clusters (K): "))
    
    model = KMeans(k=k_input)
    cluster_assignments = model.fit(X_train)

    print("\n" + "="*30)
    print("FINAL CLUSTER ASSIGNMENTS")
    print("="*30)
    
    # Organizing points by their cluster for a cleaner print
    final_groups = {}
    for idx, cluster_id in enumerate(cluster_assignments):
        if cluster_id not in final_groups:
            final_groups[cluster_id] = []
        final_groups[cluster_id].append(X_train[idx])

    for cluster_id in sorted(final_groups.keys()):
        print(f"\nCluster {cluster_id} (Centroid: {[round(c, 2) for c in model.centroids[cluster_id]]}):")
        for point in final_groups[cluster_id]:
            print(f"  -> {point}")

    print("\n" + "="*30)