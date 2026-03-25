import math
import matplotlib.pyplot as plt

def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

class ManualDendrogram:
    def __init__(self, data):
        self.data = data
        # Each cluster starts as a dictionary: {points: [idx], x_pos: float}
        self.clusters = [{"points": [i], "x": float(i)} for i in range(len(data))]
        self.history = []

    def get_cluster_dist(self, c1, c2):
        # Single Linkage: Minimum distance between any two points in the two clusters
        min_d = float('inf')
        for idx1 in c1["points"]:
            for idx2 in c2["points"]:
                d = euclidean_distance(self.data[idx1], self.data[idx2])
                if d < min_d:
                    min_d = d
        return min_d

    def run_and_plot(self):
        plt.figure(figsize=(8, 6))
        
        while len(self.clusters) > 1:
            min_d = float('inf')
            pair = (0, 1)

            # 1. Find the two closest clusters
            for i in range(len(self.clusters)):
                for j in range(i + 1, len(self.clusters)):
                    d = self.get_cluster_dist(self.clusters[i], self.clusters[j])
                    if d < min_d:
                        min_d = d
                        pair = (i, j)

            # 2. Extract the clusters to merge
            c1 = self.clusters.pop(pair[1]) # Pop higher index first to keep order
            c2 = self.clusters.pop(pair[0])

            # 3. Calculate new X position (middle of the two merged clusters)
            new_x = (c1["x"] + c2["x"]) / 2
            
            # 4. Draw the "U" shape of the Dendrogram
            # Vertical line from c1
            plt.plot([c1["x"], c1["x"]], [self._get_last_height(c1), min_d], color='blue')
            # Vertical line from c2
            plt.plot([c2["x"], c2["x"]], [self._get_last_height(c2), min_d], color='blue')
            # Horizontal line connecting them
            plt.plot([c1["x"], c2["x"]], [min_d, min_d], color='blue')

            # 5. Create the new merged cluster
            new_cluster = {
                "points": c1["points"] + c2["points"],
                "x": new_x,
                "height": min_d
            }
            self.clusters.append(new_cluster)

        plt.title("Manual Hierarchical Clustering (Single Linkage)")
        plt.xlabel("Point Index")
        plt.ylabel("Distance")
        plt.xticks(range(len(self.data)))
        plt.show()

    def _get_last_height(self, cluster):
        # Returns the height at which this cluster was last merged, or 0 if it's a leaf
        return cluster.get("height", 0)

# --- User Input Section ---
print("--- Manual Dendrogram (Matplotlib only) ---")
X_train = []
print("Enter points (e.g., '1,2'). Type 'done' to finish.")
while True:
    entry = input("Point: ").lower()
    if entry == 'done': break
    X_train.append([float(i) for i in entry.split(',')])

if len(X_train) < 2:
    print("Add more points!")
else:
    engine = ManualDendrogram(X_train)
    engine.run_and_plot()