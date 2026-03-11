#KNN ALGORITHIM
import math

# Euclidean distance
def distance(p1, p2):
    sum_val = 0
    for i in range(len(p1)):
        sum_val += (p1[i] - p2[i]) ** 2
    return math.sqrt(sum_val)


# KNN Function
def knn(train_data, train_labels, test_point, k):

    dist = []

    # Calculate distance from test point to all training points
    for i in range(len(train_data)):
        d = distance(train_data[i], test_point)
        dist.append((d, train_labels[i]))

    # Sort by distance
    dist.sort()

    # Count nearest labels
    count = {}

    for i in range(k):
        label = dist[i][1]
        if label in count:
            count[label] += 1
        else:
            count[label] = 1

    # Find majority class
    prediction = max(count, key=count.get)

    return prediction


# -------- MAIN PROGRAM --------

n = int(input("Enter number of training data points: "))
f = int(input("Enter number of features: "))

train_data = []
train_labels = []

print("\nEnter training data:")

for i in range(n):
    point = []
    for j in range(f):
        val = float(input(f"Enter feature {j+1}: "))
        point.append(val)

    label = input("Enter class label: ")

    train_data.append(point)
    train_labels.append(label)


test_point = []
print("\nEnter test data:")

for i in range(f):
    val = float(input(f"Enter feature {i+1}: "))
    test_point.append(val)


k = int(input("Enter value of K: "))

result = knn(train_data, train_labels, test_point, k)

print("\nPredicted Class:", result)