#NAIVE BAYES ALGORITHIM
# Function to calculate probability
def naive_bayes(train_data, train_labels, test):

    classes = {}

    # Count class occurrences
    for label in train_labels:
        if label in classes:
            classes[label] += 1
        else:
            classes[label] = 1

    total = len(train_labels)

    probabilities = {}

    # Calculate probability for each class
    for c in classes:

        prior = classes[c] / total
        likelihood = 1

        for i in range(len(test)):

            count = 0
            class_count = 0

            for j in range(len(train_data)):
                if train_labels[j] == c:
                    class_count += 1
                    if train_data[j][i] == test[i]:
                        count += 1

            if class_count != 0:
                likelihood *= (count / class_count)

        probabilities[c] = prior * likelihood

    # Find class with highest probability
    prediction = max(probabilities, key=probabilities.get)

    return prediction


# -------- MAIN PROGRAM --------

n = int(input("Enter number of training data: "))
f = int(input("Enter number of features: "))

train_data = []
train_labels = []

print("\nEnter training data")

for i in range(n):

    row = []

    for j in range(f):
        val = input(f"Enter feature {j+1}: ")
        row.append(val)

    label = input("Enter class label: ")

    train_data.append(row)
    train_labels.append(label)


test = []
print("\nEnter test data")

for i in range(f):
    val = input(f"Enter feature {i+1}: ")
    test.append(val)


result = naive_bayes(train_data, train_labels, test)

print("\nPredicted Class:", result)