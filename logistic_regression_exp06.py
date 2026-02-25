import math

# 1. THE SIGMOID FUNCTION
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

try:
    # 2. USER INPUT FOR DATASET
    n = int(input("Enter the number of data points: "))
    x_data = []
    y_data = []

    print("\nEnter X (Feature) and Y (Label 0 or 1) for each point:")
    for i in range(n):
        val_x, val_y = map(float, input(f"Point {i+1} (e.g., 5 1): ").split())
        x_data.append(val_x)
        y_data.append(val_y)

    # 3. CALCULATE SLOPE (m) AND INTERCEPT (c)
    # Using the Least Squares Formulas
    sum_x = sum(x_data)
    sum_y = sum(y_data)
    sum_xy = sum(x_data[i] * y_data[i] for i in range(n))
    sum_x_sq = sum(x_data[i]**2 for i in range(n))

    # Formula: m = [n*sum(xy) - sum(x)*sum(y)] / [n*sum(x^2) - (sum(x))^2]
    denominator = (n * sum_x_sq) - (sum_x**2)

    if denominator == 0:
        print("Error: Slope is undefined (Vertical Line).")
    else:
        m = (n * sum_xy - sum_x * sum_y) / denominator
        c = (sum_y - m * sum_x) / n

        print("\n" + "="*30)
        print(f"CALCULATED SLOPE (m):     {m:.4f}")
        print(f"CALCULATED INTERCEPT (c): {c:.4f}")
        print("="*30)

        # 4. PREDICT FOR THE EXISTING DATASET
        print("\n--- Dataset Predicted Values ---")
        print(f"{'X':<8} | {'Linear (z)':<12} | {'Probability':<12} | {'Class'}")
        print("-" * 50)
        for x in x_data:
            z = m * x + c
            prob = sigmoid(z)
            pred = 1 if prob > 0.5 else 0
            print(f"{x:<8} | {z:<12.4f} | {prob:<12.4f} | {pred}")

        # 5. PREDICT FOR NEW DATA
        print("\n--- Predict New Data ---")
        new_x = float(input("Enter a new X value to predict: "))
        new_z = m * new_x + c
        new_prob = sigmoid(new_z)

        print(f"Result for X={new_x}: Prob {new_prob:.4f} -> Class {'1' if new_prob >= 0.5 else '0'}")

except ValueError:
    print("Invalid input! Please enter numbers only.")