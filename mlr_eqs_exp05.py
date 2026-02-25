# --- 1. User Input ---
print("--- Multiple Linear Regression (Algebraic Method) ---")
n = int(input("Enter the number of data points (rows): "))

x1_list = []
x2_list = []
y_list = []

print("\nEnter x1, x2, and y for each row (space-separated, e.g., '10 2 25'):")
for i in range(n):
    row = list(map(float, input(f"Row {i+1}: ").split()))
    x1_list.append(row[0])
    x2_list.append(row[1])
    y_list.append(row[2])

# --- 2. Step 1: Calculate Means ---
mean_x1 = sum(x1_list) / n
mean_x2 = sum(x2_list) / n
mean_y = sum(y_list) / n

# --- 3. Step 2: Calculate Sum of Squares and Cross-Products (Deviations) ---
# s_x1y = sum((X1 - mean_X1) * (Y - mean_Y))
s_x1y = 0
s_x2y = 0
s_x1x1 = 0
s_x2x2 = 0
s_x1x2 = 0

for i in range(n):
    dx1 = x1_list[i] - mean_x1
    dx2 = x2_list[i] - mean_x2
    dy  = y_list[i] - mean_y

    s_x1y += dx1 * dy
    s_x2y += dx2 * dy
    s_x1x1 += dx1 * dx1
    s_x2x2 += dx2 * dx2
    s_x1x2 += dx1 * dx2

# --- 4. Step 3: Apply the Formulas ---
# Calculate the common denominator
denom = (s_x1x1 * s_x2x2) - (s_x1x2 ** 2)

if denom == 0:
    print("\nError: The denominator is zero. This happens if features are perfectly correlated.")
else:
    # Calculate b1 and b2 (Slopes)
    b1 = ((s_x1y * s_x2x2) - (s_x2y * s_x1x2)) / denom
    b2 = ((s_x2y * s_x1x1) - (s_x1y * s_x1x2)) / denom

    # Calculate b0 (Intercept)
    b0 = mean_y - (b1 * mean_x1) - (b2 * mean_x2)

    # --- 5. Output Results ---
    print("\n" + "="*30)
    print("FINAL REGRESSION EQUATION:")
    print(f"y = {b0:.4f} + ({b1:.4f} * x1) + ({b2:.4f} * x2)")
    print("="*30)
    print(f"Intercept (b0): {b0:.4f}")
    print(f"Slope (b1)   : {b1:.4f}")
    print(f"Slope (b2)   : {b2:.4f}")

    # --- Bonus: Simple Prediction ---
    test = input("\nWould you like to predict a value? (y/n): ")
    if test.lower() == 'y':
        tx1, tx2 = map(float, input("Enter new x1 and x2: ").split())
        prediction = b0 + (b1 * tx1) + (b2 * tx2)
        print(f"Predicted y for x1={tx1}, x2={tx2} is: {prediction:.4f}")