import numpy as np

try:
    # 1. GET DIMENSIONS
    n = int(input("Enter the number of data points (rows): "))
    k = int(input("Enter the number of independent variables (features): "))

    # 2. GET X DATA (Independent Variables)
    print(f"\nEnter the {k} features for each row separated by spaces:")
    x_list = []
    for i in range(n):
        row = list(map(float, input(f"Row {i+1} features: ").split()))
        if len(row) != k:
            raise ValueError(f"Expected {k} values, but got {len(row)}.")
        x_list.append(row)

    # 3. GET Y DATA (Target Variable)
    print(f"\nEnter the {n} target values (Y) separated by spaces:")
    y_input = input("Y values: ").split()
    if len(y_input) != n:
        raise ValueError(f"Expected {n} Y values.")
    Y = np.array([float(val) for val in y_input])

    # 4. CONVERT TO MATRIX AND ADD INTERCEPT COLUMN
    X_raw = np.array(x_list)
    # Add a column of 1s at the beginning for the Intercept (B0)
    ones = np.ones((n, 1))
    X = np.hstack((ones, X_raw))

    # 5. NORMAL EQUATION: Beta = (X^T . X)^-1 . X^T . Y
    # (X.T is transpose, np.linalg.inv is inverse, .dot is matrix multiplication)
    XT = X.T
    XTX = np.dot(XT, X)

    # Check if the matrix is singular (non-invertible)
    if np.linalg.det(XTX) == 0:
        print("\nError: The matrix is singular and cannot be inverted. Check for redundant features.")
    else:
        XTX_inv = np.linalg.inv(XTX)
        XTY = np.dot(XT, Y)
        beta = np.dot(XTX_inv, XTY)

        # 6. OUTPUT RESULTS
        print("\n" + "="*40)
        print(f"Intercept (B0): {beta[0]:.4f}")
        for i in range(1, len(beta)):
            print(f"Coefficient for X{i} (B{i}): {beta[i]:.4f}")
        print("="*40)

except ValueError as ve:
    print(f"Input Error: {ve}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")