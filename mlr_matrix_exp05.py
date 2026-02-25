import numpy as np

def run_regression():
    try:
        print("--- Multiple Linear Regression (Using NumPy) ---")

        # 1. GET DIMENSIONS
        n = int(input("Enter number of data points (rows): "))
        k = int(input("Enter number of independent variables (features): "))

        # 2. GET X DATA (Features)
        print(f"\nEnter the {k} features for each row:")
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

        # 4. PREPARE MATRICES
        X_raw = np.array(x_list)
        # Add a column of 1s for the Intercept (B0)
        X = np.hstack((np.ones((n, 1)), X_raw))

        # 5. SOLVE NORMAL EQUATION
        # .T is transpose, @ is matrix multiplication, np.linalg.inv is inverse
        XT = X.T
        XTX = XT @ X

        # Check for singularity before inverting
        if np.linalg.det(XTX) == 0:
            print("\nError: The matrix is singular and cannot be inverted.")
            return

        XTX_inv = np.linalg.inv(XTX)
        XTY = XT @ Y
        beta = XTX_inv @ XTY

        # 6. OUTPUT RESULTS
        print("\n" + "="*45)
        print(f"{'Variable':<20} | {'Coefficient':<15}")
        print("-" * 45)
        print(f"{'Intercept (B0)':<20} | {beta[0]:.4f}")
        for i in range(1, len(beta)):
            print(f"{f'Feature X{i} (B{i})':<20} | {beta[i]:.4f}")
        print("="*45)

    except ValueError as ve:
        print(f"Input Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_regression()