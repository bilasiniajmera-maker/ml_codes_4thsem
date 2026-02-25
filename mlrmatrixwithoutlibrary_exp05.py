def transpose(A):
    """Returns the transpose of a 2D list."""
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def multiply(A, B):
    """Multiplies two matrices or a matrix and a vector."""
    # Check if B is a 1D vector
    is_b_1d = isinstance(B[0], (int, float))
    if is_b_1d:
        return [sum(A[i][k] * B[k] for k in range(len(B))) for i in range(len(A))]
    else:
        # Standard matrix multiplication
        result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    result[i][j] += A[i][k] * B[k][j]
        return result

def invert_matrix(matrix):
    """Inverts a square matrix using Gauss-Jordan elimination."""
    n = len(matrix)
    # Create an identity matrix
    identity = [[float(i == j) for j in range(n)] for i in range(n)]

    # Work on a copy to avoid modifying the original
    AM = [row[:] for row in matrix]

    for i in range(n):
        # Partial Pivoting
        max_el = abs(AM[i][i])
        max_row = i
        for k in range(i + 1, n):
            if abs(AM[k][i]) > max_el:
                max_el = abs(AM[k][i])
                max_row = k

        if max_el < 1e-10:
            return None # Matrix is singular

        AM[i], AM[max_row] = AM[max_row], AM[i]
        identity[i], identity[max_row] = identity[max_row], identity[i]

        # Normalize pivot row
        pivot = AM[i][i]
        for j in range(i, n):
            AM[i][j] /= pivot
        for j in range(n):
            identity[i][j] /= pivot

        # Eliminate other rows
        for k in range(n):
            if k != i:
                factor = AM[k][i]
                for j in range(i, n):
                    AM[k][j] -= factor * AM[i][j]
                for j in range(n):
                    identity[k][j] -= factor * identity[i][j]
    return identity

# --- MAIN PROGRAM ---
try:
    print("--- Multiple Linear Regression (Manual Calculation) ---")

    # 1. GET DIMENSIONS
    n = int(input("Enter number of data points (rows): "))
    k = int(input("Enter number of features (independent variables): "))

    # 2. GET X DATA (Features)
    print(f"\nEnter the {k} features for each row:")
    X = []
    for i in range(n):
        while True:
            try:
                row_input = input(f"Row {i+1} features: ").split()
                row = [float(val) for val in row_input]
                if len(row) != k:
                    print(f"Error: Expected {k} values. Please try again.")
                    continue
                # Add 1.0 at index 0 for the Intercept (B0)
                X.append([1.0] + row)
                break
            except ValueError:
                print("Error: Invalid input. Enter numbers only.")

    # 3. GET Y DATA (Target)
    print(f"\nEnter the {n} target values (Y) separated by spaces:")
    while True:
        try:
            y_input = input("Y values: ").split()
            if len(y_input) != n:
                print(f"Error: Expected {n} values. Please try again.")
                continue
            Y = [float(val) for val in y_input]
            break
        except ValueError:
            print("Error: Invalid input. Enter numbers only.")

    # 4. MATH: Normal Equation (Beta = (X'X)^-1 X'Y)
    XT = transpose(X)
    XTX = multiply(XT, X)
    XTX_inv = invert_matrix(XTX)

    if XTX_inv is None:
        print("\n[!] Error: The matrix is singular (non-invertible). Check for redundant data.")
    else:
        XTY = multiply(XT, Y)
        beta = multiply(XTX_inv, XTY)

        # 5. OUTPUT RESULTS
        print("\n" + "="*45)
        print(f"{'Variable':<20} | {'Coefficient':<15}")
        print("-" * 45)
        print(f"{'Intercept (B0)':<20} | {beta[0]:.4f}")
        for i in range(1, len(beta)):
            print(f"{f'Feature X{i} (B{i})':<20} | {beta[i]:.4f}")
        print("="*45)

except Exception as e:
    print(f"\nAn error occurred: {e}")
