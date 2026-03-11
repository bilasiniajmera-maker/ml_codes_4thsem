# Matrix transpose
def transpose(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    t = []

    for j in range(cols):
        row = []
        for i in range(rows):
            row.append(matrix[i][j])
        t.append(row)

    return t


# Matrix multiplication
def multiply(A, B):
    result = []

    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            sum_val = 0
            for k in range(len(B)):
                sum_val += A[i][k] * B[k][j]
            row.append(sum_val)
        result.append(row)

    return result


# Matrix inverse (2x2 only)
def inverse_2x2(M):
    det = M[0][0]*M[1][1] - M[0][1]*M[1][0]

    inv = [
        [M[1][1]/det, -M[0][1]/det],
        [-M[1][0]/det, M[0][0]/det]
    ]

    return inv


# Multivariate regression
def multivariate_regression(X, Y):

    XT = transpose(X)

    XTX = multiply(XT, X)

    XTX_inv = inverse_2x2(XTX)

    XTY = multiply(XT, Y)

    B = multiply(XTX_inv, XTY)

    return B


# -------- MAIN PROGRAM --------

n = int(input("Enter number of data points: "))
m = int(input("Enter number of Y outputs: "))

X = []
Y = []

print("\nEnter values")

for i in range(n):

    x = float(input(f"Enter x value for row {i+1}: "))

    X.append([1, x])   # intercept added

    y_row = []

    for j in range(m):
        y = float(input(f"Enter y{j+1} value: "))
        y_row.append(y)

    Y.append(y_row)


coeff = multivariate_regression(X, Y)

print("\nRegression Coefficients")

for row in coeff:
    print(row)