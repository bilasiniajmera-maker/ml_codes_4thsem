#MULTIVARIATE REGRESSION
# Matrix Transpose
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


# Matrix Multiplication
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


# Matrix Inverse (Only for 2x2 matrix)
def inverse_2x2(M):
    det = M[0][0]*M[1][1] - M[0][1]*M[1][0]

    inv = [
        [M[1][1]/det, -M[0][1]/det],
        [-M[1][0]/det, M[0][0]/det]
    ]

    return inv


# Regression Function
def multivariate_regression(X, Y):

    XT = transpose(X)

    XTX = multiply(XT, X)

    XTX_inv = inverse_2x2(XTX)

    XTY = multiply(XT, Y)

    B = multiply(XTX_inv, XTY)

    return B


# ---------------- MAIN PROGRAM ----------------

n = int(input("Enter number of data points: "))

print("Enter values for x and y")

X = []
Y = []

for i in range(n):
    x = float(input("Enter x value: "))
    y = float(input("Enter y value: "))

    X.append([1, x])   # 1 for intercept
    Y.append([y])


coeff = multivariate_regression(X, Y)

print("\nRegression Coefficients")
print("b0 =", coeff[0][0])
print("b1 =", coeff[1][0])


print("\nRegression Equation:")
print("y =", coeff[0][0], "+", coeff[1][0], "* x")