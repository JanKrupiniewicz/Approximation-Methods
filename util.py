def zeros(n, m=0):
    if m == 0:
        return [0] * n
    else:
        return [[0] * m for _ in range(n)]


def eye(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def mat_mult(A, B):
    n, m = len(A), len(A[0])
    p = len(B[0])
    C = zeros(n, p)
    for i in range(n):
        for j in range(p):
            C[i][j] = sum(A[i][k] * B[k][j] for k in range(m))
    return C


def transpose(A):
    n, m = len(A), len(A[0])
    return [[A[j][i] for j in range(n)] for i in range(m)]
