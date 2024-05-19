import copy

from util import *


class SplineInterpolation:
    def __init__(self, x_points, y_points):
        assert len(x_points) == len(y_points)
        self.x_points = x_points
        self.y_points = y_points
        self.k = len(self.x_points)
        self.h = [self.x_points[i + 1] - self.x_points[i] for i in range(self.k - 1)]
        self.A = self.calculate_A()
        self.b = self.calculate_b()
        self.c = zeros(self.k)
        self.solve_c()
        self.d = self.calculate_d()
        self.b = self.calculate_b_final()

    def calculate_A(self):
        A = zeros(self.k, self.k)
        A[0][0] = 1
        A[-1][-1] = 1
        for i in range(1, self.k - 1):
            A[i][i - 1] = self.h[i - 1]
            A[i][i] = 2 * (self.h[i - 1] + self.h[i])
            A[i][i + 1] = self.h[i]
        return A

    def calculate_b(self):
        b = zeros(self.k)
        for i in range(1, self.k - 1):
            b[i] = 3 * ((self.y_points[i + 1] - self.y_points[i]) / self.h[i] - (
                    self.y_points[i] - self.y_points[i - 1]) / self.h[i - 1])
        return b

    def solve_c(self):
        U = copy.deepcopy(self.A)
        L = eye(self.k)
        P = eye(self.k)

        for i in range(self.k):
            max_row_index = max(range(i, self.k), key=lambda x: abs(U[x][i]))
            if i != max_row_index:
                U[i], U[max_row_index] = U[max_row_index], U[i]
                P[i], P[max_row_index] = P[max_row_index], P[i]
                L[i], L[max_row_index] = L[max_row_index], L[i]

            L[i][i] = 1
            for j in range(i + 1, self.k):
                factor = U[j][i] / U[i][i]
                L[j][i] = factor
                U[j][i:] = [U[j][k] - factor * U[i][k] for k in range(i, self.k)]

        Pb = [sum(P[i][j] * self.b[j] for j in range(self.k)) for i in range(self.k)]

        z = zeros(self.k)
        for i in range(self.k):
            z[i] = Pb[i]
            for j in range(i):
                z[i] -= L[i][j] * z[j]

        self.c = zeros(self.k)
        for i in range(self.k - 1, -1, -1):
            self.c[i] = z[i]
            for j in range(i + 1, self.k):
                self.c[i] -= U[i][j] * self.c[j]
            self.c[i] /= U[i][i]

    def calculate_d(self):
        d = zeros(self.k - 1)
        for i in range(self.k - 1):
            d[i] = (self.c[i + 1] - self.c[i]) / (3 * self.h[i])
        return d

    def calculate_b_final(self):
        b = zeros(self.k - 1)
        for i in range(self.k - 1):
            b[i] = (self.y_points[i + 1] - self.y_points[i]) / self.h[i] - self.h[i] * (
                    2 * self.c[i] + self.c[i + 1]) / 3
        return b

    def spline(self, x):
        for i in range(self.k - 1):
            if self.x_points[i] <= x <= self.x_points[i + 1]:
                dx = x - self.x_points[i]
                return (self.y_points[i] +
                        self.b[i] * dx +
                        self.c[i] * dx ** 2 +
                        self.d[i] * dx ** 3)
        return None

    def calculateSpline(self, x_values):
        return [self.spline(x) for x in x_values]
