class LagrangeInterpolation:
    def __init__(self, x_points, y_points):
        assert len(x_points) == len(y_points)
        self.x_points = x_points
        self.y_points = y_points
        self.k = len(self.x_points)

    def fi(self, i, x):
        fi = 1
        for j in range(self.k):
            if j != i:
                fi *= (x - self.x_points[j]) / (self.x_points[i] - self.x_points[j])
        return fi

    def lagrange(self, x):
        lagrange_sum = 0
        for i in range(self.k):
            lagrange_sum += self.y_points[i] * self.fi(i, x)
        return lagrange_sum

    def calculateLagrange(self, x_values):
        return [self.lagrange(x) for x in x_values]
