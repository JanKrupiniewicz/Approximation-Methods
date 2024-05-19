import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Lagrange import LagrangeInterpolation
from Spline import SplineInterpolation


def load_data(filename):
    file_data = pd.read_csv("data/" + filename)
    distance = file_data.iloc[:, 0].values.astype(float)
    elevation = file_data.iloc[:, 1].values.astype(float)
    return distance, elevation


def plot_results(distance, elevation, lagrange_values, spline_values, title, x_values_dense):
    plt.figure(figsize=(10, 6))
    plt.plot(distance, elevation, 'o', label='Original Data')
    plt.plot(x_values_dense, lagrange_values, label='Lagrange Interpolation', linewidth=2)
    plt.plot(x_values_dense, spline_values, label='Cubic Spline Interpolation', linewidth=1)
    plt.xlabel('Distance [m]')
    plt.ylabel('Elevation [m]')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def analyze_Lagrange_interpolation(distance, elevation, step):
    x_v = np.linspace(min(distance), max(distance), 1000)
    selected_indices = range(0, len(distance), step)
    dist_subset = distance[selected_indices]
    elev_subset = elevation[selected_indices]

    # Lagrange Interpolation
    L = LagrangeInterpolation(dist_subset, elev_subset)
    lagrange_values = L.calculateLagrange(x_v)

    return lagrange_values, selected_indices


def analyze_Lagrange_with_density_edges(distance, elevation, step, dense_factor=5):
    # Podział danych na trzy części: lewy kraniec, środek i prawy kraniec
    n = len(distance)
    left_end = distance[:n // 10]
    left_elev = elevation[:n // 10]
    middle = distance[n // 10: 9 * n // 10]
    middle_elev = elevation[n // 10: 9 * n // 10]
    right_end = distance[9 * n // 10:]
    right_elev = elevation[9 * n // 10:]

    left_indices = np.arange(0, len(left_end), max(1, step // dense_factor))
    right_indices = np.arange(0, len(right_end), max(1, step // dense_factor))

    middle_indices = np.arange(0, len(middle), step)

    selected_indices = np.concatenate([
        left_indices,
        middle_indices + len(left_end),
        right_indices + len(left_end) + len(middle)
    ])

    dist_subset = distance[selected_indices]
    elev_subset = elevation[selected_indices]

    x_v = np.linspace(min(distance), max(distance), 1000)

    # Lagrange Interpolation
    L = LagrangeInterpolation(dist_subset, elev_subset)
    lagrange_values = L.calculateLagrange(x_v)

    return lagrange_values, selected_indices


def analyze_Spline_interpolation(distance, elevation, step):
    x_v = np.linspace(min(distance), max(distance), 1000)
    selected_indices = range(0, len(distance), step)
    dist_subset = distance[selected_indices]
    elev_subset = elevation[selected_indices]

    # Spline Interpolation
    spline_interpolator = SplineInterpolation(dist_subset, elev_subset)
    spline_values = spline_interpolator.calculateSpline(x_v)

    return spline_values, selected_indices


files = ["MountEverest.csv", "Obiadek.csv", "SpacerniakGdansk.csv", "WielkiKanionKolorado.csv"]
for file in files:
    distance, elevation = load_data(file)
    x_values_dense = np.linspace(min(distance), max(distance), 1000)

    original_spline = SplineInterpolation(distance, elevation)
    original_spline_route = original_spline.calculateSpline(x_values_dense)

    # ------------------------------------------------------------------
    # The influence of the number of nodal points on the results Splain
    sv5, spline_indices_5 = analyze_Spline_interpolation(distance, elevation, 5)

    plt.plot(distance[spline_indices_5], elevation[spline_indices_5], 'o', label='Original Data')
    plt.plot(x_values_dense, original_spline_route, label='Real Path', linewidth=2)
    plt.plot(x_values_dense, sv5, label='Spline 1/5')
    plt.xlabel('Distance [m]')
    plt.ylabel('Elevation [m]')
    plt.title('Spline - influence of the number of nodal points')
    plt.legend()
    plt.grid(True)
    plt.show()

    sv25, spline_indices_25 = analyze_Spline_interpolation(distance, elevation, 25)

    plt.plot(distance[spline_indices_25], elevation[spline_indices_25], 'o', label='Original Data')
    plt.plot(x_values_dense, original_spline_route, label='Real Path', linewidth=2)
    plt.plot(x_values_dense, sv25, label='Spline 1/25')
    plt.xlabel('Distance [m]')
    plt.ylabel('Elevation [m]')
    plt.title('Spline - influence of the number of nodal points')
    plt.legend()
    plt.grid(True)
    plt.show()

    sv50, spline_indices_50 = analyze_Spline_interpolation(distance, elevation, 50)

    plt.plot(distance[spline_indices_50], elevation[spline_indices_50], 'o', label='Original Data')
    plt.plot(x_values_dense, original_spline_route, label='Real Path', linewidth=2)
    plt.plot(x_values_dense, sv50, label='Spline 1/50')
    plt.xlabel('Distance [m]')
    plt.ylabel('Elevation [m]')
    plt.title('Spline - influence of the number of nodal points')
    plt.legend()
    plt.grid(True)
    plt.show()

    # The influence of the number of nodal points on the results Spline
    plt.plot(distance[spline_indices_50], elevation[spline_indices_50], 'o', label='Original Data')
    plt.plot(x_values_dense, original_spline_route, label='Real Path', linewidth=2)
    plt.plot(x_values_dense, sv5, label='Spline 1/5')
    plt.plot(x_values_dense, sv25, label='Spline 1/25')
    plt.plot(x_values_dense, sv50, label='Spline 1/50')
    plt.xlabel('Distance [m]')
    plt.ylabel('Elevation [m]')
    plt.title('Spline - influence of the number of nodal points')
    plt.legend()
    plt.grid(True)
    plt.show()

    # ------------------------------------------------------------------
    # The influence of the number of nodal points on the results Lagrange
    lv50, lagrange_indices_50 = analyze_Lagrange_interpolation(distance, elevation, 50)

    plt.plot(distance[lagrange_indices_50], elevation[lagrange_indices_50], 'o', label='Original Data')
    plt.plot(x_values_dense, original_spline_route, label='Real Path', linewidth=2)
    plt.plot(x_values_dense, lv50, label='Lagrange 1/50 points')
    plt.xlabel('Distance [m]')
    plt.ylabel('Elevation [m]')
    plt.title('Lagrange - influence of the number of nodal points')
    plt.legend()
    plt.grid(True)
    plt.show()

    lv60, lagrange_indices_60 = analyze_Lagrange_interpolation(distance, elevation, 60)

    plt.plot(distance[lagrange_indices_60], elevation[lagrange_indices_60], 'o', label='Original Data')
    plt.plot(x_values_dense, original_spline_route, label='Real Path', linewidth=2)
    plt.plot(x_values_dense, lv60, label='Lagrange 1/60 points')
    plt.xlabel('Distance [m]')
    plt.ylabel('Elevation [m]')
    plt.title('Lagrange - influence of the number of nodal points')
    plt.legend()
    plt.grid(True)
    plt.show()

    lv80, lagrange_indices_80 = analyze_Lagrange_interpolation(distance, elevation, 80)

    plt.plot(distance[lagrange_indices_80], elevation[lagrange_indices_80], 'o', label='Original Data')
    plt.plot(x_values_dense, original_spline_route, label='Real Path', linewidth=2)
    plt.plot(x_values_dense, lv80, label='Lagrange 1/80 points')
    plt.xlabel('Distance [m]')
    plt.ylabel('Elevation [m]')
    plt.title('Lagrange - influence of the number of nodal points')
    plt.legend()
    plt.grid(True)
    plt.show()

    # The influence of the number of nodal points on the results Lagrange
    plt.plot(distance[lagrange_indices_50], elevation[lagrange_indices_50], 'o', label='Original Data')
    plt.plot(x_values_dense, original_spline_route, label='Real Path', linewidth=2)
    plt.plot(x_values_dense, lv50, label='Lagrange 1/50')
    plt.plot(x_values_dense, lv60, label='Lagrange 1/60')
    plt.plot(x_values_dense, lv80, label='Lagrange 1/80')
    plt.xlabel('Distance [m]')
    plt.ylabel('Elevation [m]')
    plt.title('Lagrange - influence of the number of nodal points')
    plt.legend()
    plt.grid(True)
    plt.show()

    # ------------------------------------------------------------------
    # Lagrangian interpolation with a density of points at the edges of the function

    lv15, lagrange_indices_15 = analyze_Lagrange_with_density_edges(distance, elevation, 15)

    plt.plot(distance[lagrange_indices_15], elevation[lagrange_indices_15], 'o', label='Original Data')
    plt.plot(x_values_dense, original_spline_route, label='Real Path', linewidth=2)
    plt.plot(x_values_dense, lv15, label='Lagrange 1/15 points')
    plt.xlabel('Distance [m]')
    plt.ylabel('Elevation [m]')
    plt.title('Lagrange - influence of the number of nodal points')
    plt.legend()
    plt.grid(True)
    plt.show()

    lv30, lagrange_indices_30 = analyze_Lagrange_with_density_edges(distance, elevation, 30)

    plt.plot(distance[lagrange_indices_30], elevation[lagrange_indices_30], 'o', label='Original Data')
    plt.plot(x_values_dense, original_spline_route, label='Real Path', linewidth=2)
    plt.plot(x_values_dense, lv30, label='Lagrange 1/30 points')
    plt.xlabel('Distance [m]')
    plt.ylabel('Elevation [m]')
    plt.title('Lagrange - influence of the number of nodal points')
    plt.legend()
    plt.grid(True)
    plt.show()

    lv80, lagrange_indices_80 = analyze_Lagrange_with_density_edges(distance, elevation, 80)

    plt.plot(distance[lagrange_indices_80], elevation[lagrange_indices_80], 'o', label='Original Data')
    plt.plot(x_values_dense, original_spline_route, label='Real Path', linewidth=2)
    plt.plot(x_values_dense, lv80, label='Lagrange 1/80 points')
    plt.xlabel('Distance [m]')
    plt.ylabel('Elevation [m]')
    plt.title('Lagrange - influence of the number of nodal points')
    plt.legend()
    plt.grid(True)
    plt.show()
