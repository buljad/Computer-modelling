import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tabulate import tabulate
import time


class Circle:
    def __init__(self, cx, cy, radius):
        self.cx = cx
        self.cy = cy
        self.radius = radius

    def is_inside(self, x, y):
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= self.radius ** 2

    def area(self):
        return np.pi * self.radius ** 2


class Square:
    def __init__(self, sx_min, sx_max, sy_min, sy_max):
        self.sx_min = sx_min
        self.sx_max = sx_max
        self.sy_min = sy_min
        self.sy_max = sy_max

    def is_inside(self, x, y):
        return self.sx_min <= x <= self.sx_max and self.sy_min <= y <= self.sy_max

    def area(self):
        return (self.sx_max - self.sx_min) * (self.sy_max - self.sy_min)


class Triangle:
    def __init__(self, vertices):
        self.vertices = vertices

    def is_inside(self, x, y):
        x1, y1 = self.vertices[0]
        x2, y2 = self.vertices[1]
        x3, y3 = self.vertices[2]

        A = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        A1 = 0.5 * np.abs(x * (y2 - y3) + x2 * (y3 - y) + x3 * (y - y2))
        A2 = 0.5 * np.abs(x1 * (y - y3) + x * (y3 - y1) + x3 * (y1 - y))
        A3 = 0.5 * np.abs(x1 * (y2 - y) + x2 * (y - y1) + x * (y1 - y2))

        return np.isclose(A, A1 + A2 + A3)

    def area(self):
        x1, y1 = self.vertices[0]
        x2, y2 = self.vertices[1]
        x3, y3 = self.vertices[2]
        return 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))


def calculate_area_monte_carlo(shape, area_domain, num_points):
    x_min, x_max, y_min, y_max = area_domain
    inside_shape_count = 0

    for _ in range(num_points):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)

        if shape.is_inside(x, y):
            inside_shape_count += 1

    area_of_domain = (x_max - x_min) * (y_max - y_min)
    estimated_area = (inside_shape_count / num_points) * area_of_domain

    return estimated_area


def calculate_error(true_area, estimated_area):
    return np.abs(estimated_area - true_area) / true_area


# Параметры и создание объектов для круга, квадрата и треугольника
circle = Circle(0, 0, 2)
square = Square(-2, 2, -2, 2)
triangle = Triangle([(0, 0), (3, 0), (1.5, 2)])

# Область измерения
area_domain = (-5, 5, -5, 5)  # (x_min, x_max, y_min, y_max)
area_domain_size = (area_domain[1] - area_domain[0]) * (area_domain[3] - area_domain[2])

# Список количеств точек для оценки площади
num_points = [2 ** i for i in range(5, 20)]

# Истинные значения площадей
true_area_circle = circle.area()
true_area_square = square.area()
true_area_triangle = triangle.area()

# Таблицы для сохранения результатов
table_circle = []
table_square = []
table_triangle = []

# Переменные для хранения итогового времени работы для каждой фигуры
total_time_circle = 0
total_time_square = 0
total_time_triangle = 0

# Начало общего времени работы программы
start_time_total = time.time()

# Установка зерна для генератора случайных чисел
random.seed(42)

# Вычисление ошибок и площадей для каждого количества точек
for npoints in num_points:
    # Circle
    start_time = time.time()
    estimated_area_circle = calculate_area_monte_carlo(circle, area_domain, npoints)
    elapsed_time_circle = time.time() - start_time
    total_time_circle += elapsed_time_circle
    error_circle = calculate_error(true_area_circle, estimated_area_circle)
    table_circle.append([npoints, estimated_area_circle, error_circle, elapsed_time_circle])

    # Square
    start_time = time.time()
    estimated_area_square = calculate_area_monte_carlo(square, area_domain, npoints)
    elapsed_time_square = time.time() - start_time
    total_time_square += elapsed_time_square
    error_square = calculate_error(true_area_square, estimated_area_square)
    table_square.append([npoints, estimated_area_square, error_square, elapsed_time_square])

    # Triangle
    start_time = time.time()
    estimated_area_triangle = calculate_area_monte_carlo(triangle, area_domain, npoints)
    elapsed_time_triangle = time.time() - start_time
    total_time_triangle += elapsed_time_triangle
    error_triangle = calculate_error(true_area_triangle, estimated_area_triangle)
    table_triangle.append([npoints, estimated_area_triangle, error_triangle, elapsed_time_triangle])

# Конец общего времени работы программы
total_elapsed_time = time.time() - start_time_total

# Вывод таблиц
print("Circle Area Estimation")
print(tabulate(table_circle, headers=["Number of Points", "Estimated Area", "Error", "Elapsed Time (s)"]))

print("\nSquare Area Estimation")
print(tabulate(table_square, headers=["Number of Points", "Estimated Area", "Error", "Elapsed Time (s)"]))

print("\nTriangle Area Estimation")
print(tabulate(table_triangle, headers=["Number of Points", "Estimated Area", "Error", "Elapsed Time (s)"]))

# Вывод итогового времени расчета для каждой фигуры
print(f"\nTotal time for Circle calculations: {total_time_circle:.4f} seconds")
print(f"Total time for Square calculations: {total_time_square:.4f} seconds")
print(f"Total time for Triangle calculations: {total_time_triangle:.4f} seconds")

# Вывод общего времени работы программы
print(f"\nTotal elapsed time for the program: {total_elapsed_time:.4f} seconds")

# Построение графика ошибок для всех фигур на одном графике
plt.figure(figsize=(10, 6))
circle_errors = [row[2] for row in table_circle]
square_errors = [row[2] for row in table_square]
triangle_errors = [row[2] for row in table_triangle]

# Применение сглаживания
smoothed_circle_errors = gaussian_filter1d(circle_errors, sigma=2)
smoothed_square_errors = gaussian_filter1d(square_errors, sigma=2)
smoothed_triangle_errors = gaussian_filter1d(triangle_errors, sigma=2)

plt.plot(num_points, smoothed_circle_errors, marker='o', linestyle='-', color='b', label='Circle')
plt.plot(num_points, smoothed_square_errors, marker='s', linestyle='-', color='r', label='Square')
plt.plot(num_points, smoothed_triangle_errors, marker='^', linestyle='-', color='g', label='Triangle')

# Теоретическая погрешность метода Монте-Карло
theoretical_error_circle = [np.sqrt(area_domain_size / (true_area_circle * n)) for n in num_points]
theoretical_error_square = [np.sqrt(area_domain_size / ( true_area_square * n)) for n in num_points]
theoretical_error_triangle = [np.sqrt(area_domain_size/ true_area_triangle / n) for n in num_points]
plt.plot(num_points, theoretical_error_circle, linestyle='--', color='r', label='Theoretical Error')
plt.plot(num_points, theoretical_error_square, linestyle='--', color='b', label='Theoretical Error')
plt.plot(num_points, theoretical_error_triangle, linestyle='--', color='g', label='Theoretical Error')

plt.title('Error in Area Estimation for Different Shapes')
plt.xlabel('Number of Points (log scale)')
plt.ylabel('Error')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.grid(True)
plt.show()
