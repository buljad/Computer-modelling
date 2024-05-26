import random
import numpy as np
import matplotlib.pyplot as plt


class Circle:
    def __init__(self, cx, cy, radius):
        self.cx = cx
        self.cy = cy
        self.radius = radius

    def is_inside(self, x, y):
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= self.radius ** 2


class Square:
    def __init__(self, sx_min, sx_max, sy_min, sy_max):
        self.sx_min = sx_min
        self.sx_max = sx_max
        self.sy_min = sy_min
        self.sy_max = sy_max

    def is_inside(self, x, y):
        return self.sx_min <= x <= self.sx_max and self.sy_min <= y <= self.sy_max


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
    return np.abs(estimated_area - true_area)


# Параметры и создание объектов для круга, квадрата и треугольника
circle = Circle(0, 0, 2)
square = Square(-2, 2, -2, 2)
triangle = Triangle([(0, 0), (3, 0), (1.5, 2)])

# Область измерения
area_domain = (-5, 5, -5, 5)  # (x_min, x_max, y_min, y_max)

# Список количеств точек для оценки площади
num_points = [2**i for i in range(5, 14)] + [20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 70000, 75000,
                                             80000, 85000, 90000, 95000, 131072]


# Списки для сохранения ошибок оценки площади
errors_circle = []
errors_square = []
errors_triangle = []

# Вычисление ошибок для каждого количества точек для круга
for npoints in num_points:
    estimated_area_circle = calculate_area_monte_carlo(circle, area_domain, npoints)
    error_circle = np.abs(np.pi * circle.radius ** 2 - estimated_area_circle)
    errors_circle.append(error_circle)

# Вычисление ошибок для каждого количества точек для квадрата
for npoints in num_points:
    estimated_area_square = calculate_area_monte_carlo(square, area_domain, npoints)
    error_square = np.abs((square.sx_max - square.sx_min) * (square.sy_max - square.sy_min) - estimated_area_square)
    errors_square.append(error_square)

# Вычисление ошибок для каждого количества точек для треугольника
for npoints in num_points:
    estimated_area_triangle = calculate_area_monte_carlo(triangle, area_domain, npoints)
    true_area_triangle = 0.5 * np.abs(
        triangle.vertices[0][0] * (triangle.vertices[1][1] - triangle.vertices[2][1]) + triangle.vertices[1][0] * (
                    triangle.vertices[2][1] - triangle.vertices[0][1]) + triangle.vertices[2][0] * (
                    triangle.vertices[0][1] - triangle.vertices[1][1]))
    error_triangle = np.abs(true_area_triangle - estimated_area_triangle)
    errors_triangle.append(error_triangle)

# Построение графика ошибок для всех фигур на одном графике
plt.figure(figsize=(10, 6))
plt.plot(num_points, errors_circle, marker='o', linestyle='-', color='b', label='Circle')
plt.plot(num_points, errors_square, marker='s', linestyle='-', color='r', label='Square')
plt.plot(num_points, errors_triangle, marker='^', linestyle='-', color='g', label='Triangle')

# Теоретическая погрешность метода Монте-Карло
theoretical_error = [1 / np.sqrt(n) for n in num_points]
plt.plot(num_points, theoretical_error, linestyle='--', color='k', label='Theoretical Error')

plt.title('Error in Area Estimation for Different Shapes')
plt.xlabel('Number of Points (log scale)')
plt.ylabel('Error')
plt.yscale('log')  # Логарифмическая шкала по оси y для наглядности
plt.legend()
plt.grid(True)
plt.show()
