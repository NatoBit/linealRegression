import numpy as np
from src.linear_regression import LinearRegression
from src.visualization import plot_data, plot_regression_line
from utils import train_and_evaluate_model

# Crear datos de ejemplo simulados (100 ejemplos)
# X: Características, y: Etiquetas de salida generadas usando una relación lineal más ruido
np.random.seed(42)  # Fijar semilla para reproducibilidad
X = 2 * np.random.rand(100, 1)  # Generar 100 valores de entrada aleatorios entre 0 y 2
y = 4 + 3 * X + np.random.randn(100, 1)  # Generar salida usando la ecuación y = 4 + 3X + ruido

# Visualizar los datos generados
plot_data(X, y)

model_normal = LinearRegression()
train_and_evaluate_model(model_normal, X, y, 'normal_equation', 'Normal Equation')

model_svd = LinearRegression()
train_and_evaluate_model(model_svd, X, y, 'svd_pseudoinverse', 'SVD Pseudoinverse')

model_gd_batch = LinearRegression()
train_and_evaluate_model(model_gd_batch, X, y, 'gd_batch', 'GD Batch')

model_gd_stochastic = LinearRegression()
train_and_evaluate_model(model_gd_stochastic, X, y, 'gd_stochastic', 'GD Stochastic')

model_gd_mini_batch = LinearRegression()
train_and_evaluate_model(model_gd_mini_batch, X, y, 'gd_mini_batch', 'GD Mini-Batch')