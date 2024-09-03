import numpy as np
from src.linear_regression import LinearRegression
from src.visualization import plot_data, plot_regression_line

# Crear datos de ejemplo simulados (100 ejemplos)
# X: Características, y: Etiquetas de salida generadas usando una relación lineal más ruido
np.random.seed(42)  # Fijar semilla para reproducibilidad
X = 2 * np.random.rand(100, 1)  # Generar 100 valores de entrada aleatorios entre 0 y 2
y = 4 + 3 * X + np.random.randn(100, 1)  # Generar salida usando la ecuación y = 4 + 3X + ruido

# Visualizar los datos generados
plot_data(X, y)

# Entrenamiento del modelo usando la ecuación normal
model_normal = LinearRegression()
model_normal.fit_normal_equation(X, y)
y_pred_normal = model_normal.predict(X)  # Predicciones usando la ecuación normal
y_theta = model_normal.theta

# Visualizar la línea de regresión ajustada por la ecuación normal
plot_regression_line(X, y, y_pred_normal)

# Calcular y mostrar el MSE y RMSE para la ecuación normal
equation_normal = model_normal.get_equation()
mse_normal = model_normal.mean_squared_error(y, y_pred_normal)
rmse_normal = model_normal.root_mean_squared_error(y, y_pred_normal)
print(f"Normal Equation - {equation_normal}, MSE: {mse_normal}, RMSE: {rmse_normal}")