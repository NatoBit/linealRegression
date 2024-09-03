import matplotlib.pyplot as plt

def plot_data(X, y):
    
    """Grafica los datos originales en un gráfico de dispersión."""
    
    """
    Parámetros:
    - X: Matriz de características de forma (m, 1), donde 'm' es el número de ejemplos.
    - y: Vector de etiquetas de salida de forma (m, 1), donde 'm' es el número de ejemplos.
    
    Resultado:
    - Muestra un gráfico de dispersión donde los puntos representan la relación entre X y y.
    """

    plt.scatter(X, y, color='blue', label='Datos Reales')
    plt.title('Datos de entrada')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def plot_regression_line(X, y, y_pred):
    
    """Grafica la línea de regresión ajustada junto con los datos originales."""
    
    """
    Parámetros:
    - X: Matriz de características de forma (m, 1), donde 'm' es el número de ejemplos.
    - y: Vector de etiquetas de salida de forma (m, 1), donde 'm' es el número de ejemplos.
    - y_pred: Vector de predicciones de salida de forma (m, 1), donde 'm' es el número de ejemplos.
    
    Resultado:
    - Muestra un gráfico con los puntos de los datos originales (X, y) y la línea de regresión ajustada (X, y_pred).
    """

    plt.scatter(X, y, color='blue', label='Datos Reales')  # Puntos reales
    plt.plot(X, y_pred, color='red', label='Línea de Regresión')  # Línea ajustada
    plt.title('Regresión Lineal Ajustada')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()