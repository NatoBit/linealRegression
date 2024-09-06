import numpy as np

class LinearRegression:
    def __init__(self):
        self.theta = None

    def fit_normal_equation(self, X, y):

        """Calcula los parámetros theta usando la Ecuación Normal"""

        """
        Parámetros:
        X: Matriz de características (ndarray de tamaño [m, n])
        - m: número de ejemplos (filas)
        - n: número de características (columnas)
        Cada fila de X corresponde a un ejemplo, y cada columna a una característica.
        
        y: Vector de salida o etiquetas (ndarray de tamaño [m, 1])
        - m: número de ejemplos (debe coincidir con las filas de X)
        Cada valor de y es el resultado esperado o etiqueta para el ejemplo correspondiente en X.

        Resultado:
        - Calcula los parámetros 'theta' resolviendo directamente la ecuación normal: (X^T * X)^(-1) * X^T * y
        """

        m = X.shape[0]

        # X_b es la matriz de características expandida
        X_b = np.c_[np.ones((m, 1)), X] # Concatena una columna de 1's a la matriz 𝑋.
        self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

        # Nota: linalg es un submódulo de la biblioteca NumPy en Python que proporciona una variedad de funciones para realizar álgebra lineal.
    
    def fit_svd_pseudoinverse(self, X, y):
        """Calcula los parámetros theta usando la seudoinversa a través de SVD (descomposición en valores singulares)."""

        """
        Parámetros:
        - X: Matriz de características de forma (m, n), donde 'm' es el número de ejemplos y 'n' es el número de características.
        - y: Vector de etiquetas de salida de forma (m, 1), donde 'm' es el número de ejemplos.
        
        Resultado:
        - Calcula los parámetros 'theta' usando la seudoinversa de X.
        """

        m = X.shape[0]  # Número de ejemplos
        X_b = np.c_[np.ones((m, 1)), X]  # Añadir columna de 1's para el intercepto
        
        # Descomposición SVD de X_b: X_b = U * Σ * V^T
        # full_matrices=False: Matrices reducidas 𝑈 y 𝑉^𝑇. Es más eficiente en términos de espacio y tiempo de cómputo, especialmente cuando la matriz 𝑋_𝑏 es grande.
        U, S, Vt = np.linalg.svd(X_b, full_matrices=False)
        
        # Invertir los valores singulares (S) y calcular la seudoinversa de X_b
        S_inv = np.diag(1 / S)  # Inversa de los valores singulares
        X_pseudo_inv = Vt.T.dot(S_inv).dot(U.T)  # Seudoinversa de X_b
        
        # Calcular los parámetros theta
        self.theta = X_pseudo_inv.dot(y)

    def predict(self, X):
        
        """Predice los valores de salida 'y' para los datos de entrada X."""
        
        """
        Parámetros:
        - X: Matriz de características de forma (m, n), donde 'm' es el número de ejemplos y 'n' es el número de características.
        
        Resultado:
        - Retorna las predicciones de forma (m, 1).
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)

    def get_equation(self):

        """Devuelve la ecuación del modelo en formato legible utilizando los valores de theta."""

        """La función item() de NumPy extrae un solo valor de un arreglo. Esto es necesario porque self.theta[i] puede ser un arreglo de un solo elemento, en cuyo caso 
        necesitamos convertirlo a un número escalar para usarlo en una cadena de formato."""
        equation = f"y = {self.theta[0].item():.2f}"
        for i in range(1, len(self.theta)):
            equation += f" + {self.theta[i].item():.2f} * x_{i}"
        return equation

    def mean_squared_error(self, y_true, y_pred):
        
        """Calcula el error cuadrático medio (MSE)."""

        """
        Parámetros:
        - y_true: Vector con los valores reales de salida (m, 1).
        - y_pred: Vector con los valores predichos de salida (m, 1).
        
        Resultado:
        - Retorna el valor del error cuadrático medio (MSE).
        """
        return np.mean((y_pred - y_true) ** 2)

    def root_mean_squared_error(self, y_true, y_pred):
        
        """Calcula la raíz del error cuadrático medio (RMSE)."""

        """
        Parámetros:
        - y_true: Vector con los valores reales de salida (m, 1).
        - y_pred: Vector con los valores predichos de salida (m, 1).
        
        Resultado:
        - Retorna el valor del error cuadrático medio (RMSE).
        """
        return np.sqrt(self.mean_squared_error(y_true, y_pred))
    
