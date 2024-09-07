from src.visualization import plot_regression_line

def train_and_evaluate_model(model, X, y, fit_method, method_name, batch_size=16):
    
    """Función para entrenar un modelo y evaluar su rendimiento."""
    
    """
    Parámetros:
    - model: instancia del modelo de regresión lineal.
    - X: matriz de características.
    - y: etiquetas/valores reales.
    - fit_method: método de ajuste del modelo.
    - method_name: nombre del método (usado en la salida).
    
    Devuelve:
    - y_pred: predicciones del modelo.
    - mse: error cuadrático medio.
    - rmse: raíz del error cuadrático medio.
    """

    # Ajustamos el modelo usando el método proporcionado
    if fit_method == 'normal_equation':
        model.fit_normal_equation(X, y)
    elif fit_method == 'svd_pseudoinverse':
        model.fit_svd_pseudoinverse(X, y)
    elif fit_method == 'gd_batch':
        model.fit_batch_gradient_descent(X, y)
    elif fit_method == 'gd_stochastic':
        model.fit_stochastic_gradient_descent(X, y)
    elif fit_method == 'gd_mini_batch':
        model.fit_mini_batch_gradient_descent(X, y, batch_size)
    
    # Hacemos las predicciones
    y_pred = model.predict(X)
    
    # Mostramos la línea de regresión ajustada
    plot_regression_line(X, y, y_pred, method_name)
    
    equation = model.get_equation()

    # Calculamos MSE y RMSE
    mse = model.mean_squared_error(y, y_pred)
    rmse = model.root_mean_squared_error(y, y_pred)
    
    print(f"{method_name} - {equation} MSE: {mse}, RMSE: {rmse}")
    return y_pred, mse, rmse