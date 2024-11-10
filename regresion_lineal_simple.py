# Importar las bibliotecas necesarias
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configuración inicial para reproducibilidad y estética de gráficos
np.random.seed(0)
plt.style.use('ggplot')  # Estilo alternativo


# Generar datos de ejemplo (predicción de ingresos en función de la experiencia laboral)
# Los ingresos se generan con una relación lineal respecto a la experiencia, más algo de ruido
X = 2 * np.random.rand(100, 1)  # Variable independiente (años de experiencia)
y = 4 + 3 * X + np.random.randn(100, 1)  # Variable dependiente con ruido (ingresos)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)  # Entrenar el modelo con datos de entrenamiento

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular el error medio cuadrático (MSE) y el coeficiente de determinación (R^2)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Visualizar los resultados con una gráfica de dispersión
plt.figure(figsize=(10, 5))
plt.scatter(X_test, y_test, color="blue", label="Datos de prueba (reales)")
plt.plot(X_test, y_pred, color="red", label="Predicción del modelo")
plt.xlabel("Experiencia (años)")
plt.ylabel("Ingresos ($)")
plt.legend()
plt.title("Regresión Lineal Simple: Datos de Prueba y Predicción")
plt.show()

# Visualizar los resultados en los datos de entrenamiento para ver la aproximación general del modelo
plt.figure(figsize=(10, 5))
plt.scatter(X_train, y_train, color="green", label="Datos de entrenamiento (reales)")
plt.plot(X_train, model.predict(X_train), color="orange", label="Predicción del modelo (entrenamiento)")
plt.xlabel("Experiencia (años)")
plt.ylabel("Ingresos ($)")
plt.legend()
plt.title("Regresión Lineal Simple: Ajuste en Datos de Entrenamiento")
plt.show()

# Imprimir métricas de desempeño
print(f"Error Medio Cuadrático (MSE) en el conjunto de prueba: {mse:.2f}")
print(f"Coeficiente de determinación (R^2) en el conjunto de prueba: {r2:.2f}")

# Extra: Comparación de valores reales vs predicciones en conjunto de prueba
plt.figure(figsize=(10, 5))
plt.plot(y_test, label="Ingresos Reales", marker='o')
plt.plot(y_pred, label="Ingresos Predichos", marker='x')
plt.xlabel("Índice de muestra")
plt.ylabel("Ingresos ($)")
plt.legend()
plt.title("Comparación de Ingresos Reales vs Predichos en el Conjunto de Prueba")
plt.show()
