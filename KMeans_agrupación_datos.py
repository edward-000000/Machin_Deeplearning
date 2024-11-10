import numpy as np
import matplotlib.pyplot as plt

# Generar datos sintéticos para el ejemplo
np.random.seed(42)
X = np.random.rand(300, 2)  # 300 puntos en 2D

# Crear 3 grupos de datos
X[:100] += 1  # Grupo 1
X[100:200] -= 1  # Grupo 2
X[200:] += [0, 2]  # Grupo 3

# Visualizar los datos generados
plt.scatter(X[:, 0], X[:, 1], s=30, c='gray', alpha=0.6)
plt.title("Datos Sintéticos")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.show()

# Función para calcular K-Means
def k_means(X, k, max_iters=100, tol=1e-4):
    """
    Implementación del algoritmo K-Means.

    Parámetros:
    - X: datos de entrada
    - k: número de clusters
    - max_iters: número máximo de iteraciones
    - tol: tolerancia para verificar convergencia

    Retorna:
    - centroids: posiciones finales de los centroides
    - labels: etiquetas asignadas a cada punto
    """
    # Paso 0: Inicializar los centroides seleccionando aleatoriamente k puntos del conjunto de datos
    np.random.seed(42)
    initial_centroids_indices = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[initial_centroids_indices]
    
    # Almacenar las etiquetas y el error inicial
    labels = np.zeros(X.shape[0])
    previous_centroids = centroids.copy()

    for i in range(max_iters):
        # Paso 1: Asignar cada punto al centroide más cercano
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  # Calcula la distancia de cada punto a los centroides
        labels = np.argmin(distances, axis=1)  # Asigna etiquetas según el centroide más cercano

        # Paso 2: Recalcular los centroides como la media de los puntos asignados a cada cluster
        new_centroids = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j] for j in range(k)])
        
        # Verificar la convergencia (si el cambio en los centroides es menor que el umbral)
        if np.linalg.norm(new_centroids - centroids) < tol:
            print(f"Convergencia alcanzada en la iteración {i+1}")
            break
        
        centroids = new_centroids

    return centroids, labels

# Ejecutar K-Means con 3 clusters
k = 3
centroids, labels = k_means(X, k)

# Visualizar los resultados
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis', alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroides')
plt.title("Resultado de K-Means")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.legend()
plt.show()
