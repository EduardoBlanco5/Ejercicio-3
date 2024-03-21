import random
import math
import matplotlib.pyplot as plt

# Definir la función objetivo
def objective_function(x1, x2):
    return 10 - math.exp(-(x1**2 + 3*x2**2))

# Gradiente de la función objetivo
def gradient(x1, x2):
    grad_x1 = 2 * x1 * math.exp(-(x1**2 + 3*x2**2))
    grad_x2 = 6 * x2 * math.exp(-(x1**2 + 3*x2**2))
    return grad_x1, grad_x2

def gradient_descent(lr, max_iterations, tolerance):
    # Inicializar valores aleatorios para x1 y x2 en el rango [-1, 1]
    x1 = random.uniform(-1, 1)
    x2 = random.uniform(-1, 1)
    
    errors = []  # Lista para almacenar los errores en cada iteración
    
    iteration = 0
    while iteration < max_iterations:
        # Calcular el gradiente
        grad_x1, grad_x2 = gradient(x1, x2)
        
        # Actualizar los valores de x1 y x2 usando el descenso del gradiente
        x1 -= lr * grad_x1
        x2 -= lr * grad_x2
        
        # Calcular el cambio en la función objetivo
        current_value = objective_function(x1, x2)
        
        # Calcular el error y agregarlo a la lista
        error = abs(current_value)
        errors.append(error)
        
        # Verificar el criterio de parada
        if error < tolerance:
            break
        
        iteration += 1
    
    return x1, x2, errors

# Parámetros
learning_rate = 0.01
max_iterations = 1000
tolerance = 1e-5

# Optimizar la función objetivo utilizando el descenso del gradiente
optimal_x1, optimal_x2, errors = gradient_descent(learning_rate, max_iterations, tolerance)

print("Valor óptimo de x1:", optimal_x1)
print("Valor óptimo de x2:", optimal_x2)
print("Valor óptimo de la función objetivo:", objective_function(optimal_x1, optimal_x2))

# Graficar la convergencia del error
plt.plot(range(len(errors)), errors, linestyle='-', marker='o')
plt.xlabel('Iteración')
plt.ylabel('Error')
plt.title('Convergencia del Error durante el Descenso del Gradiente')
plt.grid(True)
plt.show()