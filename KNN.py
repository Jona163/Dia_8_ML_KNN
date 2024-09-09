#Importacion de librerias 
import numpy as np
from collections import Counter

#Urtilizacion de distancia euclidiana 
def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

#Clase de algortitmo KNN 
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
#Definicion de la prediccion a realizar
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # calcular la distancia 
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
    
        # obteniedo el closet
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # mayoria de voye
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
