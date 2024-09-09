#Importacion de librerias 
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN import KNN

#Definiendo los campos del mapa 
cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

#Utilizacion del DataSet de iris y cargandolo para usarse y hacer el entrenamiento del algortimo de Ml.
iris = datasets.load_iris()
X, y = iris.data, iris.target

#Definiendo los parametros del entrenamiento 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

#Utilizacion de Matplotlib para graficar los parametros que nos devolvera el entramiento de Ml.
plt.figure()
plt.scatter(X[:,2],X[:,3], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()

#Se busca predecir con el algortimo KNN
clf = KNN(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

#Se imprimen las predicciones del algortimo de KNN en consola
print(predictions)

#Utilizacion de numpy para calcular y obtener  la parte del test y el acurracy 
acc = np.sum(predictions == y_test) / len(y_test)
print(acc)
