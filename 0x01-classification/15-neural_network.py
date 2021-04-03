#!/usr/bin/env python3
"""
Module contenant la classe neurone
"""
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    """
    Une classe qui définit un réseau de neurones avec une couche cachée effectuant une classification binaire
    """

    def __init__(self, nx, nodes):
        """
        constructeur de la classe
        la variable nx: est le nombre d'entités d'entrée du neurone
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        # hidden layer
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        # output neuron
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    def forward_prop(self, X):
        """
        Le calcule de la propagation directe du réseau de neurones
         Le paramètre X: tableau np avec les données d'entrée de forme (nx, m)
         elle retourne : les attributs privés __A1 et __A2
        """
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        calcule le coût du modèle à l'aide de la régression logistique
         Le paramètre Y: un tableau np avec des étiquettes de forme correctes (1, m)
         Le paramètre A: un tableau np avec la sortie activée de shape (1, m)
         elle retourne : le coût
        """
        cost = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = np.sum(cost)
        cost = - cost / A.shape[1]
        return cost

    def evaluate(self, X, Y):
        """
        évalue la prédiction du réseau neuronal
         Le paramètre X: tableau np avec des données d'entrée de forme (nx, m)
         Le paramètre Y: tableau np avec étiquette de forme correcte (1, m)
         elle retourne: prédiction des neurones et coût du réseau
        """
        self.forward_prop(X)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A2)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Le calcule une passe de descente de gradient sur le neurone
         Le paramètre X: tableau np avec des données d'entrée de forme (nx, m)
         Le paramètre Y: tableau np avec des étiquettes correctes de forme (1, m)
         Le paramètre A1: tableau np avec sortie de couche cachée activée de forme (1, m)
         Le paramètre A2: tableau np avec sortie activée de la forme (1, m)
         Le paramètre alpha: le taux d'apprentissage
         elle ne retourne rien
        """
        # gradient descent for hidden layer
        dz2 = A2 - Y
        dw2 = np.matmul(A1, dz2.T) / A1.shape[1]
        db2 = np.sum(dz2, axis=1, keepdims=True) / A2.shape[1]

        # derivative of the sigmoid function
        da1 = A1 * (1 - A1)
        # gradient descent for output layer
        dz1 = np.matmul(self.__W2.T, dz2)
        dz1 = dz1 * da1
        dw1 = np.matmul(X, dz1.T) / A1.shape[1]
        db1 = np.sum(dz1, axis=1, keepdims=True) / A1.shape[1]
        # updated value for weights and bias
        self.__W2 = self.__W2 - alpha * dw2.T
        self.__b2 = self.__b2 - alpha * db2
        self.__W1 = self.__W1 - alpha * dw1.T
        self.__b1 = self.__b1 - alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Fonction pour former le réseau neuronal
         Le paramètre X: tableau np avec des données d'entrée de forme (nx, m)
         Le paramètre Y: tableau np avec des étiquettes correctes de forme (1, m)
         Le paramètre itérations: itérations de l'entraînement
         Le paramètre alpha: taux d'apprentissage
         elle retourne : l'évaluation des données d'entraînement
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True and iterations % step == 0:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        if graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        cost = []
        for i in range(iterations + 1):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)
            cost.append(self.cost(Y, self.__A2))

            if verbose is True and i % step == 0:
                print("Cost after {} iterations: {}"
                      .format(i, cost[i]))
        if graph is True:
            plt.plot(np.arange(0, iterations + 1), cost)
            plt.title("Training cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()

        return self.evaluate(X, Y)

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2
