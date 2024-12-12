#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:51:50 2021

@author: Rafał Biedrzycki
Kodu tego mogą używać moi studenci na ćwiczeniach z przedmiotu Wstęp do Sztucznej Inteligencji.
Kod ten powstał aby przyspieszyć i ułatwić pracę studentów, aby mogli skupić się na algorytmach sztucznej inteligencji.
Kod nie jest wzorem dobrej jakości programowania w Pythonie, nie jest również wzorem programowania obiektowego, może zawierać błędy.

Nie ma obowiązku używania tego kodu.
"""

import numpy as np

#ToDo tu prosze podac pierwsze cyfry numerow indeksow
p = [3,9]

L_BOUND = -5
U_BOUND = 5

def q(x):
    return np.sin(x*np.sqrt(p[0]+1))+np.cos(x*np.sqrt(p[1]+1))

x = np.linspace(L_BOUND, U_BOUND, 100)
y = q(x)

np.random.seed(1)


# f logistyczna jako przykład sigmoidalej
def sigmoid(x):
    return 1/(1+np.exp(-x))

#pochodna fun. 'sigmoid'
def d_sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s * (1-s)

#f. straty
def nloss(y_out, y):
    return (y_out - y) ** 2

#pochodna f. straty
def d_nloss(y_out, y):
    return 2*( y_out - y )

class DlNet:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.y_out = 0

        self.HIDDEN_L_SIZE = 700
        self.LR = 0.003

        # Inicjalizacja wag dla warstwy ukrytej i wyjściowej
        self.W_h = np.random.randn(self.HIDDEN_L_SIZE, 1)
        self.W_o = np.random.randn(1, self.HIDDEN_L_SIZE)
        # Inicjalizacja biasów dla warstwy ukrytej i wyjściowej
        self.b_h = np.random.randn(self.HIDDEN_L_SIZE, 1)
        self.b_o = np.random.randn(1, 1)

    def forward(self, x):
        # Propagacja w przód przez warstwę ukrytą
        self.hidden = sigmoid(np.dot(self.W_h, x) + self.b_h)
        # Propagacja w przód przez warstwę wyjściową
        self.y_out = np.dot(self.W_o, self.hidden) + self.b_o

    def predict(self, x):
        self.forward(x)
        return self.y_out

    def backward(self, x, y):
        # Obliczanie gradientów dla wag w warstwie wyjściowej
        d_loss = d_nloss(self.y_out, y)
        d_W_o = np.dot(d_loss, self.hidden.T)
        d_b_o = d_loss
        # Obliczanie gradientów dla wag w warstwie ukrytej
        d_hidden = np.dot(self.W_o.T, d_loss)
        d_hidden *= d_sigmoid(np.dot(self.W_h, x) + self.b_h)
        d_W_h = np.dot(d_hidden, x.T)
        d_b_h = d_hidden

        # Aktualizacja wag i biasów
        self.W_o -= self.LR * d_W_o
        self.b_o -= self.LR * d_b_o
        self.W_h -= self.LR * d_W_h
        self.b_h -= self.LR * d_b_h

    def train(self, x_set, y_set, iters):
        for i in range(iters):
            for j in range(len(x_set)):
                self.forward(x_set[j])
                self.backward(x_set[j], y_set[j])


nn = DlNet(x,y)
nn.train(x, y, 30000)

yh = [nn.predict(np.array([[i]])).flatten() for i in x]

sqare_error = 0
for i in range(len(y)):
    sqare_error += nloss(y[i], yh[i])

import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
print(f'błąd średniokwadratowy aproksymacji: {sqare_error/len(y)}')
print(f'Liczba neuronów: {nn.HIDDEN_L_SIZE}')
plt.title(f'Liczba neuronów: {nn.HIDDEN_L_SIZE}')
plt.plot(x,y, 'r', label="J(x) prawdziwe")
plt.plot(x,yh, 'b', label="J(x) przewidywane")
plt.legend()
plt.savefig("figure")
plt.close()