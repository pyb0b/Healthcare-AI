from random import randint, random
from sympy import nextprime
from sage.all import *
from sklearn.model_selection import train_test_split

import pandas as pd 

def sigmoid(z):
    return 1 / (1 + exp(-z))

def logistic_loss_gradient(X, y, w, b):
    m = len(y)
    w = vector(RR, w)
    b = RR(b)
    grad_w = vector(RR, [0.0] * len(w))
    grad_b = 0.0
    for i in range(m):

        xi = vector(RR, X[i]) 
        z = w.dot_product(xi) + b
        pred = sigmoid(z)
        error = pred - y[i]
        grad_w += error * xi
        grad_b += error

        grad_w = vector(RR, grad_w)
        grad_b = RR(grad_b)

    w = vector(RR, w)
    b = RR(b)

    return grad_w / m, grad_b / m


def compute_accuracy(X_test, y_test, w, b):
    correct = 0
    for xi, yi in zip(X_test, y_test):
        xi = vector(RR, xi) 
        pred = 1 if sigmoid(w.dot_product(xi) + b) >= 0.5 else 0
        if pred == yi:
            correct += 1
    return correct / len(y_test)


def federated_learning():
    rounds = 10
    local_epochs = 10
    
    d1 = pd.read_csv("datalocal1.csv") 
    d2 = pd.read_csv("datalocal2.csv")  
    dtest = pd.read_csv("data_test.csv")  

    X1 = [vector(RR, row[:-1]) for row in d1.values]  
    y1 = [row[-1] for row in d1.values]  

    X2 = [vector(RR, row[:-1]) for row in d2.values] 
    y2 = [row[-1] for row in d2.values]  

    X_test = [vector(RR, row[:-1]) for row in dtest.values]
    y_test = [row[-1] for row in dtest.values] 
    
    c1 = len(X1)
    c2 = len(X2)
    total = c1+c2
    c1 = 1/total
    c2 = 1/total

    w_global = vector(RR, [10] * len(X1[0]))   
    b_global = 0.0
      
    
    for r in range(rounds):
        print("\nRound", r+1)

        # Client 1
        w1 = w_global[:] 
        b1 = b_global
        
        for _ in range(local_epochs):
            grad_w1, grad_b1 = logistic_loss_gradient(X1, y1, w1, b1)
            w1 -= 0.1 * grad_w1
            b1 -= 0.1 * grad_b1
        
        # Client 2
        w2 = w_global[:] 
        b2 = b_global

        for _ in range(local_epochs):
            grad_w2, grad_b2 = logistic_loss_gradient(X2, y2, w2, b2)
            w2 -= 0.1 * grad_w2
            b2 -= 0.1 * grad_b2
        
        w_avg = w1 * c1 + w2 * c2
        b_avg  = b1 * c1 + b2 * c2

        print('w:', w_avg)
        print('b:', b_avg)
        
        w_global = vector(RR, w_avg)
        b_global = b_avg

        
    acc = compute_accuracy(X_test, y_test, w_global, b_global)
    print("\nTest Accuracy:", round(acc * 100, 2), "%")

federated_learning()
