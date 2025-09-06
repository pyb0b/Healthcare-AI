from random import randint, random
from sympy import nextprime
from sage.all import *
from sklearn.model_selection import train_test_split
import time

import pandas as pd 

load("utils_xmk_ckks.sage")

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
    #start_time = time.time()
    rounds = 10
    local_epochs = 10
    sf = 15
    ctx = CKKSContext(lambda_param=8192)

    num_clients = 10
    keypairs = [keygen(ctx) for _ in range(num_clients)]
    secret_keys = [s for (s, _) in keypairs]
    public_keys = [b for (_, b) in keypairs]
    btilde = aggregate_pubkeys(ctx, public_keys)

    
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
    N = 1/total
    a = 0
    b = 0
    c = 0

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
        

        w_expected = w1 + w2
        b_expected = b1 + b2

        messages_w = [w1,w2]
        messages_b = [[b1],[b2]]

        scaled_w = [[i*10**sf for i in j] for j in messages_w]
        scaled_b = [[i*10**sf for i in j] for j in messages_b]

        start_time1 = time.time()

        plaintexts_w = [encode_plain(ctx, vec) for vec in scaled_w]
        plaintexts_b = [encode_plain(ctx, vec) for vec in scaled_b]

        ciphertexts_w = [encrypt(ctx, m, btilde) for m in plaintexts_w]
        ciphertexts_b = [encrypt(ctx, m, btilde) for m in plaintexts_b]

        end_time1 = time.time()
        #print("encryption time:", end_time1 - start_time1)
        delta1 = end_time1 - start_time1
        a = a + delta1

        start_time2 = time.time()        
        c0_agg_w, c1_agg_w = add_ciphertexts(ctx, ciphertexts_w)
        decrypt_shares_w = [compute_decrypt_share(ctx, c1_agg_w, s) for s in secret_keys]
        decoded_vector_w = final_decrypt(ctx, c0_agg_w, decrypt_shares_w)

        c0_agg_b, c1_agg_b = add_ciphertexts(ctx, ciphertexts_b)
        decrypt_shares_b = [compute_decrypt_share(ctx, c1_agg_b, s) for s in secret_keys]
        decoded_vector_b = final_decrypt(ctx, c0_agg_b, decrypt_shares_b)

        w_dec = [(i/10**sf) for i in decoded_vector_w[:8]]
        b_dec = [(i/10**sf) for i in decoded_vector_b[:1]]

        w_avg = [(i*N) for i in w_dec]
        b_avg = [(i*N) for i in b_dec]

        #print("w expected = ",w_expected)
        #print("w real = ", w_dec)

        #print("b expected = ",b_expected)
        #print("b real = ", b_dec)

        end_time2 = time.time()
        #print("aggregation time:", end_time2 - start_time2)
        delta2 = end_time2 - start_time2
        b = b + delta2

        #print('w:', w_avg)
        #print('b:', b_avg)
        
        w_global = vector(RR, w_avg)
        b_global = b_avg[0]

    a = a/10
    b = b/10

    print("avg enc:", a)
    print("avg aggregation",b)

    acc = compute_accuracy(X_test, y_test, w_global, b_global)
    print("\nTest Accuracy:", round(acc * 100, 2), "%")

    #print("w = ", w_global)
    #print("b = ", b_global)

    #end_time = time.time()
    #print(f"Execution time seq: {end_time - start_time:.4f} seconds")

federated_learning()
