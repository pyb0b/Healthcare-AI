from random import randint, random
from sympy import nextprime
from sage.all import *
from sklearn.model_selection import train_test_split
import time

import pandas as pd 

import sys
sys.set_int_max_str_digits(1000000) 

load("utils_DF_KS.sage")

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

    enc = 0
    serv = 0
    dec = 0

    ###security parameters
    lam=3
    sf= 10
    alpha = 80
    m_prime=pow(2,alpha)
    #print("m_prime=", m_prime)
    m=pow(m_prime,lam)
    d=32
    d_extended= factorial(d)/(factorial(2)*factorial(d-2))+d
    d_extended = int(d_extended)
    ###Keys generation

    #client 1
    s1=[]
    s1=Secret_Key_Generation(d,m)
    s1_Inv=Inv_Secret_Key_Generation(s1,m)

    #client 2
    s2=[]
    s2=Secret_Key_Generation(d,m)
    s2_Inv=Inv_Secret_Key_Generation(s2,m)

    #key S for key switching
    s=Secret_Key_Generation(d,m)
    s_Inv=Inv_Secret_Key_Generation(s,m)
    s_extended=[]
    for tt in range(0,s.ncols()):
        for kk in range(0,tt+1):
            s_extended.append(centered_mod(s[0,tt]*s[0,kk],m))
    s_extended_matrix=matrix(s_extended)
    s_extended_matrix_inverse=Inv_Secret_Key_Generation(s_extended_matrix,m)
    
    #matrices generation
    M1=Generating_Public_Matrix(s1_Inv,s_Inv,m,m_prime,d,d)
    M2=Generating_Public_Matrix(s2_Inv,s_Inv,m,m_prime,d,d)

    M1_final=Generating_Public_Matrix(s_extended_matrix_inverse,s1_Inv,m,m_prime,d,d_extended)
    M2_final=Generating_Public_Matrix(s_extended_matrix_inverse,s2_Inv,m,m_prime,d,d_extended)

    
    d1 = pd.read_csv("datalocal1.csv") 
    d2 = pd.read_csv("datalocal2.csv")  
    dtest = pd.read_csv("data_test.csv")  

    X1 = [vector(RR, row[:-1]) for row in d1.values]  
    y1 = [row[-1] for row in d1.values]  

    X2 = [vector(RR, row[:-1]) for row in d2.values] 
    y2 = [row[-1] for row in d2.values]  

    X_test = [vector(RR, row[:-1]) for row in dtest.values]
    y_test = [row[-1] for row in dtest.values] 
    
    l1 = len(X1)
    l2 = len(X2)
    total = l1+l2
    c1 = l1/total
    c2 = l2/total

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
    
        #encryption for client 1
        #here
        start_enc = time.time()

        c1_approx=int(round(c1, sf)*int(pow(10,sf)))
        cipher_c1_s1=DF_Enc(c1_approx,m,m_prime,r,d,s1)

        w1_approx = [int(round(i, sf)*int(pow(10,sf))) for i in w1]
        cipher_w1_s1 = [DF_Enc(i,m,m_prime,r,d,s1) for i in w1_approx]

        b1_approx = int(round(b1, sf)*int(pow(10,sf)))
        cipher_b1_s1 = DF_Enc(b1_approx,m,m_prime,r,d,s1)

        end_enc = time.time()
        delta_enc = end_enc - start_enc
        enc = enc + delta_enc

        
        # Client 2
        w2 = w_global[:] 
        b2 = b_global

        for _ in range(local_epochs):
            grad_w2, grad_b2 = logistic_loss_gradient(X2, y2, w2, b2)
            w2 -= 0.1 * grad_w2
            b2 -= 0.1 * grad_b2

        #encryption for client 2
        #here
        c2_approx = int(round(c2, sf)*pow(10,sf))
        cipher_c2_s2 = DF_Enc(c2_approx,m,m_prime,r,d,s2)

        w2_approx = [int(round(i, sf)*pow(10,sf)) for i in w2]
        cipher_w2_s2 = [DF_Enc(i,m,m_prime,r,d,s2) for i in w2_approx]

        b2_approx = int(round(b2, sf)*pow(10,sf))
        cipher_b2_s2 = DF_Enc(b2_approx,m,m_prime,r,d,s2)


        #Server side
        #here
        start_serv = time.time()

        cipher_c1_s =(M1*cipher_c1_s1.transpose()).transpose()
        cipher_w1_s =[(M1*(i.transpose())).transpose() for i in cipher_w1_s1]
        cipher_b1_s =(M1*cipher_b1_s1.transpose()).transpose()

        cipher_c2_s =(M2*cipher_c2_s2.transpose()).transpose()
        cipher_w2_s =[(M2*(i.transpose())).transpose() for i in cipher_w2_s2]
        cipher_b2_s =(M2*cipher_b2_s2.transpose()).transpose()


        cipher_mult_w1_c1_s = [cipher_multiplication(cipher_c1_s, i, m, index) for i in cipher_w1_s]
        cipher_mult_w2_c2_s = [cipher_multiplication(cipher_c2_s, i, m, index) for i in cipher_w2_s]

        cipher_mult_b1_c1_s = cipher_multiplication(cipher_c1_s, cipher_b1_s, m, index)
        cipher_mult_b2_c2_s = cipher_multiplication(cipher_c2_s, cipher_b2_s, m, index)

        cipher_w_agg_s = [i+j for i,j in zip(cipher_mult_w1_c1_s, cipher_mult_w2_c2_s)]
        cipher_b_agg_s = cipher_mult_b1_c1_s + cipher_mult_b2_c2_s

        cipher_w_agg_s1 = [(M1_final*i.transpose()).transpose() for i in cipher_w_agg_s]
        cipher_w_agg_s2 = [(M2_final*i.transpose()).transpose() for i in cipher_w_agg_s]

        cipher_b_agg_s1 = (M1_final*cipher_b_agg_s.transpose()).transpose()
        cipher_b_agg_s2 = (M2_final*cipher_b_agg_s.transpose()).transpose()


        end_serv = time.time()
        delta_serv = end_serv - start_serv
        serv = serv + delta_serv

        #decryption client 1
        #here
        start_dec = time.time()

        w_agg_dec_s1 = [float((DF_Dec(i,m_prime,d,s1_Inv))/(pow(10,(2*sf)))) for i in cipher_w_agg_s1]
        b_agg_dec_s1 = float((DF_Dec(cipher_b_agg_s1,m_prime,d,s1_Inv))/(pow(10,2*sf)))

        end_dec = time.time()
        delta_dec = end_dec - start_dec
        dec = dec + delta_dec

        #decryption client 2
        #here
        w_agg_dec_s2 = [float((DF_Dec(i,m_prime,d,s2_Inv))/(pow(10,2*sf))) for i in cipher_w_agg_s2]
        b_agg_dec_s2 = float((DF_Dec(cipher_b_agg_s2,m_prime,d,s2_Inv))/(pow(10, 2*sf)))

        ##result:
        w_agg = w_agg_dec_s1  #should not matter if s1 or s2
        b_agg = b_agg_dec_s1
        
        ###to make sure only
        
        w_avg = w1 * c1 + w2 * c2
        b_avg  = b1 * c1 + b2 * c2
    
        print('w:', w_avg)
        print('b:', b_avg)

        print('w dec:', w_agg)
        print('b dec:', b_agg)

        w_global = vector(RR, w_agg)
        b_global = b_agg

    enc = enc/rounds
    serv = serv/rounds
    dec = dec/rounds

    print("encryption time: ", enc)
    print("server time: ", serv)
    print("decryption time: ", dec)

    acc = compute_accuracy(X_test, y_test, w_global, b_global)
    print("\nTest Accuracy:", round(acc * 100, 2), "%")

federated_learning()

