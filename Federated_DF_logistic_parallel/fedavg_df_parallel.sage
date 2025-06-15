# Parallel version of the original federated learning code with Domingo-Ferrer encryption
# using multiprocessing. All encryption logic, parameters, and training procedure are kept as is.

from multiprocessing import Process, Queue
from sage.all import *
from sympy import nextprime
from random import randint, random
import time
import pandas as pd
import sys

sys.set_int_max_str_digits(1000000)

load("enc_utils_updated.sage")

def sigmoid(z):
    return 1 / (1 + exp(-z))

def logistic_loss_gradient(X, y, w, b):
    m = len(y)
    w = vector(RR, w)
    b = RR(b)
    grad_w = vector(RR, [0.0] * len(w))
    grad_b = 0.0
    for i in range(m):
        xi = vector(RR, X[i])  ######forced vector
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

def client_process(client_id, data_file, send_q, recv_q, s_global_inv):
    if client_id == 0:
        start_time = time.time()
    sf = 10
    lam = 330
    m_prime = nextprime(1000 * pow(10, 2 * sf + 1))
    m = pow(m_prime, lam)
    d = 30
    d_ext = d * (d + 1) // 2
    rounds = 10
    local_epochs = 10

    data = pd.read_csv(data_file)
    X = [vector(RR, row[:-1]) for row in data.values]
    y = [row[-1] for row in data.values]
    coef = len(X) / (len(pd.read_csv("datalocal1.csv")) + len(pd.read_csv("datalocal2.csv")))

    w = vector(RR, [10] * len(X[0]))
    b = 0.0

    s = secret_key_generation(d, m)
    s_inv = inv_secret_key_generation(s, m)
    s_ext = matrix(generate_extended_key(s, m))
    s_ext_inv = inv_secret_key_generation(s_ext, m)

    M_ext_s = generate_public_matrix(s_ext_inv, s_global_inv, m, m_prime, d, d_ext)
    M_s_client = generate_public_matrix(s_global_inv, s_inv, m, m_prime, d, d)

    for r in range(rounds):
        for _ in range(local_epochs):
            grad_w, grad_b = logistic_loss_gradient(X, y, w, b)
            w -= 0.1 * grad_w
            b -= 0.1 * grad_b

        w_enc = encrypt_values(w, sf, m, m_prime, d, s)
        b_enc = encrypt_value(b, sf, m, m_prime, d, s)
        coef_enc = encrypt_value(coef, sf, m, m_prime, d, s)

        send_q.put((client_id, w_enc, b_enc, coef_enc, M_ext_s, M_s_client))

        agg_w_enc, agg_b_enc = recv_q.get()
        w_dec = decrypt_values(agg_w_enc, 2 * sf, m_prime, d, s_inv)
        b_dec = decrypt_value(agg_b_enc, 2 * sf, m_prime, d, s_inv)

        #print(f"Client {client_id} - Decrypted weights:", w_dec)
        #print(f"Client {client_id} - Decrypted bias:", b_dec)
        w = vector(RR, w_dec)
        b = b_dec

    if client_id == 0:
        test_data = pd.read_csv("data_test.csv")
        X_test = [vector(RR, row[:-1]) for row in test_data.values]
        y_test = [row[-1] for row in test_data.values]
        acc = compute_accuracy(X_test, y_test, w, b)
        print("\nTest Accuracy:", round(acc * 100, 2), "%")

        end_time = time.time()
        print(f"Execution time (parallel): {end_time - start_time:.4f} seconds")

def server_process(q1, q2, q1_back, q2_back, s_global_inv, Index):
    rounds = 10
    for r in range(rounds):
        c1_id, w1, b1, c1_coef, M1_ext_s, M1_s_s1 = q1.get()
        c2_id, w2, b2, c2_coef, M2_ext_s, M2_s_s2 = q2.get()

        sf = 10
        d = 30

        c1_dup = duplicate_cipher_row(c1_coef, d)
        c2_dup = duplicate_cipher_row(c2_coef, d)

        w1c1 = cipher_multiplication_matrixwise(w1, c1_dup, Index)
        b1c1 = cipher_multiplication_matrixwise(b1, c1_coef, Index)
        w2c2 = cipher_multiplication_matrixwise(w2, c2_dup, Index)
        b2c2 = cipher_multiplication_matrixwise(b2, c2_coef, Index)

        w1_s = (M1_ext_s * w1c1.transpose()).transpose()
        b1_s = (M1_ext_s * b1c1.transpose()).transpose()
        w2_s = (M2_ext_s * w2c2.transpose()).transpose()
        b2_s = (M2_ext_s * b2c2.transpose()).transpose()

        w_agg_s = w1_s + w2_s
        b_agg_s = b1_s + b2_s

        w_agg_s1 = (M1_s_s1 * w_agg_s.transpose()).transpose()
        b_agg_s1 = (M1_s_s1 * b_agg_s.transpose()).transpose()
        w_agg_s2 = (M2_s_s2 * w_agg_s.transpose()).transpose()
        b_agg_s2 = (M2_s_s2 * b_agg_s.transpose()).transpose()

        q1_back.put((w_agg_s1, b_agg_s1))
        q2_back.put((w_agg_s2, b_agg_s2))

if __name__ == "__main__":
    sf = 10
    lam = 330
    m_prime = nextprime(1000 * pow(10, 2 * sf + 1))
    m = pow(m_prime, lam)
    d = 30

    s_global = secret_key_generation(d, m)
    s_global_inv = inv_secret_key_generation(s_global, m)

    q1 = Queue()
    q2 = Queue()
    q1_back = Queue()
    q2_back = Queue()

    client1 = Process(target=client_process, args=(0, "datalocal1.csv", q1, q1_back, s_global_inv))
    client2 = Process(target=client_process, args=(1, "datalocal2.csv", q2, q2_back, s_global_inv))
    server = Process(target=server_process, args=(q1, q2, q1_back, q2_back, s_global_inv, Index))

    client1.start()
    client2.start()
    server.start()

    client1.join()
    client2.join()
    server.join()
