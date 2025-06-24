from sage.all import *
import pandas as pd
import json
import subprocess
import os

# Helper: write vector to JSON
def save_vector_to_file(filename, vector_data):
    with open(filename, "w") as f:
        json.dump([float(x) for x in vector_data], f)

# Helper: read vector from JSON
def load_vector_from_file(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return vector(RR, data)

# Write the external encryption Python script (real CKKS + SMPC via TenSEAL)
ckks_script = """import sys, json
import tenseal as ts 

# Load vectors and weights
with open(sys.argv[1], 'r') as f: w1 = json.load(f)
with open(sys.argv[2], 'r') as f: w2 = json.load(f)
c1, c2 = float(sys.argv[3]), float(sys.argv[4])

# Load biases
with open("b_values.json", "r") as f: b1, b2 = json.load(f)

# CKKS context
context = context = ts.context(
    scheme=ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.generate_galois_keys()
context.global_scale = 2**40

# Encrypt
enc_w1 = ts.ckks_vector(context, w1)
enc_w2 = ts.ckks_vector(context, w2)

# Weighted average
enc_avg = enc_w1 * c1 + enc_w2 * c2

# Simulate SMPC decryption (single party for this mock)
avg = enc_avg.decrypt()

# Encrypt biases
enc_b1 = ts.ckks_vector(context, [b1])
enc_b2 = ts.ckks_vector(context, [b2])
enc_b_avg = enc_b1 * c1 + enc_b2 * c2
b_avg = enc_b_avg.decrypt()[0]

with open("w_avg.json", 'w') as f:
    json.dump(avg, f)

with open("b_avg.json", 'w') as f:
    json.dump(b_avg, f)
"""

with open("ckks_encrypt.py", "w") as f:
    f.write(ckks_script)

# Logistic functions
def sigmoid(z): return 1 / (1 + exp(-z))

def logistic_loss_gradient(X, y, w, b):
    m = len(y)
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
        pred = 1 if sigmoid(w.dot_product(vector(RR, xi)) + b) >= 0.5 else 0
        if pred == yi: correct += 1
    return correct / len(y_test)

# Main federated learning function
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

    n1, n2 = len(X1), len(X2)
    c1, c2 = n1 / (n1 + n2), n2 / (n1 + n2)

    w_global = vector(RR, [0.0] * len(X1[0]))
    b_global = 0.0

    for r in range(rounds):
        print(f"\nRound {r+1}")

        # Client 1 training
        w1, b1 = w_global[:], b_global
        for _ in range(local_epochs):
            dw1, db1 = logistic_loss_gradient(X1, y1, w1, b1)
            w1 -= 0.1 * dw1
            b1 -= 0.1 * db1

        # Client 2 training
        w2, b2 = w_global[:], b_global
        for _ in range(local_epochs):
            dw2, db2 = logistic_loss_gradient(X2, y2, w2, b2)
            w2 -= 0.1 * dw2
            b2 -= 0.1 * db2


        # Save weights to file
        save_vector_to_file("w1.json", w1)
        save_vector_to_file("w2.json", w2)

        with open("b_values.json", "w") as f:
            json.dump([float(b1), float(b2)], f)
     
        # Call Python encryption + aggregation
        subprocess.run(["python3", "ckks_encrypt.py", "w1.json", "w2.json", str(c1), str(c2), "w_avg.json"], check=True)

        # Load averaged weights back
        #w_avg = load_vector_from_file("w_avg.json")
       
        w_avg = vector(RR, load_vector_from_file("w_avg.json"))
        with open("b_avg.json", "r") as f:
            b_avg = RR(json.load(f))
        
        #b_avg = b1 * c1 + b2 * c2  # Optional: encrypt bias in future

        # Update global model
        w_global = vector(RR, w_avg)
        b_global = b_avg
    
        print("w:", w_global)
        print("b:", b_global)

    acc = compute_accuracy(X_test, y_test, w_global, b_global)
    print(f"\nTest Accuracy: {round(acc * 100, 2)} %")

federated_learning()
