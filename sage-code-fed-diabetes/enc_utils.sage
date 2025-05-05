from random import randint, random
from sympy import nextprime
from sage.all import *

# =================== DF ENCRYPTION FUNCTIONS ===================
R.<x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30> = ZZ[]
Index = [R.0, R.1, R.2, R.3, R.4, R.5, R.6, R.7, R.8, R.9, R.10, R.11, R.12, R.13, R.14,
         R.15, R.16, R.17, R.18, R.19, R.20, R.21, R.22, R.23, R.24, R.25, R.26, R.27, R.28, R.29]


def secret_key_element(m):
    """
    Generate a random secret key element that is coprime with modulus m.
    
    Parameters:
    - m: modulus

    Returns:
    - An integer coprime with m
    """
    r = randint(1, m)
    while gcd(r, m) != 1:
        r = randint(1, m)
    return r

def secret_key_generation(d, m):
    """
    Generate a secret key vector of dimension d.

    Parameters:
    - d: dimension of the secret key
    - m: modulus

    Returns:
    - A 1 × d Sage matrix of integers
    """
    s = [secret_key_element(m) for _ in range(d)]
    return matrix(s)

def inv_secret_key_generation(s, m):
    """
    Compute the modular inverse of a secret key vector.

    Parameters:
    - s: secret key matrix of shape (1 × d)
    - m: modulus

    Returns:
    - A 1 × d Sage matrix of modular inverses
    """
    return matrix([pow(s[0, i], -1, m) for i in range(s.ncols())])

def scale_round(val, sf):
    """
    Scale and round a real value using a scale factor.

    Parameters:
    - val: float value
    - sf: scale factor (decimal precision)

    Returns:
    - Integer representation
    """
    return int(round(val, sf) * 10**sf)

def rescale_float(val, sf):
    """
    Rescale an integer back to a float.

    Parameters:
    - val: scaled integer
    - sf: scale factor used during encryption

    Returns:
    - Float value
    """
    return float(val / 10**sf)

def DF_enc(x, m, m_prime, d, secret_key_vector):
    """
    Encrypt an integer value using Domingo-Ferrer encryption.

    Parameters:
    - x: integer value to encrypt
    - m: encryption modulus
    - m_prime: modular base for decomposition
    - d: dimension of encryption vector
    - secret_key_vector: 1 × d secret key matrix

    Returns:
    - Encrypted 1 × d Sage matrix
    """
    x_decomposed = [randint(1, m) for _ in range(d - 1)]
    x_decomposed.append(int(mod(x - sum(x_decomposed), m_prime)))
    cipher = [mod(x_decomposed[i] * secret_key_vector[0, i], m) for i in range(d)]
    return matrix(cipher)

def DF_dec(cipher, m_prime, d, secret_key_vector_inv):
    """
    Decrypt a DF-encrypted vector.

    Parameters:
    - cipher: 1 × d encrypted matrix
    - m_prime: base modulus used in encryption
    - d: vector dimension
    - secret_key_vector_inv: modular inverse of secret key

    Returns:
    - Integer plaintext value
    """
    plain_text_intermediate = sum(cipher[0, i] * secret_key_vector_inv[0, i] for i in range(d))
    return int(mod(plain_text_intermediate, m_prime))

def encrypt_value(val, sf, m, m_prime, d, secret_key_vector):
    """
    Encrypt a single real value.

    Parameters:
    - val: float value
    - sf: scale factor
    - m, m_prime, d: encryption parameters
    - secret_key_vector: DF secret key

    Returns:
    - Encrypted 1 × d Sage matrix
    """
    return DF_enc(scale_round(val, sf), m, m_prime, d, secret_key_vector)

def encrypt_values(val_list, sf, m, m_prime, d, secret_key_vector):
    """
    Encrypt a list of real values.

    Parameters:
    - val_list: list of float values
    - sf: scale factor
    - m, m_prime, d: encryption parameters
    - secret_key_vector: DF secret key

    Returns:
    - Encrypted matrix (n × d)
    """
    row_list = []
    for val in val_list:
        enc = DF_enc(scale_round(val, sf), m, m_prime, d, secret_key_vector)
        row_list.append(list(enc[0]))
    return matrix(row_list)

def decrypt_value(cipher, sf, m_prime, d, secret_key_vector_inv):
    """
    Decrypt a single encrypted value and rescale to float.

    Parameters:
    - cipher: 1 × d encrypted matrix
    - sf: scale factor
    - m_prime, d: decryption parameters
    - secret_key_vector_inv: inverse of secret key

    Returns:
    - Decrypted float value
    """
    int_val = DF_dec(cipher, m_prime, d, secret_key_vector_inv)
    return rescale_float(int_val, sf)

def decrypt_values(cipher_matrix, sf, m_prime, d, secret_key_vector_inv):
    """
    Decrypt a matrix of encrypted values row by row.

    Parameters:
    - cipher_matrix: matrix of shape (n × d)
    - sf, m_prime, d: decryption parameters
    - secret_key_vector_inv: inverse secret key

    Returns:
    - List of decrypted float values
    """
    decrypted = []
    for row in cipher_matrix.rows():
        int_val = DF_dec(matrix([row]), m_prime, d, secret_key_vector_inv)
        decrypted.append(rescale_float(int_val, sf))
    return decrypted

def generate_public_matrix(secret_key_vector_inv1, secret_key_vector_inv2, m, m_prime, M_rows, M_cols):
    """
    Generate a key-switching matrix.

    Parameters:
    - secret_key_vector_inv1, secret_key_vector_inv2: secret key inverses
    - m, m_prime: moduli
    - M_rows, M_cols: dimensions of matrix

    Returns:
    - Public key-switching matrix M
    """
    A = matrix(M_rows - 1, M_cols)
    for i in range(M_rows - 1):
        for j in range(M_cols):
            A[i, j] = randint(1, m)
    t_prime = secret_key_vector_inv2.submatrix(0, 1, 1, M_rows - 1)
    e_matrix = matrix(1, M_cols, [randint(1, m) for _ in range(M_cols)])
    b = (-t_prime * A + m_prime * e_matrix + secret_key_vector_inv1) * pow(secret_key_vector_inv2[0, 0], -1, m)
    return b.stack(A)


def generate_extended_key(secret_key_vector, m):
    secret_key_vector_extended = [
        mod(secret_key_vector[0, tt] * secret_key_vector[0, kk], m)
        for tt in range(secret_key_vector.ncols())
        for kk in range(tt + 1)
    ]
    secret_key_extended = matrix([secret_key_vector_extended])
    return secret_key_extended
    
def cipher_multiplication(cipher1, cipher2, index):
    """
    Multiply two encrypted vectors as polynomials.

    Parameters:
    - cipher1, cipher2: encrypted vectors (1 × d)
    - index: polynomial variables

    Returns:
    - Coefficient vector of resulting polynomial as 1 × d' matrix
    """
    poly1 = sum(cipher1[0, tt] * index[tt] for tt in range(cipher1.ncols()))
    poly2 = sum(cipher2[0, tt] * index[tt] for tt in range(cipher2.ncols()))
    return matrix(expand(poly1 * poly2).coefficients(sparse=False))

def duplicate_cipher_row(cipher_row, n):
    """
    Duplicate a single row into an n-row matrix.

    Parameters:
    - cipher_row: a 1 × d encrypted row
    - n: number of duplications

    Returns:
    - Matrix of shape (n × d)
    """
    row_values = list(cipher_row[0])
    return matrix([row_values] * n)

def cipher_multiplication_matrixwise(cipher_matrix1, cipher_matrix2, index):
    """
    Multiply two matrices of encrypted vectors row-by-row using polynomial expansion.

    Parameters:
    - cipher_matrix1, cipher_matrix2: matrices of shape (n × d)
    - index: polynomial variable list

    Returns:
    - Matrix of resulting polynomial coefficient vectors (n × d')
    """
    result_rows = []
    for row1, row2 in zip(cipher_matrix1.rows(), cipher_matrix2.rows()):
        poly1 = sum(c * x for c, x in zip(row1, index))
        poly2 = sum(c * x for c, x in zip(row2, index))
        product_poly = expand(poly1 * poly2)
        coeffs = product_poly.coefficients()
        result_rows.append(coeffs)
    max_len = max(len(row) for row in result_rows)
    padded_rows = [row + [0] * (max_len - len(row)) for row in result_rows]
    return matrix(padded_rows)



