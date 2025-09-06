from sage.all import *
import random

class CKKSContext:
    def __init__(self, lambda_param):
        param_sets = {
                      2048: {'n': 2048, 'logq': 110, 'scale': 1048576, 'q_param': 1298074214633706907132624082305051},
                      4096: {'n': 4096, 'logq': 180, 'scale': 1048576, 'q_param': 1532495540865888858358347027150309183618739122183602191},
                      8192: {'n': 8192, 'logq': 218, 'scale': 1048576, 'q_param': 421249166674228746791672110734681729275580381602196445017243910307},
                      16384: {'n': 16384, 'logq': 438, 'scale': 1048576, 'q_param': 709803441694928604052074031140629428079727891296209043243642772637343054798240159498233447962659731992932150006119314388217384402969},
                      }
        

        if lambda_param not in param_sets:
            raise ValueError("Unsupported λ. Choose from 2048, 4096, 8192, 16384")

        p = param_sets[lambda_param]
        #raw_q = 2 ** p["logq"]
        self.q = p["q_param"]
        self.scale = p["scale"]
        self.n = p["n"]
        R = PolynomialRing(Zmod(self.q), 'x')
        x = R.gen()
        self.R = R.quotient(x**self.n + 1)
        self.a = self.R.random_element()
        self.chi_params = {"type": "ternary"}
        self.psi_params = {"type": "gaussian", "sd": 3.2}
        self.phi_params = {"type": "gaussian", "sd": 3.2}

def _sample_coeff(params):
    if params['type'] == 'ternary':
        return random.choice([-1, 0, 1])
    elif params['type'] == 'gaussian':
        return Integer(round(random.gauss(0, params['sd'])))
    else:
        raise ValueError(f"Unknown distribution type {params['type']}")

def sample_small(ctx, dist):
    params = getattr(ctx, f"{dist}_params")
    coeffs = [_sample_coeff(params) for _ in range(ctx.n)]
    return ctx.R(coeffs)

def keygen(ctx):
    s = sample_small(ctx, 'chi')
    e = sample_small(ctx, 'psi')
    b = -s * ctx.a + e
    return s, b

def aggregate_pubkeys(ctx, pubkeys):
    return sum(pubkeys, ctx.R(0))

def encode_plain(ctx, vec):
    if len(vec) > ctx.n:
        raise ValueError("Vector length exceeds ring dimension.")
    coeffs = [Integer(round(v * ctx.scale)) for v in vec] + [0] * (ctx.n - len(vec))
    return ctx.R(coeffs)

def decode_plain(ctx, m):
    coeffs = m.list()
    q = ctx.q
    def center(c):
        c = Integer(c)
        return c if c <= q // 2 else c - q
    return [float(center(c) / ctx.scale) for c in coeffs]

def encrypt(ctx, m, btilde):
    v  = sample_small(ctx, 'chi')
    e0 = sample_small(ctx, 'psi')
    e1 = sample_small(ctx, 'psi')
    c0 = v * btilde + m + e0
    c1 = v * ctx.a    + e1
    return c0, c1

def add_ciphertexts(ctx, cts):
    c0_sum = sum((ct[0] for ct in cts), ctx.R(0))
    c1_sum = sum((ct[1] for ct in cts), ctx.R(0))
    return c0_sum, c1_sum

def compute_decrypt_share(ctx, c1, s):
    epsilon = sample_small(ctx, 'phi')
    return s * c1 + epsilon

def final_decrypt(ctx, c0, shares):
    M = c0 + sum(shares, ctx.R(0))
    return decode_plain(ctx, M)

def main():
    print(1-(1-6/(3.14**2))**4)
    ctx = CKKSContext(lambda_param=2048)

    print("=== Context Info ===")
    print(f"Modulus q = {ctx.q}")
    print(f"Scale = {ctx.scale}")
    print(f"Ring dimension n = {ctx.n}")

    num_clients = 10
    keypairs = [keygen(ctx) for _ in range(num_clients)]
    secret_keys = [s for (s, _) in keypairs]
    public_keys = [b for (_, b) in keypairs]

    messages = [[i + 0.5, i + 0.6] for i in range(num_clients)]
    plaintexts = [encode_plain(ctx, vec) for vec in messages]

    #print("Sample encoded plaintext coefficients (client 0):")
    #print(plaintexts[0].list()[:5])

    btilde = aggregate_pubkeys(ctx, public_keys)
    ciphertexts = [encrypt(ctx, m, btilde) for m in plaintexts]
    c0_agg, c1_agg = add_ciphertexts(ctx, ciphertexts)

    decrypt_shares = [compute_decrypt_share(ctx, c1_agg, s) for s in secret_keys]
    decoded_vector = final_decrypt(ctx, c0_agg, decrypt_shares)

    expected = [sum(msg[i] for msg in messages) for i in range(len(messages[0]))]

    print("\n=== Homomorphic Aggregation (xMK-CKKS) ===")
    print("Input messages =", messages)
    print("Expected (plaintext sum):  ", [round(val, 4) for val in expected])
    print("Decrypted (HE result):     ", [round(float(v), 4) for v in decoded_vector[:len(expected)]])

#main()