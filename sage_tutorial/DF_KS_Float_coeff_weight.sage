import random
from sympy import nextprime
R.<x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30>=ZZ[]
index=[R.0,R.1,R.2,R.3,R.4,R.5,R.6,R.7,R.8,R.9,R.10,R.11,R.12,R.13,R.14,R.15,R.16,R.17,R.18,R.19,R.20,R.21,R.22,R.23,R.24,R.25,R.26,R.27,R.28,R.29]


def Secret_Key_Element(m):
	r=randint(1,m)
	while(gcd(r,m)!=1):
		r=randint(1,m)
	return r


def Secret_Key_Generation(d,m):
	s=[]
	for i in range(0,d):
		s.append(Secret_Key_Element(m))
	return matrix(s)


def Inv_Secret_Key_Generation(s,m):
	s_Inv=[]
	for i in range(0,s.ncols()):
		s_Inv.append(pow(s[0,i],-1,m))
	return matrix(s_Inv)


def DF_Enc(x,m,m_prime,r,d,Secret_Key_Vector):
	x_decomposed=[]
	for i in range(0,d-1):
		x_decomposed.append(randint(1,m))
	x_decomposed.append(int(mod(x-sum(x_decomposed),m_prime)))
	Cipher=[]
	for i in range(0,d):
		Cipher.append(mod(x_decomposed[i]*Secret_Key_Vector[0,i],m))
	return matrix(Cipher)


def DF_Dec(Cipher,m_p,d,Secret_Key_Vector_Inv):
	plain_text_Intermediate=0
	for i in range(0,d):
		plain_text_Intermediate+=Cipher[0,i]*Secret_Key_Vector_Inv[0,i]
	return int(mod(plain_text_Intermediate,m_p))


def cipher_multiplication(cipher1,cipher2,public_modulus,index):
        cipher1_polynom=0
        for tt in range(0,cipher1.ncols()):
                cipher1_polynom=cipher1_polynom+cipher1[0,tt]*index[tt]
        cipher2_polynom=0
        for tt in range(0,cipher2.ncols()):
                cipher2_polynom=cipher2_polynom+cipher2[0,tt]*index[tt]
        c_mult_polynom=expand(cipher1_polynom*cipher2_polynom)
        c_mult=c_mult_polynom.coefficients()
        return(matrix(c_mult))


def Generating_Public_Matrix(s1_Inv,s2_Inv,m,m_prime,M_rows,M_cols):
	###Generating the public matrix M
	A=matrix(M_rows-1,M_cols)
	for i in range(0,M_rows-1):
		for j in range(0,M_cols):
			A[i,j]=randint(1,m)
	t_prime=s2_Inv.submatrix(0, 1, 1, M_rows - 1)
	e_matrix=matrix(1,M_cols)
	for i in range(0,1):
		for j in range(0,M_cols):
			e_matrix[i,j]=randint(1,m)
	b=matrix(1,M_cols)
	b=(-t_prime*A+m_prime*e_matrix+s1_Inv)*pow(s2_Inv[0,0],-1,m)
	M=b.stack(A)
	return M


def DF_KS_float_coeff():
	###Security Parameter
	lam=330
	sf=15
	###Secret Modulus
	m_prime=nextprime(1000*pow(10,2*sf+1))
	print("m_prime=", m_prime)
	###Public Modulus
	m=pow(m_prime,lam)
	#print("m=", m)
	###Generating the secret key r, the secret key r should be invertible in the public ring Z_m
	####d is the cipher-text dimension
	d=30
	d_extended=factorial(d)/(factorial(2)*factorial(d-2))+d
        ####We suppose here that we have local model 1
	s1=[]
	s1=Secret_Key_Generation(d,m)
	s1_Inv=Inv_Secret_Key_Generation(s1,m)
	c1=random.uniform(0,1)
	print("c1=",c1)
	c1_approx=int(round(c1, sf)*pow(10,sf))
	print("c1_approx=", c1_approx)
	Cipherc1=DF_Enc(c1_approx,m,m_prime,r,d,s1)
	w1=random.uniform(0,1)
	print("w1=", w1)
	w1_approx=int(round(w1, sf)*pow(10,sf))
	print("w1_approx=", w1_approx)
	Cipherw1=DF_Enc(w1_approx,m,m_prime,r,d,s1)
	####We suppose here that we have local model 2
	s2=[]
	s2=Secret_Key_Generation(d,m)
	s2_Inv=Inv_Secret_Key_Generation(s2,m)
	c2=random.uniform(0,1)
	print("c2=", c2)
	c2_approx=int(round(c2, sf)*pow(10,sf))
	print("c2_approx=", c2_approx)
	Cipherc2=DF_Enc(c2_approx,m,m_prime,r,d,s2)
	w2=random.uniform(0,1)
	print("w2=", w2)
	w2_approx=int(round(w2, sf)*pow(10,sf))
	print("w2_approx=", w2_approx)
	Cipherw2=DF_Enc(w2_approx,m,m_prime,r,d,s2)
	###############
	global_model_coeff=w1*c1+w2*c2
	print("global_model_coeff=", global_model_coeff)
	################
	s=Secret_Key_Generation(d,m)
	s_Inv=Inv_Secret_Key_Generation(s,m)
	s_extended=[]
	for tt in range(0,s.ncols()):
		for kk in range(0,tt+1):
			s_extended.append(mod(s[0,tt]*s[0,kk],m))
	s_extended_matrix=matrix(s_extended)
	s_extended_matrix_inverse=Inv_Secret_Key_Generation(s_extended_matrix,m)
	###I will generate a public matrix M1 for local model 1
	M1=Generating_Public_Matrix(s1_Inv,s_Inv,m,m_prime,d,d)
	t1=cputime()
	Cipherc1_p=(M1*Cipherc1.transpose()).transpose()
	Cipherw1_p=(M1*Cipherw1.transpose()).transpose()
	print("First KS technique execution time is", cputime()-t1)
	#####I wil generate a public matrix M2 for local model 2
	M2=Generating_Public_Matrix(s2_Inv,s_Inv,m,m_prime,d,d)
	t1=cputime()
	Cipherc2_p=(M2*Cipherc2.transpose()).transpose()
	Cipherw2_p=(M2*Cipherw2.transpose()).transpose()
	print("Second KS technique execution time is", cputime()-t1)
	##########Suppose that the global model is adding the two cipher-text
	cipher_mult_w1_c1=cipher_multiplication(Cipherc1_p,Cipherw1_p,m,index)
	cipher_mult_w2_c2=cipher_multiplication(Cipherc2_p,Cipherw2_p,m,index)
	cipher_final_extended=cipher_mult_w1_c1+cipher_mult_w2_c2
	##########
	s_final=[]
	s_final=Secret_Key_Generation(d,m)
	s_final_Inv=Inv_Secret_Key_Generation(s_final,m)
	M_final=Generating_Public_Matrix(s_extended_matrix_inverse,s_final_Inv,m,m_prime,d,d_extended)
	t1=cputime()
	cipher_final=(M_final*cipher_final_extended.transpose()).transpose()
	print("final KS time is", cputime()-t1)
	final_coefficient_decr=DF_Dec(cipher_final,m_prime,d,s_final_Inv)
	print("final_coefficient_decr=", float(final_coefficient_decr/10^(2*sf)))


	### Now we try to decrypt the final result using the key of local model 1 (s1)

	# Step 1: Create the public matrix to switch cipher_final from s_final → s1
	M_back1 = Generating_Public_Matrix(s_final_Inv, s1_Inv, m, m_prime, d, d)

	# Step 2: Apply key switching to bring cipher_final into s1's key space
	cipher_final_in_s1 = (M_back1 * cipher_final.transpose()).transpose()

	# Step 3: Decrypt using s1_Inv
	final_coefficient_decr_s1 = DF_Dec(cipher_final_in_s1, m_prime, d, s1_Inv)

	# Step 4: Print and compare the result
	print("final_coefficient_decr_s1=", float(final_coefficient_decr_s1 / 10^(2*sf)))


		### Now we try to decrypt the final result using the key of local model 2 (s2)

	# Step 1: Create the public matrix to switch cipher_final from s_final → s2
	M_back2 = Generating_Public_Matrix(s_final_Inv, s2_Inv, m, m_prime, d, d)

	# Step 2: Apply key switching to bring cipher_final into s2's key space
	cipher_final_in_s2 = (M_back2 * cipher_final.transpose()).transpose()

	# Step 3: Decrypt using s2_Inv
	final_coefficient_decr_s2 = DF_Dec(cipher_final_in_s2, m_prime, d, s2_Inv)

	# Step 4: Print and compare the result
	print("final_coefficient_decr_s2=", float(final_coefficient_decr_s2 / 10^(2*sf)))


def encrypt_local_model(d, m, m_prime, sf):
    """
    Generates secret key for a local model, encrypts a random weight, bias, and constant weight.

    Returns:
        - cipher_weight: Encrypted float weight
        - cipher_constant: Encrypted float constant for weighted average
        - secret_key, inverse_key: Key and its inverse used for encryption
        - weight_val, constant_val: Plaintext (for debug)
    """
    s = Secret_Key_Generation(d, m)
    s_Inv = Inv_Secret_Key_Generation(s, m)

    weight_val = random.uniform(0, 1)
    weight_approx = int(round(weight_val, sf) * 10**sf)
    cipher_weight = DF_Enc(weight_approx, m, m_prime, r, d, s)

    constant_val = random.uniform(0, 1)
    constant_approx = int(round(constant_val, sf) * 10**sf)
    cipher_constant = DF_Enc(constant_approx, m, m_prime, r, d, s)

    return cipher_weight, cipher_constant, s, s_Inv, weight_val, constant_val


def aggregate_encrypted_models(cipher_weights, cipher_constants, secret_keys_inv, d, m, m_prime, index):
    """
    Key-switch and aggregate encrypted weighted model parameters from multiple clients.
    
    Args:
        cipher_weights: List of encrypted weights [Cipher_w1, Cipher_w2, ...]
        cipher_constants: List of encrypted constant weights [α1, α2, ...]
        secret_keys_inv: List of inverse secret keys [s1_Inv, s2_Inv, ...]
        d: ciphertext dimension
        m: public modulus
        index: polynomial basis
    Returns:
        - cipher_final: Final aggregated ciphertext under a new key
        - s_final, s_final_Inv: Final key used after switching
    """
    s_target = Secret_Key_Generation(d, m)
    s_target_Inv = Inv_Secret_Key_Generation(s_target, m)

    mult_terms = []
    for Cipher_wi, Cipher_ai, s_inv in zip(cipher_weights, cipher_constants, secret_keys_inv):
        M = Generating_Public_Matrix(s_inv, s_target_Inv, m, m_prime, d, d)
        Cipher_wi_p = (M * Cipher_wi.transpose()).transpose()
        Cipher_ai_p = (M * Cipher_ai.transpose()).transpose()
        mult_terms.append(cipher_multiplication(Cipher_wi_p, Cipher_ai_p, m, index))

    cipher_final_extended = sum(mult_terms)
    
    # Prepare for final switch to a global key for decryption
    s_final = Secret_Key_Generation(d, m)
    s_final_Inv = Inv_Secret_Key_Generation(s_final, m)
    s_target_extended = [
        mod(s_target[0, tt] * s_target[0, kk], m)
        for tt in range(d) for kk in range(tt + 1)
    ]
    s_target_extended_matrix = matrix(s_target_extended)
    s_target_extended_inv = Inv_Secret_Key_Generation(s_target_extended_matrix, m)
    M_final = Generating_Public_Matrix(s_target_extended_inv, s_final_Inv, m, m_prime, d, len(s_target_extended))

    cipher_final = (M_final * cipher_final_extended.transpose()).transpose()
    return cipher_final, s_final, s_final_Inv


def decrypt_aggregated_model(cipher_final, s_final_Inv, m_prime, d, sf):
    """
    Decrypts the final model coefficient using the decryption key.

    Returns:
        - final_coeff: decrypted and scaled-down float result
    """
    decrypted = DF_Dec(cipher_final, m_prime, d, s_final_Inv)
    final_coeff = float(decrypted / 10**(2 * sf))
    print("final_coefficient_decr=", final_coeff)
    return final_coeff


def decrypt_with_local_key(cipher_final, s_final_Inv, s_local_Inv, m, m_prime, d, sf):
    """
    Performs key-switch from s_final to s_local, then decrypts using s_local_Inv.
    """
    M_back = Generating_Public_Matrix(s_final_Inv, s_local_Inv, m, m_prime, d, d)
    cipher_switched = (M_back * cipher_final.transpose()).transpose()
    decrypted = DF_Dec(cipher_switched, m_prime, d, s_local_Inv)
    final_coeff = float(decrypted / 10**(2 * sf))
    return final_coeff


def new_DF_KS_float_coeff():

	# Parameters
	lam = 330
	sf = 15
	m_prime = nextprime(1000 * 10**(2*sf + 1))
	m = m_prime**lam
	d = 30

	# Local models
	cipherc1, cipherw1, s1, s1_Inv, c1_val, w1_val = encrypt_local_model(d, m, m_prime, sf)
	cipherc2, cipherw2, s2, s2_Inv, c2_val, w2_val = encrypt_local_model(d, m, m_prime, sf)

	# Aggregate
	cipher_final, s_final, s_final_Inv = aggregate_encrypted_models(
		[cipherc1, cipherc2],
		[cipherw1, cipherw2],
		[s1_Inv, s2_Inv],
		d, m, m_prime, index
	)

	expected = w1_val * c1_val + w2_val * c2_val
	print("expected_plaintext_result=", expected)

	# Decrypt
	decrypt_aggregated_model(cipher_final, s_final_Inv, m_prime, d, sf)

	result_from_s1 = decrypt_with_local_key(cipher_final, s_final_Inv, s1_Inv, m, m_prime, d, sf)
	result_from_s2 = decrypt_with_local_key(cipher_final, s_final_Inv, s2_Inv, m, m_prime, d, sf)

	print("decryption with s1:", result_from_s1)
	print("decryption with s2:", result_from_s2)


