from random import randint, random
import random
from sympy import nextprime
#from sage.all import *
#R.<x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30>=ZZ[]
#index=[R.0,R.1,R.2,R.3,R.4,R.5,R.6,R.7,R.8,R.9,R.10,R.11,R.12,R.13,R.14,R.15,R.16,R.17,R.18,R.19,R.20,R.21,R.22,R.23,R.24,R.25,R.26,R.27,R.28,R.29]
R.<x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31,x32,x33,x34,x35,x36,x37,x38,x39,x40,x41,x42,x43,x44,x45,x46,x47,x48,x49,x50,x51,x52,x53,x54,x55,x56,x57,x58,x59,x60,x61,x62,x63,x64,x65,x66,x67,x68,x69,x70,x71,x72,x73,x74,x75,x76,x77,x78,x79,x80,x81,x82,x83,x84,x85,x86,x87,x88,x89,x90,x91,x92,x93,x94,x95,x96,x97,x98,x99,x100>=ZZ[]
index=[R.0,R.1,R.2,R.3,R.4,R.5,R.6,R.7,R.8,R.9,R.10,R.11,R.12,R.13,R.14,R.15,R.16,R.17,R.18,R.19,R.20,R.21,R.22,R.23,R.24,R.25,R.26,R.27,R.28,R.29,R.30,R.31,R.32,R.33,R.34,R.35,R.36,R.37,R.38,R.39,R.40,R.41,R.42,R.43,R.44,R.45,R.46,R.47,R.48,R.49,R.50,R.51,R.52,R.53,R.54,R.55,R.56,R.57,R.58,R.59,R.60,R.61,R.62,R.63,R.64,R.65,R.66,R.67,R.68,R.69,R.70,R.71,R.72,R.73,R.74,R.75,R.76,R.77,R.78,R.79,R.80,R.81,R.82,R.83,R.84,R.85,R.86,R.87,R.88,R.89,R.90,R.91,R.92,R.93,R.94,R.95,R.96,R.97,R.98,R.99]



def Secret_Key_Element(m):
	r=randint(1,m)
	while(gcd(r,m)!=1):
		r=randint(-floor(m/2),floor(m/2))
	return r


def Secret_Key_Generation(d,m):
	s=[]
	for i in range(0,d):
		s.append(Secret_Key_Element(m))
	return matrix(s)


def Inv_Secret_Key_Generation(s,m):
    s_Inv=[]
    for i in range(0,s.ncols()):
        test = int(pow(s[0,i],-1,m))
        s_Inv.append(centered_mod(test,m))
    return matrix(s_Inv)


def DF_Enc(x,m,m_prime,r,d,Secret_Key_Vector):
	x_decomposed=[]
	for i in range(0,d-1):
		x_decomposed.append(randint(-floor(m/2),floor(m/2)))
	x_decomposed.append(int(centered_mod(x-sum(x_decomposed),m_prime)))
	Cipher=[]
	for i in range(0,d):
		Cipher.append(centered_mod(x_decomposed[i]*Secret_Key_Vector[0,i],m))
	return matrix(Cipher)


def DF_Dec(Cipher,m_p,d,Secret_Key_Vector_Inv):
	plain_text_Intermediate=0
	for i in range(0,d):
		plain_text_Intermediate+=Cipher[0,i]*Secret_Key_Vector_Inv[0,i]
	return int(centered_mod(plain_text_Intermediate,m_p))


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
			A[i,j]=randint(-floor(m/2),floor(m/2))
	t_prime=s2_Inv.submatrix(0, 1, 1, M_rows - 1)
	e_matrix=matrix(1,M_cols)
	for i in range(0,1):
		for j in range(0,M_cols):
			e_matrix[i,j]=randint(-floor(m/2),floor(m/2))
	b=matrix(1,M_cols)
	b=(-t_prime*A+m_prime*e_matrix+s1_Inv)*centered_mod(pow(s2_Inv[0,0],-1,m),m)
	M=b.stack(A)
	return M


def centered_mod(x, n):
    r = int(x) % int(n)
    if r >= int(n) / 2:
        r -= int(n)
    return r


def DF_KS_float_coeff():
	###Security Parameter
    lam=3
    sf=10
    alpha = 80
    ###Secret Modulus
    m_prime=pow(2,alpha)
    print("m_prime=", m_prime)
	###Public Modulus
    m=pow(m_prime,lam)
	#print("m=", m)
	###Generating the secret key r, the secret key r should be invertible in the public ring Z_m
	####d is the cipher-text dimension
    d=8
    d_extended=factorial(d)/(factorial(2)*factorial(d-2))+d
        ####We suppose here that we have local model 1
    s1=[]
    s1=Secret_Key_Generation(d,m)
    s1_Inv=Inv_Secret_Key_Generation(s1,m)
    c1=random.uniform(-5,-1)
    print("c1=",c1)
    c1_approx=int(round(c1, sf)*pow(10,sf))
    print("c1_approx=", c1_approx)
    Cipherc1=DF_Enc(c1_approx,m,m_prime,r,d,s1)
    w1=random.uniform(0.1,1)
    print("w1=", w1)
    w1_approx=int(round(w1, sf)*pow(10,sf))
    print("w1_approx=", w1_approx)
    Cipherw1=DF_Enc(w1_approx,m,m_prime,r,d,s1)
    ####We suppose here that we have local model 2
    s2=[]
    s2=Secret_Key_Generation(d,m)
    s2_Inv=Inv_Secret_Key_Generation(s2,m)
    c2=random.uniform(-5,-1)
    print("c2=", c2)
    c2_approx=int(round(c2, sf)*pow(10,sf))
    print("c2_approx=", c2_approx)
    Cipherc2=DF_Enc(c2_approx,m,m_prime,r,d,s2)
    w2=random.uniform(0.1,1)
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
            s_extended.append(centered_mod(s[0,tt]*s[0,kk],m))
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
    ##s_final was implemeneted just as a P.O.C in order to make sure that the code is wokring.
    ##To implement the code in a better way, it is important to switch from s_extended to s1 (client#1) and to s2 (for client#2)
    #s_final=[]
    #s_final=Secret_Key_Generation(d,m)
    #s_final_Inv=Inv_Secret_Key_Generation(s_final,m)
    M1_final=Generating_Public_Matrix(s_extended_matrix_inverse,s1_Inv,m,m_prime,d,d_extended)
    t1=cputime()
    cipher_final=(M1_final*cipher_final_extended.transpose()).transpose()
    print("final KS time is", cputime()-t1)
    final_coefficient_decr=DF_Dec(cipher_final,m_prime,d,s1_Inv)
    print("final_coefficient_decr=", float((final_coefficient_decr)/10^(2*sf)))


#DF_KS_float_coeff()
	
	
	
	
	
	
