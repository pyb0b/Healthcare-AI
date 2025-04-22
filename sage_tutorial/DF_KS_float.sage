import random
from sympy import nextprime
import sys
sys.set_int_max_str_digits(1000000)  # or a higher number if needed

# Define the number
n = 336 * 10**10

# Find the next prime greater than n
prime = nextprime(n)

print(prime)

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
def Generating_Public_Matrix(s1_Inv,s2_Inv,m,m_prime,d):
	###Generating the public matrix M
	A=matrix(d-1,d)
	for i in range(0,d-1):
		for j in range(0,d):
			A[i,j]=randint(1,m)
	t_prime=s2_Inv.submatrix(0, 1, 1, d - 1)
	e_matrix=matrix(1,d)
	for i in range(0,1):
		for j in range(0,d):
			e_matrix[i,j]=randint(1,m)
	b=matrix(1,d)
	b=(-t_prime*A+m_prime*e_matrix+s1_Inv)*pow(s2_Inv[0,0],-1,m)
	M=b.stack(A)
	return M
def DF_KS_float():
	###Security Parameter
	lam=330
	sf=30
	###Secret Modulus
	m_prime=nextprime(336*pow(10,sf+1))
	#print("m_prime=", m_prime)
	###Public Modulus
	m=pow(m_prime,lam)
	#print("m=", m)
	###Generating the secret key r, the secret key r should be invertible in the public ring Z_m
	####d is the cipher-text dimension
	print ("Please enter the cipher-text dimension")
	d= 10
	#print("d=", d)
	s1=[]
	s1=Secret_Key_Generation(d,m)
	####Generating a plain-text x from the private ring m_prime
	x_float=random.uniform(1,336)
	print("x_float=",x_float)
	###I will suppose that I will take 10 digits after the floating point
	x_float_modified=int(round(x_float, 10)*pow(10,sf))
	print("s_float_modified=", x_float_modified)
	Cipher1=DF_Enc(x_float_modified,m,m_prime,r,d,s1)
	s1_Inv=Inv_Secret_Key_Generation(s1,m)
	x_float_modified_dec=DF_Dec(Cipher1,m_prime,d,s1_Inv)
	print("x_float_modified_dec=", x_float_modified_dec)
	if(x_float_modified==x_float_modified_dec):
		print("Encryption and Decryption operations are working well")
	x_dec_retrieved=float(x_float_modified_dec/10^sf)
	print("x_dec_retrieved=", x_dec_retrieved)
	###Genrating the new secret key s2
	s2=[]
	s2=Secret_Key_Generation(d,m)
	s2_Inv=Inv_Secret_Key_Generation(s2,m)
	M=matrix(d,d)
	M=Generating_Public_Matrix(s1_Inv,s2_Inv,m,m_prime,d)
	Cipher2=(M*Cipher1.transpose()).transpose()
	x_dec_2=DF_Dec(Cipher2,m_prime,d,s2_Inv)
	x_dec_2_retrieved=float(x_dec_2/10^sf)
	print("x_dec_2_retrieved=", x_dec_2_retrieved)
	
