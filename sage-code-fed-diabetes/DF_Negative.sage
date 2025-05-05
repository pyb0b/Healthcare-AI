R.<x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30>=ZZ[]
index=[R.0,R.1,R.2,R.3,R.4,R.5,R.6,R.7,R.8,R.9,R.10,R.11,R.12,R.13,R.14,R.15,R.16,R.17,R.18,R.19,R.20,R.21,R.22,R.23,R.24,R.25,R.26,R.27,R.28,R.29]
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
		test=int(pow(s[0,i],-1,m))
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
	return int(centered_mod(int(mod(plain_text_Intermediate,m_p)),m_p))
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
	b=(-t_prime*A+m_prime*e_matrix+s1_Inv)*centered_mod(int(pow(s2_Inv[0,0],-1,m)),m)
	M=b.stack(A)
	return M
def centered_mod(x, n):
    r = x % n
    if r >= n / 2:
        r -= n
    return r

def DF_Neg():
	###Security Parameter
	lam=330
	###Secret Modulus
	m_prime=500
	print("m_prime=", m_prime)
	###Public Modulus
	m=pow(m_prime,lam)
	print("m=", m)
	###Generating the secret key r, the secret key r should be invertible in the public ring Z_m
	####d is the cipher-text dimension
	print ("Please enter the cipher-text dimension")
	d=30
	d_extended=factorial(d)/(factorial(2)*factorial(d-2))+d
	#print("d=", d)
	s=[]
	s=Secret_Key_Generation(d,m)
	####Generating a plain-text x from the private ring m_prime
	x=randint(-20,20)
	#print("test0")
	print("x=",x)
	Cipher=DF_Enc(x,m,m_prime,r,d,s)
	#print("Cipher1=", Cipher1)
	#print("test1")
	#print("s=", s)
	s_Inv=Inv_Secret_Key_Generation(s,m)
	#print("s1_Inv=", s1_Inv)
	x_dec=DF_Dec(Cipher,m_prime,d,s_Inv)
	print("x_dec=", x_dec)
	if(x==x_dec):
		print("Encryption and Decryption operations are working well!!!")
	###Verifying the addition property
	x1=randint(-20,20)
	#print("x1=", x1)
	Cipher1=DF_Enc(x1,m,m_prime,r,d,s)
	x2=randint(-20,20)
	#print("x2=", x2)
	Cipher2=DF_Enc(x2,m,m_prime,r,d,s)
	x_add=centered_mod(x1+x2,m_prime)
	print("x_add=", x_add)
	Cipher_add=Cipher1+Cipher2
	x_add_Dec=DF_Dec(Cipher_add,m_prime,d,s_Inv)
	print("x_add_Dec=", x_add_Dec)
	if(x_add==x_add_Dec):
		print("Homomorphic Addition is working well!!!")
	####Verifying the multiplication property
	x1=randint(-20,20)
	Cipher1=DF_Enc(x1,m,m_prime,r,d,s)
	x2=randint(-20,20)
	Cipher2=DF_Enc(x2,m,m_prime,r,d,s)
	Cipher_mult=cipher_multiplication(Cipher1,Cipher2,m,index)
	x_mult=centered_mod(x1*x2,m_prime)
	print("x_mult=", x_mult)
	###Generating the public matrix M
	s1=Secret_Key_Generation(d,m)
	s1_Inv=Inv_Secret_Key_Generation(s1,m)
	s_extended=[]
	for tt in range(0,s.ncols()):
		for kk in range(0,tt+1):
			s_extended.append(centered_mod((int(mod(s[0,tt]*s[0,kk],m))),m))
	s_extended_matrix=matrix(s_extended)
	s_extended_matrix_inverse=Inv_Secret_Key_Generation(s_extended_matrix,m)
	M=Generating_Public_Matrix(s_extended_matrix_inverse,s1_Inv,m,m_prime,d,d_extended)
	Cipher_Mult_fresh=(M*Cipher_mult.transpose()).transpose()
	x_mult_decr=DF_Dec(Cipher_Mult_fresh,m_prime,d,s1_Inv)
	print("x_mult_decr=", x_mult_decr)
	if(x_mult_decr==x_mult):
		print("Decryption is working well after well!!")
	
	
	
	
