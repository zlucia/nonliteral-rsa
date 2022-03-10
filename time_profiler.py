"""
Lucia Zheng
A Computational Implementation of an Extension of the Rational Speech Acts Model for Nonliteral Number Words
LINGUIST230A: Introduction to Semantics and Pragmatics, Winter 2022
"""
import numpy as np
from nonliteral_rsa_nv import NonliteralNumberRSANV
from nonliteral_rsa import NonliteralNumberRSA
import timeit
import matplotlib.pyplot as plt

if __name__ == '__main__':
	x = list(range(1, 11))
	rsa_nv_data = []
	rsa_data = []
	for n in x:
		print(n)
		# Core lexicon
		U = np.linspace(0, 10000, num=n, dtype=int)
		S = np.linspace(0, 10000, num=n, dtype=int)
		A = np.array([0, 1])
		lex = np.identity(len(U))
		s_prior = np.ones(S.shape) / len(S)
		as_prior = np.zeros((S.shape[0], A.shape[0])) 
		as_prior[:, 1] = np.linspace(0.01, 0.99, num=len(S))
		as_prior[:, 0] = 1 - as_prior[:, 1]

		rsa_nv = NonliteralNumberRSANV(lexicon=lex, utterances=U, states=S, affects=A, s_prior=s_prior, 
								  as_prior=as_prior, round_cost=1, sharp_cost=5, precision=1)

		rsa = NonliteralNumberRSA(lexicon=lex, utterances=U, states=S, affects=A, s_prior=s_prior, 
								  as_prior=as_prior, round_cost=1, sharp_cost=5, precision=1)

		rsa_nv_times = timeit.repeat("rsa_nv.pragmatic_listener()", "from __main__ import rsa_nv", repeat=3, number=1)
		rsa_times = timeit.repeat("rsa.pragmatic_listener()", "from __main__ import rsa", repeat=3, number=1)

		rsa_nv_time = np.mean(rsa_nv_times)
		rsa_time = np.mean(rsa_times)

		rsa_nv_data.append(rsa_nv_time)
		rsa_data.append(rsa_time)

	plt.plot(x, rsa_nv_data, label='Non-vectorized')
	plt.plot(x, rsa_data, label='Vectorized')
	plt.title("Time vs. Number of States / Utterances")
	plt.xlabel("Number of States / Utterances")
	plt.ylabel("Time (seconds)")
	plt.legend(title="Nonliteral RSA Implementation")
	plt.savefig('time/time_comparison.png')
	# plt.show()
