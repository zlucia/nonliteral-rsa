"""
Lucia Zheng
A Computational Implementation of an Extension of the Rational Speech Acts Model for Nonliteral Number Words
LINGUIST230A: Introduction to Semantics and Pragmatics, Winter 2022
"""
import numpy as np
import matplotlib.pyplot as plt
from nonliteral_rsa import NonliteralNumberRSA

class NonliteralNumberRSATemp(NonliteralNumberRSA):
	"""Implementation of the Rational Speech Acts model extension proposed in Kao et al. (2014) for nonliteral number words.
	Implementation with temperature parameter, controlling the strength of pragmatic inference.

	Parameters
	----------
	lexicon: np.array
		The semantic interpretation function [[.]]: U -> S -> {0, 1}, truth condition of [[u]](s)
	utterances: np.array
		Possible utterances
	states: np.array
		Possible states
	affects: np.array
		Possible affects, A = {0, 1} (0 means no affect and 1 means with affect), based on Kao et al. (2014)
	s_prior: np.array
		Prior probabilities of each state
	as_prior: np.array
		Prior probabilities of each affect given a state
	round_cost: float
		Cost of a rounded number utterance
	sharp_cost: float
		Cost of a non-rounded number utterance
	precision: int
		Rounding place value, rounding to the nearest 10^precision
	alpha: float
		Temperature, controls the strength of pragmatic inference, larger alpha results in stronger pragmatic inferences
	interpretation_space_dims: int
		The number of interpretation space dimensions, defaults to 2, state and affect, based on Kao et al. (2014)
	"""

	def __init__(self, lexicon, utterances, states, affects, s_prior, as_prior, round_cost, sharp_cost, precision, alpha, interpretation_space_dims=2):
		self.lexicon = lexicon
		self.utterances = utterances
		self.states = states
		self.affects = affects
		self.s_prior = s_prior
		self.as_prior = as_prior
		self.round_cost = round_cost
		self.sharp_cost = sharp_cost
		self.precision = precision
		self.alpha = alpha
		self.interpretation_space_dims = interpretation_space_dims

	def literal_listener(self):
		"""Literal listener predictions for all possible states and affects, given an utterance.
		Returns
		-------
		3D np.array. The first dimension corresponds to utterances, the second dimension corresponds
		to states, and the third dimension corresponds to affects.
		"""
		as_prior_each_u = np.zeros((self.utterances.shape[0], self.states.shape[0], self.affects.shape[0]))
		# Broadcast P(a|s) across utterances
		as_prior_each_u[:] = self.as_prior

		# L_0(s, a u): literal listener's joint probability of a state s, affect a, utterance u, for each s, a, u
		# shape: (|U|, |S|, |A|)
		L_0_sau = self.lexicon[:, :, np.newaxis] * as_prior_each_u
		
		# L_0(u): literal listener's probability of an utterance u, L_0(s, a, u) summed over all s' in S, a' in A
		# shape: (|U|, 1, 1)
		L_0_u = np.sum(L_0_sau, axis=(1, 2), keepdims=True)

		# Literal listener's prediction probability of state s, affect a, given utterance u
		# L_0(s, a | u) = L_0(s, a, u) / L_0(u) for each state s, affect a, utterance u
		# shape: (|U|, |S|, |A|)
		literal_listener = L_0_sau / L_0_u

		return literal_listener

	def pragmatic_speaker(self):
		"""Pragmatic predictions for all possible utterances, given states, affects, and goals.
		Returns
		-------
		4D np.array. The first dimension corresponds to goals, the second dimension corresponds to affects,
		the third dimension corresponds to states, and the fourth dimension corresponds to utterances.
		"""
		# L_0(s, a | u): literal listener's prediction probability of s, a, given u
		# shape: (|U|, |S|, |A|)
		literal_listener = self.literal_listener()
		# Tranpose literal listener
		# shape: (|A|, |S|, |U|)
		literal_listener_T = literal_listener.transpose(2, 1, 0)

		# Round utterances
		round_utterances = self.Round(self.utterances)
		# C(u): if utterance is a rounded number, round cost, otherwise, sharp cost
		C = np.where(self.utterances == round_utterances, self.round_cost, self.sharp_cost)

		# Possible communicative goals
		goals = self.goals()

		# Cache of g(s, a), for each goal g
		# shape: (|G|, |A|, |S|, self.interpretation_space_dims)
		gsa_each_g = self.gsa_each_g(goals)

		# Boolean array indicating whether g(s, a) = g(s', a') for each s' in S, a' in A, for a given g, a, s
		# shape: (|G|, |A|, |S|, |A|, |S|)
		goal_indicator = self.goal_indicator(goals, gsa_each_g)

		# Condition the literal listener prediction probability L_0(s', a' | U) on g(s, a) = g(s', a') for all s' in S, a' in A
		# shape: (|G|, |A|, |S|, |A|, |S|, |U|)
		L_0_goal_conditioned = goal_indicator[:, :, :, :, :, np.newaxis] * literal_listener_T

		# S_1(u, s, a, g): pragmatic speaker's joint probability of an utterance u, state s, affect a, goal g
		# shape: (|G|, |A|, |S|, |U|)
		S_1_usag = np.power(np.sum(L_0_goal_conditioned, axis=(3, 4)), self.alpha) * np.exp(self.alpha * -C[np.newaxis, np.newaxis, np.newaxis, :])
		
		# S_1(s, a, g): pragmatic speaker's probability of a state s, affect a, goal g, S_1(u, s, a, g) summed over all u' in U
		# shape: (|G|, |A|, |S|, 1)
		S_1_sag = np.sum(S_1_usag, axis=3, keepdims=True)

		# Pragmatic speaker's prediction probability of utterance u, given state s, affect a, goal g
		# S_1(u | s, a, g) = S_1(u, s, a, g) / S_1(s, a, g) for each utterance u, state s, affect a, goal g
		# shape: (|G|, |A|, |S|, |U|)
		speaker = S_1_usag / S_1_sag

		return speaker

if __name__ == '__main__':
	# Core lexicon
	U = np.array([50, 51, 10000])
	S = np.array([50, 51, 10000])
	A = np.array([0, 1])
	lex = np.identity(len(U))
	# s_prior[s] = P(s)
	s_prior = np.array([0.495, 0.495, 0.01])
	# as_prior[s][a] = P(a|s)
	as_prior = np.array(
				[[0.9, 0.1],
				 [0.9, 0.1],
				 [0.01, 0.99]]
			   )

	rsa = NonliteralNumberRSATemp(lexicon=lex, utterances=U, states=S, affects=A, s_prior=s_prior, 
							  as_prior=as_prior, round_cost=1, sharp_cost=5, precision=1, alpha=1)

	# rsa.display_listener(rsa.literal_listener(), title="Literal Listener", visual=True, save=False) #save=True, path='basic/literal_listener.png')
	# rsa.display_speaker(rsa.pragmatic_speaker(), title="Pragmatic Speaker", visual=True, save=False) #save=True, path='basic/pragmatic_speaker.png')
	# rsa.display_listener(rsa.pragmatic_listener(), title="Pragmatic Listener", visual=True, save=False) #save=True, path='basic/pragmatic_listener.png')

	probs_u2s0a1 = []
	alphas = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
	for alpha in alphas:
		rsa = NonliteralNumberRSATemp(lexicon=lex, utterances=U, states=S, affects=A, s_prior=s_prior,
							  as_prior=as_prior, round_cost=1, sharp_cost=1, precision=1, alpha=alpha)
		pragmatic_listener = rsa.pragmatic_listener()
		probs_u2s0a1.append(pragmatic_listener[2, 0, 1])

	plt.plot(alphas, probs_u2s0a1)
	plt.title('P(s = 50, a = 1 | u = 10000) vs. ⍺')
	plt.xlabel('⍺')
	plt.ylabel('P(s = 50, a = 1 | u = 10000)')
	plt.savefig('temp/prag_prob_alpha.png')
	# plt.show()




	