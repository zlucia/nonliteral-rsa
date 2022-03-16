"""
Lucia Zheng
A Computational Implementation of an Extension of the Rational Speech Acts Model for Nonliteral Number Words
LINGUIST230A: Introduction to Semantics and Pragmatics, Winter 2022
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

class NonliteralNumberRSA():
	"""Implementation of the Rational Speech Acts model extension proposed in Kao et al. (2014) for nonliteral number words.
	Vectorized implementation.

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
	interpretation_space_dims: int
		The number of interpretation space dimensions, defaults to 2, state and affect, based on Kao et al. (2014)
	"""

	def __init__(self, lexicon, utterances, states, affects, s_prior, as_prior, round_cost, sharp_cost, precision, interpretation_space_dims=2):
		self.lexicon = lexicon
		self.utterances = utterances
		self.states = states
		self.affects = affects
		self.s_prior = s_prior
		self.as_prior = as_prior
		self.round_cost = round_cost
		self.sharp_cost = sharp_cost
		self.precision = precision
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
		S_1_usag = np.sum(L_0_goal_conditioned, axis=(3, 4)) * np.exp(-C[np.newaxis, np.newaxis, np.newaxis, :])
		
		# S_1(s, a, g): pragmatic speaker's probability of a state s, affect a, goal g, S_1(u, s, a, g) summed over all u' in U
		# shape: (|G|, |A|, |S|, 1)
		S_1_sag = np.sum(S_1_usag, axis=3, keepdims=True)

		# Pragmatic speaker's prediction probability of utterance u, given state s, affect a, goal g
		# S_1(u | s, a, g) = S_1(u, s, a, g) / S_1(s, a, g) for each utterance u, state s, affect a, goal g
		# shape: (|G|, |A|, |S|, |U|)
		speaker = S_1_usag / S_1_sag

		return speaker

	def pragmatic_listener(self):
		"""Pragmatic listener predictions for all possible states and affects, given an utterance.
		Returns
		-------
		3D np.array. The first dimension corresponds to utterances, the second dimension corresponds to
		states, and the third dimension corresponds to affects.
		"""
		# S_1(u | s, a, g): pragmatic speaker's prediction probability of u, given s, a, g
		pragmatic_speaker = self.pragmatic_speaker()
		# Transpose pragmatic speaker
		# shape: (|U|, |S|, |A|, |G|)
		pragmatic_speaker_T = pragmatic_speaker.transpose(3, 2, 1, 0)

		# Possible communicative goals
		goals = self.goals()

		# L_1(s, a, u): pragmatic listener's joint probability of a state s, affect a, utterance u, for each s, a, u
		# shape: (|U|, |S|, |A|)
		L_1_sau = np.sum(self.s_prior[np.newaxis, :, np.newaxis, np.newaxis] * self.as_prior[np.newaxis, :, :, np.newaxis] * self.P_G() * pragmatic_speaker_T, axis=3)
		
		# L_1(u): pragmatic listener's probability of an utterance u, L_1(s, a, u) summed over all s' in S, a' in A
		# shape: (|U|, 1, 1)
		L_1_u = np.sum(L_1_sau, axis=(1, 2), keepdims=True)

		# Pragmatic listener's prediction probability of state s, affect a, given utterance u
		# L_1(s, a | u) = L_1(s, a, u) / L_1(u) for each state s, affect a, utterance u
		# shape: (|U|, |S|, |A|)
		pragmatic_listener = L_1_sau / L_1_u
		
		return pragmatic_listener

	def gsa_each_g(self, goals):
		"""Returns the function mapping of (s, a) input pairs under communicative goal function g, for each goal g
		"""
		gsa_each_g = np.zeros((len(goals), self.affects.shape[0], self.states.shape[0], self.interpretation_space_dims), dtype=object)
		for i in range(len(goals)):
			g = goals[i]
			for j in range(self.affects.shape[0]):
				for k in range(self.states.shape[0]):
					a = self.affects[j]
					s = self.states[k]
					gsa_each_g[i, j, k] = g(s, a)

		return gsa_each_g

	def goal_indicator(self, goals, gsa_each_g):
		"""Returns a boolean array indicator whether g(s', a') = g(s, a) for each s' in S, a' in A, for a given g, a, s
		"""
		goal_indicator = np.zeros((len(goals), self.affects.shape[0], self.states.shape[0], self.affects.shape[0], self.states.shape[0]))
		for i in range(len(goals)):
			for j in range(self.affects.shape[0]):
				for k in range(self.states.shape[0]):
					goal_indicator[i, j, k, :, :] = np.prod(np.equal(gsa_each_g[i, j, k], gsa_each_g[i]), axis=2)

		return goal_indicator

	def P_G(self):
		"""Returns the probability of a particular communicative goal. Based on Kao et al. (2014),
		we implement this as a uniform prior.
		"""
		return 1 / len(self.goals())

	def goals(self):
		"""Returns a list corresponding to the different communicative goals described in Kao et al. (2014).
		   This function returns communicative goals represented by functions of all possible combinations of 
		   three  possible functions r from state, affect to [state, None], [None, affect] or [state, affect] and 
		   two functions f from state to exact or approximate state.
		"""
		return [lambda s, a, r=r, f=f: r(f(s), a) for r in self.possible_r() for f in self.possible_f()]

	def possible_r(self):
		"""Returns a list of functions r that map (state, affect) to (state,), (affect,), or (state, affect)
		"""
		return [lambda s, a: np.array([s, None]),
				lambda s, a: np.array([None, a]),
				lambda s, a: np.array([s, a])]

	def possible_f(self):
		"""Returns a list of functions s that map state to state or rounded state
		"""
		return [lambda s: s,
				lambda s: self.Round(s)]

	def Round(self, x):
		"""Returns a rounded version of utterances for the approximate state communicative goal, rounded to self.precision
		   number of decimals
		"""
		return np.around(x, -self.precision)

	def display_listener(self, listener, title, visual, save=False, path=None):
		"""Display the probability distribution for a listener. If visual=True, displays heatmap of interpretation probabilities.
		   Otherwise, displays table of interpretation probabilities.
		"""
		print(title)

		if visual:
			sns.set_theme()
			f, axes = plt.subplots(1, len(self.utterances), figsize=(12, 5))
			f.suptitle(title, fontsize=16)
			f.subplots_adjust(left=0.05, wspace=0.2, bottom=0.15, top=0.85)
			cbar_ax_base = f.add_axes([.93, .15, .03, .7])

		for u, given_u in enumerate(listener):
			lex = pd.DataFrame(index=self.states, columns=self.affects, data=given_u)
			d = lex.copy()
			if visual:
				if u == len(listener) - 1:
					cbar = True
					cbar_ax = cbar_ax_base
				else:
					cbar = False
					cbar_ax = None
				fig = sns.heatmap(d, annot=True, fmt='.2g', ax=axes[u], vmin=0, vmax=1,
					linewidths=2, cmap="BuGn", cbar=cbar, cbar_ax=cbar_ax, annot_kws={"fontsize": 10})
				fig.tick_params(axis='x', bottom=False, top=False, labelsize=10)
				fig.tick_params(axis='y', left=False, right=False, labelsize=10)
				fig.set_xlabel("Affects", fontsize=12)
				fig.set_ylabel("States", fontsize=12)
				fig.set_title("Utterance: " + str(self.utterances[u]), fontsize=12)
			else:
				d.loc['utterance'] = [self.utterances[u]] + [" "]
				print(d)

		if visual:
			if save:
				directories = '/'.join(path.split('/')[:-1])
				if not os.path.exists(directories):
					os.makedirs(directories)
				plt.savefig(path)
			else:
				plt.show()

	def display_speaker(self, speaker, title, visual, save=False, path=None):
		"""Displays the probability distribution for a speaker. If visual=True, displays heatmap of interpretation probabilities.
		   Otherwise, displays table of interpretation probabilities.
		"""
		print(title)
		goals = [f"r_{r}(f_{f}(s),a)" for r in ['s','a','sa'] for f in ['e','a']]

		if visual:
			sns.set_theme()
			f, axes = plt.subplots(len(goals), len(self.affects), figsize=(8, 12))
			f.suptitle(title, fontsize=16)
			f.subplots_adjust(left=0.1, bottom=0.08, top=0.92, wspace=0.3, hspace=0.8)
			cbar_ax_base = f.add_axes([0.93, 0.15, 0.02, 0.7])
			cbar_ax_base.tick_params(labelsize=10)

		for g, given_g in enumerate(speaker):
			for a, given_ag in enumerate(given_g):
				lex = pd.DataFrame(index=self.states, columns=self.utterances, data=given_ag)
				d = lex.copy()
				if visual:
					if g == len(goals) - 1 and a == len(self.affects) - 1:
						cbar = True
						cbar_ax = cbar_ax_base
					else:
						cbar = False
						cbar_ax = None
					fig = sns.heatmap(d, annot=True, fmt='.2g', ax=axes[g][a], vmin=0, vmax=1,
						linewidths=2, cmap="RdPu", cbar=cbar, cbar_ax=cbar_ax, annot_kws={"fontsize": 10})
					fig.tick_params(axis='x', bottom=False, top=False, labelsize=10)
					fig.tick_params(axis='y', left=False, right=False, labelsize=10)
					fig.set_xlabel("Utterances", fontsize=12)
					fig.set_ylabel("States", fontsize=12)
					fig.set_title("Goal: " + str(goals[g]) + "; Affect: " + str(self.affects[a]), fontsize=12)
				else:
					d.loc['goal'] = [goals[g]] + [" ", " "]
					d.loc['affect'] = [rsa.affects[a]] + [" ", " "]
					print(d)

		if visual:
			if save:
				directories = '/'.join(path.split('/')[:-1])
				if not os.path.exists(directories):
					os.makedirs(directories)
				plt.savefig(path)
			else:
				plt.show()

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

	rsa = NonliteralNumberRSA(lexicon=lex, utterances=U, states=S, affects=A, s_prior=s_prior, 
							  as_prior=as_prior, round_cost=1, sharp_cost=5, precision=1)

	rsa.display_listener(rsa.literal_listener(), title="Literal Listener", visual=True, save=True, path='basic/literal_listener.png')
	rsa.display_speaker(rsa.pragmatic_speaker(), title="Pragmatic Speaker", visual=True, save=True, path='basic/pragmatic_speaker.png')
	rsa.display_listener(rsa.pragmatic_listener(), title="Pragmatic Listener", visual=True, save=True, path='basic/pragmatic_listener.png')

	# Uncomment the following lines to run nonliteral RSA for a hyperbolic utterance involving positive attitudes
	# Model hyperbolic utterances involving positive attitudes with an additional affect value a = 2, A = {0, 1, 2}
	# Core lexicon
	U = np.array([0, 50, 10000])
	S = np.array([0, 50, 10000])
	A = np.array([0, 1, 2])
	lex = np.identity(len(U))
	# s_prior[s] = P(s)
	s_prior = np.array([0.01, 0.985, 0.005])
	# as_prior[s][a] = P(a|s)
	as_prior = np.array(
				[[0.01, 0.01, 0.98],
				 [0.89, 0.07, 0.04],
				 [0.01, 0.98, 0.01]]
			   )

	rsa = NonliteralNumberRSA(lexicon=lex, utterances=U, states=S, affects=A, s_prior=s_prior, 
							  as_prior=as_prior, round_cost=1, sharp_cost=5, precision=1)

	rsa.display_listener(rsa.literal_listener(), title="Literal Listener", visual=True, save=True, path='positive/literal_listener.png')
	rsa.display_speaker(rsa.pragmatic_speaker(), title="Pragmatic Speaker", visual=True, save=True, path='positive/pragmatic_speaker.png')
	rsa.display_listener(rsa.pragmatic_listener(), title="Pragmatic Listener", visual=True, save=True, path='positive/pragmatic_listener.png')


