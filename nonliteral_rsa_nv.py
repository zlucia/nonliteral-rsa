"""
Lucia Zheng
A Computational Implementation of an Extension of the Rational Speech Acts Model for Nonliteral Number Words
LINGUIST230A: Introduction to Semantics and Pragmatics, Winter 2022
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class NonliteralNumberRSANV():
	"""Implementation of the Rational Speech Acts model extension proposed in Kao et al. (2014) for nonliteral number words.
	Non-vectorized implementation.

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

		self.u2i = {u: i for i, u in enumerate(self.utterances)}
		self.s2i = {s: i for i, s in enumerate(self.states)}
		self.a2i = {a: i for i, a in enumerate(self.affects)}

	def literal_listener(self):
		"""Literal listener predictions for all possible states and affects, given an utterance.
		Returns
		-------
		3D np.array. The first dimension corresponds to utterances, the second dimension corresponds
		to states, and the third dimension corresponds to affects.
		"""
		literal_listener = np.zeros((self.utterances.shape[0], self.states.shape[0], self.affects.shape[0]))
		for i, u in enumerate(self.utterances):
			for j, s in enumerate(self.states):
				for k, a in enumerate(self.affects):
					literal_listener[i][j][k] = self.L_0(s, a, u)

		return literal_listener

	def pragmatic_speaker(self):
		"""Pragmatic predictions for all possible utterances, given states, affects, and goals.
		Returns
		-------
		4D np.array. The first dimension corresponds to goals, the second dimension corresponds to affects,
		the third dimension corresponds to states, and the fourth dimension corresponds to utterances.
		"""
		goals = self.goals()
		speaker = np.zeros((len(goals), self.affects.shape[0], self.states.shape[0], self.utterances.shape[0]))
		for i, g in enumerate(goals):
			for j, a in enumerate(self.affects):
				for k, s in enumerate(self.states):
					for l, u in enumerate(self.utterances):
						speaker[i][j][k][l] = self.S_1(u, s, a, g)

		return speaker

	def pragmatic_listener(self):
		"""Pragmatic listener predictions for all possible states and affects, given an utterance.
		Returns
		-------
		3D np.array. The first dimension corresponds to utterances, the second dimension corresponds to
		states, and the third dimension corresponds to affects.
		"""
		pragmatic_listener = np.zeros((self.utterances.shape[0], self.states.shape[0], self.affects.shape[0]))
		for i, u in enumerate(self.utterances):
			for j, s in enumerate(self.states):
				for k, a in enumerate(self.affects):
					pragmatic_listener[i][j][k] = self.L_1(s, a, u)

		return pragmatic_listener

	def L_0(self, s, a, u):
		"""Returns the listeral listener's prediction probability of a state s, affect a, given utterance u
		"""
		# If s = u
		if self.lexicon[self.u2i[u]][self.s2i[s]] == 1:
			return self.P_A(a, s)
		else:
			return 0

	def S_1(self, u, s, a, g):
		"""Returns the speaker's prediction probability of an utterance u, given state s, affect a, goal g
		"""
		numerator = self.S_1_joint(u, s, a, g)
		denominator = sum([self.S_1_joint(u_p, s, a, g) for u_p in self.utterances])
		return numerator / denominator

	def S_1_joint(self, u, s, a, g):
		"""Returns the speaker's joint probability of an utterance u, state s, affect a, goal g
		"""
		result = 0
		for s_p in self.states:
			for a_p in self.affects:
				if g(s, a) == g(s_p, a_p):
					result += self.L_0(s_p, a_p, u) * np.exp(-self.C(u))
		return result

	def L_1(self, s, a, u):
		"""Returns the pragmatic listener's prediction probability of a state s, affect a, given utterance u
		"""
		numerator = self.L_1_joint(s, a, u)
		denominator = sum([self.L_1_joint(s_p, a_p, u) for s_p in self.states for a_p in self.affects])
		return numerator / denominator

	def L_1_joint(self, s, a, u):
		"""Returns the pragmatic speaker's joint probability of a state s, affect a, utterance u
		"""
		result = 0
		for g in self.goals():
			result += self.P_S(s) * self.P_A(a, s) * self.P_G(g) * self.S_1(u, s, a, g)
		return result

	def P_A(self, a, s):
		"""Returns the prior probability of affect a given state s
		"""
		return self.as_prior[self.s2i[s]][self.a2i[a]]

	def P_S(self, s):
		"""Returns the prior probability of state s
		"""
		return self.s_prior[self.s2i[s]]

	def C(self, u):
		"""Returns the cost of an utterance based on whether it is a rounded or non-rounded number
		"""
		if u == self.Round(u):
			return self.round_cost
		else:
			return self.sharp_cost
	
	def P_G(self, g):
		"""Returns the probability of a particular conversational goal. Based on Kao et al. (2014),
		we implement this as a uniform prior.
		"""
		return 1 / len(self.goals())

	def goals(self):
		"""Returns a list corresponding to the different conversational goals described in Kao et al. (2014).
		   This function returns conversational goals represented by functions of all possible combinations of 
		   three  possible functions r from state, affect to [state, None], [None, affect] or [state, affect] and 
		   two functions f from state to exact or approximate state.
		"""
		return [lambda s, a, r=r, f=f: r(f(s), a) for r in self.possible_r() for f in self.possible_f()]

	def possible_r(self):
		"""Returns a list of functions r that map (state, affect) to (state,), (affect,), or (state, affect)
		"""
		return [lambda s, a: (s, ),
				lambda s, a: (a, ),
				lambda s, a: (s, a)]

	def possible_f(self):
		"""Returns a list of functions s that map state to state or rounded state
		"""
		return [lambda s: s,
				lambda s: self.Round(s)]

	def Round(self, x):
		"""Returns a rounded version of utterances for the approximate state conversational goal, rounded to self.precision
		   number of digits
		"""
		return np.around(x, -self.precision)

	def display_listener(self, listener, title, visual):
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
			plt.show()

	def display_speaker(self, speaker, title, visual):
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

	rsa = NonliteralNumberRSANV(lexicon=lex, utterances=U, states=S, affects=A, s_prior=s_prior, 
							  as_prior=as_prior, round_cost=1, sharp_cost=5, precision=1)

	rsa.display_listener(rsa.literal_listener(), title="Literal Listener", visual=True)
	rsa.display_speaker(rsa.pragmatic_speaker(), title="Pragmatic Speaker", visual=True)
	rsa.display_listener(rsa.pragmatic_listener(), title="Pragmatic Listener", visual=True)
