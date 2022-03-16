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
from nonliteral_rsa import NonliteralNumberRSA

class NonliteralNumberRSAAltRoundGoals(NonliteralNumberRSA):
	"""Implementation of the Rational Speech Acts model extension proposed in Kao et al. (2014) for nonliteral number words.
	Implementation with three approximate / round communicative goals, modeling rounding to different levels of precision.

	Extends the class NonliteralNumberRSA with alternative approximate / round communicative goals, modeling rounding to different levels of precision specified in self.precisions
	and modeling differential costs for different rounding functions of different precision. Slight adjustments made to the display_listener and display_speaker methods to visualize
	prediction probabilities incorporating these alternative approximate / round communicative goals.

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
	rounded_costs: np.array[float]
		NumPy array of costs of number utterances rounded to different levels of precision, cost should have same shape and ordering as precisions
	sharp_cost: float
		Cost of a non-rounded number utterance
	precisions: np.array[int]
		NumPy array of rounding place value, rounding to the nearest 10^precision, assumes precisions are sorted in decreasing order
	interpretation_space_dims: int
		The number of interpretation space dimensions, defaults to 2, state and affect, based on Kao et al. (2014)
	"""
	def __init__(self, lexicon, utterances, states, affects, s_prior, as_prior, round_costs, sharp_cost, precisions, interpretation_space_dims=2):
		self.lexicon = lexicon
		self.utterances = utterances
		self.states = states
		self.affects = affects
		self.s_prior = s_prior
		self.as_prior = as_prior
		self.round_costs = round_costs
		self.sharp_cost = sharp_cost
		self.precisions = precisions
		self.interpretation_space_dims = interpretation_space_dims

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
		# C(u): if utterance is a rounded number, most inexpensive round cost, otherwise sharp cost
		C = np.zeros(self.utterances.shape)
		C[:] = self.sharp_cost
		for i, precision in enumerate(self.precisions):
			for j, u in enumerate(self.utterances):
				if u == self.Round(u, precision):
					C[j] = self.round_costs[i]

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

	def possible_f(self):
		"""Returns a list of functions s that map state to state or rounded state
		"""
		fs = [lambda s: s]
		for precision in self.precisions:
			fs.append(lambda s: self.Round(s, precision))
		return fs

	def Round(self, x, precision):
		"""Returns a rounded version of utterances for the approximate state communicative goal, rounded to precision
		   number of decimals
		"""
		return np.around(x, precision)

	def display_listener(self, listener, title, visual, save=False, path=None):
		"""Display the probability distribution for a listener. If visual=True, displays heatmap of interpretation probabilities.
		   Otherwise, displays table of interpretation probabilities.
		"""
		print(title)

		if visual:
			sns.set_theme()
			f, axes = plt.subplots(1, len(self.utterances), figsize=(12, 5))
			f.suptitle(title, fontsize=16)
			f.subplots_adjust(left=0.05, wspace=0.3, bottom=0.15, top=0.85)
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
		goals = [f"r_{r}(f_{f}(s),a)" for r in ['s','a','sa'] for f in ['e','a1', 'a0', 'a-1']]

		if visual:
			sns.set_theme()
			f, axes = plt.subplots(len(goals), len(self.affects), figsize=(8, 20))
			f.suptitle(title, fontsize=16)
			f.subplots_adjust(left=0.1, bottom=0.08, top=0.92, wspace=0.3, hspace=0.95)
			cbar_ax_base = f.add_axes([0.93, 0.1, 0.02, 0.8])
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
	U = np.array([50, 51, 51.10, 51.11])
	S = np.array([50, 51, 51.10, 51.11])
	A = np.array([0, 1])
	lex = np.identity(len(U))
	# s_prior[s] = P(s)
	s_prior = np.array([0.25, 0.25, 0.25, 0.25])
	# as_prior[s][a] = P(a|s)
	as_prior = np.array(
				[[0.9, 0.1],
				 [0.9, 0.1],
				 [0.9, 0.1],
				 [0.9, 0.1]]
			   )
	round_costs = np.array([3, 2, 1])
	precisions = np.array([1, 0, -1])

	rsa = NonliteralNumberRSAAltRoundGoals(lexicon=lex, utterances=U, states=S, affects=A, s_prior=s_prior, 
							  as_prior=as_prior, round_costs=round_costs, sharp_cost=5, precisions=precisions)

	rsa.display_listener(rsa.literal_listener(), title="Literal Listener", visual=True, save=True, path='alt_round_goals/literal_listener.png')
	rsa.display_speaker(rsa.pragmatic_speaker(), title="Pragmatic Speaker", visual=True, save=True, path='alt_round_goals/pragmatic_speaker.png')
	rsa.display_listener(rsa.pragmatic_listener(), title="Pragmatic Listener", visual=True, save=True, path='alt_round_goals/pragmatic_listener.png')


	