import math

import numpy as np
import torch


class Imp_smpling():
	list_of_candidates: list
	prob_dist: list

	def __init__(self, num_train_timesteps: int, scaled_cosine_alpha: int):
		list_of_candidates = [
			x for x in range(num_train_timesteps)
		]
		prob_dist = [
			self.importance_sampling_fn(
				x,
				num_train_timesteps,
				scaled_cosine_alpha
			) for x in list_of_candidates
		]
		prob_sum = 0
		# normalize the prob_list so that sum of prob is 1
		for i in prob_dist:
			prob_sum += i
		prob_dist = [x / prob_sum for x in prob_dist]

		self.list_of_candidates = list_of_candidates
		self.prob_dist = prob_dist

	@staticmethod
	def importance_sampling_fn(t, max_t, alpha):
		"""Importance Sampling Function f(t)"""
		return 1 / max_t * (1 - alpha * math.cos(math.pi * t / max_t))

	def sample(self, size: int):
		timesteps = np.random.choice(
			self.list_of_candidates,
			size=size,
			replace=True,
			p=self.prob_dist
		)
		timesteps = torch.tensor(timesteps).cuda()

		return timesteps
