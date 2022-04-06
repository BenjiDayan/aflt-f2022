from matplotlib.pyplot import thetagrids
import numpy as np
from numpy import linalg as LA
from frozendict import frozendict
from path import Path

from rayuela.base.datastructures import PriorityQueue
from rayuela.base.semiring import Real

from typing import TYPE_CHECKING
from rayuela.base.symbol import Sym

from rayuela.fsa.scc import SCC
if TYPE_CHECKING:
	from rayuela.fsa.fsa import FSA
	from rayuela.fsa.state import State

class Strategy:
	VITERBI = 1
	BELLMANFORD = 2
	DIJKSTRA = 3
	LEHMANN = 4
	JOHNSON = 5
	FIXPOINT = 6
	DECOMPOSED_LEHMANN = 7

class Pathsum:

	def __init__(self, fsa):

		# basic FSA stuff
		self.fsa: FSA = fsa
		self.R = fsa.R
		self.N = self.fsa.num_states

		# state dictionary
		self.I = {}
		for n, q in enumerate(self.fsa.Q):
			self.I[q] = n

		# lift into the semiring
		self.W = self.lift()

	def _convert(self):
		mat = np.zeros((self.N, self.N))
		for n in range(self.N):
			for m in range(self.N):
				mat[n, m] = self.W[n, m].score
		return mat

	def max_eval(self):
		# computes the largest eigenvalue
		mat = self._convert()
		if len(mat) == 0:
			return 0.0
		vals = []
		for val in LA.eigvals(mat):
			vals.append(np.abs(val))
		return np.max(vals)

	def lift(self):
		""" creates the weight matrix from the automaton """
		W = self.R.zeros(self.N, self.N)
		for p in self.fsa.Q:
			for a, q, w in self.fsa.arcs(p):
				W[self.I[p], self.I[q]] += w
		return W

	def pathsum(self, strategy):
		if strategy == Strategy.DIJKSTRA:
			assert self.R.superior, "Dijkstra's requires a superior semiring"
			return self.dijkstra_early()

		elif strategy == Strategy.VITERBI:
			assert self.fsa.acyclic, "Viterbi requires an acyclic FSA"
			return self.viterbi_pathsum()

		elif strategy == Strategy.BELLMANFORD:
			assert self.R.idempotent, "Bellman-Ford requires an idempotent semiring"
			return self.bellmanford_pathsum()

		elif strategy == Strategy.JOHNSON:
			assert self.R.idempotent, "Johnson's requires an idempotent semiring"
			return self.johnson_pathsum()

		elif strategy == Strategy.LEHMANN:
			return self.lehmann_pathsum()

		elif strategy == Strategy.FIXPOINT:
			return self.fixpoint_pathsum()

		elif strategy == Strategy.DECOMPOSED_LEHMANN:
			return self.decomposed_lehmann_pathsum()

		else:
			raise NotImplementedError

	def forward(self, strategy):
		
		if strategy == Strategy.DIJKSTRA:
			assert self.R.superior, "Dijkstra's requires a superior semiring"
			return self.dijkstra_fwd()

		if strategy == Strategy.VITERBI:
			assert self.fsa.acyclic, "Viterbi requires an acyclic FSA"
			return self.viterbi_fwd()

		elif strategy == Strategy.BELLMANFORD:
			assert self.R.idempotent, "Bellman-Ford requires an idempotent semiring"
			return self.bellmanford_fwd()

		elif strategy == Strategy.JOHNSON:
			assert self.R.idempotent, "Johnson's requires an idempotent semiring"
			return self.johnson_fwd()

		elif strategy == Strategy.LEHMANN:
			return self.lehmann_fwd()

		elif strategy == Strategy.FIXPOINT:
			return self.fixpoint_fwd()

		else:
			raise NotImplementedError

	def backward(self, strategy):
		if strategy == Strategy.VITERBI:
			assert self.fsa.acyclic, "Viterbi requires an acyclic FSA"
			return self.viterbi_bwd()

		elif strategy == Strategy.BELLMANFORD:
			assert self.R.idempotent, "Bellman-Ford requires an idempotent semiring"
			return self.bellmanford_bwd()

		elif strategy == Strategy.JOHNSON:
			assert self.R.idempotent, "Johnson's requires an idempotent semiring"
			return self.johnson_bwd()

		elif strategy == Strategy.LEHMANN:
			return self.lehmann_bwd()

		elif strategy == Strategy.FIXPOINT:
			return self.fixpoint_bwd()

		else:
			raise NotImplementedError

	def allpairs(self, strategy):
		if strategy == Strategy.JOHNSON:
			assert self.R.idempotent, "Johnson's requires an idempotent semiring"
			return self.johnson()

		elif strategy == Strategy.LEHMANN:
			return self.lehmann()
		
		elif strategy == Strategy.FIXPOINT:
			raise self.fixpoint()

		else:
			raise NotImplementedError

	def allpairs_pathsum(self, W):
		pathsum = self.R.zero
		for p in self.fsa.Q:
			for q in self.fsa.Q:
				pathsum += self.fsa.λ[p] * W[p, q] * self.fsa.ρ[q]
		return pathsum

	def allpairs_fwd(self, W):
		α = self.R.chart()
		for p in self.fsa.Q:
			for q in self.fsa.Q:
				α[q] += self.fsa.λ[p] * W[p, q]
		return frozendict(α)

	def allpairs_bwd(self, W):
		𝜷 = self.R.chart()
		W = self.lehmann()
		for p in self.fsa.Q:
			for q in self.fsa.Q:
				𝜷[p] += W[p, q] * self.fsa.ρ[q]
		return frozendict(𝜷)

	def viterbi_pathsum(self):
		pathsum = self.R.zero
		𝜷 = self.viterbi_bwd()
		for q in self.fsa.Q:
			pathsum += self.fsa.λ[q] * 𝜷[q]
		return pathsum


	def viterbi_fwd(self):
		assert self.fsa.acyclic
		
		α = self.R.chart()
		for q, w in self.fsa.I:
			α[q] = w

		rev_fsa = self.fsa.reverse()  # solely for access to δ[q_start][sym][q_end] dictionary
		for q_dest in self.fsa.toposort(rev=False):
			for _sym, q_start, w in rev_fsa.arcs(q_dest):
				α[q_dest] += w * α[q_start]
		return frozendict(α)

	def viterbi_bwd(self):
		""" The Viterbi algorithm run backwards. """

		assert self.fsa.acyclic

		# chart
		𝜷 = self.R.chart()

		# base case (paths of length 0)
		for q, w in self.fsa.F:
			𝜷[q] = w

		# recursion
		for p in self.fsa.toposort(rev=True):
			for _, q, w in self.fsa.arcs(p):
				𝜷[p] += 𝜷[q] * w

		return frozendict(𝜷)

	def dijkstra_early(self):
		""" Dijkstra's algorithm with early stopping."""
		raise NotImplementedError


	def dijkstra_fwd(self, I=None):
		""" Dijkstra's algorithm without early stopping. """

		assert self.fsa.R.superior

		# initialization
		α = self.R.chart()
		agenda = PriorityQueue(R=self.fsa.R)
		popped = set([]) 

		# base case
		if I is None:
			for q, w in self.fsa.I:
				agenda.push(q, w)
		else:
			for q in I:
				agenda.push(q, self.R.one)

		# main loop
		while agenda:
			i, v = agenda.pop()
			popped.add(i)
			α[i] += v

			for _, j, w in self.fsa.arcs(i):
				if j not in popped:
					agenda.push(j, v * w)

		return α

	def _lehmann(self, zero=True):
		"""
		Lehmann's (1977) algorithm.
		"""

		# initialization
		V = self.W.copy()
		U = self.W.copy()


		# basic iteration
		for j in range(self.N):
			V, U = U, V
			V = self.R.zeros(self.N, self.N)
			for i in range(self.N):
				for k in range(self.N):
					# i ➙ j ⇝ j ➙ k
					V[i,k] = U[i,k] + U[i,j] * U[j,j].star() * U[j,k]

		# post-processing (paths of length zero)
		if zero:
			for i in range(self.N):
				V[i,i] += self.R.one


		return V

	def lehmann(self, zero=True):

		V = self._lehmann(zero=zero)

		W = {}
		for p in self.fsa.Q:
			for q in self.fsa.Q:
				if p in self.I and q in self.I:
					W[p, q] = V[self.I[p], self.I[q]]
				elif p == q and zero:
					W[p, q] = self.R.one
				else:
					W[p, q] = self.R.zero

		return frozendict(W)

	def lehmann_pathsum(self): return self.allpairs_pathsum(self.lehmann())
	def lehmann_fwd(self): return self.allpairs_fwd(self.lehmann())
	def lehmann_bwd(self): return self.allpairs_bwd(self.lehmann())

	def decomposed_lehmann_pathsum(self):
		# Homework 3: Question 4
		sccs = SCC(self.fsa).scc()
		fsa = self.fsa.copy()
		for scc_states in sccs:
			scc_restricted_fsa = fsa.restricted(scc_states)
			scc_restricted_fsa
			scc_restricted_fsa_ps = Pathsum(scc_restricted_fsa)
			closure_matrix = scc_restricted_fsa_ps.lehmann()
			new_states, new_arcs, new_I, new_F, in_map, out_map = fsa.blob_surgery_in_out(scc_states)
			# replace new_arcs with matrix weights between every node
			new_arcs = set()
			for i in scc_restricted_fsa.Q:
				for j in scc_restricted_fsa.Q:
					w = closure_matrix[i, j]
					i_new = in_map[i]
					j_new = out_map[j]
					new_arcs.add((i_new, Sym('matrix'), j_new, w))
	
			scc_restricted_fsa_ps._lehmann()
			fsa.blob_surgery(scc_states, new_states, in_map, out_map, new_arcs, new_I, new_F)

		# return fsa
		ps = Pathsum(fsa)
		return ps.pathsum(Strategy.LEHMANN)
	
		

	def bellmanford_pathsum(self):
		pathsum = self.R.zero
		𝜷 = self.bellmanford_bwd()
		for q in self.fsa.Q:
			pathsum += self.fsa.λ[q] * 𝜷[q]
		return pathsum

	def bellmanford_fwd(self):
		raise NotImplementedError


	def bellmanford_bwd(self):
		raise NotImplementedError


	def johnson(self):
		raise NotImplementedError

	def johnson_pathsum(self): return self.allpairs_pathsum(self.johnson())
	def johnson_fwd(self): return self.allpairs_fwd(self.johnson())
	def johnson_bwd(self): return self.allpairs_bwd(self.johnson())

	def fixpoint(self):
		raise NotImplementedError

	def fixpoint_pathsum(self): return self.allpairs_pathsum(self.fixpoint())
	def fixpoint_fwd(self): return self.allpairs_fwd(self.fixpoint())
	def fixpoint_bwd(self): return self.allpairs_bwd(self.fixpoint())
