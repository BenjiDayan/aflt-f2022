from __future__ import annotations
import copy
from typing import List, Set
import numpy as np
from frozendict import frozendict
from itertools import chain, product

from collections import Counter
from collections import defaultdict as dd

from rayuela.base.semiring import Boolean, String, ProductSemiring, Semiring
from rayuela.base.misc import epsilon_filter
from rayuela.base.symbol import Sym, ε, ε_1, ε_2

from rayuela.fsa.state import State, PairState
from rayuela.fsa.pathsum import Pathsum, Strategy

class FSA:

	def __init__(self, R=Boolean):

		# DEFINITION
		# A weighted finite-state automaton is a 5-tuple <R, Σ, Q, δ, λ, ρ> where
		# • R is a semiring;
		# • Σ is an alphabet of symbols;
		# • Q is a finite set of states;
		# • δ is a finite relation Q × Σ × Q × R;
		# • λ is an initial weight function;
		# • ρ is a final weight function.

		# NOTATION CONVENTIONS
		# • single states (elements of Q) are denoted q
		# • multiple states not in sequence are denoted, p, q, r, ...
		# • multiple states in sequence are denoted i, j, k, ...
		# • symbols (elements of Σ) are denoted lowercase a, b, c, ...
		# • single weights (elements of R) are denoted w
		# • multiple weights (elements of R) are denoted u, v, w, ...

		# semiring
		self.R = R

		# alphabet of symbols
		self.Sigma = set([])

		# a finite set of states
		self.Q = set([])

		# transition function : Q × Σ × Q → R
		self.δ = dd(lambda : dd(lambda : dd(lambda : self.R.zero)))

		# initial weight function
		self.λ = R.chart()

		# final weight function
		self.ρ = R.chart()
	
	def add_state(self, q):
		self.Q.add(q)

	def add_states(self, Q):
		for q in Q:
			self.add_state(q)

	def add_arc(self, i, a, j, w=None):
		if w is None: w = self.R.one

		if not isinstance(i, State): i = State(i)
		if not isinstance(j, State): j = State(j)
		if not isinstance(a, Sym): a = Sym(a)
		if not isinstance(w, self.R): w = self.R(w)

		self.add_states([i, j])
		self.Sigma.add(a)
		self.δ[i][a][j] += w

	def set_arc(self, i, a, j, w):
		self.add_states([i, j])
		self.Sigma.add(a)
		self.δ[i][a][j] = w

	def set_I(self, q, w=None):
		if w is None: w = self.R.one
		self.add_state(q)
		self.λ[q] = w

	def set_F(self, q, w=None):
		if w is None: w = self.R.one
		self.add_state(q)
		self.ρ[q] = w

	def add_I(self, q, w):
		self.add_state(q)
		self.λ[q] += w

	def add_F(self, q, w):
		self.add_state(q)
		self.ρ[q] += w

	def freeze(self):
		self.Sigma = frozenset(self.Sigma)
		self.Q = frozenset(self.Q)
		self.δ = frozendict(self.δ)
		self.λ = frozendict(self.λ)
		self.ρ = frozendict(self.ρ)
	
	@property
	def I(self):
		for q, w in self.λ.items():
			if w != self.R.zero:
				yield q, w

	@property
	def F(self):
		for q, w in self.ρ.items():
			if w != self.R.zero:
				yield q, w

	@property
	def D_tuples(self, in_state=None):
		for q1 in (self.δ if in_state is None else [in_state]):
			for a in self.δ[q1]:
				for q2 in self.δ[q1][a]:
					yield (q1, a, q2), self.δ[q1][a][q2]

	def arcs(self, i, no_eps=False):
		for a, T in self.δ[i].items():
			if no_eps and a == ε:
				continue
			for j, w in T.items():
				if w == self.R.zero:
					continue
				yield a, j, w
	
	@staticmethod
	def string_to_fsa(R: Semiring, string: str) -> FSA:
		assert isinstance(string, str)

		fsa = FSA(R=R)
		for i, x in enumerate(list(string)):
			fsa.add_arc(State(i), Sym(x), State(i+1), R.one)
		
		fsa.set_I(State(0), R.one)
		fsa.add_F(State(len(string)), R.one)
		return fsa

	def accept(self, string):
		""" determines whether a string is in the language """
		string_fsa = self.string_to_fsa(self.R, string)
		return self.intersect(string_fsa).pathsum()

	@property
	def num_states(self):
		return len(self.Q)

	def copy(self):
		""" deep copies the machine """
		return copy.deepcopy(self)

	def spawn(self, keep_init=False, keep_final=False):
		""" returns a new FSA in the same semiring """
		F = FSA(R=self.R)

		if keep_init:
			for q, w in self.I:
				F.set_I(q, w)
		if keep_final:
			for q, w in self.F:
				F.set_F(q, w)

		return F
	
	def push(self):
		from rayuela.fsa.transformer import Transformer
		return Transformer.push(self)

	def minimize(self, strategy=None):
		# Homework 5: Question 3
		raise NotImplementedError

	def dfs(self, ll=False):
		""" Depth-first search (Cormen et al. 2019; Section 22.3) """

		in_progress, finished = set([]), {}
		cyclic, counter = False, 0
		if ll:
			trees = []

		def _dfs(p):
			nonlocal in_progress
			nonlocal finished
			nonlocal cyclic
			nonlocal counter
			if ll:
				nonlocal trees

			in_progress.add(p)

			for _, q, _ in self.arcs(p):
				if q in in_progress:
					cyclic = True
				elif q not in finished:
					
					_dfs(q)

			in_progress.remove(p)
			finished[p] = counter
			counter += 1

		for q, _ in self.I: _dfs(q)

		return cyclic, finished

	def finish(self, rev=False, acyclic_check=False):
		"""
		Returns the nodes in order of their finishing time.
		"""

		cyclic, finished = self.dfs()

		if acyclic_check:
			assert self.acyclic

		sort = {}
		for s, n in finished.items():
			sort[n] = s
		if rev:
			for n in sorted(list(sort.keys())):
				yield sort[n]
		else:
			for n in reversed(sorted(list(sort.keys()))):
				yield sort[n]

	def toposort(self, rev=False):
		return self.finish(rev=rev, acyclic_check=True)

	@property
	def acyclic(self) -> bool:
		cyclic, _ = self.dfs()
		return not cyclic

	@property
	def deterministic(self) -> bool:

		# Homework 1: Question 2
			for v1 in self.δ.values():
				for k2, v2 in v1.items():
					if k2 == Sym("ε"):
						return False
					if len(v2.keys()) > 1:
						return False
			return True

	@property
	def pushed(self) -> bool:
			
		# Homework 1: Question 2
		out = self.R.chart()
		for q in self.Q:
			for a, dest_state_to_weight in self.δ[q].items():
				for weight in dest_state_to_weight.values():
					out[q] += weight

			out[q] += self.ρ[q]
			if out[q] != self.R.one:
				return False
		return True

	def reverse(self) -> FSA:
		""" computes the reverse of the FSA """

		# Homework 1: Question 3
		rev_fsa = FSA(R=self.R)
		rev_fsa.λ = self.ρ.copy()
		rev_fsa.ρ = self.λ.copy()
		rev_fsa.Q = self.Q.copy()

		for q, adict in self.δ.items():
			for a, q2dict in adict.items():
				for q2, w in q2dict.items():
					rev_fsa.add_arc(q2, a, q, w)

		return rev_fsa
	
	def accessible(self) -> set:
		""" computes the set of acessible states """

		# Homework 1: Question 3
		stack = list([k for k,v in self.λ.items() if v != self.R.zero])
		accessible_states = set(stack)

		# we pop off the stack to find the next state to check out. This does a depth first search.
		# we add to the stack new found which have never been in the stack
		while len(stack) > 0:
			q = stack.pop()
			for a, q2dict in self.δ[q].items():
				for q2 in q2dict.keys():
					if q2 not in accessible_states:  # it's never been in the stack
						accessible_states.add(q2)
						stack.append(q2)

		return accessible_states

	def coaccessible(self) -> set:
		""" computes the set of co-acessible states """

		# Homework 1: Question 3
		rev_fsa = self.reverse()
		return rev_fsa.accessible()

	def restricted(self, states_to_keep: set):
		fsa = FSA(R=self.R)
		fsa.Q = states_to_keep

		for q in states_to_keep.intersection(set(self.λ.keys())):
			fsa.add_I(q, self.λ[q])
		for q in states_to_keep.intersection(set(self.ρ.keys())):
			fsa.add_F(q, self.ρ[q])

		for q in states_to_keep:
			for a, q2dict in self.δ[q].items():
				for q2, weight in q2dict.items():
					if q2 in states_to_keep:
						fsa.add_arc(q, a, q2, weight)
		return fsa

	def blob_surgery(self, blob_states: Set[State], new_states, in_map, out_map, new_arcs, new_I, new_F):
		rev_fsa = self.reverse()
		for state in blob_states:
			# remove state from intial/final, if they're in the dict
			self.λ.pop(state, None)
			self.ρ.pop(state, None)
			# replace arcs going out from state
			for sym, qdest, weight in self.arcs(state):
				if not qdest in blob_states:
					self.add_arc(out_map[state], sym, qdest, weight)

			self.δ.pop(state)
			# replace arcs going into state
			for sym, qstart, weight in rev_fsa.arcs(state):
				if not qstart in blob_states:
					self.δ[qstart][sym].pop(state, None)
					self.add_arc(qstart, sym, in_map[state], weight)

			# remove state from states set
			self.Q.remove(state)
				
		self.λ.update(new_I)
		self.ρ.update(new_F)

		# add arcs between new states
		for qstart, sym, qdest, w in new_arcs:
			self.add_arc(qstart, sym, qdest, w)

		


	def blob_surgery_test(self, states):
		new_states, new_arcs, new_I, new_F, in_map, out_map = self.blob_surgery_in_out(states)
		self.blob_surgery(states, new_states, in_map, out_map, new_arcs, new_I, new_F)

	def blob_surgery_in_out(self, states):
		new_states = set()
		in_map = {}
		out_map = {}
		new_arcs = set()
		new_I = self.R.chart()
		new_F = self.R.chart()
		for state in states:
			in_state, out_state = State(str(state.idx) + 'in'), State(str(state.idx) + 'out')
			new_states.add(in_state)
			new_states.add(out_state)
			if in_state in self.Q or out_state in self.Q:
				raise("haven't dealt with state idx collisions here")
			in_map[state] = in_state
			new_I[in_state] = self.λ[state]
			new_F[out_state] = self.ρ[state]
			out_map[state] = out_state
		
		for in_state in in_map.values():
			for out_state in out_map.values():
				new_arcs.add((in_state, Sym("surgery"), out_state, self.R.one))

		return new_states, new_arcs, new_I, new_F, in_map, out_map



	def trim(self) -> FSA:
		""" keeps only those states that are both accessible and co-accessible """

		# Homework 1: Question 3
		states_to_keep = self.accessible().intersection(self.coaccessible())
		return self.restricted(states_to_keep)
	

	def prefixed_states_fsa(self, prefix):
		fsa = FSA(R=self.R)
		for (q1, a, q2), w in self.D_tuples:
			fsa.add_arc((prefix, q1.idx), a, (prefix, q2.idx), w)
		for q, w in self.I:
			fsa.add_I(State((prefix, q.idx)), w)
		for q, w in self.F:
			fsa.add_F(State((prefix, q.idx)), w)
		return fsa

	def union(self, fsa) -> FSA:
		""" construct the union of the two FSAs """

		# Homework 1: Question 4
		fsa1, fsa2 = self.prefixed_states_fsa("fsa1"), fsa.prefixed_states_fsa("fsa2")
		for (q1, a, q2), w in fsa2.D_tuples:
			fsa1.add_arc(q1, a, q2, w)
		for q, w in fsa2.F:
			fsa1.add_F(q, w)

		old_I = [pairs for pairs in fsa1.I]
		fsa1.λ = self.R.chart()  # reset this to just one input
		fsa1.add_I(State("init"), self.R.one)
		for q, w in chain(old_I, fsa2.I):
			fsa1.add_arc(State("init"), ε, q, w)

		return fsa1


	def concatenate(self, fsa: FSA) -> FSA:
		""" construct the concatenation of the two FSAs """

		# Homework 1: Question 4
		fsa1, fsa2 = self.prefixed_states_fsa("fsa1"), fsa.prefixed_states_fsa("fsa2")
		for (q1, a, q2), w in fsa2.D_tuples:
			fsa1.add_arc(q1, a, q2, w)

		old_F = [pairs for pairs in fsa1.F]
		fsa1.ρ = self.R.chart()
		link_state = State("link")
		for q, w in old_F:
			fsa1.add_arc(q, ε, link_state, w)
		for q, w in fsa2.I:
			fsa1.add_arc(link_state, ε, q, w)

		for q, w in fsa2.F:
			fsa1.add_F(q, w)

		return fsa1

	def kleene_closure(self) -> FSA:
		""" compute the Kleene closure of the FSA """

		# Homework 1: Question 4
		fsa = self.prefixed_states_fsa("fsa")
		init, final = State("init"), State("final")
		fsa.add_arc(init, ε, final, self.R.one)

		for q, w in fsa.I:
			fsa.add_arc(init, ε, q, w)
		
		for q, w in fsa.F:
			fsa.add_arc(q, ε, final, w)
			fsa.add_arc(q, ε, init, w)
		
		fsa.λ = self.R.chart()
		fsa.set_I(init, self.R.one)
		fsa.ρ = self.R.chart()
		fsa.set_F(final, self.R.one)

		return fsa



	def pathsum(self, strategy=Strategy.LEHMANN):
		if self.acyclic:
			strategy = Strategy.VITERBI
		pathsum = Pathsum(self)
		return pathsum.pathsum(strategy)

	def viterbi_arcsum(self, qstart, a, qdest):
		w = self.δ[qstart][a][qdest]
		pathsum = Pathsum(self)
		α = pathsum.viterbi_fwd()
		β = pathsum.viterbi_bwd()
		return α[qstart] * w * β[qdest]

	def viterbi_arcsums(self):
		pathsum = Pathsum(self)
		α = pathsum.viterbi_fwd()
		β = pathsum.viterbi_bwd()
		output = dd(lambda : dd(lambda : dd(lambda : self.R.zero)))
		for (qstart, a, qdest), w in self.D_tuples:
			output[qstart][a][qdest] = α[qstart] * w * β[qdest]
		
		return output



	def edge_marginals(self) -> dict:
		""" computes the edge marginals μ(q→q') """

		# Homework 2: Question 2
		raise NotImplementedError

	def intersect(self, fsa):
		"""
		on-the-fly weighted intersection
		"""

		# the two machines need to be in the same semiring
		assert self.R == fsa.R

		# add initial states
		product_fsa = FSA(R=self.R)
		for (q1, w1), (q2, w2) in product(self.I, fsa.I):
			product_fsa.add_I(PairState(q1, q2), w=w1 * w2)
		
		self_initials = {q: w for q, w in self.I}
		fsa_initials = {q: w for q, w in fsa.I}

		visited = set([(i1, i2, State('0')) for i1, i2 in product(self_initials, fsa_initials)])
		stack = [(i1, i2, State('0')) for i1, i2 in product(self_initials, fsa_initials)]

		self_finals = {q: w for q, w in self.F}
		fsa_finals = {q: w for q, w in fsa.F}

		while stack:
			q1, q2, qf = stack.pop()

			E1 = [(a if a != ε else ε_2, j, w) for (a, j, w) in self.arcs(q1)] + \
                            [(ε_1, q1, self.R.one)]
			E2 = [(a if a != ε else ε_1, j, w) for (a, j, w) in fsa.arcs(q2)] + \
                            [(ε_2, q2, self.R.one)]

			M = [((a1, j1, w1), (a2, j2, w2))
				 for (a1, j1, w1), (a2, j2, w2) in product(E1, E2)
				 if epsilon_filter(a1, a2, qf) != State('⊥')]

			for (a1, j1, w1), (a2, j2, w2) in M:

				product_fsa.set_arc(
					PairState(q1, q2), a1,
					PairState(j1, j2), w=w1*w2)

				_qf = epsilon_filter(a1, a2, qf)
				if (j1, j2, _qf) not in visited:
					stack.append((j1, j2, _qf))
					visited.add((j1, j2, _qf))

			# final state handling
			if q1 in self_finals and q2 in fsa_finals:
				product_fsa.add_F(
					PairState(q1, q2), w=self_finals[q1] * fsa_finals[q2])

		return product_fsa

	def topologically_equivalent(self, fsa):
		""" Tests topological equivalence. """
		
		# Homework 5: Question 4
		raise NotImplementedError

	def tikz(self, max_per_row=4):

		tikz_string = []
		previous_ids, positioning = [], ''
		rows = {}

		initial = {q: w for q, w in self.I}
		final = {q: w for q, w in self.F}

		for jj, q in enumerate(self.Q):
			options = 'state'
			additional = ''

			if q in initial:
				options += ', initial'
				additional = f' / {initial[q]}'
			if q in final:
				options += ', accepting'
				additional = f' / {final[q]}'

			if jj >= max_per_row:
				positioning = f'below = of {previous_ids[jj - max_per_row]}'
			elif len(previous_ids) > 0:
				positioning = f'right = of {previous_ids[-1]}'
			previous_ids.append(f'q{q.idx}')
			rows[q] = jj // max_per_row

			tikz_string.append(f'\\node[{options}] (q{q.idx}) [{positioning}] {{ ${q.idx}{additional}$ }}; \n')

		tikz_string.append('\\draw')

		seen_pairs, drawn_pairs = set(), set()

		for jj, q in enumerate(self.Q):
			target_edge_labels = dict()
			for a, j, w in self.arcs(q):
				if j not in target_edge_labels:
					target_edge_labels[j] = f'{a}/{w}'
				else:
					target_edge_labels[j] += f'\\\\{a}/{w}'
				seen_pairs.add(frozenset([q, j]))

			for ii, (target, label) in enumerate(target_edge_labels.items()):

				edge_options = 'align=left'
				if q == target:
					edge_options += ', loop above'
				elif frozenset([q, target]) not in seen_pairs:
					edge_options += 'a, bove'
				elif frozenset([q, target]) not in drawn_pairs:
					if rows[q] == rows[target]:
						edge_options += ', bend left, above'
					else:
						edge_options += ', bend left, right'
				else:
					if rows[q] == rows[target]:
						edge_options += ', bend left, below'
					else:
						edge_options += ', bend left, right'
				end = '\n'
				if jj == self.num_states - 1 and ii == len(target_edge_labels) - 1:
					end = '; \n'
				tikz_string.append(f'(q{q.idx}) edge[{edge_options}] node{{ ${label}$ }} (q{target.idx}) {end}')
				drawn_pairs.add(frozenset([q, j]))

		if not len(list(self.arcs(list(self.Q)[-1]))) > 0:
			tikz_string.append(';')

		return ''.join(tikz_string)

	def __truediv__(self, other):
		return self.intersect(other)

	def __add__(self, other):
		return self.concatenate(other)

	def __sub__(self, other):
		return self.difference(other)

	def __repr__(self):
		return f'WFSA({self.num_states} states, {self.R})'

	def __str__(self):
		""" ascii visualize """

		output = []
		for q, w in self.I:
			output.append(f"initial state:\t{q.idx}\t{w}")
		for q, w in self.F:
			output.append(f"final state:\t{q.idx}\t{w}")
		for p in self.Q:
			for a, q, w in self.arcs(p):
				output.append(f"{p}\t----{a}/{w}---->\t{q}")
		return "\n".join(output)

	def _repr_html_(self):
		"""
		When returned from a Jupyter cell, this will generate the FST visualization
		Based on: https://github.com/matthewfl/openfst-wrapper
		"""
		from uuid import uuid4
		import json
		from collections import defaultdict
		ret = []
		if self.num_states == 0:
			return '<code>Empty FST</code>'

		if self.num_states > 64:
			return f'FST too large to draw graphic, use fst.ascii_visualize()<br /><code>FST(num_states={self.num_states})</code>'

		finals = {q for q, _ in self.F}
		initials = {q for q, _ in self.I}

		# print initial
		for q, w in self.I:
			if q in finals:
				label = f'{str(q)} / [{str(w)} / {str(self.ρ[q])}]'
				color = 'af8dc3'
			else:
				label = f'{str(q)} / {str(w)}'
				color = '66c2a5'

			ret.append(
				f'g.setNode("{repr(q)}", {{ label: {json.dumps(label)} , shape: "circle" }});\n')
				# f'g.setNode("{repr(q)}", {{ label: {json.dumps(hash(label) // 1e8)} , shape: "circle" }});\n')

			ret.append(f'g.node("{repr(q)}").style = "fill: #{color}"; \n')

		# print normal
		for q in (self.Q - finals) - initials:

			label = str(q)

			ret.append(
				f'g.setNode("{repr(q)}", {{ label: {json.dumps(label)} , shape: "circle" }});\n')
				# f'g.setNode("{repr(q)}", {{ label: {json.dumps(hash(label) // 1e8)} , shape: "circle" }});\n')
			ret.append(f'g.node("{repr(q)}").style = "fill: #8da0cb"; \n')

		# print final
		for q, w in self.F:
			# already added
			if q in initials:
				continue

			if w == self.R.zero:
				continue
			label = f'{str(q)} / {str(w)}'

			ret.append(
				f'g.setNode("{repr(q)}", {{ label: {json.dumps(label)} , shape: "circle" }});\n')
				# f'g.setNode("{repr(q)}", {{ label: {json.dumps(hash(label) // 1e8)} , shape: "circle" }});\n')
			ret.append(f'g.node("{repr(q)}").style = "fill: #fc8d62"; \n')

		for q in self.Q:
			to = defaultdict(list)
			for a, j, w in self.arcs(q):
				if self.R is ProductSemiring and isinstance(w.score[0], String):
					# the imporant special case of encoding transducers
					label = f'{str(a)}:{str(w)}'
				else:
					label = f'{str(a)} / {str(w)}'
				to[j].append(label)

			for dest, values in to.items():
				if len(values) > 4:
					values = values[0:3] + ['. . .']
				label = '\n'.join(values)
				ret.append(
					f'g.setEdge("{repr(q)}", "{repr(dest)}", {{ arrowhead: "vee", label: {json.dumps(label)} }});\n')

		# if the machine is too big, do not attempt to make the web browser display it
		# otherwise it ends up crashing and stuff...
		if len(ret) > 256:
			return f'FST too large to draw graphic, use fst.ascii_visualize()<br /><code>FST(num_states={self.num_states})</code>'

		ret2 = ['''
		<script>
		try {
		require.config({
		paths: {
		"d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/4.13.0/d3",
		"dagreD3": "https://cdnjs.cloudflare.com/ajax/libs/dagre-d3/0.6.1/dagre-d3.min"
		}
		});
		} catch {
		  ["https://cdnjs.cloudflare.com/ajax/libs/d3/4.13.0/d3.js",
		   "https://cdnjs.cloudflare.com/ajax/libs/dagre-d3/0.6.1/dagre-d3.min.js"].forEach(function (src) {
			var tag = document.createElement('script');
			tag.src = src;
			document.body.appendChild(tag);
		  })
		}
		try {
		requirejs(['d3', 'dagreD3'], function() {});
		} catch (e) {}
		try {
		require(['d3', 'dagreD3'], function() {});
		} catch (e) {}
		</script>
		<style>
		.node rect,
		.node circle,
		.node ellipse {
		stroke: #333;
		fill: #fff;
		stroke-width: 1px;
		}

		.edgePath path {
		stroke: #333;
		fill: #333;
		stroke-width: 1.5px;
		}
		</style>
		''']

		obj = 'fst_' + uuid4().hex
		ret2.append(
			f'<center><svg width="850" height="600" id="{obj}"><g/></svg></center>')
		ret2.append('''
		<script>
		(function render_d3() {
		var d3, dagreD3;
		try { // requirejs is broken on external domains
		  d3 = require('d3');
		  dagreD3 = require('dagreD3');
		} catch (e) {
		  // for google colab
		  if(typeof window.d3 !== "undefined" && typeof window.dagreD3 !== "undefined") {
			d3 = window.d3;
			dagreD3 = window.dagreD3;
		  } else { // not loaded yet, so wait and try again
			setTimeout(render_d3, 50);
			return;
		  }
		}
		//alert("loaded");
		var g = new dagreD3.graphlib.Graph().setGraph({ 'rankdir': 'LR' });
		''')
		ret2.append(''.join(ret))

		ret2.append(f'var svg = d3.select("#{obj}"); \n')
		ret2.append(f'''
		var inner = svg.select("g");

		// Set up zoom support
		var zoom = d3.zoom().scaleExtent([0.3, 5]).on("zoom", function() {{
		inner.attr("transform", d3.event.transform);
		}});
		svg.call(zoom);

		// Create the renderer
		var render = new dagreD3.render();

		// Run the renderer. This is what draws the final graph.
		render(inner, g);

		// Center the graph
		var initialScale = 0.75;
		svg.call(zoom.transform, d3.zoomIdentity.translate(
		    (svg.attr("width") - g.graph().width * initialScale) / 2, 20).scale(initialScale));

		svg.attr('height', g.graph().height * initialScale + 50);
		}})();

		</script>
		''')

		return ''.join(ret2)
