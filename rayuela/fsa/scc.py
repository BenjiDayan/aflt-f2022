import numpy as np
from numpy import linalg as LA

from collections import deque

from rayuela.base.semiring import Boolean, Real
#from rayuela.fsa.pathsum import Pathsum, Strategy
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rayuela.fsa.fsa import FSA
# from rayuela.fsa.state import MinimizeState

class SCC:

    def __init__(self, fsa):
        self.fsa: FSA = fsa

    def scc(self):
        """
        Computes the SCCs of the FSA.
        Currently uses Kosaraju's algorithm.

        Guarantees SCCs come back in topological order.
        """
        for scc in self._kosaraju():
            yield scc

    def _kosaraju(self):
        """
        Kosaraju's algorithm [https://en.wikipedia.org/wiki/Kosaraju%27s_algorithm]
        Runs in O(E + V) time.
        Returns in the SCCs in topologically sorted order.
        """
		# Homework 3: Question 4
        visited = set()
        rev_fsa = self.fsa.reverse()
        my_sccs = []
        # dfs in order of decreasing finish time. So f(q) > f(q') => "q' ---> q <=> q ---> q' too"
        # hence dfsing from q, all nodes we hit are in the same SCC as q.
        for root_node in self.fsa.finish():
            if root_node in visited:
                continue
            my_scc = set()
            dfs_queue = deque([root_node])

            while len(dfs_queue) > 0:
                node = dfs_queue.pop()
                visited.add(node)
                my_scc.add(node)  # first element will be my_scc = {node}
                # qdest points to node in original fsa
                for _sym, qdest, _weight  in rev_fsa.arcs(node):
                    if not qdest in visited:
                        dfs_queue.append(qdest)
            my_sccs.append(my_scc)  # We've got through everything that points (multiple hops) to q so no more in SCC
        # precede to other nodes that had an earlier finish time in original dfs, 
        return my_sccs


