from rayuela.base.semiring import Boolean, Real, Tropical, \
    String, Integer, Rational
from rayuela.base.symbol import Sym, ε
from rayuela.fsa.fsa import FSA
from rayuela.fsa.pathsum import Pathsum, Strategy
from rayuela.fsa.scc import SCC
from rayuela.fsa.state import State

def real_fsa(cyclic=True):
    fsa = FSA(Real)

    # We can directly add edges between the states without adding the states first.
    # The states will be created automatically.
    fsa.add_arc(State(1), Sym('a'), State(2), Real(0.5))
    fsa.add_arc(State(1), Sym('b'), State(3), Real(0.42))

    if cyclic:
        fsa.add_arc(State(2), Sym('b'), State(2), Real(0.63))
    fsa.add_arc(State(2), Sym('c'), State(4), Real(0.9))

    fsa.add_arc(State(3), Sym('c'), State(4), Real(0.21))
    fsa.add_arc(State(3), Sym('b'), State(5), Real(0.13))

    fsa.add_arc(State(4), Sym('a'), State(6), Real(0.72))
    fsa.add_arc(State(5), Sym('a'), State(6), Real(0.29))

    # Add initial and final states
    # This time, we also add weights to the inital / final states.
    fsa.set_I(State(1), Real(0.3))
    fsa.set_F(State(6), Real(0.1))

    return fsa

def real_scc_fsa():
    fsa = FSA(R=Real)

    fsa.add_arc(State(2), Sym('a'), State(5), Real(0.1))

    fsa.add_arc(State(3), Sym('a'), State(6), Real(0.1))
    fsa.add_arc(State(3), Sym('b'), State(1), Real(0.7))

    fsa.add_arc(State(1), Sym('c'), State(3), Real(0.3))
    fsa.add_arc(State(1), Sym('c'), State(5), Real(0.3))
    fsa.add_arc(State(1), Sym('b'), State(4), Real(0.1))

    fsa.add_arc(State(4), Sym('a'), State(2), Real(0.5))

    fsa.add_arc(State(5), Sym('a'), State(7), Real(0.8))

    fsa.add_arc(State(6), Sym('a'), State(7), Real(0.7))
    fsa.add_arc(State(6), Sym('a'), State(3), Real(0.1))

    fsa.add_arc(State(7), Sym('a'), State(8), Real(0.2))

    fsa.add_arc(State(8), Sym('a'), State(7), Real(0.2))
    fsa.add_arc(State(8), Sym('a'), State(2), Real(0.2))

    fsa.set_I(State(3), Real(0.3))
    fsa.set_F(State(2), Real(0.4))
    fsa.set_F(State(7), Real(0.2))

    return fsa


def test_union():
    fsa = FSA(Tropical)

    # We can directly add edges between the states without adding the states first.
    # The states will be created automatically.
    fsa.add_arc(State(1), Sym('a'), State(2), Tropical(0.5))
    fsa.add_arc(State(1), Sym('b'), State(3), Tropical(0.42))

    fsa.add_arc(State(2), Sym('b'), State(2), Tropical(0.63))
    fsa.add_arc(State(2), Sym('c'), State(4), Tropical(0.9))

    fsa.add_arc(State(3), Sym('c'), State(4), Tropical(0.21))
    fsa.add_arc(State(3), Sym('b'), State(5), Tropical(0.13))

    fsa.add_arc(State(4), Sym('a'), State(6), Tropical(0.72))
    fsa.add_arc(State(5), Sym('a'), State(6), Tropical(0.29))

    # Add initial and final states
    # This time, we also add weights to the inital / final states.
    fsa.set_I(State(1), Tropical(0.3))
    fsa.set_F(State(6), Tropical(0.1))

    fsa_union = fsa.union(fsa)

    print(fsa.accept("abbca"))
    print(fsa_union.accept("abbca"))

def test_fwd():
    fsa = real_fsa(cyclic=False)

    pathsum = Pathsum(fsa)

    beta = pathsum.viterbi_bwd()
    backward_ps = pathsum.pathsum(Strategy.VITERBI)
    alpha = pathsum.viterbi_fwd()
    ps = fsa.R.zero
    for q, w in fsa.F:
        ps += w * alpha[q]
    assert backward_ps == ps



def test_reverse():
    fsa = real_fsa(cyclic=False)
    rev_fsa = fsa.reverse()
    assert rev_fsa != fsa
    rev_rev_fsa = rev_fsa.reverse()
    assert rev_rev_fsa == fsa

def test_arcsums():
    fsa = real_fsa(cyclic=False)
    ps = fsa.pathsum()
    arcsums = fsa.viterbi_arcsums()
    qstart = State(3)
    ps2 = fsa.R.zero
    for a, qdest, w in fsa.arcs(qstart):
        ps2 += arcsums[qstart][a][qdest]

    assert ps == ps2

def test_lehmann():
    fsa = FSA(R=Real)
    fsa.add_arc(5, 'a', 6, 0.9)
    fsa.add_arc(6, 'a', 5, 0.5)
    ps = Pathsum(fsa)
    Ms = ps._lehmann()
    return Ms

def test_kosaraju():
    fsa = real_scc_fsa()
    scc = SCC(fsa)
    print(list(scc.scc()))

def test_blob_surgery():
    fsa = real_scc_fsa()
    states = list(fsa.Q)
    print(states)
    subset = set([states[i] for i in (0, 2, 5)])
    print(subset)
    fsa2 = fsa.copy()
    fsa2.blob_surgery_test(subset)

def test_lehmann_decompose():
    fsa = real_scc_fsa()
    ps = Pathsum(fsa)
    out = ps.decomposed_lehmann_pathsum()
    return out

if __name__ == '__main__':
    test_lehmann_decompose()