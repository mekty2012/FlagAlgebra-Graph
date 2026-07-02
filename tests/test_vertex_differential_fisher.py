import math
import itertools
import networkx as nx
import flag_algebra.flag_algebra as fa
from fractions import Fraction


def induced_graph_density_weighted_exact(H, weights):
    """Exact Fraction induced density of H in a graphon with rational block weights.

    weights: list of Fraction values (must sum to 1).
    Returns 0 if H is not a complete multipartite graph, else a Fraction.
    Multiplies by n!/|Aut(H)| to give the unlabeled (isomorphism-class) density.
    """
    from collections import Counter
    H_bar = nx.complement(H)
    components = list(nx.connected_components(H_bar))
    m = len(components)

    for comp in components:
        subgraph = H_bar.subgraph(comp)
        num_nodes = len(comp)
        if subgraph.number_of_edges() != num_nodes * (num_nodes - 1) // 2:
            return Fraction(0)

    if m > len(weights):
        return Fraction(0)

    sizes = [len(comp) for comp in components]
    total_density = Fraction(0)

    for indices in itertools.permutations(range(len(weights)), m):
        term = Fraction(1)
        for j in range(m):
            term *= weights[indices[j]] ** sizes[j]
        total_density += term

    n = sum(sizes)
    unlabeled_factor = math.factorial(n)
    for s in sizes:
        unlabeled_factor //= math.factorial(s)
    for mult in Counter(sizes).values():
        unlabeled_factor //= math.factorial(mult)

    return total_density * unlabeled_factor


def test_vertex_differential_fisher():
    k3 = nx.from_edgelist([(0, 1), (1, 2), (0, 2)])
    edge = nx.from_edgelist([(0, 1)])
    atlas = fa.get_graph_atlas(7)

    failures = []

    # S = n/10 gives rational weights and g_prime = (20+n)/20.
    # Scale the objective by 20 so coefficients are integers.
    # x = (400 - n^2) / 600 ranges from 2/3 (n=0) to 1/2 (n=10).
    for n in range(11):
        S = Fraction(n, 10)
        weights = [Fraction(20 + n, 60), Fraction(20 + n, 60), Fraction(10 - n, 30)]
        objective = [('sub', k3, 20), ('sub', edge, -(20 + n))]

        grad_num, _ = fa.vertex_differential(objective, atlas, integer_mode=True)

        density_fracs = [induced_graph_density_weighted_exact(H, weights) for H in atlas]

        lcm_denom = 1
        for f in density_fracs:
            lcm_denom = lcm_denom * f.denominator // math.gcd(lcm_denom, f.denominator)
        density_nums = [f.numerator * (lcm_denom // f.denominator) for f in density_fracs]

        # Exact integer dot product, shape [len_flags1]
        res_int = [
            sum(density_nums[j] * int(grad_num[j, fi]) for j in range(len(atlas)))
            for fi in range(grad_num.shape[1])
        ]

        if not all(r == 0 for r in res_int):
            bad = [(i, res_int[i]) for i in range(len(res_int)) if res_int[i] != 0]
            x = Fraction(400 - n * n, 600)
            failures.append(f"S={S} (x={x}): nonzero at {bad}")

    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    test_vertex_differential_fisher()
    print("All tests passed for Razborov's vertex differential (exact rational mode).")
