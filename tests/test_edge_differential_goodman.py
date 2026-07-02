import flag_algebra.flag_algebra as fa
import networkx as nx
import numpy as np
import math


def falling_factorial(x, m):
    res = 1
    for i in range(m):
        res *= (x - i)
    return res


def induced_graph_density_int(H, k):
    """Integer numerator of induced density of H in balanced k-partite graphon.

    Returns falling_factorial(k, m) * (n! / |Aut(H)|) where n = |V(H)| and
    m = number of parts in H. The n!/|Aut(H)| factor converts labeled density
    to unlabeled (isomorphism-class) density. Returns 0 if H is not a complete
    multipartite graph.
    """
    from collections import Counter
    H_bar = nx.complement(H)
    components = list(nx.connected_components(H_bar))
    m = len(components)
    for comp in components:
        subgraph = H_bar.subgraph(comp)
        num_nodes = len(comp)
        if subgraph.number_of_edges() != num_nodes * (num_nodes - 1) // 2:
            return 0
    sizes = [len(comp) for comp in components]
    n = sum(sizes)
    unlabeled_factor = math.factorial(n)
    for s in sizes:
        unlabeled_factor //= math.factorial(s)
    for mult in Counter(sizes).values():
        unlabeled_factor //= math.factorial(mult)
    return falling_factorial(k, m) * unlabeled_factor


def test_edge_differential_goodman():
    k3 = nx.from_edgelist([(0, 1), (1, 2), (0, 2)])
    edge = nx.from_edgelist([(0, 1)])
    e2 = nx.from_edgelist([(0, 1), (2, 3)])

    objective = [('sub', k3, 1), ('sub', e2, -2), ('sub', edge, 1)]
    atlas = fa.get_graph_atlas(7)

    # Integer mode: exact rational result
    grad_num, _ = fa.edge_differential(objective, atlas, integer_mode=True)

    failures = []
    for k in [3, 4, 5, 6, 7]:
        # Integer density numerators — denominator k^7 is common and cancels in the check
        density_nums = np.array([induced_graph_density_int(H, k) for H in atlas], dtype=np.int64)
        # Exact integer dot product: negative iff edge differential is negative
        res_int = density_nums @ grad_num
        if not np.all(res_int >= 0):
            bad = np.where(res_int < 0)[0]
            failures.append(f"W_{k}: negative at flag indices {bad.tolist()}, values {res_int[bad].tolist()}")

    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    test_edge_differential_goodman()
    print("Test passed: The edge differential for Goodman expression is non-negative at W_k for k=3,4,5,6,7.")
