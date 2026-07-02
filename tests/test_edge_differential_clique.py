import flag_algebra.flag_algebra as fa
import networkx as nx
import numpy as np


def test_edge_differential_clique():
    failures = []
    for k in range(2, 6):
        G = nx.complete_graph(k)
        atlas = fa.get_graph_atlas(k + 1)
        objective = [('ind', G, 1)]

        derivative = fa.rat_to_float(*fa.edge_differential(objective, atlas, integer_mode=True, return_average=False))

        expected_last = -k * (k - 1) / 2
        if not np.isclose(derivative[-1], expected_last):
            failures.append(
                f"k={k}: expected coefficient of K_{{k+1}} to be {expected_last}, got {derivative[-1]:.6f}"
            )
        if not np.allclose(derivative[:-1], 0):
            failures.append(
                f"k={k}: expected all other coefficients to be 0, got max|coeff|={np.max(np.abs(derivative[:-1])):.3e}"
            )

    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    test_edge_differential_clique()
    print("Test passed: The edge differential of K_r is correct.")
