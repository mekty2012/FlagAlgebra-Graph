import flag_algebra.flag_algebra as fa
import networkx as nx
import numpy as np


def is_clique(graph):
    n = len(graph.nodes)
    m = len(graph.edges)
    return m == n * (n - 1) / 2


def test_vertex_differential_clique():
    failures = []
    for k in range(2, 6):
        G = nx.complete_graph(k)
        atlas = fa.get_graph_atlas(k + 1)
        objective = [('ind', G, 1)]

        derivative = fa.rat_to_float(*fa.vertex_differential(objective, atlas, integer_mode=True, return_average=False))

        flags = fa.get_partially_labeled_graph_atlas(n=k + 1, k=1)
        res = np.zeros(len(flags))

        for i, flag in enumerate(flags):
            unlabeled_subgraph = flag.subgraph([v for v in flag.nodes if 'label' not in flag.nodes[v]])

            if is_clique(unlabeled_subgraph):
                res[i] = 1

            labeled_vertex = [v for v in flag.nodes if 'label' in flag.nodes[v]][0]
            labeled_deg = flag.degree(labeled_vertex)

            if labeled_deg == k - 1:
                neighbors = list(flag.neighbors(labeled_vertex))
                if is_clique(flag.subgraph([labeled_vertex] + neighbors)):
                    res[i] -= 1 / k
            elif labeled_deg == k:
                unlabeled_vertices = [v for v in flag.nodes if 'label' not in flag.nodes[v]]
                for v in unlabeled_vertices:
                    rest_vertices = set(unlabeled_vertices) - {v}
                    if is_clique(flag.subgraph(rest_vertices.union({labeled_vertex}))):
                        res[i] -= 1 / k

        res = res * k

        if not np.allclose(derivative, res):
            max_err = np.max(np.abs(derivative - res))
            failures.append(f"k={k}: max|derivative - expected|={max_err:.3e}")

    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    test_vertex_differential_clique()
    print("Test passed: The vertex differential for K_r cliques is correct.")
