import flag_algebra.flag_algebra as fa
import networkx as nx
import math
import numpy as np
import itertools
from fractions import Fraction


def induced_graph_density_exact(H, G):
    """Exact Fraction version of induced density of H in blowup of G.

    Returns a Fraction representing the exact induced density.
    """
    m = len(H)
    k = len(G)

    densities = {}
    all_subsets = []
    for r in range(1, m + 1):
        all_subsets.extend(itertools.combinations(range(m), r))

    for subset_tuple in all_subsets:
        subset = frozenset(subset_tuple)
        size = len(subset)

        if size == 1:
            densities[subset] = Fraction(1)
            continue

        sum_split_val = Fraction(0)
        ordered_subset = sorted(list(subset))

        for mapping in itertools.product(range(k), repeat=size):
            unique_buckets = set(mapping)

            if len(unique_buckets) == 1:
                continue

            consistent = True
            for i in range(size):
                for j in range(i + 1, size):
                    u_idx = ordered_subset[i]
                    v_idx = ordered_subset[j]
                    bucket_u_idx = mapping[i]
                    bucket_v_idx = mapping[j]

                    if bucket_u_idx != bucket_v_idx:
                        is_edge_H = H.has_edge(u_idx, v_idx)
                        is_edge_G = G.has_edge(bucket_u_idx, bucket_v_idx)

                        if is_edge_H != is_edge_G:
                            consistent = False
                            break
                if not consistent:
                    break

            if consistent:
                term = Fraction(1)
                bucket_groups = {b: [] for b in unique_buckets}
                for i, bucket_idx in enumerate(mapping):
                    bucket_groups[bucket_idx].append(ordered_subset[i])

                for b_nodes in bucket_groups.values():
                    if len(b_nodes) > 0:
                        term *= densities[frozenset(b_nodes)]

                sum_split_val += term

        densities[subset] = Fraction(sum_split_val, k ** size - k)

    prob_density = densities[frozenset(range(m))]

    GM = nx.isomorphism.GraphMatcher(H, H)
    automorphisms = sum(1 for _ in GM.isomorphisms_iter())
    unlabelled_density = prob_density * Fraction(math.factorial(m), automorphisms)

    return unlabelled_density


def test_vertex_differential_c5():
    C5 = nx.cycle_graph(5)
    density = induced_graph_density_exact(C5, C5)
    expected = Fraction(math.factorial(5), 5 ** 5 - 5)
    assert density == expected, f"Expected density {expected}, got {density}"

    objective = [('ind', C5, -1)]
    atlas = fa.get_graph_atlas(7)

    # Integer mode: exact rational result
    grad_num, _ = fa.vertex_differential(objective, atlas, integer_mode=True)

    # Exact Fraction densities; find common denominator then use integer arithmetic
    density_fracs = [induced_graph_density_exact(G, C5) for G in atlas]
    lcm_denom = 1
    for f in density_fracs:
        lcm_denom = lcm_denom * f.denominator // math.gcd(lcm_denom, f.denominator)
    density_nums = np.array(
        [f.numerator * (lcm_denom // f.denominator) for f in density_fracs], dtype=object
    )

    # Exact integer dot product: shape [len_flags1]
    res_int = np.array(
        [int(sum(density_nums[j] * int(grad_num[j, fi]) for j in range(len(atlas))))
         for fi in range(grad_num.shape[1])]
    )

    failures = []
    if not np.all(res_int == 0):
        bad = np.where(res_int != 0)[0]
        failures.append(f"vertex differential not exactly zero at flag indices {bad.tolist()}")

    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    test_vertex_differential_c5()
    print("Test passed: The vertex differential for C5 inducibility is exactly correct.")
