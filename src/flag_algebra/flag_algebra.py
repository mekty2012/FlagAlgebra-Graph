import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
from itertools import permutations, combinations, product
import math
import tqdm
import subprocess
import shutil
import sys
import io
import warnings

ATLAS = nx.graph_atlas_g()

######################################################################
###                           Graph Utils                          ###
######################################################################
def _get_invariants(g, label_name='label'):
  """
  An invariant function that accelerates the isomorphism testing for partially labeled graphs
  Here, we use the degree sequence grouped by labels as the invariant.

  Args:
    g (networkx.Graph): Input graph with node labels
  Returns:
    tuple: Invariant representation of the graph
  """
  deg_by_label = {}
  g_labels = nx.get_node_attributes(g, label_name)

  for node, degree in g.degree():
    label = g_labels.get(node, -1)

    if label not in deg_by_label:
      deg_by_label[label] = []
    deg_by_label[label].append(degree)

  final_invariant_list = []
  for label, degrees in deg_by_label.items():
    sorted_degrees = tuple(sorted(degrees))
    final_invariant_list.append((label, sorted_degrees))

  final_invariant_list.sort()

  return tuple(final_invariant_list)

def graph_equal(g1, g2, label_name='label'):
  # Define edge set based on labels
  g1_edges = set()
  for g1_edge in g1.edges:
    u_label = g1.nodes[g1_edge[0]].get(label_name, -1)
    v_label = g1.nodes[g1_edge[1]].get(label_name, -1)
    g1_edges.add((min(u_label, v_label), max(u_label, v_label)))
  g2_edges = set()
  for g2_edge in g2.edges:
    u_label = g2.nodes[g2_edge[0]].get(label_name, -1)
    v_label = g2.nodes[g2_edge[1]].get(label_name, -1)
    g2_edges.add((min(u_label, v_label), max(u_label, v_label)))

  return g1_edges == g2_edges

def check_graph_isomorphism(g1, g1_inv, g2, g2_inv, label_name='label'):
  if g1_inv != g2_inv:
    return False

  nm = isomorphism.categorical_node_match(label_name, -1)
  GM = isomorphism.GraphMatcher(g1, g2, node_match=nm)
  return GM.is_isomorphic()

def _sort_key(G):
  """
  Sorting key to approximate 'geng' canonical order:
  1. Number of edges (ascending)
  2. Degree sequence (descending, lexicographical)
  """
  deg_seq = sorted((d for n, d in G.degree()), reverse=True)
  return (G.number_of_edges(), deg_seq)

######################################################################
###         Integer/Rational Arithmetic for Flag Algebras          ###
######################################################################
#
# A "rational array" is a tuple (num, denom) where:
#   num   : np.ndarray of dtype int64
#   denom : Python int (positive)
# The element at index i represents the value  num[i] / denom.
# All elements share a single common denominator.

def _gcd_of_array(arr: np.ndarray) -> int:
  """GCD of absolute values of all elements in an integer ndarray. Returns 1 if array is empty."""
  g = 0
  for x in arr.flat:
    xi = int(abs(int(x)))
    g = math.gcd(g, xi)
    if g == 1:
      return 1
  return g if g != 0 else 1


def rat_reduce(num: np.ndarray, denom: int):
  """
  Divide (num, denom) by their common GCD so the representation is in lowest terms.
  Always keeps denom positive.
  Returns (num_reduced, denom_reduced).
  """
  if denom < 0:
    num = -num.copy()
    denom = -denom
  if np.all(num == 0):
    return num.copy(), 1
  g = math.gcd(abs(denom), _gcd_of_array(num))
  if g > 1:
    return num // g, denom // g
  return num, denom


def rat_to_float(num: np.ndarray, denom: int) -> np.ndarray:
  """Convert a rational array (num, denom) to a float64 ndarray."""
  return num.astype(np.float64) / float(denom)


def _rat_scalar_reduce(num: int, denom: int):
  """Reduce a rational scalar (num, denom) to lowest terms. Keeps denom positive."""
  if denom < 0:
    num, denom = -num, -denom
  if num == 0:
    return 0, 1
  g = math.gcd(abs(num), denom)
  return num // g, denom // g


def _rat_scalar_add(a_num: int, a_denom: int, b_num: int, b_denom: int):
  """
  Add two rational scalars: a_num/a_denom + b_num/b_denom.
  Returns (num, denom) in reduced form.
  """
  if a_denom == b_denom:
    total = a_num + b_num
    if total == 0:
      return 0, 1
    g = math.gcd(abs(total), a_denom)
    return total // g, a_denom // g
  gcd_d = math.gcd(a_denom, b_denom)
  lcm = a_denom * b_denom // gcd_d
  total = a_num * (lcm // a_denom) + b_num * (lcm // b_denom)
  if total == 0:
    return 0, 1
  g = math.gcd(abs(total), lcm)
  return total // g, lcm // g


def _lcm(a: int, b: int) -> int:
  """Least common multiple of two positive integers."""
  return a * b // math.gcd(a, b)


def _lcm_list(values) -> int:
  """LCM of a list of positive integers."""
  result = 1
  for v in values:
    result = _lcm(result, v)
  return result


def _count_induced_labeled_subgraph_isos(other, flag, nm, aut_F):
  """
  Count label-preserving injective maps from flag to induced subgraphs of other.

  Flag algebra requires INDUCED densities.  NetworkX's subgraph_isomorphisms_iter()
  is non-induced, so we enumerate all same-size subsets of the unlabeled vertices of
  `other`, form the induced subgraph, and check full isomorphism.

  Result = #{good subsets} * aut_F, matching the normalisation formula in Step 3.
  """
  labeled_in_other = [v for v in other.nodes if 'label' in other.nodes[v]]
  unlabeled_in_other = [v for v in other.nodes if 'label' not in other.nodes[v]]
  n_unlabeled_flag = sum(1 for v in flag.nodes if 'label' not in flag.nodes[v])

  good_subsets = 0
  for subset in combinations(unlabeled_in_other, n_unlabeled_flag):
    induced = other.subgraph(labeled_in_other + list(subset))
    GM_iso = isomorphism.GraphMatcher(induced, flag, node_match=nm)
    if GM_iso.is_isomorphic():
      good_subsets += 1
  return good_subsets * aut_F

######################################################################
###          Graph Atlas and Partially Labeled Graph Atlas         ###
######################################################################

def get_graph_atlas(n):
  """
  Returns a list of all graphs with n vertices from the NetworkX graph atlas.

  Args:
    n (int): Number of vertices
  Returns:
    List of networkx.Graph objects with n vertices
  """
  if n <= 7:
    return [G.copy() for G in ATLAS if len(G) == n]
  else:
    current_graphs = [G.copy() for G in ATLAS if len(G) == 7]

    for current_size in range(7, n):
      next_size = current_size + 1
      next_graphs = []

      grouped_next_graphs = {}

      for G in current_graphs:
        new_node = current_size
        nodes = list(G.nodes())

        for i in range(1 << current_size):
          candidate = G.copy()
          candidate.add_node(new_node)

          for bit_idx in range(current_size):
            if (i >> bit_idx) & 1:
              candidate.add_edge(new_node, nodes[bit_idx])

          cand_inv = _get_invariants(candidate)
          if cand_inv not in grouped_next_graphs:
            grouped_next_graphs[cand_inv] = []

          is_iso = False
          for other, other_inv in grouped_next_graphs[cand_inv]:
            if check_graph_isomorphism(candidate, cand_inv, other, other_inv):
              is_iso = True
              break

          if not is_iso:
            grouped_next_graphs[cand_inv].append((candidate, cand_inv))
            next_graphs.append(candidate)
      current_graphs = next_graphs

    current_graphs.sort(key=_sort_key)
    return current_graphs

def get_partially_labeled_graph_atlas(n, k):
  """
  Returns a list of all partially labeled graphs with n vertices and k labeled vertices.

  Args:
    n (int): Number of vertices
    k (int): Number of labeled vertices
  Returns:
    List of networkx.Graph objects with n vertices and k labeled vertices
  """

  atlas = get_graph_atlas(n)

  res = []
  for g in atlas:
    labeled_g_list = []

    for labeled_vertices in permutations(g.nodes, k):
      g_labeled = g.copy()
      label_dict = {
        v: i for i, v in enumerate(labeled_vertices)
      }
      nx.set_node_attributes(g_labeled, label_dict, 'label')

      g_labeled_inv = _get_invariants(g_labeled)

      for other, other_invariant in labeled_g_list:
        # Check isomorphism considering labels
        if check_graph_isomorphism(g_labeled, g_labeled_inv, other, other_invariant):
          break
      else:
        labeled_g_list.append((g_labeled, g_labeled_inv))

    res.extend(labeled_g_list)

  return [g for g, _ in res]

######################################################################
###            Computation of Subgraph Coefficients            ###
######################################################################

def compute_hom_coefficients(H, atlas=None):
  """
  Computes the homomorphism coefficients c_g^OPT for a given graph H over an atlas of graphs.

  Args:
    H (networkx.Graph): Target graph
    atlas (list of networkx.Graph, optional): List of graphs to compute coefficients for. If None, uses the graph atlas for graphs with the same number of nodes as H.
  Returns:
    np.array: Array of homomorphism coefficients c_g^OPT for each graph g in the atlas such that
              t(H, G) ≃ sum_g c_g^OPT * t(ind g, G)
  """
  if atlas is None:
    atlas = get_graph_atlas(len(H.nodes))

  res = []
  for g in atlas:
    matcher = isomorphism.GraphMatcher(g, H)
    num_monomorphisms = sum(1 for _ in matcher.subgraph_monomorphisms_iter())
    c_g_OPT = num_monomorphisms / math.perm(len(g.nodes), len(H.nodes))
    res.append(c_g_OPT)
  return np.array(res)

def compute_ind_hom_coefficients(H, atlas=None):
  """
  Computes the induced homomorphism coefficients c_g^IND for a given graph H over an atlas of graphs.

  Args:
    H (networkx.Graph): Target graph
    atlas (list of networkx.Graph, optional): List of graphs to compute coefficients for. If None, uses the graph atlas for graphs with the same number of nodes as H.
  Returns:
    np.array: Array of induced homomorphism coefficients c_g^IND for each graph g in the atlas such that
              t_ind(H, G) ≃ sum_g c_g^IND * p(ind g, G)
  """
  if atlas is None:
    atlas = get_graph_atlas(len(H.nodes))

  res = []
  for g in atlas:
    matcher = isomorphism.GraphMatcher(g, H)
    num_monomorphisms = sum(1 for _ in matcher.subgraph_isomorphisms_iter())
    c_g_OPT = num_monomorphisms / math.perm(len(g.nodes), len(H.nodes))
    res.append(c_g_OPT)
  return np.array(res)

def compute_ind_subgraph_coefficients(H, atlas=None):
  """
  Computes the induced subgraph coefficients c_g^OPT for a given graph H over an atlas of graphs.

  Args:
    H (networkx.Graph): Target graph
    atlas (list of networkx.Graph, optional): List of graphs to compute coefficients for. If None, uses the graph atlas for graphs with the same number of nodes as H.
  Returns:
    np.array: Array of induced subgraph coefficients c_g^OPT for each graph g in the atlas such that
              p(ind H, G) ≃ sum_g c_g^OPT * p(ind g, G)
  """
  if atlas is None:
    atlas = get_graph_atlas(len(H.nodes))

  res = []
  for g in atlas:
    subgraph_set = set()
    matcher = isomorphism.GraphMatcher(g, H)

    for sub_mono in matcher.subgraph_isomorphisms_iter():
      subg_nodes = frozenset(sub_mono.keys())
      subgraph_set.add(subg_nodes)

    c_G_OPT = len(subgraph_set) / math.comb(len(g.nodes), len(H.nodes))
    res.append(c_G_OPT)

  return np.array(res)

def subgraph_to_ind_subgraph(H, integer_mode=False):
  """
  Returns {graph: coefficient} where t_sub(H) = sum_g coeff_g * t_ind(g).

  The correct formula is: t_sub(H) = sum_g (#mono H->g) / |H|! * t_ind(g),
  where the sum is over same-size graphs g and mono H->g counts injective
  edge-preserving maps from H into g (not the reverse).

  Args:
    H (networkx.Graph): Source graph.
    integer_mode (bool): If True, return {graph: (num_monomorphisms, |H|!)} instead of floats.

  Returns:
    dict: When integer_mode=False, maps graph -> float coefficient.
          When integer_mode=True,  maps graph -> (int numerator, int denominator).
  """
  atlas = get_graph_atlas(len(H.nodes))
  n_fact = math.factorial(len(H.nodes))
  res = dict()
  for g in atlas:
    matcher = isomorphism.GraphMatcher(g, H)
    num_monomorphisms = sum(1 for _ in matcher.subgraph_monomorphisms_iter())

    if num_monomorphisms > 0:
      if integer_mode:
        res[g] = (num_monomorphisms, n_fact)
      else:
        res[g] = num_monomorphisms / n_fact

  return res

def compute_subgraph_coefficients(H, atlas=None, integer_mode=False):
  """
  Compute induced-subgraph expansion coefficients for t(H, ·).

  Args:
    H (networkx.Graph): Graph whose subgraph density is to be expanded.
    atlas (list, optional): Atlas of graphs to expand into. Defaults to same-size atlas.
    integer_mode (bool): If True, return (np.ndarray[int64], int) rational array.

  Returns:
    When integer_mode=False: np.ndarray of float64.
    When integer_mode=True:  (np.ndarray[int64], int) — numerators and common denominator.
  """
  ind_subgraph_coefficients = subgraph_to_ind_subgraph(H, integer_mode=integer_mode)

  if atlas is None:
    atlas = get_graph_atlas(len(H.nodes))

  if not integer_mode:
    res = np.zeros(len(atlas))
    for g, c in ind_subgraph_coefficients.items():
      ind_subgraph_coeff = compute_ind_subgraph_coefficients(g, atlas)
      res += c * ind_subgraph_coeff
    return res

  # integer_mode=True
  # ind_subgraph_coefficients: {g: (num_mono, aut_g)}
  # For same-size atlas, compute_ind_subgraph_coefficients returns a 0/1 indicator array
  # (since |g| == |atlas elements|, comb(n,n)=1 so coeff = len(subgraph_set) in {0,1}).
  # So res[atlas_idx_of_g] += num_mono / aut_g.

  atlas_invs = [_get_invariants(ag) for ag in atlas]

  # Map each g to its atlas index and collect (num_mono, aut_g) per atlas slot
  indexed_rationals = {}  # atlas_idx -> list of (num_mono, aut_g)
  for g, (num_mono, aut_g) in ind_subgraph_coefficients.items():
    g_inv = _get_invariants(g)
    for idx, inv in enumerate(atlas_invs):
      if check_graph_isomorphism(g, g_inv, atlas[idx], inv):
        if idx not in indexed_rationals:
          indexed_rationals[idx] = []
        indexed_rationals[idx].append((num_mono, aut_g))
        break

  if not indexed_rationals:
    return np.zeros(len(atlas), dtype=np.int64), 1

  # Common denominator = LCM of all aut_g values
  all_denoms = [aut_g for pairs in indexed_rationals.values() for _, aut_g in pairs]
  lcm_denom = _lcm_list(all_denoms)

  result_num = np.zeros(len(atlas), dtype=np.int64)
  for idx, pairs in indexed_rationals.items():
    for num_mono, aut_g in pairs:
      result_num[idx] += num_mono * (lcm_denom // aut_g)

  return rat_reduce(result_num, lcm_denom)

def compute_edge_densities(atlas):
  """
  Computes the edge density for each graph in the atlas.

  Args:
    atlas (list of networkx.Graph): List of graphs to compute edge densities for
  Returns:
    np.array: Array of edge densities for each graph in the atlas
  """
  res = []
  for g in atlas:
    e = len(g.edges)
    res.append(e / (len(g.nodes) * (len(g.nodes) - 1) / 2))

  return np.array(res)

######################################################################
###                Computation of SDP coefficients                 ###
######################################################################

def compute_averaged_flag_product_coefficients(atlas, n, k, verbose=False, integer_mode=False):
  """
  Computes the averaged flag product coefficients for a given atlas.

  Args:
    atlas (list of networkx.Graph): List of graphs to compute coefficients for (Should be graphs with 2*n-k vertices)
    n (int): Number of vertices in the flags
    k (int): Number of labeled vertices
    verbose (bool): Whether to print progress information
    integer_mode (bool): If True, return (np.ndarray[int64], int) rational array.

  Returns:
    When integer_mode=False: np.ndarray of shape [1, len(atlas), len(partial_atlas_n_k), len(partial_atlas_n_k)]
    When integer_mode=True:  (np.ndarray[int64] of same shape, int denominator)
  """
  partial_atlas_n2_k = get_partially_labeled_graph_atlas(n, k)

  atlas_invariants = [_get_invariants(pg) for pg in partial_atlas_n2_k]

  # Compute constant denominator for integer mode:
  # denom = perm(2n-k, k) * comb(2n-2k, n-k)
  if integer_mode:
    int_denom = math.perm(2 * n - k, k) * math.comb(2 * n - 2 * k, n - k)
    A_int = np.zeros((len(atlas), len(partial_atlas_n2_k), len(partial_atlas_n2_k)), dtype=np.int64)
  else:
    A_results = np.zeros((len(atlas), len(partial_atlas_n2_k), len(partial_atlas_n2_k)))

  pbar = atlas if not verbose else tqdm.tqdm(atlas)

  nm = isomorphism.categorical_node_match('label', -1)

  for atlas_idx, g in enumerate(pbar):
    labeled_gs_data = []
    labeled_count = {}

    if not integer_mode:
      res = np.zeros((len(partial_atlas_n2_k), len(partial_atlas_n2_k)))

    for labeled_vertices in permutations(g.nodes, k):
      g_labeled = g.copy()
      label_dict = {
        v: i for i, v in enumerate(labeled_vertices)
      }
      nx.set_node_attributes(g_labeled, label_dict, 'label')

      g_labeled_inv = _get_invariants(g_labeled)

      found_match = False
      for i, (other_g, other_inv) in enumerate(labeled_gs_data):
        if check_graph_isomorphism(g_labeled, g_labeled_inv, other_g, other_inv):
          labeled_count[i] += 1
          found_match = True
          break

      if not found_match:
        new_idx = len(labeled_gs_data)
        labeled_gs_data.append((g_labeled, g_labeled_inv))
        labeled_count[new_idx] = 1

    total_labeled = sum(labeled_count.values())

    for label_idx, (labeled_g, _) in enumerate(labeled_gs_data):
      unlabeled_vertices = [
        v for v in labeled_g.nodes if 'label' not in labeled_g.nodes[v]
      ]
      labeled_vertices_nodes = [
        v for v in labeled_g.nodes if 'label' in labeled_g.nodes[v]
      ]
      m = len(unlabeled_vertices)
      base_labeled_nodes = list(labeled_vertices_nodes)

      for part1 in combinations(unlabeled_vertices, m // 2):
        part2 = [v for v in unlabeled_vertices if v not in part1]

        g1 = labeled_g.subgraph(list(part1) + base_labeled_nodes)
        g2 = labeled_g.subgraph(list(part2) + base_labeled_nodes)

        inv1 = _get_invariants(g1)
        inv2 = _get_invariants(g2)

        idx1 = None
        idx2 = None

        for i, pg_inv in enumerate(atlas_invariants):

          if idx1 is None and inv1 == pg_inv:
            pg = partial_atlas_n2_k[i]
            if check_graph_isomorphism(g1, inv1, pg, pg_inv):
              idx1 = i

          if idx2 is None and inv2 == pg_inv:
            pg = partial_atlas_n2_k[i]
            if check_graph_isomorphism(g2, inv2, pg, pg_inv):
              idx2 = i

          if idx1 is not None and idx2 is not None:
            break

        assert idx1 is not None, f"Failed to find index in partial atlas for g1: {g1.nodes(data=True)} {g1.edges()}"
        assert idx2 is not None, f"Failed to find index in partial atlas for g2: {g2.nodes(data=True)} {g2.edges()}"

        if integer_mode:
          # Numerator contribution: labeled_count[label_idx]
          # Full value = labeled_count / total_labeled / comb(m, m//2)
          # = labeled_count / (perm(2n-k, k) * comb(2n-2k, n-k))
          A_int[atlas_idx, idx1, idx2] += labeled_count[label_idx]
        else:
          res[idx1, idx2] += labeled_count[label_idx] / total_labeled / math.comb(m, m // 2)

    if not integer_mode:
      A_results[atlas_idx] = res

  if integer_mode:
    result_num = A_int[np.newaxis, :, :, :]
    return rat_reduce(result_num, int_denom)
  else:
    return A_results[np.newaxis, :, :, :]

def compute_grouped_averaged_flag_product_coefficients(atlas, n, k, verbose=False, integer_mode=False):
  """
  Computes the averaged flag product coefficients for grouped partially labeled graphs.

  Args:
    atlas: List of graphs (should have 2n-k vertices each)
    n (int): Number of vertices in the flags
    k (int): Number of labeled vertices
    verbose (bool): Whether to print progress
    integer_mode (bool): If True, return (np.ndarray[int64], int) rational array.

  Returns:
    When integer_mode=False: np.ndarray of shape [len(k-types), len(atlas), len(flags_i), len(flags_i)]
    When integer_mode=True:  (np.ndarray[int64] same shape, int denominator)
  """

  flags = get_partially_labeled_graph_atlas(n, k)
  types = get_graph_atlas(k)

  # label all the types
  for type in types:
    label_dict = {v: i for i, v in enumerate(type.nodes)}
    nx.set_node_attributes(type, label_dict, 'label')

  type_invariants = [_get_invariants(t) for t in types]

  # 1. Group flags by type
  flag_indices = [[] for _ in types]

  for f_index, flag in enumerate(flags):
    labeled_nodes = [v for v in flag.nodes if 'label' in flag.nodes[v]]
    flag_type = flag.subgraph(labeled_nodes)

    for t_index, ty in enumerate(types):
      is_type = graph_equal(flag_type, ty, label_name='label')

      if is_type:
        flag_indices[t_index].append(f_index)
        break

  # Constant denominator for integer mode:
  # denom = perm(2n-k, k) * comb(2n-2k, n-k)
  if integer_mode:
    int_denom = math.perm(2 * n - k, k) * math.comb(2 * n - 2 * k, n - k)
    results_int = [
      np.zeros((len(atlas), len(flag_indices[ti]), len(flag_indices[ti])), dtype=np.int64)
      for ti in range(len(types))
    ]
  else:
    results = [
      np.zeros((len(atlas), len(flag_indices[ti]), len(flag_indices[ti])))
      for ti in range(len(types))
    ]

  pbar = atlas if not verbose else tqdm.tqdm(atlas)

  nm = isomorphism.categorical_node_match('label', -1)

  for g_idx, g in enumerate(pbar):
    labeled_gs_data = [[] for _ in types]
    labeled_count = [{} for _ in types]

    for labeled_vertices in permutations(g.nodes, k):
      g_copy = g.copy()
      label_dict = {v: i for i, v in enumerate(labeled_vertices)}
      nx.set_node_attributes(g_copy, label_dict, 'label')
      labeled_subgraph = g_copy.subgraph(labeled_vertices)

      for type_idx, ty in enumerate(types):
        if graph_equal(labeled_subgraph, ty, label_name='label'):
          break
      else:
        continue
      g_labeled = g.copy()
      label_dict = {
        v: i for i, v in enumerate(labeled_vertices)
      }
      nx.set_node_attributes(g_labeled, label_dict, 'label')

      g_labeled_inv = _get_invariants(g_labeled)

      found_match = False
      for i, (other_g, other_inv) in enumerate(labeled_gs_data[type_idx]):
        if g_labeled_inv == other_inv:
          GM = isomorphism.GraphMatcher(g_labeled, other_g, node_match=nm)
          if GM.is_isomorphic():
            labeled_count[type_idx][i] += 1
            found_match = True
            break

      if not found_match:
        new_idx = len(labeled_gs_data[type_idx])
        labeled_gs_data[type_idx].append((g_labeled, g_labeled_inv))
        labeled_count[type_idx][new_idx] = 1

    for type_idx in range(len(types)):

      for label_idx, (labeled_g, _) in enumerate(labeled_gs_data[type_idx]):
        unlabeled_vertices = [
          v for v in labeled_g.nodes if 'label' not in labeled_g.nodes[v]
        ]
        labeled_vertices_nodes = [
          v for v in labeled_g.nodes if 'label' in labeled_g.nodes[v]
        ]

        m = len(unlabeled_vertices)
        base_labeled_nodes = list(labeled_vertices_nodes)

        for part1 in combinations(unlabeled_vertices, m // 2):
          part2 = [v for v in unlabeled_vertices if v not in part1]

          g1 = labeled_g.subgraph(list(part1) + base_labeled_nodes)
          g2 = labeled_g.subgraph(list(part2) + base_labeled_nodes)

          inv1 = _get_invariants(g1)
          inv2 = _get_invariants(g2)

          idx1 = None
          idx2 = None

          for i in flag_indices[type_idx]:

            pg = flags[i]
            pg_inv = _get_invariants(pg)
            if idx1 is None and inv1 == pg_inv:
              if check_graph_isomorphism(g1, inv1, pg, pg_inv):
                idx1 = flag_indices[type_idx].index(i)

            if idx2 is None and inv2 == pg_inv:
              if check_graph_isomorphism(g2, inv2, pg, pg_inv):
                idx2 = flag_indices[type_idx].index(i)

            if idx1 is not None and idx2 is not None:
              break

          assert idx1 is not None, f"Failed to find index in partial atlas for g1: {g1.nodes(data=True)} {g1.edges()}"
          assert idx2 is not None, f"Failed to find index in partial atlas for g2: {g2.nodes(data=True)} {g2.edges()}"

          if integer_mode:
            results_int[type_idx][g_idx, idx1, idx2] += labeled_count[type_idx][label_idx]
          else:
            results[type_idx][g_idx, idx1, idx2] += labeled_count[type_idx][label_idx] / math.perm(2 * n - k, k) / math.comb(m, m // 2)

  if integer_mode:
    stacked = np.stack(results_int, axis=0)
    return rat_reduce(stacked, int_denom)
  else:
    return np.stack(results, axis=0)

def compute_grouped_averaged_flag_product_coefficients_asymmetric(atlas, n1, n2, k, verbose=False, integer_mode=False):
  """
  Computes the averaged flag product coefficients for grouped partially labeled graphs
  with asymmetric flag sizes n1, n2.

  Args:
    atlas: List of graphs (should have n1+n2-k vertices each)
    n1 (int): Number of vertices in the first flag set
    n2 (int): Number of vertices in the second flag set
    k (int): Number of labeled vertices
    verbose (bool): Whether to print progress
    integer_mode (bool): If True, return (np.ndarray[int64], int) rational array.

  Returns:
    When integer_mode=False: np.ndarray of shape [len(types), len(atlas), len(flags1_i), len(flags2_i)]
    When integer_mode=True:  (np.ndarray[int64] same shape, int denominator)
  """
  flags1 = get_partially_labeled_graph_atlas(n1, k)
  flags2 = get_partially_labeled_graph_atlas(n2, k)
  types = get_graph_atlas(k)

  for type in types:
    label_dict = {v: i for i, v in enumerate(type.nodes)}
    nx.set_node_attributes(type, label_dict, 'label')

  type_invariants = [_get_invariants(t) for t in types]

  flag1_indices = [[] for _ in types]
  flag2_indices = [[] for _ in types]

  for f_index, flag in enumerate(flags1):
    labeled_nodes = [v for v in flag.nodes if 'label' in flag.nodes[v]]
    flag_type = flag.subgraph(labeled_nodes)

    for t_index, ty in enumerate(types):
      is_type = graph_equal(flag_type, ty, label_name='label')

      if is_type:
        flag1_indices[t_index].append(f_index)
        break

  for f_index, flag in enumerate(flags2):
    labeled_nodes = [v for v in flag.nodes if 'label' in flag.nodes[v]]
    flag_type = flag.subgraph(labeled_nodes)

    for t_index, ty in enumerate(types):
      is_type = graph_equal(flag_type, ty, label_name='label')

      if is_type:
        flag2_indices[t_index].append(f_index)
        break

  # Constant denominator for integer mode:
  # Each contribution is labeled_count / perm(n1+n2-k, k) / comb(m, n1-k)
  # where m = n1+n2-2k (constant).
  if integer_mode:
    m_unlabeled = n1 + n2 - 2 * k
    int_denom = math.perm(n1 + n2 - k, k) * math.comb(m_unlabeled, n1 - k)
    results_int = [
      np.zeros((len(atlas), len(flag1_indices[ti]), len(flag2_indices[ti])), dtype=np.int64)
      for ti in range(len(types))
    ]
  else:
    results = [
      np.zeros((len(atlas), len(flag1_indices[ti]), len(flag2_indices[ti])))
      for ti in range(len(types))
    ]

  pbar = atlas if not verbose else tqdm.tqdm(atlas)

  nm = isomorphism.categorical_node_match('label', -1)

  for g_idx, g in enumerate(pbar):
    labeled_gs_data = [[] for _ in types]
    labeled_count = [{} for _ in types]

    for labeled_vertices in permutations(g.nodes, k):
      g_copy = g.copy()
      label_dict = {v: i for i, v in enumerate(labeled_vertices)}
      nx.set_node_attributes(g_copy, label_dict, 'label')
      labeled_subgraph = g_copy.subgraph(labeled_vertices)

      for type_idx, ty in enumerate(types):
        if graph_equal(labeled_subgraph, ty, label_name='label'):
          break
      else:
        continue
      g_labeled = g.copy()
      label_dict = {
        v: i for i, v in enumerate(labeled_vertices)
      }
      nx.set_node_attributes(g_labeled, label_dict, 'label')

      g_labeled_inv = _get_invariants(g_labeled)

      found_match = False
      for i, (other_g, other_inv) in enumerate(labeled_gs_data[type_idx]):
        if g_labeled_inv == other_inv:
          GM = isomorphism.GraphMatcher(g_labeled, other_g, node_match=nm)
          if GM.is_isomorphic():
            labeled_count[type_idx][i] += 1
            found_match = True
            break

      if not found_match:
        new_idx = len(labeled_gs_data[type_idx])
        labeled_gs_data[type_idx].append((g_labeled, g_labeled_inv))
        labeled_count[type_idx][new_idx] = 1

    for type_idx in range(len(types)):
      for label_idx, (labeled_g, _) in enumerate(labeled_gs_data[type_idx]):
        unlabeled_vertices = [
          v for v in labeled_g.nodes if 'label' not in labeled_g.nodes[v]
        ]
        labeled_vertices_nodes = [
          v for v in labeled_g.nodes if 'label' in labeled_g.nodes[v]
        ]

        m = len(unlabeled_vertices)
        base_labeled_nodes = list(labeled_vertices_nodes)

        for part1 in combinations(unlabeled_vertices, n1 - k):
          part2 = [v for v in unlabeled_vertices if v not in part1]

          g1 = labeled_g.subgraph(list(part1) + base_labeled_nodes)
          g2 = labeled_g.subgraph(list(part2) + base_labeled_nodes)

          inv1 = _get_invariants(g1)
          inv2 = _get_invariants(g2)

          idx1 = None
          idx2 = None

          for i in flag1_indices[type_idx]:
            pg = flags1[i]
            pg_inv = _get_invariants(pg)
            if inv1 == pg_inv and check_graph_isomorphism(g1, inv1, pg, pg_inv):
              idx1 = flag1_indices[type_idx].index(i)
              break

          for i in flag2_indices[type_idx]:
            pg = flags2[i]
            pg_inv = _get_invariants(pg)
            if inv2 == pg_inv and check_graph_isomorphism(g2, inv2, pg, pg_inv):
              idx2 = flag2_indices[type_idx].index(i)
              break

          assert idx1 is not None, f"Failed to find index in partial atlas for g1: {g1.nodes(data=True)} {g1.edges()}"
          assert idx2 is not None, f"Failed to find index in partial atlas for g2: {g2.nodes(data=True)} {g2.edges()}"

          if integer_mode:
            results_int[type_idx][g_idx, idx1, idx2] += labeled_count[type_idx][label_idx]
          else:
            results[type_idx][g_idx, idx1, idx2] += labeled_count[type_idx][label_idx] / math.perm(n1 + n2 - k, k) / math.comb(m, n1 - k)

  if integer_mode:
    stacked = np.stack(results_int, axis=0)
    return rat_reduce(stacked, int_denom)
  else:
    return np.stack(results, axis=0)

######################################################################
###             Differential Operators in Flag Algebras            ###
######################################################################

def labeling_vertex_mu(G):
  """
  Computes labeling operator mu^1(G) in flag algebras.

  Args:
    G (networkx.Graph): Input graph

  Returns:
    list of (networkx.Graph, float): List of tuples (F, c_F) where F is G with one vertex labeled (up to isomorphism) and c_F is the number of isomorphic copies.
  """

  flags = []
  flags_invariants = []
  counts = []
  for v in G.nodes:
    G_labeled = G.copy()
    nx.set_node_attributes(G_labeled, {v: 0}, 'label')

    G_labeled_inv = _get_invariants(G_labeled)
    for i, inv in enumerate(flags_invariants):
      if check_graph_isomorphism(G_labeled, G_labeled_inv, flags[i], inv):
        counts[i] += 1
        break
    else:
      flags.append(G_labeled)
      flags_invariants.append(G_labeled_inv)
      counts.append(1)

  return list(zip(flags, counts))

def adding_vertex_pi(G):
  """
  Computes adding vertex operator pi^1(G) in flag algebras.

  Args:
    G (networkx.Graph): Input graph

  Returns:
    list of (networkx.Graph, float): List of tuples (F, c_F) where F is G with one additional labeled vertex (up to isomorphism) and c_F is the number of isomorphic copies, where removing the labeled vertex from F results in G.
  """

  flags = []
  flags_invariants = []
  counts = []

  new_vertex_id = max(G.nodes) + 1 if len(G.nodes) > 0 else 0

  new_graph_base = G.copy()
  new_graph_base.add_node(new_vertex_id)
  label = {new_vertex_id: 0}
  nx.set_node_attributes(new_graph_base, label, 'label')

  for edges in product([False, True], repeat=len(G.nodes)):
    new_graph = new_graph_base.copy()
    for include, u in zip(edges, G.nodes):
      if include:
        new_graph.add_edge(new_vertex_id, u)

    new_graph_inv = _get_invariants(new_graph)

    for i, inv in enumerate(flags_invariants):
      if check_graph_isomorphism(new_graph, new_graph_inv, flags[i], inv):
        counts[i] += 1
        break
    else:
      flags.append(new_graph)
      flags_invariants.append(new_graph_inv)
      counts.append(1)

  return list(zip(flags, counts))

def vertex_differential(objective, atlas, return_average=True, integer_mode=False):
  """
  Computes the vertex differential operator partial_1(objective) in flag algebras.

  Args:
    objective (list of ('ind'|'sub', networkx.Graph, number)): Linear objective.
    atlas (list of networkx.Graph): Basis graphs.
    return_average (bool): If True, return the averaged result. If False, return raw flag coefficients.
    integer_mode (bool): If True, compute entirely in integer arithmetic.
      Objective coefficients must be (convertible to) integers in this mode.
      Returns (np.ndarray[int64], int) instead of float array.

  Returns:
    When integer_mode=False: np.ndarray of float.
    When integer_mode=True:  (np.ndarray[int64], int) rational array.
  """

  # ------------------------------------------------------------------
  # Step 1. Convert objective to induced-graph form.
  # ------------------------------------------------------------------
  # In float mode: ind_coeffs[i] is a float.
  # In integer mode: ind_coeffs[i] is a (num: int, denom: int) pair.
  ind_graphs = []
  ind_coeffs = []
  ind_graph_invs = []

  for density_type, g, c in objective:
    if density_type == 'ind':
      if integer_mode:
        c_num = int(round(float(c)))
        c_denom = 1
        g_inv = _get_invariants(g)
        for i, inv in enumerate(ind_graph_invs):
          if check_graph_isomorphism(g, g_inv, ind_graphs[i], inv):
            ind_coeffs[i] = _rat_scalar_add(ind_coeffs[i][0], ind_coeffs[i][1], c_num, c_denom)
            break
        else:
          ind_graphs.append(g)
          ind_graph_invs.append(g_inv)
          ind_coeffs.append((c_num, c_denom))
      else:
        g_inv = _get_invariants(g)
        for i, inv in enumerate(ind_graph_invs):
          if check_graph_isomorphism(g, g_inv, ind_graphs[i], inv):
            ind_coeffs[i] += c
            break
        else:
          ind_graphs.append(g)
          ind_graph_invs.append(g_inv)
          ind_coeffs.append(c)
    else:  # 'sub'
      graphs = get_graph_atlas(len(g.nodes))
      if integer_mode:
        c_int = int(round(float(c)))
        subg_num, subg_denom = compute_subgraph_coefficients(g, graphs, integer_mode=True)
        for i, graph in enumerate(graphs):
          if subg_num[i] == 0:
            continue
          add_num = c_int * int(subg_num[i])
          add_denom = subg_denom
          graph_inv = _get_invariants(graph)
          for j, inv in enumerate(ind_graph_invs):
            if check_graph_isomorphism(graph, graph_inv, ind_graphs[j], inv):
              ind_coeffs[j] = _rat_scalar_add(ind_coeffs[j][0], ind_coeffs[j][1], add_num, add_denom)
              break
          else:
            ind_graphs.append(graph)
            ind_graph_invs.append(graph_inv)
            ind_coeffs.append(_rat_scalar_reduce(add_num, add_denom))
      else:
        subg_coeffs = compute_subgraph_coefficients(g, graphs)
        for i, graph in enumerate(graphs):
          if subg_coeffs[i] == 0:
            continue
          graph_inv = _get_invariants(graph)
          for j, inv in enumerate(ind_graph_invs):
            if check_graph_isomorphism(graph, graph_inv, ind_graphs[j], inv):
              ind_coeffs[j] += c * subg_coeffs[i]
              break
          else:
            ind_graphs.append(graph)
            ind_graph_invs.append(graph_inv)
            ind_coeffs.append(c * subg_coeffs[i])

  # ------------------------------------------------------------------
  # Step 2. Apply partial_1: for each induced graph G_i,
  #   partial_1(G_i) = |G_i| * (pi^1(G_i) - mu^1(G_i))
  # ------------------------------------------------------------------
  # In float mode: flag_coeffs[i] is a float.
  # In integer mode: flag_coeffs[i] is a (num: int, denom: int) pair.
  flags = []
  flag_coeffs = []
  flag_invs = []

  for i in range(len(ind_graphs)):
    g = ind_graphs[i]
    n_vertex = len(g.nodes)

    if integer_mode:
      c_num, c_denom = ind_coeffs[i]
      pi_num = c_num * n_vertex
      pi_denom = c_denom
      mu_num = -c_num * n_vertex
      mu_denom = c_denom
    else:
      c_val = ind_coeffs[i]

    pi_terms = adding_vertex_pi(g)
    for H, coeff in pi_terms:
      H_inv = _get_invariants(H)
      if integer_mode:
        for j, inv in enumerate(flag_invs):
          if check_graph_isomorphism(H, H_inv, flags[j], inv):
            flag_coeffs[j] = _rat_scalar_add(flag_coeffs[j][0], flag_coeffs[j][1], pi_num, pi_denom)
            break
        else:
          flags.append(H)
          flag_invs.append(H_inv)
          flag_coeffs.append(_rat_scalar_reduce(pi_num, pi_denom))
      else:
        for j, inv in enumerate(flag_invs):
          if check_graph_isomorphism(H, H_inv, flags[j], inv):
            flag_coeffs[j] += c_val * n_vertex
            break
        else:
          flags.append(H)
          flag_invs.append(H_inv)
          flag_coeffs.append(c_val * n_vertex)

    mu_terms = labeling_vertex_mu(g)
    for H, coeff in mu_terms:
      H_inv = _get_invariants(H)
      if integer_mode:
        for j, inv in enumerate(flag_invs):
          if check_graph_isomorphism(H, H_inv, flags[j], inv):
            flag_coeffs[j] = _rat_scalar_add(flag_coeffs[j][0], flag_coeffs[j][1], mu_num, mu_denom)
            break
        else:
          flags.append(H)
          flag_invs.append(H_inv)
          flag_coeffs.append(_rat_scalar_reduce(mu_num, mu_denom))
      else:
        for j, inv in enumerate(flag_invs):
          if check_graph_isomorphism(H, H_inv, flags[j], inv):
            flag_coeffs[j] -= c_val * n_vertex
            break
        else:
          flags.append(H)
          flag_invs.append(H_inv)
          flag_coeffs.append(-c_val * n_vertex)

  # ------------------------------------------------------------------
  # Step 3. Embed all flags to the same max size, build coeff_res.
  # ------------------------------------------------------------------
  max_vertex = 0
  for flag in flags:
    max_vertex = max(max_vertex, len(flag.nodes))

  partial_atlas = get_partially_labeled_graph_atlas(max_vertex, 1)
  partial_atlas_invs = [_get_invariants(pg) for pg in partial_atlas]

  nm = isomorphism.categorical_node_match('label', -1)

  if integer_mode:
    # Collect all (partial_atlas_idx, add_num, add_denom) contributions
    all_contributions = []

    for i in range(len(flags)):
      flag = flags[i]
      c_num, c_denom = flag_coeffs[i]
      flag_inv = _get_invariants(flag)

      if len(flag.nodes) != max_vertex:
        GM_self = isomorphism.GraphMatcher(flag, flag, node_match=nm)
        aut_F = sum(1 for _ in GM_self.isomorphisms_iter())
        k_label = 1
        normalisation = aut_F * math.comb(max_vertex - k_label, len(flag.nodes) - k_label)

        for j, other in enumerate(partial_atlas):
          count = _count_induced_labeled_subgraph_isos(other, flag, nm, aut_F)
          if count != 0:
            all_contributions.append((j, c_num * count, c_denom * normalisation))
      else:
        for j, inv in enumerate(partial_atlas_invs):
          if check_graph_isomorphism(flag, flag_inv, partial_atlas[j], inv):
            all_contributions.append((j, c_num, c_denom))
            break

    if not all_contributions:
      coeff_res_num = np.zeros(len(partial_atlas), dtype=np.int64)
      coeff_res_denom = 1
    else:
      lcm_denom = _lcm_list([d for _, _, d in all_contributions])
      coeff_res_num = np.zeros(len(partial_atlas), dtype=np.int64)
      for j, add_num, add_denom in all_contributions:
        coeff_res_num[j] += add_num * (lcm_denom // add_denom)
      coeff_res_num, coeff_res_denom = rat_reduce(coeff_res_num, lcm_denom)

  else:
    coeff_res = np.zeros(len(partial_atlas))

    for i in range(len(flags)):
      flag = flags[i]
      c = flag_coeffs[i]
      flag_inv = _get_invariants(flag)

      if len(flag.nodes) != max_vertex:
        GM_self = isomorphism.GraphMatcher(flag, flag, node_match=nm)
        aut_F = sum(1 for _ in GM_self.isomorphisms_iter())
        k_label = 1
        normalisation = aut_F * math.comb(max_vertex - k_label, len(flag.nodes) - k_label)

        for j, other in enumerate(partial_atlas):
          coeff = _count_induced_labeled_subgraph_isos(other, flag, nm, aut_F)
          coeff_res[j] += c * coeff / normalisation
      else:
        for j, inv in enumerate(partial_atlas_invs):
          if check_graph_isomorphism(flag, flag_inv, partial_atlas[j], inv):
            coeff_res[j] += c
            break

  if not return_average:
    if integer_mode:
      return coeff_res_num, coeff_res_denom
    else:
      return coeff_res

  # ------------------------------------------------------------------
  # Step 4. Multiply by the averaged flag-product matrix.
  # ------------------------------------------------------------------
  atlas_vertex_size = len(atlas[0].nodes)

  if integer_mode:
    mat_num, mat_denom = compute_grouped_averaged_flag_product_coefficients_asymmetric(
      atlas, atlas_vertex_size - max_vertex + 1, max_vertex, 1, integer_mode=True)
    # mat_num[0]: shape [len_atlas, len_flags_small, len_partial_atlas_max]
    result_num = mat_num[0].dot(coeff_res_num)
    result_denom = mat_denom * coeff_res_denom
    return rat_reduce(result_num, result_denom)
  else:
    mat = compute_grouped_averaged_flag_product_coefficients_asymmetric(
      atlas, atlas_vertex_size - max_vertex + 1, max_vertex, 1)
    return mat[0].dot(coeff_res)

def labeling_edge_mu(G):
  """
  Computes labeling operator mu^E(G) in flag algebras.

  Args:
    G (networkx.Graph): Input graph

  Returns:
    list of (networkx.Graph, float): List of tuples (F, c_F) where F is G with one edge labeled (up to isomorphism) and c_F is the number of isomorphic copies.
  """

  flags = []
  flags_invariants = []
  counts = []

  for u in G.nodes:
    for v in G.nodes:
      if u == v:
        continue
      if G.has_edge(u, v):
        G_labeled = G.copy()
        nx.set_node_attributes(G_labeled, {u: 0, v: 1}, 'label')

        G_labeled_inv = _get_invariants(G_labeled)
        for i, inv in enumerate(flags_invariants):
          if check_graph_isomorphism(G_labeled, G_labeled_inv, flags[i], inv):
            counts[i] += 1
            break
        else:
          flags.append(G_labeled)
          flags_invariants.append(G_labeled_inv)
          counts.append(1)

  return list(zip(flags, counts))

def labeling_nonedge_mu(G):
  """
  Computes labeling operator mu^{\bar{E}}(G) in flag algebras.

  Args:
    G (networkx.Graph): Input graph

  Returns:
    list of (networkx.Graph, float): List of tuples (F, c_F) where F is G with one non-edge labeled (up to isomorphism) then that edge filled and c_F is the number of isomorphic copies.
  """

  flags = []
  flags_invariants = []
  counts = []

  for u in G.nodes:
    for v in G.nodes:
      if u == v:
        continue
      if not G.has_edge(u, v):
        G_labeled = G.copy()
        G_labeled.add_edge(u, v)
        nx.set_node_attributes(G_labeled, {u: 0, v: 1}, 'label')

        G_labeled_inv = _get_invariants(G_labeled)
        for i, inv in enumerate(flags_invariants):
          if check_graph_isomorphism(G_labeled, G_labeled_inv, flags[i], inv):
            counts[i] += 1
            break
        else:
          flags.append(G_labeled)
          flags_invariants.append(G_labeled_inv)
          counts.append(1)

  return list(zip(flags, counts))

def edge_differential(objective, atlas, return_average=True, integer_mode=False, return_sdp_matrix=False):
  """
  Computes the edge differential operator partial_E(objective) in flag algebras.

  Args:
    objective (list of ('ind'|'sub', networkx.Graph, number)): Linear objective.
    atlas (list of networkx.Graph): Basis graphs.
    return_average (bool): If True, return the averaged result. If False, return raw flag coefficients.
      Ignored when return_sdp_matrix=True.
    integer_mode (bool): If True, compute entirely in integer arithmetic.
      Objective coefficients must be (convertible to) integers in this mode.
      Returns (np.ndarray[int64], int) instead of float array.
    return_sdp_matrix (bool): If True, return the full 3-D averaged flag-product tensor
      mat[1] of shape [len(atlas), m_small, m_large] without contracting with coeff_res.
      This is the tensor needed to build an SDP certificate for the edge differential.
      When integer_mode=False: returns np.ndarray of shape [len(atlas), m_small, m_large].
      When integer_mode=True:  returns (np.ndarray[int64], int) with shape [1, len(atlas), m_small, m_large]
        (the leading axis comes from compute_grouped_averaged_flag_product_coefficients_asymmetric).

  Returns:
    When return_sdp_matrix=False and integer_mode=False: np.ndarray of float.
    When return_sdp_matrix=False and integer_mode=True:  (np.ndarray[int64], int) rational array.
    When return_sdp_matrix=True  and integer_mode=False: np.ndarray of shape [len(atlas), m_small, m_large].
    When return_sdp_matrix=True  and integer_mode=True:  (np.ndarray[int64], int).
  """

  # ------------------------------------------------------------------
  # Step 1. Convert objective to induced-graph form.
  # ------------------------------------------------------------------
  ind_graphs = []
  ind_coeffs = []
  ind_graph_invs = []

  for density_type, g, c in objective:
    if density_type == 'ind':
      if integer_mode:
        c_num = int(round(float(c)))
        c_denom = 1
        g_inv = _get_invariants(g)
        for i, inv in enumerate(ind_graph_invs):
          if check_graph_isomorphism(g, g_inv, ind_graphs[i], inv):
            ind_coeffs[i] = _rat_scalar_add(ind_coeffs[i][0], ind_coeffs[i][1], c_num, c_denom)
            break
        else:
          ind_graphs.append(g)
          ind_graph_invs.append(g_inv)
          ind_coeffs.append((c_num, c_denom))
      else:
        g_inv = _get_invariants(g)
        for i, inv in enumerate(ind_graph_invs):
          if check_graph_isomorphism(g, g_inv, ind_graphs[i], inv):
            ind_coeffs[i] += c
            break
        else:
          ind_graphs.append(g)
          ind_graph_invs.append(g_inv)
          ind_coeffs.append(c)
    else:
      graphs = get_graph_atlas(len(g.nodes))
      if integer_mode:
        c_int = int(round(float(c)))
        subg_num, subg_denom = compute_subgraph_coefficients(g, graphs, integer_mode=True)
        for i, graph in enumerate(graphs):
          if subg_num[i] == 0:
            continue
          add_num = c_int * int(subg_num[i])
          add_denom = subg_denom
          graph_inv = _get_invariants(graph)
          for j, inv in enumerate(ind_graph_invs):
            if check_graph_isomorphism(graph, graph_inv, ind_graphs[j], inv):
              ind_coeffs[j] = _rat_scalar_add(ind_coeffs[j][0], ind_coeffs[j][1], add_num, add_denom)
              break
          else:
            ind_graphs.append(graph)
            ind_graph_invs.append(graph_inv)
            ind_coeffs.append(_rat_scalar_reduce(add_num, add_denom))
      else:
        subg_coeffs = compute_subgraph_coefficients(g, graphs)
        for i, graph in enumerate(graphs):
          if subg_coeffs[i] == 0:
            continue
          graph_inv = _get_invariants(graph)
          for j, inv in enumerate(ind_graph_invs):
            if check_graph_isomorphism(graph, graph_inv, ind_graphs[j], inv):
              ind_coeffs[j] += c * subg_coeffs[i]
              break
          else:
            ind_graphs.append(graph)
            ind_graph_invs.append(graph_inv)
            ind_coeffs.append(c * subg_coeffs[i])

  # ------------------------------------------------------------------
  # Step 2. Apply partial_E: for each induced graph G_i,
  #   partial_E(G_i) = |G_i|(|G_i|-1)/2 * (Fill(mu^Ebar(G_i)) - mu^E(G_i))
  # ------------------------------------------------------------------
  flags = []
  flag_coeffs = []
  flag_invs = []

  for i in range(len(ind_graphs)):
    g = ind_graphs[i]
    n_vertex = len(g.nodes)

    if integer_mode:
      c_num, c_denom = ind_coeffs[i]
      # n_vertex*(n_vertex-1)/2 must be integer (it always is for integer n_vertex)
      factor = n_vertex * (n_vertex - 1) // 2
      base_num = c_num * factor
      base_denom = c_denom
      mu_e_num = -base_num   # mu^E contributes negatively
      mu_ne_num = base_num   # Fill(mu^Ebar) contributes positively
    else:
      c_base = ind_coeffs[i] * (n_vertex * (n_vertex - 1) / 2)

    muE_terms = labeling_edge_mu(g)
    for H, coeff in muE_terms:
      H_inv = _get_invariants(H)
      if integer_mode:
        for j, inv in enumerate(flag_invs):
          if check_graph_isomorphism(H, H_inv, flags[j], inv):
            flag_coeffs[j] = _rat_scalar_add(flag_coeffs[j][0], flag_coeffs[j][1], mu_e_num, base_denom)
            break
        else:
          flags.append(H)
          flag_invs.append(H_inv)
          flag_coeffs.append(_rat_scalar_reduce(mu_e_num, base_denom))
      else:
        for j, inv in enumerate(flag_invs):
          if check_graph_isomorphism(H, H_inv, flags[j], inv):
            flag_coeffs[j] -= c_base
            break
        else:
          flags.append(H)
          flag_invs.append(H_inv)
          flag_coeffs.append(-c_base)

    muNE_terms = labeling_nonedge_mu(g)
    for H, coeff in muNE_terms:
      H_inv = _get_invariants(H)
      if integer_mode:
        for j, inv in enumerate(flag_invs):
          if check_graph_isomorphism(H, H_inv, flags[j], inv):
            flag_coeffs[j] = _rat_scalar_add(flag_coeffs[j][0], flag_coeffs[j][1], mu_ne_num, base_denom)
            break
        else:
          flags.append(H)
          flag_invs.append(H_inv)
          flag_coeffs.append(_rat_scalar_reduce(mu_ne_num, base_denom))
      else:
        for j, inv in enumerate(flag_invs):
          if check_graph_isomorphism(H, H_inv, flags[j], inv):
            flag_coeffs[j] += c_base
            break
        else:
          flags.append(H)
          flag_invs.append(H_inv)
          flag_coeffs.append(c_base)

  # ------------------------------------------------------------------
  # Step 3. Embed all flags to the same max size, build coeff_res.
  #         Only keep E-type flags (labeled edge exists).
  # ------------------------------------------------------------------
  max_vertex = 0
  for flag in flags:
    max_vertex = max(max_vertex, len(flag.nodes))

  partial_atlas_full = get_partially_labeled_graph_atlas(max_vertex, 2)
  partial_atlas = []
  for flag in partial_atlas_full:
    u = [v for v in flag.nodes if flag.nodes[v].get('label', -1) == 0][0]
    v = [v for v in flag.nodes if flag.nodes[v].get('label', -1) == 1][0]
    if flag.has_edge(u, v):
      partial_atlas.append(flag)
  partial_atlas_invs = [_get_invariants(pg) for pg in partial_atlas]

  nm = isomorphism.categorical_node_match('label', -1)

  if integer_mode:
    all_contributions = []

    for i in range(len(flags)):
      flag = flags[i]
      c_num, c_denom = flag_coeffs[i]
      flag_inv = _get_invariants(flag)

      if len(flag.nodes) != max_vertex:
        GM_self = isomorphism.GraphMatcher(flag, flag, node_match=nm)
        aut_F = sum(1 for _ in GM_self.isomorphisms_iter())
        k_label = 2
        normalisation = aut_F * math.comb(max_vertex - k_label, len(flag.nodes) - k_label)

        for j, other in enumerate(partial_atlas):
          count = _count_induced_labeled_subgraph_isos(other, flag, nm, aut_F)
          if count != 0:
            all_contributions.append((j, c_num * count, c_denom * normalisation))
      else:
        for j, inv in enumerate(partial_atlas_invs):
          if check_graph_isomorphism(flag, flag_inv, partial_atlas[j], inv):
            all_contributions.append((j, c_num, c_denom))
            break

    if not all_contributions:
      coeff_res_num = np.zeros(len(partial_atlas), dtype=np.int64)
      coeff_res_denom = 1
    else:
      lcm_denom = _lcm_list([d for _, _, d in all_contributions])
      coeff_res_num = np.zeros(len(partial_atlas), dtype=np.int64)
      for j, add_num, add_denom in all_contributions:
        coeff_res_num[j] += add_num * (lcm_denom // add_denom)
      coeff_res_num, coeff_res_denom = rat_reduce(coeff_res_num, lcm_denom)

  else:
    coeff_res = np.zeros(len(partial_atlas))

    for i in range(len(flags)):
      flag = flags[i]
      c = flag_coeffs[i]
      flag_inv = _get_invariants(flag)

      if len(flag.nodes) != max_vertex:
        GM_self = isomorphism.GraphMatcher(flag, flag, node_match=nm)
        aut_F = sum(1 for _ in GM_self.isomorphisms_iter())
        k_label = 2
        normalisation = aut_F * math.comb(max_vertex - k_label, len(flag.nodes) - k_label)

        for j, other in enumerate(partial_atlas):
          coeff = _count_induced_labeled_subgraph_isos(other, flag, nm, aut_F)
          coeff_res[j] += c * coeff / normalisation
      else:
        for j, inv in enumerate(partial_atlas_invs):
          if check_graph_isomorphism(flag, flag_inv, partial_atlas[j], inv):
            coeff_res[j] += c
            break

  if not return_average and not return_sdp_matrix:
    if integer_mode:
      return coeff_res_num, coeff_res_denom
    else:
      return coeff_res

  # ------------------------------------------------------------------
  # Step 4. Multiply by the averaged flag-product matrix and rescale.
  # ------------------------------------------------------------------
  atlas_vertex_size = len(atlas[0].nodes)
  N = atlas_vertex_size

  if integer_mode:
    mat_num, mat_denom = compute_grouped_averaged_flag_product_coefficients_asymmetric(
      atlas, atlas_vertex_size - max_vertex + 2, max_vertex, 2, integer_mode=True)
    # mat_num[1]: shape [len_atlas, len_flags_small, len_partial_atlas_max]
    # Use type index 1 (E-type).
    if return_sdp_matrix:
      # Return the raw 3-D tensor without contracting with coeff_res.
      # Shape: [len(atlas), m_small, m_large] (after stripping the leading type axis).
      return rat_reduce(mat_num[1], mat_denom)
    result_num = mat_num[1].dot(coeff_res_num)
    result_denom = mat_denom * coeff_res_denom
    return rat_reduce(result_num, result_denom)

  else:
    mat = compute_grouped_averaged_flag_product_coefficients_asymmetric(
      atlas, atlas_vertex_size - max_vertex + 2, max_vertex, 2)
    if return_sdp_matrix:
      # Return the raw 3-D tensor without contracting with coeff_res.
      # Shape: [len(atlas), m_small, m_large].
      return mat[1]
    result = mat[1].dot(coeff_res)
    return result
