import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
from itertools import permutations, combinations
import math
import tqdm
import subprocess
import shutil
import sys
import io
import warnings

ATLAS = nx.graph_atlas_g() 

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

def get_graph_atlas(n):
  """
  Returns a list of all graphs with n vertices from the NetworkX graph atlas.

  Args: 
    n (int): Number of vertices
  Returns:
    List of networkx.Graph objects with n vertices
  """
  if n <= 7:
    return [G for G in ATLAS if len(G) == n]
  else:
    # Warning if n >= 10
    if n >= 10:
      warnings.warn("n>=10 requires at least 10 GB of memory, and may result out of memory error.")
    
    if not shutil.which('geng'):
      raise EnvironmentError("The 'geng' command in nauty suite is required for n > 7, but not found.")
    
    cmd = ['geng', '-q', str(n)]

  
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

def compute_hom_coefficient(H, F):
  """
  Computes the homomorphism coefficient c_F^OPT for graphs H and F
  This function allows to translate the (non-induced) density t(H, G) into a linear combination of homomorphism densities t(ind F, G) over F in the atlas.

  Args:
    H (networkx.Graph): Target graph
    F (networkx.Graph): Source graph
  Returns:
    int: Number of (non-induced) copies of H in F
  """
  matcher = isomorphism.GraphMatcher(F, H)
  num_monomorphisms = sum(1 for _ in matcher.subgraph_monomorphisms_iter())

  return num_monomorphisms / math.perm(len(F.nodes), len(H.nodes))

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
    c_g_OPT = compute_hom_coefficient(H, g)
    res.append(c_g_OPT)
  return np.array(res)

def compute_ind_hom_coefficient(H, F):
  """
  Computes the induced homomorphism coefficient c_F^IND for graphs H and F
  This function allows to translate the induced density t_ind(H, G) into a linear combination of homomorphism densities t(ind F, G) over F in the atlas.

  Args:
    H (networkx.Graph): Target graph
    F (networkx.Graph): Source graph
  Returns:
    int: Number of induced copies of H in F
  """
  matcher = isomorphism.GraphMatcher(F, H)
  num_isomorphisms = sum(1 for _ in matcher.subgraph_isomorphisms_iter())

  return num_isomorphisms / math.perm(len(F.nodes), len(H.nodes))

def compute_ind_hom_coefficients(H, atlas=None):
  """
  Computes the induced homomorphism coefficients c_g^IND for a given graph H over an atlas of graphs.

  Args:
    H (networkx.Graph): Target graph
    atlas (list of networkx.Graph, optional): List of graphs to compute coefficients for. If None, uses the graph atlas for graphs with the same number of nodes as H.
  Returns:
    np.array: Array of induced homomorphism coefficients c_g^IND for each graph g in the atlas such that 
              t_ind(H, G) ≃ sum_g c_g^IND * t(ind g, G)
  """
  if atlas is None:
    atlas = get_graph_atlas(len(H.nodes))
  
  res = []
  for g in atlas:
    c_g_IND = compute_ind_hom_coefficient(H, g)
    res.append(c_g_IND)
  return np.array(res)

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

def compute_averaged_flag_product_coefficients(atlas, n, k, verbose=False):
  """
  Computes the averaged flag product coefficients for a given atlas.

  Args:
    atlas (list of networkx.Graph): List of graphs to compute coefficients for (Should be graphs with 2*n-k vertices)
    n (int): Number of vertices in the graphs
    k (int): Number of labeled vertices
    verbose (bool): Whether to print progress information

  Returns:
    np.array: 4D array A_results of size [1, len(atlas), len(partial_atlas_n_k), len(partial_atlas_n_k)]
                where partial_atlas_n_k is the list of partially labeled graphs with n vertices and k labeled vertices.
              
              A_results[F_idx, G1_idx, G2_idx] = probability of following event:
                1. Take a random labeling of k vertices of F {v1, ..., vk}
                2. Split the remaining 2n-2k vertices into two equal parts randomly U1, U2
                3. F[U1 + {v1, ..., vk}] is isomorphic to G1 and F[U2 + {v1, ..., vk}] is isomorphic to G2
              
              In terms of Flag Algebra, 
                sum_{F with v(F) ≃ 2n-k} A_results[F_idx, G1_idx, G2_idx] = [G1 * G2]_k
  """
  partial_atlas_n2_k = get_partially_labeled_graph_atlas(n, k)

  # --- Optimisation 1: Pre-calculate invariants for the target atlas ---
  atlas_invariants = [_get_invariants(pg) for pg in partial_atlas_n2_k]
  
  A_results = np.zeros((len(atlas), len(partial_atlas_n2_k), len(partial_atlas_n2_k)))

  pbar = atlas if not verbose else tqdm.tqdm(atlas)

  nm = isomorphism.categorical_node_match('label', -1)

  for atlas_idx, g in enumerate(pbar):
    labeled_gs_data = [] 
    labeled_count = {}

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
      labeled_vertices = [
        v for v in labeled_g.nodes if 'label' in labeled_g.nodes[v]
      ]
      m = len(unlabeled_vertices)
      base_labeled_nodes = list(labeled_vertices)

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

        res[idx1, idx2] += labeled_count[label_idx] / total_labeled / math.comb(m, m // 2)
    
    A_results[atlas_idx] = res
  
  return A_results[np.newaxis, :, :, :] # Add a dummy first dimension for compatibility

def compute_grouped_averaged_flag_product_coefficients(atlas, n, k, verbose=False):
  """
  Computes the averaged flag product coefficients for grouped partially labeled graphs.

  Args:
    n (int): Number of vertices in the graphs
    k (int): Number of labeled vertices 
    verbose (bool): Whether to print progress information

  Returns:
    np.array of size 
    [len(k-types), len(atlas), len(partial_atlas_n_k_i), len(partial_atlas_n_k_i)]
    where
    results[type_idx, F_idx, G1_idx, G2_idx] = probability of following event:
      1. Randomly label k vertices of F {v1, ..., vk}
      2. The labeled subgraph is isomorphic to type type_idx
      3. Split the remaining 2n-2k vertices into two equal parts randomly U1, U2
      4. F[U1 + {v1, ..., vk}] is isomorphic to G1 and F[U2 + {v1, ..., vk}] is isomorphic to G2

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
    # Take subgraph induced by labeled vertices
    labeled_nodes = [v for v in flag.nodes if 'label' in flag.nodes[v]]
    flag_type = flag.subgraph(labeled_nodes)

    for t_index, ty in enumerate(types):
      is_type = graph_equal(flag_type, ty, label_name='label')
      
      if is_type:
        flag_indices[t_index].append(f_index)
        break
   
  results = [np.zeros((len(atlas), len(flag_indices[type_idx]), len(flag_indices[type_idx]))) for type_idx in range(len(types))]

  pbar = atlas if not verbose else tqdm.tqdm(atlas)

  nm = isomorphism.categorical_node_match('label', -1)

  # Enumerate over all graphs in the atlas
  for g_idx, g in enumerate(pbar):
    labeled_gs_data = [[] for _ in types]
    labeled_count = [{} for _ in types]
    
    # Merge the labeled graphs if isomorphic
    for labeled_vertices in permutations(g.nodes, k):
      g_copy = g.copy()
      # First, check if the labeled subgraph matches any type strictly
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
    
    # For each type,
    for type_idx in range(len(types)):
      
      for label_idx, (labeled_g, _) in enumerate(labeled_gs_data[type_idx]):
        unlabeled_vertices = [
          v for v in labeled_g.nodes if 'label' not in labeled_g.nodes[v]
        ]
        labeled_vertices = [
          v for v in labeled_g.nodes if 'label' in labeled_g.nodes[v]
        ]

        m = len(unlabeled_vertices)
        base_labeled_nodes = list(labeled_vertices)

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

          results[type_idx][g_idx, idx1, idx2] += labeled_count[type_idx][label_idx] / math.perm(2 * n - k, k) / math.comb(m, m // 2)
  
  return np.stack(results, axis=0)