import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
from itertools import permutations, combinations
import math
import tqdm

ATLAS = nx.graph_atlas_g() 

def get_graph_atlas(n):
  """
  Returns a list of all graphs with n vertices from the NetworkX graph atlas.
  For n > 7, uses igraph to generate the graphs.

  Args: 
    n (int): Number of vertices
  Returns:
    List of networkx.Graph objects with n vertices
  """
  if n <= 7:
    return [G for G in ATLAS if len(G) == n]
  else:
    raise NotImplementedError("Graph atlas for n > 7 is not implemented yet.")
  
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
      
      for other in labeled_g_list:
        # Check isomorphism considering labels
        nm = isomorphism.categorical_node_match('label', -1)
        GM = isomorphism.GraphMatcher(g_labeled, other, node_match=nm)
        if GM.is_isomorphic():
          break
      else:
        labeled_g_list.append(g_labeled)
  
    res.extend(labeled_g_list)
  
  return res

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
  if verbose:
    print("Pre-calculating atlas invariants...")
  atlas_invariants = [_get_invariants(pg) for pg in partial_atlas_n2_k]
  if verbose:
    print("...done.")

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
        if g_labeled_inv == other_inv:
          GM = isomorphism.GraphMatcher(g_labeled, other_g, node_match=nm)
          if GM.is_isomorphic():
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
            GM1 = isomorphism.GraphMatcher(g1, pg, node_match=nm)
            if GM1.is_isomorphic():
              idx1 = i
          
          if idx2 is None and inv2 == pg_inv:
            pg = partial_atlas_n2_k[i]
            GM2 = isomorphism.GraphMatcher(g2, pg, node_match=nm)
            if GM2.is_isomorphic():
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

  type_copies = []
  # Compute the number of automorphisms of the type without labels
  for t in types:
    GM = isomorphism.GraphMatcher(t, t)
    num_aut = sum(1 for _ in GM.isomorphisms_iter())
    type_copies.append(num_aut)
  
  # Setup the default label for each type
  for type in types:
    labels = {
      v: i for i, v in enumerate(type.nodes)
    }
    nx.set_node_attributes(type, labels, 'label')

  # Store the invariant of types for future comparison
  type_invariants = [_get_invariants(t) for t in types]
  
  # For each flag, determine its type and store the index
  type_indices = [[] for _ in range(len(types))]
  for flag_idx, flag in enumerate(flags):
    labeled_vertices = [v for v in flag.nodes if 'label' in flag.nodes[v]]
    type_subgraph = flag.subgraph(labeled_vertices).copy() 
    type_inv = _get_invariants(type_subgraph)

    for type_idx, t_inv in enumerate(type_invariants):
      if type_inv == t_inv:
        nm = isomorphism.categorical_node_match('label', -1)
        GM = isomorphism.GraphMatcher(type_subgraph, types[type_idx], node_match=nm)
        if GM.is_isomorphic():
          type_indices[type_idx].append(flag_idx)
          break
    # Note: this removes flags that do not correspond to any type, due to labeling.
    #       This will be handled later, by multiplying with the appropriate factor.
  
  # Predetermine the invariants for each flag
  flag_invs = [[_get_invariants(flags[flag_idx]) for flag_idx in type_idx_list] for type_idx_list in type_indices]
  
  # This will be the final result
  results = [np.zeros((len(atlas), len(type_indices[i]), len(type_indices[i]))) for i in range(len(types))]
  for graph_index, graph in enumerate(atlas):
    
    labeled_gs_data = [] 
    labeled_count = {}

    for labeled_vertices in permutations(graph.nodes, k):
      g_labeled = graph.copy()
      label_dict = {
        v: i for i, v in enumerate(labeled_vertices)
      }

      # Before checking isomorphisms,...
      labeled_subgraph = g_labeled.subgraph(labeled_vertices).copy()

      labeled_subgraph_inv = _get_invariants(labeled_subgraph)
      # Iterate the types to see if the labeled subgraph matches any type
      subgraph_type_idx = -1
      for i, type_inv in enumerate(type_invariants):
        if labeled_subgraph_inv == type_inv:
          GM = isomorphism.GraphMatcher(labeled_subgraph, types[i])
          if GM.is_isomorphic():
            subgraph_type_idx = i
            break
      if subgraph_type_idx == -1:
        print("Warning: Labeled subgraph does not match any type. Skipping...")
        continue
      
      g_labeled_inv = _get_invariants(g_labeled)

      found_match = False
      for i, (other_g, other_inv) in enumerate(labeled_gs_data):
        if g_labeled_inv == other_inv:
          GM = isomorphism.GraphMatcher(g_labeled, other_g, node_match=nm)
          if GM.is_isomorphic():
            labeled_count[i] += math.factorial(k) // type_copies[subgraph_type_idx]
            found_match = True
            break
      
      if not found_match:
        new_idx = len(labeled_gs_data)
        labeled_gs_data.append((g_labeled, g_labeled_inv))
        labeled_count[new_idx] = math.factorial(k) // type_copies[subgraph_type_idx]
    
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

        for type_idx, type_idx_list in enumerate(type_indices):
          for local_idx, flag_idx in enumerate(type_idx_list):
            flag = flags[flag_idx]
            flag_inv = flag_invs[type_idx][local_idx]
            
            if idx1 is None and inv1 == flag_inv:
              nm = isomorphism.categorical_node_match('label', -1)
              GM1 = isomorphism.GraphMatcher(g1, flag, node_match=nm)
              if GM1.is_isomorphic():
                idx1 = local_idx
            
            if idx2 is None and inv2 == flag_inv:
              nm = isomorphism.categorical_node_match('label', -1)
              GM2 = isomorphism.GraphMatcher(g2, flag, node_match=nm)
              if GM2.is_isomorphic():
                idx2 = local_idx
            
            if idx1 is not None and idx2 is not None:
              break
          
          if idx1 is not None and idx2 is not None:
            break

        assert idx1 is not None, f"Failed to find index in partial atlas for g1: {g1.nodes(data=True)} {g1.edges()}"
        assert idx2 is not None, f"Failed to find index in partial atlas for g2: {g2.nodes(data=True)} {g2.edges()}"

        results[type_idx][graph_index, idx1, idx2] += labeled_count[label_idx] / total_labeled / math.comb(m, m // 2)

  return np.stack(results, axis=0)