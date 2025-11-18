import cvxpy as cp
import numpy as np
from . import flag_algebra as _fa

def build_problem(
  objectives,
  constraints,
  sdp_configs,
  lowerbound=True,
  atlas=None,
):
  """
  Defines the problem and builds the SDP.

  objectives:
    Tuple (term_type, graph, coefficient, hom_coefficients (optional))
    term_type : hom, ind
    graph : networkx.Graph
    coefficient : float
    hom_coefficients : np.array
      (Result of compute_hom_coefficients(graph, atlas))
  
  constraints:
    List of tuples (term_type, graph, target, hom_coefficients (optional))
    term_type : hom, ind
    graph : networkx.Graph
    target : float
    hom_coefficients : np.array (only for hom term_type)
      (Result of compute_hom_coefficients(graph, atlas))

  sdp_configs:
    List of tuple (n, k, matrix_coefficients (optional))
    n : int
    k : int
    matrix_coefficients : np.array
      3D array of size [len(atlas), len(partial_atlas_n_k), len(partial_atlas_n_k)]
      (Result of compute_averaged_flag_product_coefficients(atlas, n, k))
  """

  g_sizes = set()
  for sdp_config in sdp_configs:
    n, k = sdp_config[0], sdp_config[1]
    g_sizes.add(2 * n - k)
  
  if len(g_sizes) != 1:
    raise ValueError("All SDPs must correspond to the same graph size.")
  
  g_size = g_sizes.pop()
  
  if atlas is None:
    atlas = _fa.get_graph_atlas(g_size)
  else:
    for g in atlas:
      if len(g.nodes) != g_size:
        raise ValueError("All graphs in the atlas must have the same number of nodes as required by the SDP configurations.")
  
  variable_dict = {}

  t = cp.Variable() # Objective variable
  variable_dict['t'] = t

  objective_terms = []
  constraint_terms = []
  sdp_terms = []

  for objective in objectives:
    if len(objective) == 3:
      term_type, H, coefficient = objective
      if term_type == 'hom':
        hom_coefficients = _fa.compute_hom_coefficients(H, atlas)
      else:
        hom_coefficients = _fa.compute_ind_hom_coefficients(H, atlas)
      objective_terms.append(hom_coefficients * coefficient)
    else:
      objective_terms.append(objective[2] * objective[3]) # hom_coefficients
  
  for i, constraint in enumerate(constraints):
    if len(constraint) == 3:
      term_type, H, target = constraint
      if term_type == 'hom':
        hom_coefficients = _fa.compute_hom_coefficients(H, atlas)
      else:
        hom_coefficients = _fa.compute_ind_hom_coefficients(H, atlas)
    else:
      term_type, H, target, hom_coefficients = constraint # hom_coefficients

    x = cp.Variable() # Constraint variable
    constraint_terms.append(x * (hom_coefficients - target))
    variable_dict[f'constraint_{i}'] = x

  for i, sdp_config in enumerate(sdp_configs):
    if len(sdp_config) == 2:
      n, k = sdp_config
      matrix_coefficients = _fa.compute_averaged_flag_product_coefficients(atlas, n, k)
    else:
      n, k, matrix_coefficients = sdp_config

    x_nk = cp.Variable((matrix_coefficients.shape[1], matrix_coefficients.shape[2]), PSD=True)
    variable_dict[f'sdp_{n}_{k}'] = x_nk

    sdp_terms.append((matrix_coefficients, x_nk))

  objectives = cp.sum(objective_terms)
  constraints = cp.sum(constraint_terms) if len(constraint_terms) > 0 else None

  final_constraints = []
  
  for i in range(len(atlas)):
    const_obj = objectives[i] if constraints is None else objectives[i] + constraints[i]
    for sdp_term in sdp_terms:
      if lowerbound:
        const_obj += -cp.sum(cp.multiply(sdp_term[0][i, :, :], sdp_term[1]), axis=(0, 1))
      else:
        const_obj += cp.sum(cp.multiply(sdp_term[0][i, :, :], sdp_term[1]), axis=(0, 1))
    
    if lowerbound:
      final_constraints.append(const_obj >= t)
    else:
      final_constraints.append(const_obj <= t)

  if lowerbound:
    problem = cp.Problem(cp.Maximize(t), final_constraints)
  else:
    problem = cp.Problem(cp.Minimize(t), final_constraints)
  
  return problem, variable_dict