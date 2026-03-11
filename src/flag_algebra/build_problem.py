import cvxpy as cp
import numpy as np
from . import flag_algebra as _fa

def build_problem(
  objectives,
  constraints,
  sdp_configs,
  lowerbound=True,
  use_vertex_differential=False,
  use_edge_differential=False,
  atlas=None,
):
  """
  Defines the problem and builds the SDP.

  objectives:
    Tuple (term_type, graph, coefficient, subg_coefficients (optional))
    term_type : sub, ind
    graph : networkx.Graph
    coefficient : float
    subg_coefficients : np.array
      (Result of compute_subgraph_coefficients(graph, atlas))
  
  constraints:
    List of tuples (term_type, graph, target, subg_coefficients (optional))
    term_type : sub, ind
    graph : networkx.Graph
    target : float
    subg_coefficients : np.array
      (Result of compute_subgraph_coefficients(graph, atlas))

  sdp_configs:
    List of tuple (n, k, matrix_coefficients (optional))
    n : int
    k : int
    matrix_coefficients : np.array
      3D array of size [len(atlas), len(partial_atlas_n_k), len(partial_atlas_n_k)]
      (Result of compute_averaged_flag_product_coefficients(atlas, n, k))

  lowerbound: bool
    Whether to build a lower bound SDP (True) or upper bound SDP (False).

  use_vertex_differential: bool
    Whether to include the vertex differential in the SDP.
  
  use_edge_differential: bool
    Whether to include the edge differential in the SDP.

  atlas: list of networkx.Graph
    If None, the graph atlas for the required size will be generated.
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

  if len(constraints) != 0 and use_vertex_differential:
    raise NotImplementedError("Vertex differential does not support constraints.")

  for objective in objectives:
    if len(objective) == 3:
      term_type, H, coefficient = objective
      if term_type == 'sub':
        subg_coefficients = _fa.compute_subgraph_coefficients(H, atlas)
      else:
        subg_coefficients = _fa.compute_ind_subgraph_coefficients(H, atlas)
      objective_terms.append(subg_coefficients * coefficient)
    else:
      objective_terms.append(objective[2] * objective[3]) # subg_coefficients
  
  for i, constraint in enumerate(constraints):
    if len(constraint) == 3:
      term_type, H, target = constraint
      if term_type == 'sub':
        subg_coefficients = _fa.compute_subgraph_coefficients(H, atlas)
      else:
        subg_coefficients = _fa.compute_ind_subgraph_coefficients(H, atlas)
    else:
      term_type, H, target, subg_coefficients = constraint # subg_coefficients

    x = cp.Variable() # Constraint variable
    constraint_terms.append(x * (subg_coefficients - target))
    variable_dict[f'constraint_{i}'] = x

  for i, sdp_config in enumerate(sdp_configs):
    if len(sdp_config) == 2:
      n, k = sdp_config
      matrix_coefficients = _fa.compute_grouped_averaged_flag_product_coefficients(atlas, n, k)
    else:
      n, k, matrix_coefficients = sdp_config

    for i in range(matrix_coefficients.shape[0]):
      x_nk = cp.Variable((matrix_coefficients[i].shape[1], matrix_coefficients[i].shape[2]), PSD=True)
      variable_dict[f'sdp_{n}_{k}_{i}'] = x_nk

      sdp_terms.append((matrix_coefficients[i], x_nk))

  if use_vertex_differential is not False:
    if use_vertex_differential is True:
      derivative_mat = _fa.vertex_differential([(t[0], t[1], t[2]) for t in objectives], atlas)
    else:
      derivative_mat = use_vertex_differential
    derivative_variable = cp.Variable(derivative_mat.shape[1])
    variable_dict['vertex_differential'] = derivative_variable

  if use_edge_differential is not False:
    if use_edge_differential is True:
      derivative_mat = _fa.edge_differential([(t[0], t[1], t[2]) for t in objectives], atlas)
    else:
      derivative_mat = use_edge_differential
    derivative_variable = cp.Variable(derivative_mat.shape[1], nonneg=True)
    variable_dict['edge_differential'] = derivative_variable

  objective_sum = cp.sum(objective_terms)
  constraint_sum = cp.sum(constraint_terms) if len(constraint_terms) > 0 else None

  final_constraints = []
  
  for i in range(len(atlas)):
    const_obj = objective_sum[i] if constraint_sum is None else objective_sum[i] + constraint_sum[i]
    for sdp_term in sdp_terms:
      if lowerbound:
        const_obj += -cp.sum(cp.multiply(sdp_term[0][i, :, :], sdp_term[1]))
      else:
        const_obj += cp.sum(cp.multiply(sdp_term[0][i, :, :], sdp_term[1]))
    
    if use_vertex_differential is not False:
      const_obj += derivative_mat[i, :] @ derivative_variable
    if use_edge_differential is not False:
      const_obj += -derivative_mat[i, :] @ derivative_variable
    
    if lowerbound:
      final_constraints.append(const_obj >= t)
    else:
      final_constraints.append(const_obj <= t)

  if lowerbound:
    problem = cp.Problem(cp.Maximize(t), final_constraints)
  else:
    problem = cp.Problem(cp.Minimize(t), final_constraints)
  
  return problem, variable_dict