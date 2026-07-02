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

  use_edge_differential: bool | np.ndarray
    Controls inclusion of an edge-type certificate in the SDP.

    - False (default): no edge certificate.
    - True: automatically compute the E-type SDP certificate from the objectives.
      Uses the full 3-D averaged flag-product tensor M[i, j, l] of shape
      [len(atlas), m_small, m_large] (edge type, k=2).  When m_small == m_large
      a symmetric PSD matrix variable Q is used; otherwise a non-negative matrix
      variable Q is used.  The contribution to each atlas-graph constraint is
        cp.sum(cp.multiply(M[i], Q))
      which is always non-negative, giving a strictly richer certificate than
      the legacy nonneg-vector approach.
    - 2-D np.ndarray of shape [len(atlas), m]: legacy behaviour — the pre-computed
      contracted matrix is used with a non-negative vector variable of length m.
    - 3-D np.ndarray of shape [len(atlas), m_small, m_large]: new SDP behaviour
      using the supplied tensor directly (useful when the tensor is pre-computed
      outside build_problem).

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
      vertex_deriv_mat = _fa.vertex_differential([(t[0], t[1], t[2]) for t in objectives], atlas)
    else:
      vertex_deriv_mat = use_vertex_differential
    if lowerbound:
      vertex_deriv_variable = cp.Variable(vertex_deriv_mat.shape[1])
    else:
      vertex_deriv_variable = cp.Variable(vertex_deriv_mat.shape[1], nonneg=True)
    variable_dict['vertex_differential'] = vertex_deriv_variable

  # edge_sdp_tensor: 3-D tensor for the SDP approach (shape [N_atlas, m_small, m_large])
  # edge_deriv_mat: 2-D contracted matrix for the legacy nonneg-vector approach
  edge_sdp_tensor = None
  edge_deriv_mat = None

  if use_edge_differential is not False:
    if use_edge_differential is True:
      # New SDP approach: obtain the full 3-D E-type flag-product tensor.
      edge_sdp_tensor = _fa.edge_differential(
        [(t[0], t[1], t[2]) for t in objectives], atlas, return_sdp_matrix=True)
    elif isinstance(use_edge_differential, np.ndarray) and use_edge_differential.ndim == 3:
      # Pre-computed 3-D tensor supplied directly.
      edge_sdp_tensor = use_edge_differential
    else:
      # Legacy: pre-computed 2-D contracted matrix → nonneg vector variable.
      edge_deriv_mat = use_edge_differential

    if edge_sdp_tensor is not None:
      m_small, m_large = edge_sdp_tensor.shape[1], edge_sdp_tensor.shape[2]
      if m_small == m_large:
        # Symmetric case: use a PSD matrix variable (as in vanilla flag algebra).
        edge_variable = cp.Variable((m_small, m_large), PSD=True)
      else:
        # Asymmetric case: use an entrywise non-negative matrix variable.
        # This is strictly more general than the legacy nonneg-vector approach.
        edge_variable = cp.Variable((m_small, m_large), nonneg=True)
      variable_dict['edge_differential'] = edge_variable
    else:
      edge_variable = cp.Variable(edge_deriv_mat.shape[1], nonneg=True)
      variable_dict['edge_differential'] = edge_variable

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
      if lowerbound:
        const_obj += vertex_deriv_mat[i, :] @ vertex_deriv_variable
      else:
        const_obj -= vertex_deriv_mat[i, :] @ vertex_deriv_variable
    if use_edge_differential is not False:
      if edge_sdp_tensor is not None:
        # SDP approach: inner product of the 3-D tensor slice with the matrix variable.
        if lowerbound:
          const_obj += cp.sum(cp.multiply(edge_sdp_tensor[i], edge_variable))
        else:
          const_obj -= cp.sum(cp.multiply(edge_sdp_tensor[i], edge_variable))
      else:
        # Legacy nonneg-vector approach.
        if lowerbound:
          const_obj += edge_deriv_mat[i, :] @ edge_variable
        else:
          const_obj -= edge_deriv_mat[i, :] @ edge_variable
    
    if lowerbound:
      final_constraints.append(const_obj >= t)
    else:
      final_constraints.append(const_obj <= t)

  if lowerbound:
    problem = cp.Problem(cp.Maximize(t), final_constraints)
  else:
    problem = cp.Problem(cp.Minimize(t), final_constraints)
  
  return problem, variable_dict