// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#include "ocuda-fec/phy/upper/channel_coding/ldpc_decoder_cuda_backend.h"
#include "ldpc_graph_impl.h"

using namespace ocudu;

cuda_ldpc_decoder_backend::cuda_ldpc_decoder_backend()

{
  // Full base graph descriptions.
  std::array<cuda::base_graph_description, TOTAL_NOF_GRAPHS> base_graph_descriptions = {};
  for (unsigned i_base_graph = 0; i_base_graph != TOTAL_NOF_GRAPHS; ++i_base_graph) {
    // Select LDPC base graph.
    const ldpc_graph_impl& base_graph = ldpc::graph_array[i_base_graph];

    // Get number of check nodes.
    unsigned nof_check_nodes = base_graph.get_nof_BG_check_nodes();

    // Select the flat base graph description.
    cuda::base_graph_description& bg_info = base_graph_descriptions[i_base_graph];
    bg_info.nof_check_nodes               = nof_check_nodes;

    // Iterate each of the check nodes.
    for (unsigned check_node = 0; check_node != nof_check_nodes; ++check_node) {
      // Retrieve list of variable nodes connected to this check node.
      const ldpc::BG_adjacency_row_t& current_var_indices = base_graph.get_adjacency_row(check_node);

      // Find first NO_EDGE in current_var_indices.
      const auto* this_var_index_end = std::find(current_var_indices.begin(), current_var_indices.end(), ldpc::NO_EDGE);

      cuda::base_graph_check_node_info& bg_check_node_info = bg_info.check_nodes_info[check_node];

      bg_check_node_info = {};
      std::for_each(current_var_indices.cbegin(),
                    this_var_index_end,
                    [&base_graph, &bg_check_node_info, check_node](uint16_t element) {
                      unsigned idx                           = bg_check_node_info.size++;
                      bg_check_node_info.bg_check_edges[idx] = element;
                      bg_check_node_info.shifts[idx]         = base_graph.get_lifted_node(check_node, element);
                    });
    }
  }

  // Copy all LDPC base graph descriptions to the GPU.
  d_bg_info.copy_to_device(base_graph_descriptions);
}
