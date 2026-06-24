// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#pragma once

#include <cstdint>

namespace ocudu {
namespace cuda {

struct base_graph_check_node_info {
  uint16_t bg_check_edges[20];
  uint16_t shifts[20];
  uint16_t size;
};

struct base_graph_description {
  base_graph_check_node_info check_nodes_info[46];
  unsigned                   nof_check_nodes;
};

} // namespace cuda
} // namespace ocudu
