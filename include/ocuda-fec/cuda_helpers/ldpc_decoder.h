// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.
#pragma once

#include "ocudu/adt/span.h"
#include <cinttypes>
#include <memory>

namespace ocudu::cuda {

struct base_graph_check_node_info {
  uint16_t bg_check_edges[20];
  uint16_t shifts[20];
  uint16_t size;
};

struct base_graph_description {
  base_graph_check_node_info check_nodes_info[46];
  unsigned                   nof_check_nodes;
};

struct ldpc_decoder_cb_configuration {
  int8_t*  soft_bits;
  int8_t*  check_to_var;
  unsigned bg_info_pos;
  unsigned message_size_bytes;
  unsigned nof_llr;
  unsigned nof_layers;
  unsigned lifting_size;
  unsigned bg_N_high_rate;
  unsigned bg_N_full;
  unsigned max_nof_iterations;
};

struct ldpc_decoder_cb_arguments {
  uint8_t*                      output;
  ldpc_decoder_cb_configuration cb_configuration;
};

class ldpc_decoder
{
public:
  static constexpr unsigned max_nof_codeblocks = 32;

  virtual ~ldpc_decoder() = default;

  virtual void ldpc_decode(span<const ldpc_decoder_cb_arguments> codeblocks, cuda_stream& stream) = 0;

  static std::unique_ptr<ldpc_decoder> create(span<const base_graph_description> base_graph_descriptions);
};

} // namespace ocudu::cuda
