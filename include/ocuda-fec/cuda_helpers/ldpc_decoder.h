// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#pragma once

#include "ocuda-fec/cuda_helpers/base_graph_description.h"
#include "ocudu/adt/span.h"
#include <cinttypes>
#include <memory>

namespace ocudu::cuda {

struct ldpc_decoder_cb_configuration {
  int8_t*  input;
  unsigned input_size;
  int8_t*  soft_bits;
  int8_t*  check_to_var;
  int8_t*  rm_buffer;
  unsigned bg_info_pos;
  unsigned message_size_bytes;
  unsigned nof_layers;
  unsigned lifting_size;
  unsigned bg_N_high_rate;
  unsigned bg_N_full;
  unsigned max_nof_iterations;
  unsigned nof_systematic_bits;
  unsigned nof_filler_bits;
  unsigned rm_buffer_size;
  unsigned shift_k0;
  unsigned modulation_order;
  bool     new_data;
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
