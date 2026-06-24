// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#pragma once

#include "ocuda-fec/cuda_helpers/base_graph_description.h"
#include "ocuda-fec/cuda_helpers/cuda_stream.h"
#include "ocuda-fec/cuda_helpers/device_vector.h"
#include "ocuda-fec/cuda_helpers/ldpc_decoder.h"

namespace ocudu::cuda {

class ldpc_decoder_impl : public ldpc_decoder
{
public:
  ldpc_decoder_impl(span<const base_graph_description> base_graph_descriptions_);

  void ldpc_decode(span<const ldpc_decoder_cb_arguments> codeblocks, cuda_stream& stream) override;

private:
  device_vector<ldpc_decoder_cb_arguments, max_nof_codeblocks> d_codeblocks;
  span<const base_graph_description>                           base_graph_descriptions;
};

} // namespace ocudu::cuda
