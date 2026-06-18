// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#include "pusch_codeblock_cuda_decoder.h"

using namespace ocudu;

void pusch_codeblock_cuda_decoder::rate_match(span<log_likelihood_ratio>       rm_buffer,
                                              span<const log_likelihood_ratio> cb_llrs,
                                              bool                             new_data,
                                              const codeblock_metadata&        metadata)
{
  dematcher->rate_dematch(rm_buffer, cb_llrs, new_data, metadata);
}

void pusch_codeblock_cuda_decoder::decode(bit_buffer                                   cb_data_,
                                          span<log_likelihood_ratio>                   rm_buffer,
                                          span<const log_likelihood_ratio>             cb_llrs,
                                          bool                                         new_data,
                                          ocudu::crc_generator_poly                    crc_poly,
                                          unsigned                                     nof_ldpc_iterations,
                                          const codeblock_metadata&                    metadata,
                                          pusch_codeblock_cuda_decoder_callback_func&& callback_)
{
  callback = std::move(callback_);
  cb_data  = cb_data_;

  rate_match(rm_buffer, cb_llrs, new_data, metadata);

  // Prepare LDPC decoder configuration.
  ldpc_decoder_cuda::configuration decoder_config = {
      .base_graph      = metadata.tb_common.base_graph,
      .lifting_size    = metadata.tb_common.lifting_size,
      .nof_filler_bits = metadata.cb_specific.nof_filler_bits,
      .max_iterations  = nof_ldpc_iterations,
  };

  // Decode with early stop.
  decoder->decode(
      cb_data, rm_buffer, decoder_config, [this, crc_poly, nof_filler_bits = metadata.cb_specific.nof_filler_bits]() {
        // Select CRC calculator.
        crc_calculator* crc = select_crc(crc_poly);
        ocudu_assert(crc != nullptr, "Invalid CRC calculator.");

        // Discard filler bits for the CRC.
        unsigned nof_significant_bits = cb_data.size() - nof_filler_bits;
        callback(crc->calculate(cb_data.first(nof_significant_bits)) == 0);
      });
}
