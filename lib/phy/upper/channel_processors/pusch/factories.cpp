// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

#include "ocudu/phy/upper/channel_processors/pusch/factories.h"
#include "pusch_codeblock_cuda_decoder.h"
#include "pusch_decoder_cuda_impl.h"
#include "ocudu/phy/upper/channel_processors/pusch/pusch_processor_result_notifier.h"

using namespace ocudu;

namespace {

class pusch_decoder_factory_cuda : public pusch_decoder_factory
{
public:
  explicit pusch_decoder_factory_cuda(pusch_decoder_factory_sw_configuration config) :
    crc_factory(std::move(config.crc_factory)),
    segmenter_factory(std::move(config.segmenter_factory)),
    executor(config.executor),
    nof_prb(config.nof_prb),
    nof_layers(config.nof_layers)
  {
    ocudu_assert(crc_factory, "Invalid CRC calculator factory.");
    ocudu_assert(config.decoder_factory, "Invalid LDPC decoder factory.");
    ocudu_assert(config.dematcher_factory, "Invalid LDPC dematcher factory.");
    ocudu_assert(segmenter_factory, "Invalid LDPC segmenter factory.");

    cuda_backend = std::make_shared<cuda_ldpc_decoder_asynchronous_backend>(*executor);

    std::vector<std::unique_ptr<pusch_codeblock_cuda_decoder>> codeblock_decoders(
        std::max(1U, config.nof_pusch_decoder_threads * MAX_NOF_SEGMENTS));
    for (std::unique_ptr<pusch_codeblock_cuda_decoder>& codeblock_decoder : codeblock_decoders) {
      pusch_codeblock_cuda_decoder::sch_crc crcs1;
      crcs1.crc16  = crc_factory->create(crc_generator_poly::CRC16);
      crcs1.crc24A = crc_factory->create(crc_generator_poly::CRC24A);
      crcs1.crc24B = crc_factory->create(crc_generator_poly::CRC24B);

      codeblock_decoder = std::make_unique<pusch_codeblock_cuda_decoder>(
          config.dematcher_factory->create(), std::make_unique<ldpc_decoder_cuda>(cuda_backend), crcs1);
    }

    decoder_pool = std::make_unique<pusch_decoder_cuda_impl::codeblock_decoder_pool>(codeblock_decoders);
  }

  std::unique_ptr<pusch_decoder> create() override
  {
    pusch_decoder_cuda_impl::sch_crc crcs;
    crcs.crc16  = crc_factory->create(crc_generator_poly::CRC16);
    crcs.crc24A = crc_factory->create(crc_generator_poly::CRC24A);
    crcs.crc24B = crc_factory->create(crc_generator_poly::CRC24B);

    return std::make_unique<pusch_decoder_cuda_impl>(
        segmenter_factory->create(), decoder_pool, std::move(crcs), executor, nof_prb, nof_layers);
  }

private:
  std::shared_ptr<cuda_ldpc_decoder_backend>                       cuda_backend;
  std::shared_ptr<pusch_decoder_cuda_impl::codeblock_decoder_pool> decoder_pool;
  std::shared_ptr<crc_calculator_factory>                          crc_factory;
  std::shared_ptr<ldpc_segmenter_rx_factory>                       segmenter_factory;
  task_executor*                                                   executor;
  unsigned                                                         nof_prb;
  unsigned                                                         nof_layers;
};

} // namespace

std::shared_ptr<pusch_decoder_factory>
ocudu::create_pusch_decoder_factory_plugin(pusch_decoder_factory_sw_configuration config)
{
  return std::make_shared<pusch_decoder_factory_cuda>(std::move(config));
}
