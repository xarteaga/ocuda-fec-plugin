# ocuda-fec-plugin

CUDA-accelerated Forward Error Correction (FEC) plugin for [OCUDU](https://ocudu.org) — a 5G CU/DU software-defined radio platform.

This plugin provides GPU-offloaded LDPC decoding for the 5G PUSCH (Physical Uplink Shared Channel), leveraging CUDA to accelerate the rate dematching and message-passing LDPC decoder kernels.

## Features

- GPU-accelerated LDPC decoding using 3GPP 5G LDPC base graphs BG1 (rate 1/3) and BG2 (rate 1/5)
- Asynchronous execution with a pool of 128 CUDA streams for concurrent codeblock processing
- Callback-driven pipeline — phases are chained via CUDA stream callbacks instead of CPU polling, minimising executor thread usage
- Support for HARQ soft combining across retransmissions
- Full PUSCH codeblock segmentation, rate dematching, decoding, and CRC verification
- Pluggable architecture — integrates with OCUDU's PUSCH decoder factory pattern

## Project Structure

```
include/ocuda-fec/          Public headers (consumed by OCUDU core)
├── cuda_helpers/           RAII CUDA streams, device memory, host-to-device promise, LDPC decoder interface
├── instrumentation/
│   └── traces/
│       └── l1_cuda_traces.h  L1-level CUDA event tracer (conditional on OCUDU_L1_DL_TRACE / OCUDU_L1_UL_TRACE)
└── phy/upper/channel_coding/
    ├── ldpc_decoder_cuda.h   Per-codeblock decoder (config + decode)
    └── ldpc_decoder_cuda_backend.h  Base class + create_asynchronous_backend() factory

lib/                        Source files
├── cuda_helpers/           CUDA kernels (.cu) and device helpers (device_*.h)
│   ├── ldpc_decoder_impl.cu          Main GPU kernel: rate dematch → LDPC iterations → hard decision + decoder factory
│   ├── ldpc_decoder_impl.h           ldpc_decoder_impl class declaration
│   ├── ldpc_decoder_cuda_helpers.cu  cudaMalloc/memcpy/memset wrappers
│   ├── cuda_stream.cu                cudaStream_t lifecycle
│   ├── device_ldpc_decoder.h         __device__ check-node / variable-node processing
│   ├── device_ldpc_rate_dematcher.h  __device__ rate dematching
│   └── device_math_helpers.h         __device__ soft-bit loading
├── instrumentation/            CUDA tracing support
│   └── traces/
│       └── l1_cuda_traces.cpp  L1 event tracer definition
└── phy/upper/
    ├── channel_coding/             LDPC decoder implementation
    │   ├── ldpc_decoder_cuda_backend.cpp                   Base graph upload to GPU
    │   ├── ldpc_decoder_cuda_asynchronous_backend.h        cuda_ldpc_decoder_batch (callback-chained pipeline) + async backend
    │   ├── ldpc_decoder_cuda_asynchronous_backend.cpp      128-stream pool, callback-driven pipeline (no CPU polling)
    │   └── ldpc_decoder_cuda_impl.cpp                      Per-codeblock: config, H2D queue, backend call
    └── channel_processors/pusch/     PUSCH integration layer
        ├── factories.cpp                       Factory → creates pusch_decoder instances
        ├── pusch_codeblock_cuda_decoder.h/.cpp  Single codeblock: rate dematch + LDPC decode + CRC
        ├── pusch_decoder_cuda_impl.h/.cpp      Full PUSCH decoder: state machine, segment, dispatch, join
        └── pusch_decoder_buffer_dummy.h        Dummy buffer for testing (no-op PUSCH decoder)
```

Four static libraries are built:

| Library | Purpose |
|---------|---------|
| `ocuda_ldpc_decoder` | Core CUDA kernels: stream management, H2D/D2H helpers, LDPC solver |
| `ocuda_cuda_traces` | L1-level CUDA event tracing for latency analysis |
| `ocudu_cuda_ldpc` | Base graph upload, async backend (128-stream pool), per-codeblock LDPC facade |
| `ocuda_pusch_decoder` | PUSCH decoder integration (factory, codeblock dispatch, CRC) |

## Requirements

- CUDA Toolkit (tested on architectures 7.5, 8.0, 8.6 — configurable via `CMAKE_CUDA_ARCHITECTURES`)
- OCUDU parent project (version matching the plugin branch)

## Building

This plugin is built as part of the OCUDU project:

```bash
git clone git@github.com:xarteaga/ocuda-fec-plugin.git
cd ocuda-fec-plugin
# Build within the parent OCUDU project
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DENABLE_PLUGINS=ON -DENABLE_CUDA=ON ..
cmake --build build
```

Key CMake variables:

- `ENABLE_CUDA` — when OFF, the plugin is skipped entirely
- `CMAKE_CUDA_ARCHITECTURES` — defaults to `75;80;86`; override for your GPU (e.g. `CMAKE_CUDA_ARCHITECTURES=80` for Ampere-only)

## License

BSD 3-Clause-Open-MPI — see [LICENSE](LICENSE) for details.

Portions of this software may implement 3GPP specifications, which may be subject to additional licensing requirements.

## Copyright

Copyright (c) 2026, Xavier Arteaga.
Portions Copyright (C) 2021–2026 Software Radio Systems Limited.