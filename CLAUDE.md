# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**ocuda-fec-plugin** — a CUDA-accelerated LDPC decoder plugin for the [OCUDU](https://ocudu.org) 5G CU/DU project. It GPU-offloads the PUSCH (Physical Uplink Shared Channel) LDPC decoding chain.

## Building

This plugin is built within the parent OCUDU project. From the plugin directory:

```bash
cd /home/xavier/workspace/ocudu          # parent repo
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_CUDA=ON
cmake --build build
```

Key CMake variables:
- `ENABLE_CUDA` — when OFF, the plugin is skipped entirely (CMake condition on line 20–21 of root CMakeLists.txt: `ENABLE_CUDA AND CUDAToolkit_FOUND`)
- `CMAKE_CUDA_ARCHITECTURES` — defaults to `75;80;86` in `lib/cuda_helpers/CMakeLists.txt`; override for your GPU

### CMake Library Structure

| Library | Sources | Links |
|---------|---------|-------|
| `ocuda_ldpc_decoder` | `cuda_stream.cu`, `ldpc_decoder_cuda_helpers.cu`, `ldpc_decoder_impl.cu` | `CUDA::cudart` |
| `ocudu_cuda_ldpc` | `ldpc_decoder_cuda_backend.cpp`, `ldpc_decoder_cuda_asynchronous_backend.cpp`, `ldpc_decoder_cuda_impl.cpp` | `ocuda_ldpc_decoder` |
| `ocuda_pusch_decoder` | `factories.cpp`, `pusch_codeblock_cuda_decoder.cpp`, `pusch_decoder_cuda_impl.cpp` | `ocudu_upper_phy_support`, `ocudu_ran` |

## Code Architecture

### Structure

```
include/ocuda-fec/                      Public headers (consumed by OCUDU core)
├── cuda_helpers/
│   ├── base_graph_description.h        LDPC base graph adjacency description (host-side)
│   ├── ldpc_decoder.h                  Abstract GPU LDPC decoder interface (batch ldpc_decode)
│   ├── ldpc_decoder_cuda_helpers.h     H2D/D2H copy, memset, zero helpers (templated)
│   ├── cuda_stream.h                   RAII CUDA stream + sync token
│   ├── device_vector.h                 Auto-sized device memory wrapper
│   └── host_to_device_promise.h        Deferred async H2D transfer (RAII promise)
└── phy/upper/channel_coding/
    ├── ldpc_decoder_cuda.h             Per-codeblock decoder (config + decode)
    └── ldpc_decoder_cuda_backend.h     Base class + create_asynchronous_backend() factory

lib/
├── cuda_helpers/                       CUDA device code (.cu) and device helpers (device_*.h)
│   ├── ldpc_decoder_impl.cu              Main GPU kernel: ldpc_decoder::create factory
│   │                                   → rate dematch → load soft bits → LDPC iterations → hard decision
│   ├── ldpc_decoder_impl.h               ldpc_decoder_impl class declaration
│   ├── ldpc_decoder_cuda_helpers.cu      cudaMalloc/memcpy/memset wrappers
│   ├── cuda_stream.cu                    cudaStream_t lifecycle
│   ├── device_ldpc_decoder.h             __device__ check-node / variable-node processing
│   ├── device_ldpc_rate_dematcher.h      __device__ rate dematching
│   └── device_math_helpers.h             __device__ soft-bit loading
└── phy/upper/
    ├── channel_coding/                     LDPC decoder implementation
    │   ├── ldpc_decoder_cuda_backend.cpp               Base graph upload to GPU
    │   ├── ldpc_decoder_cuda_asynchronous_backend.h    cuda_ldpc_decoder_batch + async backend (128-stream pool)
    │   ├── ldpc_decoder_cuda_asynchronous_backend.cpp  128-stream pool, deferred decode + wait loop
    │   └── ldpc_decoder_cuda_impl.cpp                  Per-codeblock: config, H2D queue, backend call
    └── channel_processors/pusch/                   PUSCH integration layer
        ├── factories.cpp                           Factory → creates pusch_decoder instances
        ├── pusch_codeblock_cuda_decoder.h/.cpp     Single codeblock: rate dematch + LDPC decode + CRC
        ├── pusch_decoder_buffer_dummy.h            Dummy buffer for testing (no-op PUSCH decoder)
        └── pusch_decoder_cuda_impl.h/.cpp          Full PUSCH decoder: state machine, segment, dispatch, join
```

### Data Flow (PUSCH decode)

```
softbits → pusch_decoder_cuda_impl (state machine)
  │
  ├─ new_data()          – allocate buffers, set config
  ├─ on_new_softbits()   – accumulate LLRs, dispatch available codeblocks as CUDA tasks
  ├─ set_nof_softbits()  – known size: segment all codeblocks upfront, dispatch eagerly
  └─ on_end_softbits()   – segment remaining codeblocks, wait for all decodes → join & notify
         │
         └─ pusch_codeblock_cuda_decoder (per codeblock)
              ├─ rate_match()  – GPU rate dematching + HARQ combining
              └─ decode()      – GPU LDPC message-passing → CRC check → async callback
                    │
                    └─ cuda_sch_decode kernel (ldpc_decoder_impl.cu)
                         rate_dematch → load_soft_bits → LDPC iterations → hard_bits
```

### Key Types

| Type | Purpose |
|------|---------|
| `cuda_ldpc_decoder_backend` | Base class: owns base graph descriptions on device, pure-virtual `decode()`, exposes `create_asynchronous_backend()` factory |
| `cuda_ldpc_decoder_asynchronous_backend` | Pool of 128 CUDA streams, deferred decode + wait loop (moved to local header `lib/.../ldpc_decoder_cuda_asynchronous_backend.h`) |
| `cuda_ldpc_decoder_batch` | Internal: batches up to 32 codeblocks per stream, enqueues, checks completion (moved to local header) |
| `ldpc_decoder_cuda` | Per-codeblock facade: computes LDPC config, queues H2D, calls backend |
| `ldpc_decoder` (cuda namespace) | Abstract batch decoder interface; implemented by `ldpc_decoder_impl` |
| `pusch_codeblock_cuda_decoder` | Wraps rate dematcher + decoder + CRC for one codeblock |
| `pusch_decoder_cuda_impl` | Full PUSCH decoder: collects softbits, segments, dispatches codeblocks, joins results |
| `host_to_device_promise<T>` | RAII wrapper that delays an async H2D transfer until `.transfer()` is called |

### Concurrency Model

- `cuda_ldpc_decoder_asynchronous_backend` uses a pool of 128 CUDA streams protected by `backend_mutex` — created via `create_asynchronous_backend(task_executor&)` factory
- Codeblocks are dispatched to streams; when a stream's batch fills (max 32), it dequeues and runs
- A polling loop (`timed_decode`) waits for an idle stream
- Completion callbacks (`check_and_complete`) are deferred back to the `task_executor`

## Editing Guidelines

- All files carry `SPDX-License-Identifier: BSD-3-Clause-Open-MPI` and the SRS copyright header
- This plugin depends on the OCUDU parent project's headers — do not modify parent paths
- New `.cu` files need to be added to `lib/cuda_helpers/CMakeLists.txt`
- New C++ sources need to be added to their respective CMakeLists.txt (`channel_coding/` or `channel_processors/pusch/`)
- CUDA kernels live in `lib/cuda_helpers/ldpc_decoder_impl.cu`; device helpers in `device_*.h`
- Host-side code lives in `lib/cuda_helpers/ldpc_decoder_cuda_helpers.cu` and `.cpp` files
- All libraries are registered with the parent build via `add_to_exported_libs()`
- Use the existing `ocudu_assert()` pattern consistently; avoid `printf` (the codebase already has a stray `fmt::println` in `pusch_decoder_cuda_impl.cpp` that should probably use the logger)