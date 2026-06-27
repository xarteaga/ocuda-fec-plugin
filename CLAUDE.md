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
| `ocuda_cuda_traces` | `l1_cuda_traces.cpp` | `CUDA::cudart`, `ocudu_support` |
| `ocudu_cuda_ldpc` | `ldpc_decoder_cuda_backend.cpp`, `ldpc_decoder_cuda_asynchronous_backend.cpp`, `ldpc_decoder_cuda_impl.cpp` | `ocuda_ldpc_decoder`, `ocuda_cuda_traces` |
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
├── instrumentation/
│   └── traces/
│       └── l1_cuda_traces.h  L1-level CUDA event tracer (conditional on OCUDU_L1_DL_TRACE / OCUDU_L1_UL_TRACE)
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
├── instrumentation/                    CUDA tracing support
│   └── traces/
│       └── l1_cuda_traces.cpp            L1 event tracer definition
└── phy/upper/
    ├── channel_coding/                     LDPC decoder implementation
    │   ├── ldpc_decoder_cuda_backend.cpp               Base graph upload to GPU
    │   ├── ldpc_decoder_cuda_asynchronous_backend.h    cuda_ldpc_decoder_batch (callback-chained pipeline) + async backend
    │   ├── ldpc_decoder_cuda_asynchronous_backend.cpp  128-stream pool, callback-driven pipeline (no CPU polling)
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
| `cuda_ldpc_decoder_asynchronous_backend` | Pool of 128 CUDA streams; pipeline phases are chained via CUDA stream callbacks (no CPU polling) |
| `cuda_ldpc_decoder_batch` | One batch: stores task executor ref, owns `cuda_stream`, chains `launch_decode_kernel` → `unload_output` → `complete` via `set_callback_next_complete()` |
| `cuda_stream_callback` | Zero-allocation `unique_function<void()>` (32-byte small buffer, `ForbidAlloc=true`) used for stream completion callbacks |
| `cuda_stream` | RAII CUDA stream with completion callback support via `set_callback_next_complete()` / `on_complete()` |
| `ldpc_decoder_cuda` | Per-codeblock facade: computes LDPC config, queues H2D, calls backend |
| `ldpc_decoder` (cuda namespace) | Abstract batch decoder interface; implemented by `ldpc_decoder_impl` |
| `pusch_codeblock_cuda_decoder` | Wraps rate dematcher + decoder + CRC for one codeblock |
| `pusch_decoder_cuda_impl` | Full PUSCH decoder: collects softbits, segments, dispatches codeblocks, joins results |
| `host_to_device_promise<T>` | RAII wrapper that delays an async H2D transfer until `.transfer()` is called |

### Concurrency Model

- `cuda_ldpc_decoder_asynchronous_backend` uses a pool of 128 CUDA streams protected by `backend_mutex` — created via `create_asynchronous_backend(task_executor&)` factory
- Each batch (`cuda_ldpc_decoder_batch`) owns a `task_executor&` and a single `cuda_stream`. Pipeline phases are chained via CUDA stream callbacks instead of executor-thread polling:
  1. `start_asynch_decoding()` — H2D transfers on the stream, then `set_callback_next_complete()` fires the next phase when idle.
  2. `launch_decode_kernel()` — defers the kernel launch to the executor, then arms a stream callback that fires `unload_output()`.
  3. `unload_output()` — defers D2H transfer, then arms a callback that fires `complete()`.
  4. `complete()` — defers per-codeblock result copying and callback invocation, then fires the external completion callback.
- Each phase uses `executor.defer()` for device-accessing work (kernel launch, D2H transfer, result processing). No executor thread is held waiting for the GPU — callbacks fire directly from the CUDA runtime when the stream is idle.
- A timeout fallback (`timed_decode`) handles batch reuse when all streams are occupied.

### Tracing Events

L1 CUDA trace events (enabled when `OCUDU_L1_DL_TRACE` or `OCUDU_L1_UL_TRACE` is defined):

| Event | Phase |
|-------|-------|
| `ldpc_dispatch` | Batch fills, pipeline starts |
| `ldpc_decode_cuda_input` | Input H2D transfer enqueued |
| `ldpc_decode_cuda_decode` | Kernel launched on stream |
| `ldpc_decode_cuda_output` | Output D2H transfer enqueued |
| `ldpc_decode_cuda_prepare` | Backwards-compatible reference (start of batch) |
| `ldpc_decode_cuda_async` | Stream became idle (GPU work finished) |
| `ldpc_decode_cuda_complete` | All callbacks fired |

## Editing Guidelines

- All files carry `SPDX-License-Identifier: BSD-3-Clause-Open-MPI` and the SRS copyright header
- This plugin depends on the OCUDU parent project's headers — do not modify parent paths
- New `.cu` files need to be added to `lib/cuda_helpers/CMakeLists.txt`
- New C++ sources need to be added to their respective CMakeLists.txt (`channel_coding/` or `channel_processors/pusch/`)
- CUDA kernels live in `lib/cuda_helpers/ldpc_decoder_impl.cu`; device helpers in `device_*.h`
- Host-side code lives in `lib/cuda_helpers/ldpc_decoder_cuda_helpers.cu` and `.cpp` files
- All libraries are registered with the parent build via `add_to_exported_libs()`
- Use the existing `ocudu_assert()` pattern consistently; avoid `printf` (the codebase already has a stray `fmt::println` in `pusch_decoder_cuda_impl.cpp` that should probably use the logger)
- The `l1_cuda_tracer` event tracer in `include/ocuda-fec/instrumentation/traces/l1_cuda_traces.h` is the standard way to add timing points; trace events flow through `file_event_tracer<T>`

### Pipeline Design Note

The async backend uses callback-chained pipeline phases inside each `cuda_stream` (rather than CPU polling with `is_idle()`). Each phase defers work to the executor via `executor.defer()` and arms a CUDA stream callback to trigger the next phase. This means **zero executor threads are held waiting for GPU completion** — callbacks fire directly from the CUDA runtime when the stream is idle. A `task_executor` is needed only for the device-accessing operations within each phase (kernel launch, D2H transfer, result processing). The executor still needs enough threads to handle the dispatch workload, but not the wait workload.
