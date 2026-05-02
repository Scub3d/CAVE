# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Cave is a GPU-accelerated 3D cellular automata simulation engine. C++20, Vulkan 1.3, Windows (MSVC).

## Build Commands

```bash
# Configure (first time only)
cmake -B build

# Build
cmake --build build --config Debug
cmake --build build --config Release

# Run
./build/Debug/Cave.exe
```

Offline-built shaders are compiled via `shaders/compileShaders.bat` (hardcoded Vulkan SDK path — update if SDK version differs). Runtime shader compilation goes through Slang (`SlangCompiler`); see `shaders/slang/` for source modules and `shaders/slang/searchSimulationShapeTemplate.slang` for the per-shape template + recipe when adding a new shape.

There are no tests, linting, or formatting tools configured.

## Dependencies

CMake FetchContent: GLFW 3.3.8, GLM 1.0.1, spdlog v1.17.0, nlohmann/json 3.12.0. Vendored in `lib/`: ImGui (with GLFW+Vulkan backends), stb, tinyobjloader, VMA, RenderDoc API. System: Vulkan SDK 1.3.290.0.

## Application Modes

The application has three distinct modes, each with its own GUI for parameter selection:

1. **Rendering mode** — Run a single cellular automata simulation with real-time visuals. The user configures simulation parameters via GUI, then watches the simulation play out on screen.
2. **Search mode** — Run hundreds to thousands of simulations in parallel using headless compute. The user configures search parameters via GUI, monitors results in real-time, and output is saved to JSON files.
3. **Video encoding mode** — Given a JSON file (from search mode), generate video files for each simulation in the file. The user selects parameters via GUI and the application encodes videos.

## Architecture

**Entry point:** `main.cpp` creates `Engine`, calls `Init() -> Run() -> Cleanup()`.

**Engine** (`src/engine.h`) orchestrates everything. It owns:
- `VulkanContext` — Vulkan device, queues, command pools, synchronization primitives
- `Simulation` — CPU-side grid state, spawn configuration, shape topology
- `VulkanSimulationRenderer` — GPU buffers for cell data, vertices, indices, indirect draw commands
- `ComputeSystem` — dispatches the CA simulation compute shader; shared across all three modes
- `RenderSystem` — renders cell instances via graphics pipeline (rendering mode)
- `SearchSystem` — orchestrates massively parallel headless simulations (search mode)
- `VideoEncoderSystem` — encodes simulation results to video files (video encoding mode)

**Rendering mode flow:** `ComputeSystem::Tick()` dispatches the simulation compute shader, writing cell state buffers (ping-pong) and indirect draw command buffers, then signals a semaphore. `RenderSystem::RenderFrame()` waits on that semaphore, renders cell instances via `drawIndexedIndirect` using dynamic rendering, and outputs to off-screen color attachments. Double-buffered (2 frames in flight), indexed by `currentFrameIndex % framesInFlight`.

**Search mode flow:** Two-stage GPU compute pipeline per tick: (1) simulation dispatch advances one CA tick, writes per-tick stats to `GridSnapshot` via atomics, and ping-pongs cell buffers; (2) manager dispatch reads `GridSnapshot`, updates cumulative `GridInfo`, decides if the permutation survived or died, and auto-advances to the next permutation on GPU. A semaphore synchronizes the two dispatches. The CPU reads results from host-visible copy buffers (non-blocking) and saves completed results to JSON.

**Video encoding mode flow:** `ComputeSystem::Tick()` runs the simulation (same as rendering mode). `VideoEncoderSystem` renders each frame using its own graphics pipeline (independent of `RenderSystem`), then synchronously encodes the color attachment to H.264 video.

**Shader generation:** All runtime-compiled shaders are Slang modules under `shaders/slang/`, compiled to SPIR-V via `SlangCompiler` (a thin wrapper around the vendored `slang.dll`). Rendering, search-simulation, search-manager, cell-init, ghost-exchange, and dual-GPU compositing all go through Slang. Per-shape and per-grid parameterization happens via Slang preprocessor `#define` injection (e.g. `GRID_X_SIZE`, `MAX_RULE_BITS`, `WORKGROUP_SIZE`, `SHAPE_TYPE`); shape-specific neighbor deltas live as `static const int3` arrays in each shape's `.slang` module (see `cube.slang`, `elongatedDodecahedron.slang`). Shader modules are cached per `(shape, grid, …)` key and regenerated only when those inputs change. Adding a new shape: copy `searchSimulationShapeTemplate.slang`, follow the recipe in its header.

**Key abstractions:**
- `Shape` (abstract) / `Cube` — defines grid topology (neighbor positions, rotations)
- `Pipeline` (base) / `ComputePipeline` / `GraphicsPipeline` — Vulkan pipeline wrappers
- `Descriptor` — descriptor set layout and binding management
- `Buffer`, `Image` — GPU resource wrappers using VMA

**Bit-packing:** `SimulationParameters` in `structs.h` uses three `uint64_t` fields to encode all simulation data (survival rules, birth rules, grid dimensions, neighborhood config, shape config, etc) into packed bit fields. See the bit shift/mask constants in `simulation.h`.

## Code Conventions

- All code lives in `namespace Cave`
- Non-copyable classes: deleted copy constructor and assignment operator
- Logging via macros in `common/logger.h` (`LOG_FATAL`, `LOG_ERROR`, `LOG_INFO`, `LOG_WARNING`, `LOG_DEBUG`), wrapping spdlog
- GPU resources use smart pointers (`std::unique_ptr`, `std::shared_ptr`)
- CMake copies `shaders/`, `data/`, `models/`, `videos/` directories into the build output
- Variable names should never be shortened and should be explicit about what purpose it servers. For example a vk::DescriptorBufferInfo variable responsible for indirect drawing commands should be named drawIndirectIndexCommandsDescriptorBufferInfo
- When initializing a variable to a vulkan struct, format it so it has one line per input with a comment at the end of that line that is the input parameter/variable name like the example below:
    """vk::DependencyInfoKHR dependencyInfo = vk::DependencyInfoKHR(
            {},						// dependencyFlags
            {},						// memoryBarrierCount
            {},						// pMemoryBarriers
            {},						// bufferMemoryBarrierCount
            {},						// pBufferMemoryBarriers
            1,						// imageMemoryBarrierCount
            &imageMemoryBarrier		// pImageMemoryBarriers
                                    // pNext
        );
    """
