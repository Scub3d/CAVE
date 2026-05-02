# Cave

GPU-accelerated 3D cellular automata simulation engine. C++20, Vulkan 1.3, Windows (MSVC) primary.

## Build

```bash
cmake -B build
cmake --build build --config Release
./build/Release/Cave.exe
```

First configure downloads vendored deps via CMake FetchContent (Slang, ImGui, stb, tinyobjloader, VMA, RenderDoc API). Internet required on first configure; cached afterwards under `build/_deps/`.

System requirements:
- **Vulkan SDK 1.3+** installed and on PATH (for `Vulkan::Vulkan` and headers).
- **CMake 3.24+**.
- **MSVC** (Windows) or recent GCC/Clang (Linux). C++20.

## Application Modes

- **Rendering** — single CA simulation with real-time visuals.
- **Search** — hundreds-to-thousands of parallel simulations via headless compute, output to JSONL.
- **Video encoding** — given a JSONL from search mode, encode H.264 video files for each simulation.

## Layout

| Directory | Purpose |
|-----------|---------|
| `src/` | C++ sources |
| `shaders/slang/` | Slang shader modules (runtime-compiled via SlangCompiler) |
| `models/` | `.obj` meshes |
| `tools/` | Build helpers (compileShaders.{bat,sh}) and dev utilities |
| `benchmarks/` | Performance bench scripts + reference results |
| `.vscode/` | IDE config (shared) |
| `build/` | CMake build output (gitignored) |
| `lib/` | Vendored deps fetched by CMake (gitignored) |
| `data/`, `videos/`, `images/` | Runtime output (gitignored) |

## CLI quickstart

```bash
# Render mode (default)
./build/Release/Cave.exe --mode render

# Search mode (parallel rule sweep, output to JSONL)
./build/Release/Cave.exe --mode search --shape cube --grid 49 --neighborhood FE \
    --max-rule-bits 2 --min-cs 2 --max-cs-range 3 --ticks 20 \
    --chunks-per-config 32 --search-workgroup-size 256 --dual-gpu --headless

# Video encode mode (from a JSONL of viable rules)
./build/Release/Cave.exe --mode video --json data/search/<run>/aggregate.jsonl --encode-all
```

`--help` for the full flag list.

## Tests

```bash
./build/Release/Cave.exe --test all
```

## Documentation

See `CLAUDE.md` for architecture notes, code conventions, and per-feature design context.

## License

(Add when known.)
