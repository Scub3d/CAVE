#pragma once

#include <vulkan/vulkan.hpp>
#include <string>

#include "deviceContext.h"
#include "shader.h"
#include "slangCompiler.h"
#include "../shapes/shape.h"

namespace Cave
{

// Generates and caches the compute shaders used during search mode.
//
// Simulation shader:
//   Bakes the shape's face/edge/corner neighbor deltas AND grid dimensions as GLSL
//   constants, eliminating runtime buffer lookups for neighbor positions and grid
//   indexing math. Rules and cell state are read from GPU buffers so the same
//   shader module works for every rule permutation of a given (shape, grid) pair.
//   Regenerated when shape or grid dimensions change.
//
//   Expected descriptor set layout (binding order):
//     0 — SimulationParameters (readonly)
//     1 — GridSnapshot         (read-write, atomicAdd)
//     2 — CellsIn              (readonly)
//     3 — CellsOut             (read-write)
//
// Manager shader:
//   Bakes grid dimensions and the max-ticks-to-survive threshold as GLSL constants.
//   Reads snapshot and permutation data from GPU buffers, updates cumulative stats,
//   and auto-advances to the next permutation when a simulation completes.
//   Regenerated when grid dimensions or the tick threshold change.
//
//   Expected descriptor set layout (binding order):
//     0 — InitialCellsData     (read-write)
//     1 — CellsData            (read-write)
//     2 — GridSnapshot          (read-write)
//     3 — GridMaxPermutation    (read-write)
//     4 — GridPermutations      (read-write)
//     5 — GridPermutationsCopy  (read-write)
//     6 — GridInfo              (read-write)
//     7 — GridInfoCopy          (read-write)
class SearchShaderGenerator
{
private:
	DeviceContext &_deviceContext;
	SlangCompiler _slangCompiler;

	// Simulation shader members
	std::string _cachedShapeName;
	uint32_t _cachedSimulationGridXSize{0};
	uint32_t _cachedSimulationGridYSize{0};
	uint32_t _cachedSimulationGridZSize{0};
	uint32_t _cachedSimulationMaxRuleBits{0};
	uint32_t _cachedSimulationWorkgroupSize{0};
	vk::ShaderModule _shaderModule{};
	std::string _shaderCode;

	// Manager shader members
	vk::ShaderModule _managerShaderModule{};
	std::string _managerShaderCode;
	uint32_t _cachedManagerGridXSize{0};
	uint32_t _cachedManagerGridYSize{0};
	uint32_t _cachedManagerGridZSize{0};
	uint32_t _cachedManagerMaxNumberOfTicksToSurvive{0};
	uint32_t _cachedManagerMaxRuleBits{0};

	void GenerateSimulationShader(Shape &shape, uint32_t gridXSize, uint32_t gridYSize, uint32_t gridZSize, uint32_t maxRuleBits, uint32_t workgroupSize);
	void GenerateManagerShader(uint32_t gridXSize, uint32_t gridYSize, uint32_t gridZSize, uint32_t maxNumberOfTicksToSurvive, uint32_t maxRuleBits);

public:
	explicit SearchShaderGenerator(DeviceContext &deviceContext);
	~SearchShaderGenerator();

	SearchShaderGenerator(const SearchShaderGenerator &) = delete;
	SearchShaderGenerator &operator=(const SearchShaderGenerator &) = delete;

	// Returns the cached simulation shader module, rebuilding it only when the shape,
	// grid dimensions, maxRuleBits cap, or workgroup size has changed.
	vk::ShaderModule GetShaderModule(Shape &shape, uint32_t gridXSize, uint32_t gridYSize, uint32_t gridZSize, uint32_t maxRuleBits, uint32_t workgroupSize);

	// Returns the cached manager shader module, rebuilding it only when the grid
	// dimensions, max-ticks-to-survive threshold, or max-rule-bits cap have changed.
	vk::ShaderModule GetManagerShaderModule(uint32_t gridXSize, uint32_t gridYSize, uint32_t gridZSize,
		uint32_t maxNumberOfTicksToSurvive, uint32_t maxRuleBits = 0);

	const std::string &GetShaderCode() const { return _shaderCode; }
	const std::string &GetManagerShaderCode() const { return _managerShaderCode; }
};

} // namespace Cave
