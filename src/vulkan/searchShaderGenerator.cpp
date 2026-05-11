#include "searchShaderGenerator.h"
#include "../common/logger.h"

namespace Cave
{

SearchShaderGenerator::SearchShaderGenerator(DeviceContext &deviceContext)
	: _deviceContext{deviceContext}
{
}

SearchShaderGenerator::~SearchShaderGenerator()
{
	if (_shaderModule)
	{
		_deviceContext.GetDevice().destroyShaderModule(_shaderModule);
	}
	if (_managerShaderModule)
	{
		_deviceContext.GetDevice().destroyShaderModule(_managerShaderModule);
	}
}

vk::ShaderModule SearchShaderGenerator::GetShaderModule(Shape &shape, uint32_t gridXSize, uint32_t gridYSize, uint32_t gridZSize, uint32_t maxRuleBits, uint32_t workgroupSize)
{
	if (shape.GetName() == _cachedShapeName
		&& gridXSize == _cachedSimulationGridXSize
		&& gridYSize == _cachedSimulationGridYSize
		&& gridZSize == _cachedSimulationGridZSize
		&& maxRuleBits == _cachedSimulationMaxRuleBits
		&& workgroupSize == _cachedSimulationWorkgroupSize
		&& _shaderModule)
	{
		return _shaderModule;
	}

	if (_shaderModule)
	{
		_deviceContext.GetDevice().destroyShaderModule(_shaderModule);
		_shaderModule = nullptr;
	}

	GenerateSimulationShader(shape, gridXSize, gridYSize, gridZSize, maxRuleBits, workgroupSize);
	_cachedShapeName = shape.GetName();
	_cachedSimulationGridXSize = gridXSize;
	_cachedSimulationGridYSize = gridYSize;
	_cachedSimulationGridZSize = gridZSize;
	_cachedSimulationMaxRuleBits = maxRuleBits;
	_cachedSimulationWorkgroupSize = workgroupSize;
	return _shaderModule;
}

void SearchShaderGenerator::GenerateSimulationShader(Shape &shape, uint32_t gridXSize, uint32_t gridYSize, uint32_t gridZSize, uint32_t maxRuleBits, uint32_t workgroupSize)
{
	LOG_DEBUG("Generating search compute shader for shape: {} grid: {}x{}x{} maxRuleBits: {} workgroupSize: {}",
		shape.GetName(), gridXSize, gridYSize, gridZSize, maxRuleBits, workgroupSize);

	// Both supported shapes route through Slang. The runtime-selectable F/E/C
	// neighborhood subset is handled INSIDE the per-shape Slang shader via
	// bitmask-flag branching (the simulation buffer's neighborhoodFlags). When
	// adding a new shape, follow the recipe in
	// shaders/slang/searchSimulationShapeTemplate.slang.
	std::map<std::string, std::string> defines;
	defines["GRID_X_SIZE"]    = std::to_string(gridXSize);
	defines["GRID_Y_SIZE"]    = std::to_string(gridYSize);
	defines["GRID_Z_SIZE"]    = std::to_string(gridZSize);
	defines["MAX_RULE_BITS"]  = std::to_string(maxRuleBits) + "u";
	defines["WORKGROUP_SIZE"] = std::to_string(workgroupSize);

	_shaderCode.clear();

	if (shape.GetName() == "cube")
	{
		_shaderModule = _slangCompiler.Compile(
			_deviceContext, "shaders/slang", "searchSimulationCube", "computeMain", defines);
		LOG_DEBUG("Search simulation compute shader generated via Slang (cube)");
		return;
	}
	if (shape.GetName() == "ElongatedRhombicDodecahedron")
	{
		_shaderModule = _slangCompiler.Compile(
			_deviceContext, "shaders/slang", "searchSimulationERD", "computeMain", defines);
		LOG_DEBUG("Search simulation compute shader generated via Slang (ERD)");
		return;
	}

	LOG_FATAL("Unsupported shape '{}' for search simulation. Supported: 'cube', 'ElongatedRhombicDodecahedron'.", shape.GetName());
	throw std::runtime_error("Unsupported shape for search simulation: " + shape.GetName());
}

vk::ShaderModule SearchShaderGenerator::GetManagerShaderModule(uint32_t gridXSize, uint32_t gridYSize, uint32_t gridZSize, uint32_t maxNumberOfTicksToSurvive, uint32_t maxRuleBits)
{
	if (gridXSize == _cachedManagerGridXSize
		&& gridYSize == _cachedManagerGridYSize
		&& gridZSize == _cachedManagerGridZSize
		&& maxNumberOfTicksToSurvive == _cachedManagerMaxNumberOfTicksToSurvive
		&& maxRuleBits == _cachedManagerMaxRuleBits
		&& _managerShaderModule)
	{
		return _managerShaderModule;
	}

	if (_managerShaderModule)
	{
		_deviceContext.GetDevice().destroyShaderModule(_managerShaderModule);
		_managerShaderModule = nullptr;
	}

	GenerateManagerShader(gridXSize, gridYSize, gridZSize, maxNumberOfTicksToSurvive, maxRuleBits);
	_cachedManagerGridXSize = gridXSize;
	_cachedManagerGridYSize = gridYSize;
	_cachedManagerGridZSize = gridZSize;
	_cachedManagerMaxNumberOfTicksToSurvive = maxNumberOfTicksToSurvive;
	_cachedManagerMaxRuleBits = maxRuleBits;
	return _managerShaderModule;
}

void SearchShaderGenerator::GenerateManagerShader(uint32_t gridXSize, uint32_t gridYSize, uint32_t gridZSize, uint32_t maxNumberOfTicksToSurvive, uint32_t maxRuleBits)
{
	LOG_DEBUG("Generating search manager compute shader (grid: {}x{}x{}, maxTicks: {}, maxRuleBits: {})",
			  gridXSize, gridYSize, gridZSize, maxNumberOfTicksToSurvive, maxRuleBits);

	uint32_t totalNumberOfCells = gridXSize * gridYSize * gridZSize;

	// Manager shader is a Slang module at shaders/slang/searchManager.slang.
	std::map<std::string, std::string> defines;
	defines["GRID_X_SIZE"] = std::to_string(gridXSize);
	defines["GRID_Y_SIZE"] = std::to_string(gridYSize);
	defines["GRID_Z_SIZE"] = std::to_string(gridZSize);
	defines["MAX_NUMBER_OF_TICKS_TO_SURVIVE"] = std::to_string(maxNumberOfTicksToSurvive);
	defines["MAX_RULE_BITS"] = std::to_string(maxRuleBits) + "u";

	_managerShaderCode.clear();
	_managerShaderModule = _slangCompiler.Compile(
		_deviceContext, "shaders/slang", "searchManager", "computeMain", defines);

	LOG_DEBUG("Search manager compute shader generated (grid: {}x{}x{}, totalCells: {}, maxTicks: {})",
			  gridXSize, gridYSize, gridZSize, totalNumberOfCells, maxNumberOfTicksToSurvive);
}

} // namespace Cave
