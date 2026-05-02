#include "simulation.h"
#include "../vulkan/objMesh.h"

namespace Cave
{
	Simulation::Simulation()
		: Simulation(
			glm::uvec3(225, 225, 225),
			glm::uvec3(25, 25, 25),
			1,
			SimulationParameters{
				(EncodeMaxCellState(5) << BIT_SHIFT) | 0x39,   // maxCellState=5, birth=1,4,5,6
				(FACE_NEIGHBORS_MASK << BIT_SHIFT) | 0x8,      // face neighbors, survival=4
				0                                               // shapeAndGridConfiguration
			},
			ColorRules{ConvertHexToVec(0x5f4b8b), ConvertHexToVec(0xe69a8d)},
			Cube::Create())
	{
	}

	Simulation::Simulation(glm::uvec3 gridDimensions, glm::uvec3 spawnDimensions, int spawnMode,
						   SimulationParameters simulationParameters, ColorRules colorRules,
						   std::shared_ptr<Shape> shape, float spawnRandomDensity)
	{
		_dimensions = glm::uvec4(gridDimensions, gridDimensions.x * gridDimensions.y * gridDimensions.z);
		_centerPosition = glm::vec4(0, 0, 0, glm::length(glm::vec3(_dimensions.x, _dimensions.y, _dimensions.z)));
		_spawnAreaDimensions = glm::uvec4(spawnDimensions, spawnMode);
		_spawnRandomDensity = spawnRandomDensity;

		_colorRules = colorRules;
		_simulationParameters = simulationParameters;

		_shape = shape;

		bool faces = (_simulationParameters.survivalAndNeighborhoodRules >> BIT_SHIFT) & FACE_NEIGHBORS_MASK;
		bool edges = (_simulationParameters.survivalAndNeighborhoodRules >> BIT_SHIFT) & EDGE_NEIGHBORS_MASK;
		bool corners = (_simulationParameters.survivalAndNeighborhoodRules >> BIT_SHIFT) & CORNER_NEIGHBORS_MASK;

		_neighborDeltaConfigurations = _shape->CalculateNeighborDeltaConfigurations(faces, edges, corners);
		_neighborConfigurationCount = static_cast<uint32_t>(_neighborDeltaConfigurations.size());
		_numberOfNeighbors = _neighborDeltaConfigurations.empty() ? 0
			: static_cast<uint32_t>(_neighborDeltaConfigurations[0].size());

		ObjMesh shapeMesh(("models/" + _shape->GetName() + ".obj").c_str(), ("models/" + _shape->GetName() + ".mtl").c_str());
		_vertexLump = shapeMesh.GetMeshVertices();
		_indexLump = shapeMesh.GetMeshIndices();

		// All GPU buffers (cells, draw commands, orientations) are initialized on the GPU
		// via a compute shader in VulkanSimulationRenderer::InitCellsAndDrawCommandsOnGPU().
		// No large CPU allocations needed here.
	}
}