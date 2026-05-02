#pragma once

#include <memory>
#include <vector>

#include "../shapes/shape.h"
#include "../shapes/cube.h"

#include "../common/structs.h"

namespace Cave
{
	class Simulation
	{
	private:
		glm::uvec4 _dimensions;
		glm::uvec4 _spawnAreaDimensions;
		glm::vec4 _centerPosition;
		float _spawnRandomDensity = 0.5f;

		ColorRules _colorRules;
		SimulationParameters _simulationParameters;

		std::shared_ptr<Shape> _shape;
		std::vector<Vertex> _vertexLump;
		std::vector<uint32_t> _indexLump;

		uint32_t _numberOfNeighbors;
		uint32_t _neighborConfigurationCount;
		std::vector<std::vector<glm::ivec4>> _neighborDeltaConfigurations{};

	private:
		glm::vec4 ConvertHexToVec(uint32_t color)
		{
			float red   = ((color >> 16) & 0xFF) / 255.0;
			float green = ((color >> 8) & 0xFF) / 255.0;
			float blue  = ((color) & 0xFF) / 255.0;

			return glm::vec4(red, green, blue, 1.f);
		}

	public:
		static const uint64_t EXISTENCE_PERMUTATION_BIT_MASK = 0xFFFFFFFFFFFFFFF;
		static const uint64_t EXISTENCE_PERMUTATION_INVERTED_BIT_MASK = 0XF000000000000000;
		static const uint64_t EXISTENCE_MIN_PERMUTATION = 0x1000000000000000;
		static const uint64_t BIT_SHIFT = 60;
		static const uint64_t FACE_NEIGHBORS_MASK = 1;
		static const uint64_t EDGE_NEIGHBORS_MASK = 2;
		static const uint64_t CORNER_NEIGHBORS_MASK = 4;
		static const uint64_t WRAP_NEIGHBORS_MASK = 8;

		// maxCellState uses shift-by-1 encoding: the 4 bits store (actualValue - 1),
		// so the representable range is 1..16 (not 0..15). maxCellState == 0 is
		// meaningless (every cell would be dead from birth), so this shift recovers
		// one usable value from the bit budget.
		static const uint64_t MAX_CELL_STATE_MIN = 1;
		static const uint64_t MAX_CELL_STATE_MAX = 16;
		static constexpr uint64_t EncodeMaxCellState(uint64_t actual) { return (actual - 1) & 0xFULL; }
		static constexpr uint64_t DecodeMaxCellState(uint64_t storedBits) { return (storedBits & 0xFULL) + 1; }

	public:
		Simulation();
		Simulation(glm::uvec3 gridDimensions, glm::uvec3 spawnDimensions, int spawnMode,
				   SimulationParameters simulationParameters, ColorRules colorRules,
				   std::shared_ptr<Shape> shape, float spawnRandomDensity = 0.5f);
		~Simulation() = default;

	public:
		glm::uvec4 *GetDimensions() { return &_dimensions; }

		glm::vec4 *GetCenterPosition() { return &_centerPosition; }

		glm::uvec4 *GetCenterSpawnAreaDimensions() { return &_spawnAreaDimensions; }

		float GetSpawnRandomDensity() const { return _spawnRandomDensity; }

		ColorRules *GetColorRules() { return &_colorRules; }

		SimulationParameters *GetSimulationParameters() { return &_simulationParameters; }

		std::vector<Vertex> *GetVertexLump() { return &_vertexLump; }
		std::vector<uint32_t> *GetIndexLump() { return &_indexLump; }

		std::shared_ptr<Shape> GetShape() const { return _shape; };

		uint32_t GetNumberOfNeighbors() { return _numberOfNeighbors; }
		uint32_t GetNeighborConfigurationCount() { return _neighborConfigurationCount; }
		const std::vector<std::vector<glm::ivec4>>& GetNeighborDeltaConfigurations() { return _neighborDeltaConfigurations; }
	};
}