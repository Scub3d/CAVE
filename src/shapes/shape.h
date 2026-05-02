#pragma once

#include <string>
#include <memory>
#include <vector>
#include <limits>

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>

namespace Cave
{
	class Shape
	{
	protected:
		std::string _name;
		float _meshScaleRatio;

		// Neighbor deltas indexed by configuration.
		// Outer vector: one entry per configuration (e.g., even-y vs odd-y).
		// Inner vector: deltas for that configuration.
		// All configurations must have the same count per type.
		std::vector<std::vector<glm::ivec4>> _faceNeighborDeltaConfigurations;
		std::vector<std::vector<glm::ivec4>> _edgeNeighborDeltaConfigurations;
		std::vector<std::vector<glm::ivec4>> _cornerNeighborDeltaConfigurations;

	public:
		Shape(std::string name, float meshScaleRatio)
		{
			_name = name;
			_meshScaleRatio = meshScaleRatio;
		};

		virtual bool IsCellViable(glm::ivec3 cellIndices, glm::ivec3 simulationDimensions) const = 0;

		virtual glm::vec3 ComputePosition(glm::ivec3 cellIndices, glm::ivec3 simulationDimensions) const = 0;
		virtual glm::vec3 ComputeRotation(int x, int y, int z) const = 0;

		virtual glm::vec3 ComputeCenteringOffset(glm::ivec3 simulationDimensions) const
		{
			float minimumX = std::numeric_limits<float>::max();
			float minimumY = std::numeric_limits<float>::max();
			float minimumZ = std::numeric_limits<float>::max();
			float maximumX = std::numeric_limits<float>::lowest();
			float maximumY = std::numeric_limits<float>::lowest();
			float maximumZ = std::numeric_limits<float>::lowest();

#pragma omp parallel for reduction(min:minimumX, minimumY, minimumZ) reduction(max:maximumX, maximumY, maximumZ)
			for (int x = 0; x < simulationDimensions.x; x++)
			{
				for (int y = 0; y < simulationDimensions.y; y++)
				{
					for (int z = 0; z < simulationDimensions.z; z++)
					{
						glm::ivec3 cellIndices(x, y, z);
						if (!IsCellViable(cellIndices, simulationDimensions))
							continue;

						glm::vec3 rawPosition = ComputePosition(cellIndices, simulationDimensions);

						if (rawPosition.x < minimumX) minimumX = rawPosition.x;
						if (rawPosition.y < minimumY) minimumY = rawPosition.y;
						if (rawPosition.z < minimumZ) minimumZ = rawPosition.z;
						if (rawPosition.x > maximumX) maximumX = rawPosition.x;
						if (rawPosition.y > maximumY) maximumY = rawPosition.y;
						if (rawPosition.z > maximumZ) maximumZ = rawPosition.z;
					}
				}
			}

			return (glm::vec3(minimumX, minimumY, minimumZ) + glm::vec3(maximumX, maximumY, maximumZ)) * 0.5f;
		}

		virtual std::string GetName() const { return _name; }
		float GetMeshScaleRatio() const { return _meshScaleRatio; }

		uint32_t GetNeighborConfigurationCount() const
		{
			return static_cast<uint32_t>(_faceNeighborDeltaConfigurations.size());
		}

		const std::vector<std::vector<glm::ivec4>>& GetFaceNeighborDeltaConfigurations() const { return _faceNeighborDeltaConfigurations; }
		const std::vector<std::vector<glm::ivec4>>& GetEdgeNeighborDeltaConfigurations() const { return _edgeNeighborDeltaConfigurations; }
		const std::vector<std::vector<glm::ivec4>>& GetCornerNeighborDeltaConfigurations() const { return _cornerNeighborDeltaConfigurations; }

		// Returns combined neighbor deltas per configuration.
		// Result[configIndex] = concatenation of selected face+edge+corner deltas for that config.
		virtual std::vector<std::vector<glm::ivec4>> CalculateNeighborDeltaConfigurations(bool faces, bool edges, bool corners)
		{
			std::vector<std::vector<glm::ivec4>> result;
			uint32_t configurationCount = GetNeighborConfigurationCount();
			for (uint32_t configurationIndex = 0; configurationIndex < configurationCount; configurationIndex++)
			{
				std::vector<glm::ivec4> configurationDeltas;
				if (faces)
					for (const auto& delta : _faceNeighborDeltaConfigurations[configurationIndex])
						configurationDeltas.push_back(delta);
				if (edges)
					for (const auto& delta : _edgeNeighborDeltaConfigurations[configurationIndex])
						configurationDeltas.push_back(delta);
				if (corners)
					for (const auto& delta : _cornerNeighborDeltaConfigurations[configurationIndex])
						configurationDeltas.push_back(delta);
				result.push_back(std::move(configurationDeltas));
			}
			return result;
		}

		virtual bool operator==(const Shape &other) const
		{
			return _name == other.GetName();
		}

		// Neighbor count for the given (F, E, C) selection. Assumes all neighbor
		// configurations of a shape have matching counts per delta type (shape invariant),
		// so it's safe to read from configuration 0 only.
		virtual uint32_t GetNeighborCount(bool faces, bool edges, bool corners) const
		{
			uint32_t count = 0;
			if (faces && !_faceNeighborDeltaConfigurations.empty())
				count += static_cast<uint32_t>(_faceNeighborDeltaConfigurations[0].size());
			if (edges && !_edgeNeighborDeltaConfigurations.empty())
				count += static_cast<uint32_t>(_edgeNeighborDeltaConfigurations[0].size());
			if (corners && !_cornerNeighborDeltaConfigurations.empty())
				count += static_cast<uint32_t>(_cornerNeighborDeltaConfigurations[0].size());
			return count;
		}
	};
}
