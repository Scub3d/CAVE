#pragma once

#include "shape.h"
#include <vector>

namespace Cave
{
	class Cube : public Shape
	{
	public:
		Cube() : Shape("cube", 0.1f)
		{
			_faceNeighborDeltaConfigurations = { {
				glm::ivec4(1, 0, 0, 0),
				glm::ivec4(0, 1, 0, 0),
				glm::ivec4(0, 0, 1, 0),
				glm::ivec4(-1, 0, 0, 0),
				glm::ivec4(0, -1, 0, 0),
				glm::ivec4(0, 0, -1, 0)
			} };

			_edgeNeighborDeltaConfigurations = { {
				glm::ivec4(1, 1, 0, 0),
				glm::ivec4(1, -1, 0, 0),
				glm::ivec4(-1, 1, 0, 0),
				glm::ivec4(-1, -1, 0, 0),
				glm::ivec4(1, 0, 1, 0),
				glm::ivec4(1, 0, -1, 0),
				glm::ivec4(-1, 0, 1, 0),
				glm::ivec4(-1, 0, -1, 0),
				glm::ivec4(0, 1, 1, 0),
				glm::ivec4(0, 1, -1, 0),
				glm::ivec4(0, -1, 1, 0),
				glm::ivec4(0, -1, -1, 0)
			} };

			_cornerNeighborDeltaConfigurations = { {
				glm::ivec4(1, 1, 1, 0),
				glm::ivec4(-1, 1, 1, 0),
				glm::ivec4(1, -1, 1, 0),
				glm::ivec4(-1, -1, 1, 0),
				glm::ivec4(1, 1, -1, 0),
				glm::ivec4(-1, 1, -1, 0),
				glm::ivec4(1, -1, -1, 0),
				glm::ivec4(-1, -1, -1, 0)
			} };
		}

		static std::shared_ptr<Shape> Create() { return std::make_shared<Cube>(); }

		bool IsCellViable(glm::ivec3 cellIndices, glm::ivec3 simulationDimensions) const override
		{
			return true;
		}

		glm::vec3 ComputeCenteringOffset(glm::ivec3 simulationDimensions) const override
		{
			glm::vec3 minimumPosition = ComputePosition(glm::ivec3(0, 0, 0), simulationDimensions);
			glm::vec3 maximumPosition = ComputePosition(simulationDimensions - glm::ivec3(1), simulationDimensions);
			return (minimumPosition + maximumPosition) * 0.5f;
		}

		glm::vec3 ComputePosition(glm::ivec3 cellIndices, glm::ivec3 simulationDimensions) const override
		{
			return glm::vec3(
				cellIndices.x * (2.0f * _meshScaleRatio),
				cellIndices.y * (2.0f * _meshScaleRatio),
				cellIndices.z * (2.0f * _meshScaleRatio));
		}

		glm::vec3 ComputeRotation(int x, int y, int z) const override
		{
			return glm::vec3(glm::radians(90.0), 0, 0);
		}
	};
}
