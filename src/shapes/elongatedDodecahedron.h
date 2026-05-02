#pragma once

#include "shape.h"
#include <vector>

namespace Cave
{
	class ElongatedDodecahedron : public Shape
	{
	public:
		ElongatedDodecahedron() : Shape("ElongatedDodecahedron", 0.1f)
		{
			// Face neighbors: 2 configurations (even-y, odd-y), 12 deltas each
			_faceNeighborDeltaConfigurations = {
				// Config 0: Even row (y % 2 == 0)
				{
					glm::ivec4( 1,  0,  0, 0),
					glm::ivec4( 0,  1,  0, 0),
					glm::ivec4( 0,  0,  1, 0),
					glm::ivec4(-1,  0,  0, 0),
					glm::ivec4( 0, -1,  0, 0),
					glm::ivec4( 0,  0, -1, 0),
					glm::ivec4(-1, -1,  0, 0),
					glm::ivec4(-1, -1, -1, 0),
					glm::ivec4( 0, -1, -1, 0),
					glm::ivec4(-1,  1,  0, 0),
					glm::ivec4(-1,  1, -1, 0),
					glm::ivec4( 0,  1, -1, 0),
				},
				// Config 1: Odd row (y % 2 == 1)
				{
					glm::ivec4( 1,  0,  0, 0),
					glm::ivec4( 0,  1,  0, 0),
					glm::ivec4( 0,  0,  1, 0),
					glm::ivec4(-1,  0,  0, 0),
					glm::ivec4( 0, -1,  0, 0),
					glm::ivec4( 0,  0, -1, 0),
					glm::ivec4( 1, -1,  0, 0),
					glm::ivec4( 1, -1,  1, 0),
					glm::ivec4( 0, -1,  1, 0),
					glm::ivec4( 1,  1,  0, 0),
					glm::ivec4( 1,  1,  1, 0),
					glm::ivec4( 0,  1,  1, 0),
				},
			};

			// Edge neighbors: same for both configs, 4 deltas
			std::vector<glm::ivec4> edgeDeltas = {
				glm::ivec4( 1, 0,  1, 0),
				glm::ivec4( 1, 0, -1, 0),
				glm::ivec4(-1, 0,  1, 0),
				glm::ivec4(-1, 0, -1, 0),
			};
			_edgeNeighborDeltaConfigurations = { edgeDeltas, edgeDeltas };

			// Corner neighbors: none
			_cornerNeighborDeltaConfigurations = { {}, {} };
		}

		static std::shared_ptr<Shape> Create() { return std::make_shared<ElongatedDodecahedron>(); }

		glm::vec3 ComputeCenteringOffset(glm::ivec3 simulationDimensions) const override
		{
			// All cells are viable, so the bounding box is determined by corner cells.
			// Odd rows are offset by halfOffset in X and Z, which may extend the max bound.
			glm::vec3 minimumPosition = ComputePosition(glm::ivec3(0, 0, 0), simulationDimensions);
			glm::vec3 maximumPosition = ComputePosition(simulationDimensions - glm::ivec3(1), simulationDimensions);

			// Check an odd-row corner in case it extends further than the even-row corner
			if (simulationDimensions.y > 1)
			{
				glm::vec3 oddRowCorner = ComputePosition(
					glm::ivec3(simulationDimensions.x - 1, 1, simulationDimensions.z - 1), simulationDimensions);
				maximumPosition = glm::max(maximumPosition, oddRowCorner);
			}

			return (minimumPosition + maximumPosition) * 0.5f;
		}

		bool IsCellViable(glm::ivec3 cellIndices, glm::ivec3 simulationDimensions) const override
		{
			return true;
		}

		glm::vec3 ComputePosition(glm::ivec3 cellIndices, glm::ivec3 simulationDimensions) const override
		{
			float xSpacing = 1.73204f * _meshScaleRatio;
			float ySpacing = 2.0f * _meshScaleRatio;
			float zSpacing = 1.73204f * _meshScaleRatio;
			float halfOffset = 0.86602f * _meshScaleRatio;

			return glm::vec3(
				cellIndices.x * xSpacing + (cellIndices.y % 2 == 1 ? halfOffset : 0.0f),
				cellIndices.y * ySpacing,
				cellIndices.z * zSpacing + (cellIndices.y % 2 == 1 ? halfOffset : 0.0f));
		}

		glm::vec3 ComputeRotation(int x, int y, int z) const override
		{
			return glm::vec3(glm::radians(90.0f), 0.0f, 0.0f);
		}
	};
}
