#include "simulationContext.h"

#include "../vulkan/simulationRenderer.h"
#include "../vulkan/systems/computeSystem.h"
#include "../vulkan/systems/guiSystem.h"
#include "../camera.h"

#include <glm/glm.hpp>

namespace Cave
{

	SimulationContext::~SimulationContext() = default;

	SimulationContext::SimulationContext(SimulationContext&&) noexcept = default;

	SimulationContext& SimulationContext::operator=(SimulationContext&&) noexcept = default;

	void SimulationContext::DestroyGpuResources()
	{
		computeSystem.reset();
		simulationRenderer.reset();
		// Camera is intentionally kept alive — preserves user's camera settings
		// across resets. It's destroyed separately when leaving a mode.
	}

	RayMarchPushConstants SimulationContext::BuildRayMarchPushConstants(const GuiState& guiState)
	{
		const glm::uvec4* dimensions = simulation.GetDimensions();
		float meshScaleRatio = simulation.GetShape()->GetMeshScaleRatio();

		// gridDimensionZ in push constants must be localZSize for correct buffer/image
		// indexing in the fragment shader. But centering and volume bounds must use the
		// GLOBAL grid dimensions so both GPUs' renders align in world space.
		uint32_t renderGridZ = simulationRenderer->IsDomainDecomposed()
			? simulationRenderer->GetLocalZSize()
			: dimensions->z;

		// Centering uses global dimensions — both GPUs share the same world-space frame
		glm::ivec3 globalGridDimensions(dimensions->x, dimensions->y, dimensions->z);
		glm::vec3 centeringOffset = simulation.GetShape()->ComputeCenteringOffset(globalGridDimensions);

		// When domain decomposed, shift centering Z to account for the domain's position
		// within the global grid. Local cell Z=0 maps to global Z = zDomainOffset - ghostDepth.
		// The fragment shader computes worldPos = cellCoord * spacing - centeringOffset,
		// so we adjust centering: effectiveCentering.z = globalCentering.z - domainShift * spacing
		if (simulationRenderer->IsDomainDecomposed())
		{
			int32_t domainZShift = static_cast<int32_t>(simulationRenderer->GetZDomainOffset())
								 - static_cast<int32_t>(simulationRenderer->GetGhostDepth());
			centeringOffset.z -= domainZShift * (2.0f * meshScaleRatio);
		}

		// Volume bounds: computed from local grid corners with the adjusted centering
		// so each GPU's AABB covers its portion of the global world space. Cell indexing
		// in the fragment shader derives directly from (worldPos - volumeMin)/cellSize, so
		// volumeMin must stay aligned with local cell (0,0,0) — we can't shrink it to owned
		// range. Ghost cells are instead excluded by an owned-range filter inside the shader.
		glm::ivec3 localGridDimensions(dimensions->x, dimensions->y, renderGridZ);
		glm::vec3 posMin = simulation.GetShape()->ComputePosition(glm::ivec3(0, 0, 0), localGridDimensions) - centeringOffset;
		glm::vec3 posMax = simulation.GetShape()->ComputePosition(localGridDimensions - glm::ivec3(1), localGridDimensions) - centeringOffset;

		// For shapes with alternating offsets, check odd-row corners that may extend the bounds
		if (dimensions->y > 1)
		{
			glm::vec3 oddCorner = simulation.GetShape()->ComputePosition(
				glm::ivec3(localGridDimensions.x - 1, 1, localGridDimensions.z - 1), localGridDimensions) - centeringOffset;
			posMax = glm::max(posMax, oddCorner);
		}

		const ColorRules* colorRules = simulation.GetColorRules();
		uint64_t maxCellState = Simulation::DecodeMaxCellState(simulation.GetSimulationParameters()->birthAndMaxCellStateRules >> Simulation::BIT_SHIFT);

		RayMarchPushConstants pushConstants{};
		pushConstants.gridDimensionX = dimensions->x;
		pushConstants.gridDimensionY = dimensions->y;
		pushConstants.gridDimensionZ = renderGridZ;
		pushConstants.maxCellState = static_cast<uint32_t>(maxCellState);
		pushConstants.volumeMinX = posMin.x - meshScaleRatio;
		pushConstants.volumeMinY = posMin.y - meshScaleRatio;
		pushConstants.volumeMinZ = posMin.z - meshScaleRatio;
		pushConstants.volumeMaxX = posMax.x + meshScaleRatio;
		pushConstants.volumeMaxY = posMax.y + meshScaleRatio;
		pushConstants.volumeMaxZ = posMax.z + meshScaleRatio;
		pushConstants.aliveColorR = colorRules->aliveColor.r;
		pushConstants.aliveColorG = colorRules->aliveColor.g;
		pushConstants.aliveColorB = colorRules->aliveColor.b;
		pushConstants.deadColorR = colorRules->deadColor.r;
		pushConstants.deadColorG = colorRules->deadColor.g;
		pushConstants.deadColorB = colorRules->deadColor.b;
		pushConstants.occlusionStrength = 0.5f;
		pushConstants.maxNeighborCount = 26.0f;
		pushConstants.chunkSize = simulationRenderer->GetSkipGridChunkSize();
		pushConstants.computeNeighborCount = guiState.computeNeighborCount ? 1u : 0u;
		pushConstants.debugRandomCellColors = guiState.debugRandomCellColors ? 1u : 0u;
		pushConstants.meshScaleRatio = meshScaleRatio;
		pushConstants.centeringOffsetX = centeringOffset.x;
		pushConstants.centeringOffsetY = centeringOffset.y;
		pushConstants.centeringOffsetZ = centeringOffset.z;

		// Owned Z range in local cell coords. For single-GPU, full local grid is owned.
		if (simulationRenderer->IsDomainDecomposed())
		{
			uint32_t ghostDepth = simulationRenderer->GetGhostDepth();
			uint32_t ownedZSize = renderGridZ - 2 * ghostDepth;
			pushConstants.ownedZMin = ghostDepth;
			pushConstants.ownedZMax = ghostDepth + ownedZSize;
		}
		else
		{
			pushConstants.ownedZMin = 0;
			pushConstants.ownedZMax = renderGridZ;
		}

		return pushConstants;
	}

	CameraData SimulationContext::BuildCameraData(const GuiState& guiState)
	{
		CameraData cameraData = camera->GetCameraData();
		cameraData.lightDirection = glm::vec4(
			guiState.lightDirection[0], guiState.lightDirection[1], guiState.lightDirection[2],
			guiState.enableLighting ? 1.0f : 0.0f);
		cameraData.lightingParameters = glm::vec4(
			guiState.ambientStrength, guiState.diffuseStrength,
			guiState.specularStrength, guiState.shininess);
		cameraData.cameraPosition = glm::vec4(camera->GetPosition(), 0.0f);

		uint32_t cullingMask = 0;
		for (int planeIndex = 0; planeIndex < 4; planeIndex++)
		{
			if (!guiState.cullingPlaneEnabled[planeIndex])
			{
				cameraData.cullingPlanes[planeIndex] = glm::vec4(0.0f);
				continue;
			}
			glm::vec3 normal(
				guiState.cullingPlaneNormal[planeIndex][0],
				guiState.cullingPlaneNormal[planeIndex][1],
				guiState.cullingPlaneNormal[planeIndex][2]);
			float lengthSquared = glm::dot(normal, normal);
			if (lengthSquared < 1e-8f)
			{
				// All-zero normal — plane has no orientation; skip rather than divide by zero.
				cameraData.cullingPlanes[planeIndex] = glm::vec4(0.0f);
				continue;
			}
			normal = glm::normalize(normal);
			cameraData.cullingPlanes[planeIndex] = glm::vec4(normal, guiState.cullingPlaneOffset[planeIndex]);
			cullingMask |= (1u << planeIndex);
		}
		cameraData.cullingControl = glm::uvec4(cullingMask, 0u, 0u, 0u);

		return cameraData;
	}

} // namespace Cave
