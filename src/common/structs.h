#pragma once

#include <optional>
#include <string>
#include <unordered_map>

#include <vulkan/vulkan.hpp>

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace Cave
{
	struct QueueFamily
	{
		uint32_t QueueCount;
		std::optional<uint32_t> Index;
		float Priority;
	};

	struct QueueFamilies
	{
		QueueFamily ComputeFamily;
		QueueFamily GraphicsFamily;
		QueueFamily PresentFamily;
		QueueFamily VideoEncodeFamily;

		bool IsComplete()
		{
			return ComputeFamily.Index.has_value() && GraphicsFamily.Index.has_value() && PresentFamily.Index.has_value() && VideoEncodeFamily.Index.has_value();
		}
	};

	struct GLTFVertex
	{
		glm::vec3 position;
		glm::vec3 normal;
		glm::vec2 uv;
		glm::vec3 color;

		bool operator==(const GLTFVertex &other) const
		{
			return position == other.position && normal == other.normal && uv == other.uv && color == other.color;
		}
	};

	struct Vertex
	{
		glm::vec4 position{};

		bool operator==(const Vertex &other) const
		{
			return position == other.position;
		}
	};

	struct InstanceCellData
	{
		uint64_t chunk1;
		uint64_t chunk2;

		bool operator==(const InstanceCellData &other) const
		{
			return chunk1 == other.chunk1 && chunk2 == other.chunk2;
		}
	};

	struct ColorRules
	{
		glm::vec4 aliveColor;
		glm::vec4 deadColor;
	};

	struct GridInfo
	{
		uint32_t totalNumberOfCellsThatDied;
		uint32_t totalNumberOfCellsThatWereBorn;
		uint32_t numberOfSimulationTicksSurvived;
		uint32_t status;
		uint32_t maxAliveCount;          // peak alive cell count seen during the run
		uint32_t finalAliveCount;        // alive cell count at the last completed tick
	};

	struct SimulationParameters
	{
		uint64_t birthAndMaxCellStateRules;
		uint64_t survivalAndNeighborhoodRules;

		// Rule-identity metadata — bit layout (LSB to MSB):
		//   bits  0-4  : shape ID   (5 bits, 0-31)
		//   bits  5-63 : reserved   (grid geometry has moved to ChunkConfiguration;
		//                            spawn placement / area dims also live there now)
		uint64_t shapeAndGridConfiguration;
	};

	// Per-chunk configuration for search mode sweeps (and the natural home for grid
	// geometry in rendering / video modes as well). A "chunk" is one (grid, spawn,
	// initial-condition) setting under which all 210 rule permutation slots of a
	// SearchSystem are evaluated. Grid dims are full 32-bit values so axis sizes
	// can scale independently — no more symmetric bit budget forcing X=Y=Z sizing.
	//
	// Uses plain uint32_t fields to avoid GLM_FORCE_DEFAULT_ALIGNED_GENTYPES layout drift
	// across translation units.
	struct ChunkConfiguration
	{
		uint32_t shapeId;                   // 0 = Cube, 1 = ERD, … ; baked into shader neighbor deltas at compile time
		uint32_t gridDimensionX;            // per-axis up to 4B; real cap is Vulkan's maxImageDimension3D = 16384
		uint32_t gridDimensionY;
		uint32_t gridDimensionZ;
		uint32_t spawnAreaDimensionX;       // max spawn region (can be asymmetric relative to grid)
		uint32_t spawnAreaDimensionY;
		uint32_t spawnAreaDimensionZ;
		uint32_t spawnPlacementId;          // center / 8 corners / 6 face centers
		uint32_t spawnRegionShapeId;        // filled cube, random cube, sphere, slab, ring, noise, ...
		uint32_t spawnSizeIndex;            // search-mode sweep axis (0 = default = full spawn region)
		uint32_t spawnDensityIndex;         // search-mode sweep axis (0..15 → density in (1/16, 16/16])
		uint32_t runSeedIndex;              // search-mode sweep axis (per-chunk seed salt for random spawn)
		// Pre-partition the slot's survival rule range upfront (--chunks-per-config). Slot i
		// of N takes survival range [1 + i × maxPermutation/N, (i+1) × maxPermutation/N].
		// partitionCount=1 means no partitioning (default, single chunk per cross-product cell).
		uint32_t partitionIndex = 0;
		uint32_t partitionCount = 1;
	};                                       // 14 × uint32 = 56 bytes; clean std140 / push-constant layout

	struct GridSnapshot
	{
		uint32_t numberOfDeadCells;
		uint32_t numberOfDyingCells;
		uint32_t numberOfCellsBorn;
		uint32_t numberOfSurvivingCells;
	};

	// Configuration for Z-axis domain decomposition across multiple GPUs.
	// When provided to VulkanSimulationRenderer, buffers are sized for the local
	// domain (ghostDepth + ownedZSize + ghostDepth) instead of the full grid Z.
	struct DomainConfig
	{
		uint32_t globalGridDimensionZ; // full global Z dimension
		uint32_t zDomainOffset;        // global Z of the first owned cell in this domain
		uint32_t ownedZSize;           // number of owned Z slices in this domain
		uint32_t ghostDepth;           // ghost slices on each side of the owned range

		uint32_t GetLocalZSize() const { return ghostDepth + ownedZSize + ghostDepth; }
	};

	struct CameraData
	{
		glm::mat4 model;
		glm::mat4 view;
		glm::mat4 projection;
		glm::vec4 lightDirection;      // xyz = direction, w = enableLighting (0.0 or 1.0)
		glm::vec4 lightingParameters;  // x = ambient, y = diffuse, z = specular, w = shininess
		glm::vec4 cameraPosition;      // xyz = camera world position (for specular), w = unused
		glm::vec4 cullingPlanes[4];    // xyz = plane normal (normalized), w = signed offset; cells where dot(normal, worldPos) > offset are hidden
		glm::uvec4 cullingControl;     // x = bitmask of enabled planes (bit i = plane i active), yzw = unused
	};

	struct VideoEncodeStatus
	{
		uint32_t bitstreamStartOffset;
		uint32_t bitstreamSize;
		vk::QueryResultStatusKHR status;
	};

	// Push constants for the graphics pipeline (grid layout + simulation parameters)
	// Uses plain types instead of glm to avoid GLM_FORCE_DEFAULT_ALIGNED_GENTYPES layout mismatches.
	struct GridPushConstants
	{
		uint32_t gridDimensionX;
		uint32_t gridDimensionY;
		uint32_t gridDimensionZ;
		uint32_t totalCellCount;
		float meshScaleRatio;
		float centeringOffsetX;
		float centeringOffsetY;
		float centeringOffsetZ;
		SimulationParameters simulationParameters;
	};

	struct PerformanceTimings
	{
		float computeMs = 0.0f;
		float renderMs = 0.0f;
		float presentMs = 0.0f;
		float searchSimulationMs = 0.0f;
		float searchManagerMs = 0.0f;
		float searchCopyMs = 0.0f;
		float encodeComputeMs = 0.0f;
		float encodeRenderMs = 0.0f;
		float encodePresentMs = 0.0f;
	};

	struct RayMarchComputePushConstants
	{
		SimulationParameters simulationParameters;
		uint32_t enableComputeSkip;
		uint32_t padding0;
	};

	struct RayMarchPushConstants
	{
		uint32_t gridDimensionX;
		uint32_t gridDimensionY;
		uint32_t gridDimensionZ;
		uint32_t maxCellState;
		float volumeMinX;
		float volumeMinY;
		float volumeMinZ;
		float volumeMaxX;
		float volumeMaxY;
		float volumeMaxZ;
		float aliveColorR;
		float aliveColorG;
		float aliveColorB;
		float deadColorR;
		float deadColorG;
		float deadColorB;
		float occlusionStrength;
		float maxNeighborCount;
		uint32_t chunkSize;
		uint32_t computeNeighborCount;
		uint32_t debugRandomCellColors;
		float meshScaleRatio;
		float centeringOffsetX;
		float centeringOffsetY;
		float centeringOffsetZ;
		// Owned Z range in local cell coords. For single-GPU, this is [0, gridDimensionZ).
		// For domain-decomposed rendering, ghost slices (outside this range) must be
		// excluded from hit tests to prevent double-rendering on the domain boundary.
		uint32_t ownedZMin;
		uint32_t ownedZMax; // exclusive upper bound
	};

	enum class OverlayPosition : uint32_t
	{
		TopLeft = 0,
		TopRight = 1,
	};

	// Snapshot of simulation metadata rendered each encoded frame in the corner overlay.
	// Built once per encode tick from the live simulation params + GuiState; consumed by
	// VideoOverlay::SetInfo before the frame is recorded.
	struct OverlayInfo
	{
		uint32_t gridX = 0;
		uint32_t gridY = 0;
		uint32_t gridZ = 0;
		uint32_t spawnX = 0;
		uint32_t spawnY = 0;
		uint32_t spawnZ = 0;
		std::string birthRules;          // e.g. "1,4-6"
		std::string survivalRules;       // e.g. "4"
		std::string shapeName;           // e.g. "cube"
		uint32_t maxCellState = 0;
		bool faceNeighbors = false;
		bool edgeNeighbors = false;
		bool cornerNeighbors = false;
		bool wrapAtBoundary = false;
		uint32_t currentTick = 0;
		uint32_t maxTicks = 0;           // total ticks for this encode job
		uint32_t ticksPerSecond = 0;     // = encoded video FPS (one tick per frame)
		OverlayPosition position = OverlayPosition::TopLeft;
	};

} // namespace Cave

namespace std
{
	template <>
	struct hash<Cave::Vertex>
	{
		size_t operator()(Cave::Vertex const &vertex) const
		{
			return hash<glm::vec4>()(vertex.position);
		}
	};
}