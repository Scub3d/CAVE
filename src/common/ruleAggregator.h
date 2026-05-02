#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>

#include "structs.h"
#include "../simulation/simulation.h"

namespace Cave
{
	// Groups SearchResults across chunks by *rule identity* — same rule evaluated at
	// different (grid, density, seed) combinations lands in the same aggregate entry.
	//
	// Rule identity = (birth bitmask, survival bitmask, maxCellState, neighborhoodFlags, shapeId).
	// Everything else on the SimulationParameters / ChunkConfiguration side is "how we
	// tested this rule" and gets bucketed into the stats for that rule.
	class RuleAggregator
	{
	public:
		enum class Category { Viable, Possible, Unviable };

		struct RuleKey
		{
			uint64_t birthBitmask;
			uint64_t survivalBitmask;
			uint32_t maxCellState;
			uint32_t neighborhoodFlags;
			uint32_t shapeId;

			bool operator==(const RuleKey& other) const
			{
				return birthBitmask       == other.birthBitmask
					&& survivalBitmask    == other.survivalBitmask
					&& maxCellState       == other.maxCellState
					&& neighborhoodFlags  == other.neighborhoodFlags
					&& shapeId            == other.shapeId;
			}
		};

		struct RuleKeyHash
		{
			size_t operator()(const RuleKey& k) const noexcept
			{
				// Mix the rule bits into a single 64-bit hash. Birth and survival already
				// cover the bulk of entropy; the remaining uint32 fields salt the hash.
				size_t h = std::hash<uint64_t>{}(k.birthBitmask);
				h ^= std::hash<uint64_t>{}(k.survivalBitmask) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
				h ^= std::hash<uint32_t>{}(k.maxCellState) + 0x9e3779b9u + (h << 6) + (h >> 2);
				h ^= std::hash<uint32_t>{}(k.neighborhoodFlags) + 0x9e3779b9u + (h << 6) + (h >> 2);
				h ^= std::hash<uint32_t>{}(k.shapeId) + 0x9e3779b9u + (h << 6) + (h >> 2);
				return h;
			}
		};

		struct RuleStats
		{
			uint32_t totalRuns = 0;
			uint32_t viableRuns = 0;
			uint32_t possibleRuns = 0;
			uint32_t unviableRuns = 0;
			uint32_t maxTicksSurvived = 0;
			uint32_t minTicksSurvived = UINT32_MAX;
			uint64_t sumTicksSurvived = 0;
		};

		// Record one SearchResult into the aggregate. Called once per (rule × chunk) result.
		void Record(const SimulationParameters& params, uint32_t ticksSurvived, Category category, uint32_t shapeId)
		{
			RuleKey key{
				params.birthAndMaxCellStateRules    & Simulation::EXISTENCE_PERMUTATION_BIT_MASK,
				params.survivalAndNeighborhoodRules & Simulation::EXISTENCE_PERMUTATION_BIT_MASK,
				static_cast<uint32_t>(Simulation::DecodeMaxCellState(params.birthAndMaxCellStateRules >> Simulation::BIT_SHIFT)),
				static_cast<uint32_t>(params.survivalAndNeighborhoodRules >> Simulation::BIT_SHIFT),
				shapeId
			};

			RuleStats& stats = _stats[key];
			stats.totalRuns++;
			switch (category)
			{
				case Category::Viable:   stats.viableRuns++;   break;
				case Category::Possible: stats.possibleRuns++; break;
				case Category::Unviable: stats.unviableRuns++; break;
			}
			if (ticksSurvived > stats.maxTicksSurvived) stats.maxTicksSurvived = ticksSurvived;
			if (ticksSurvived < stats.minTicksSurvived) stats.minTicksSurvived = ticksSurvived;
			stats.sumTicksSurvived += ticksSurvived;
		}

		size_t UniqueRuleCount() const { return _stats.size(); }

		// Iterate every (rule, stats) pair. Provided so visualization code can read
		// the aggregate without needing access to the underlying container type.
		template <typename Func>
		void ForEachRule(Func&& func) const
		{
			for (const auto& [key, stats] : _stats)
				func(key, stats);
		}

		// Write one JSONL line per unique rule with its aggregated stats. Uses the same
		// atomic-single-line-per-rule format as per-chunk JSONL files.
		void WriteJsonl(const std::string& filePath) const
		{
			std::ofstream output(filePath, std::ios::out | std::ios::trunc);
			if (!output.is_open()) return;

			for (const auto& [key, stats] : _stats)
			{
				float averageTicks = stats.totalRuns > 0
					? static_cast<float>(stats.sumTicksSurvived) / static_cast<float>(stats.totalRuns)
					: 0.0f;
				float viabilityRate = stats.totalRuns > 0
					? static_cast<float>(stats.viableRuns) / static_cast<float>(stats.totalRuns)
					: 0.0f;

				output
					<< "{\"birth\":" << key.birthBitmask
					<< ",\"survival\":" << key.survivalBitmask
					<< ",\"maxCellState\":" << key.maxCellState
					<< ",\"neighborhoodFlags\":" << key.neighborhoodFlags
					<< ",\"shapeId\":" << key.shapeId
					<< ",\"totalRuns\":" << stats.totalRuns
					<< ",\"viableRuns\":" << stats.viableRuns
					<< ",\"possibleRuns\":" << stats.possibleRuns
					<< ",\"unviableRuns\":" << stats.unviableRuns
					<< ",\"viabilityRate\":" << viabilityRate
					<< ",\"minTicksSurvived\":" << (stats.minTicksSurvived == UINT32_MAX ? 0 : stats.minTicksSurvived)
					<< ",\"maxTicksSurvived\":" << stats.maxTicksSurvived
					<< ",\"avgTicksSurvived\":" << averageTicks
					<< "}\n";
			}
		}

		void Clear() { _stats.clear(); }

	private:
		std::unordered_map<RuleKey, RuleStats, RuleKeyHash> _stats;
	};
}
