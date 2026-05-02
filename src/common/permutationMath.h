#pragma once

#include <cstdint>
#include <algorithm>
#include <functional>
#include <vector>
#include "../shapes/shape.h"

namespace Cave::PermutationMath
{
	// Number of (birth, survival) rule pairs representable with the given neighbor count.
	//
	// A rule bitmask is a uint with `neighborCount` meaningful bits. The all-zero bitmask
	// is degenerate (no birth or no survival count ever matches), so search treats values
	// 1..(2^n - 1) as the iteration range, giving (2^n - 1)^2 rule pairs per (maxCS, neighborhood) slot.
	constexpr uint64_t RulePermutationCount(uint32_t neighborCount)
	{
		if (neighborCount == 0) return 0;
		if (neighborCount >= 60) neighborCount = 60; // saturate at the 60-bit rule mask
		uint64_t rulesPerAxis = (uint64_t(1) << neighborCount) - 1;
		return rulesPerAxis * rulesPerAxis;
	}

	// Rule-pair count for a specific (shape, F, E, C) neighborhood selection.
	inline uint64_t RulePermutationCount(const Shape &shape, bool faces, bool edges, bool corners)
	{
		return RulePermutationCount(shape.GetNeighborCount(faces, edges, corners));
	}

	// Number of parallel SearchSystem slots that will be spawned for the given sweep.
	// Each slot runs one (maxCS, neighborhoodFlags) combo and iterates its rule range.
	//   neighborhoodConfigCount = (2^selected - 1) × wrapMultiplier
	//     where selected = count({F, E, C} that are true), wrapMultiplier = 2 if wrap else 1
	//
	// Precondition: at least one of faces/edges/corners is true. Returns 0 if none.
	inline uint32_t SearchSlotCount(int minMaxCellState, int maxMaxCellState,
		bool faces, bool edges, bool corners, bool wrap)
	{
		int maxCellStateCount = maxMaxCellState - minMaxCellState + 1;
		if (maxCellStateCount <= 0) return 0;

		int selected = (faces ? 1 : 0) + (edges ? 1 : 0) + (corners ? 1 : 0);
		if (selected == 0) return 0;

		int neighborhoodConfigCount = (1 << selected) - 1;
		if (wrap) neighborhoodConfigCount *= 2;

		return static_cast<uint32_t>(maxCellStateCount * neighborhoodConfigCount);
	}

	// Total number of rule permutations the search will evaluate for the given sweep —
	// summed across all (maxCS, neighborhoodFlags) slots.
	//
	// The neighbor count varies per neighborhoodFlags combo (e.g., F only = 6 neighbors
	// on cube; F+E = 18; F+E+C = 26). We iterate every valid combination and sum.
	inline uint64_t SearchTotalPermutationCount(const Shape &shape,
		int minMaxCellState, int maxMaxCellState,
		bool facesEnabled, bool edgesEnabled, bool cornersEnabled, bool wrapEnabled)
	{
		int maxCellStateCount = maxMaxCellState - minMaxCellState + 1;
		if (maxCellStateCount <= 0) return 0;

		uint64_t total = 0;

		// Iterate over all non-empty subsets of {F, E, C} that intersect the user selection,
		// matching how SearchMode builds its gridConfigs list.
		for (int face = 0; face <= (facesEnabled ? 1 : 0); face++)
		for (int edge = 0; edge <= (edgesEnabled ? 1 : 0); edge++)
		for (int corner = 0; corner <= (cornersEnabled ? 1 : 0); corner++)
		{
			if (!face && !edge && !corner) continue;
			uint64_t pairs = RulePermutationCount(shape,
				face != 0, edge != 0, corner != 0);
			total += pairs;
		}

		total *= static_cast<uint64_t>(maxCellStateCount);
		if (wrapEnabled) total *= 2;

		return total;
	}

	// Convenience: "how many rule permutations does this shape have across every possible
	// neighborhood config and wrap option?" — useful as a headline number (e.g., "Cube has
	// N reachable rules") without sweeping maxCS.
	inline uint64_t TotalPermutationsForShape(const Shape &shape)
	{
		return SearchTotalPermutationCount(shape,
			/* minMaxCellState */ 1, /* maxMaxCellState */ 1,
			/* F */ true, /* E */ true, /* C */ true, /* wrap */ true);
	}

	// Enumerate all bitmasks with 1..maxBits bits set drawn from the low N bits,
	// sorted ascending. Returns a fresh vector each call. Typical sizes:
	//   N=6  K=2:    21 entries
	//   N=14 K=2:   105 entries
	//   N=26 K=2:   351 entries
	//   N=26 K=4: 17901 entries
	// Used by work-stealing to pick a split point that guarantees both halves of the
	// stolen range contain valid bitmasks, preventing livelock on sparse-K configs.
	inline std::vector<uint64_t> EnumerateValidBitmasks(uint32_t neighborCount, uint32_t maxBits)
	{
		std::vector<uint64_t> out;
		if (maxBits == 0 || neighborCount == 0) return out;
		uint32_t effectiveMaxBits = std::min(maxBits, neighborCount);

		// Recursive lambda — pick `remainingBits` more bit positions ≥ `startIndex`
		// to append to the accumulator.
		std::function<void(uint64_t, uint32_t, uint32_t)> emit =
			[&](uint64_t acc, uint32_t startIndex, uint32_t remainingBits)
		{
			if (remainingBits == 0)
			{
				if (acc != 0) out.push_back(acc);
				return;
			}
			for (uint32_t i = startIndex; i + remainingBits <= neighborCount + 1 - 1; i++)
			{
				emit(acc | (uint64_t(1) << i), i + 1, remainingBits - 1);
			}
		};

		for (uint32_t p = 1; p <= effectiveMaxBits; p++)
			emit(0, 0, p);

		std::sort(out.begin(), out.end());
		return out;
	}

	// Estimate the VRAM a single SearchSystem instance will allocate on its device.
	// Used by the chunk scheduler to decide how many concurrent chunks fit in the
	// user's GPU budget.
	//
	// Breakdown per slot (3 big buffers + several tiny):
	//   cellsIn       = 4 bytes × ceil(gridVolume / 8)   (4-bit packed, 8 cells per uint32_t)
	//   cellsOut      = 4 bytes × ceil(gridVolume / 8)
	//   initialCells  = 4 bytes × ceil(gridVolume / 8)
	//   small buffers = ~120 bytes (permutations / snapshot / info / copies)
	// Plus per-SearchSystem overhead for pipeline objects, descriptor pools, command
	// buffers. Call that ~16 MB based on measurements in earlier sessions.
	//
	// slotCount is typically ≤ 210 (14 neighborhood × 15 maxCS combos) but depends on
	// the user's sweep scope; the caller passes the actual count.
	inline uint64_t EstimateChunkVramBytes(uint32_t gridX, uint32_t gridY, uint32_t gridZ,
		uint32_t slotCount)
	{
		uint64_t gridVolume = uint64_t(gridX) * uint64_t(gridY) * uint64_t(gridZ);
		uint64_t packedUints = (gridVolume + 7) / 8;
		uint64_t perSlotBig = uint64_t(3) * uint64_t(4) * packedUints; // 3 × 4B × packedUints
		uint64_t perSlotSmall = 128;
		uint64_t systemOverhead = uint64_t(16) * 1024 * 1024; // ~16 MB pipeline + descriptors
		return (perSlotBig + perSlotSmall) * uint64_t(slotCount) + systemOverhead;
	}
}
