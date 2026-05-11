// Search-mode manager compute shader. One thread per dispatch.
//
// Each tick: reads the simulation's per-tick GridSnapshot, accumulates running
// totals into GridInfo, decides if the current rule died/survived/still-running,
// copies state out for the CPU, and (when a rule completes) advances to the
// next permutation in the (birth, survival) iteration space.
//
// Required defines (injected via SlangCompiler):
//   GRID_X_SIZE, GRID_Y_SIZE, GRID_Z_SIZE   — grid dimensions
//   MAX_NUMBER_OF_TICKS_TO_SURVIVE          — tick threshold for "viable"
//   MAX_RULE_BITS                           — 0 or K (K-bit cap on rule popcount)
//
// IMPORTANT — Slang 64-bit literal gotcha: `0xF000000000000000UL` truncates to
// uint32 (=0). Always build uint64 constants from bit shifts or explicit casts.

static const uint TOTAL_NUMBER_OF_CELLS = GRID_X_SIZE * GRID_Y_SIZE * GRID_Z_SIZE;

static const uint64_t PERMUTATION_MASK          = (uint64_t(1) << 60) - uint64_t(1);
static const uint64_t PERMUTATION_INVERTED_MASK = uint64_t(0xF) << 60;
static const uint64_t NULL_MASK                 = uint64_t(0);

struct GridSnapshot
{
    uint numberOfDeadCells;
    uint numberOfDyingCells;
    uint numberOfCellsBorn;
    uint numberOfSurvivingCells;
};

struct GridInfo
{
    uint totalNumberOfCellsThatDied;
    uint totalNumberOfCellsThatWereBorn;
    uint numberOfSimulationTicksSurvived;
    uint status;
    uint maxAliveCount;
    uint finalAliveCount;
};

struct GridPermutations
{
    uint64_t birthAndMaxCellStateRules;
    uint64_t survivalAndNeighborhoodRules;
};

[vk::binding(0, 0)] RWStructuredBuffer<uint>             initialCells;
[vk::binding(1, 0)] RWStructuredBuffer<uint>             cells;
[vk::binding(2, 0)] RWStructuredBuffer<GridSnapshot>     gridSnapshotBuffer;
[vk::binding(3, 0)] RWStructuredBuffer<uint64_t>         gridMaxPermutationBuffer;
[vk::binding(4, 0)] RWStructuredBuffer<GridPermutations> gridPermutationsBuffer;
[vk::binding(5, 0)] RWStructuredBuffer<GridPermutations> gridPermutationsCopyBuffer;
[vk::binding(6, 0)] RWStructuredBuffer<GridInfo>         gridInfoBuffer;
[vk::binding(7, 0)] RWStructuredBuffer<GridInfo>         gridInfoCopyBuffer;
[vk::binding(8, 0)] RWStructuredBuffer<uint64_t>         gridSurvivalEndPermutationBuffer;

void NextPermutation()
{
    GridPermutations permutations = gridPermutationsBuffer[0];
    if (permutations.survivalAndNeighborhoodRules == NULL_MASK
        || permutations.birthAndMaxCellStateRules == NULL_MASK)
    {
        return;
    }

    uint64_t maxPermutation         = gridMaxPermutationBuffer[0];
    uint64_t survivalEndPermutation = gridSurvivalEndPermutationBuffer[0];

    if ((permutations.birthAndMaxCellStateRules & PERMUTATION_MASK) < maxPermutation)
    {
        permutations.birthAndMaxCellStateRules++;
    }
    else if ((permutations.survivalAndNeighborhoodRules & PERMUTATION_MASK) < survivalEndPermutation)
    {
        permutations.survivalAndNeighborhoodRules++;
        permutations.birthAndMaxCellStateRules =
            (permutations.birthAndMaxCellStateRules & PERMUTATION_INVERTED_MASK) + uint64_t(1);
    }
    else
    {
        permutations.survivalAndNeighborhoodRules = NULL_MASK;
        permutations.birthAndMaxCellStateRules    = NULL_MASK;
        gridPermutationsBuffer[0] = permutations;
        return;
    }

    gridPermutationsBuffer[0] = permutations;

    GridInfo freshInfo;
    freshInfo.totalNumberOfCellsThatDied      = 0;
    freshInfo.totalNumberOfCellsThatWereBorn  = 0;
    freshInfo.numberOfSimulationTicksSurvived = 0;
    freshInfo.status                          = 0;
    freshInfo.maxAliveCount                   = 0;
    freshInfo.finalAliveCount                 = 0;
    gridInfoBuffer[0] = freshInfo;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{
    GridPermutations permutations = gridPermutationsBuffer[0];

    if (permutations.survivalAndNeighborhoodRules == NULL_MASK
        || permutations.birthAndMaxCellStateRules == NULL_MASK)
    {
        gridInfoCopyBuffer[0]         = gridInfoBuffer[0];
        gridPermutationsCopyBuffer[0] = permutations;
        return;
    }

    GridInfo info         = gridInfoBuffer[0];
    GridSnapshot snapshot = gridSnapshotBuffer[0];

    info.numberOfSimulationTicksSurvived++;
    info.totalNumberOfCellsThatDied     += snapshot.numberOfDeadCells;
    info.totalNumberOfCellsThatWereBorn += snapshot.numberOfCellsBorn;
    info.finalAliveCount                 = snapshot.numberOfSurvivingCells;
    info.maxAliveCount                   = max(info.maxAliveCount, snapshot.numberOfSurvivingCells);

    // Early-death detection: a rule is doomed once no cells reach `maxCellState`
    // (survivedCount==0) AND no new cells were born (bornCount==0) on a single tick.
    // From that tick onward, all surviving cells just decay through 1..maxCS-1 to 0
    // — no future ticks can produce alive neighbors, so no future births. The original
    // check (numberOfDeadCells == TOTAL_NUMBER_OF_CELLS) only fired after the full
    // decay tail completed (`maxCellState` extra ticks of bookkeeping), which let
    // rules in their decay tail reach MAX_TICKS and get misclassified as VIABLE.
    if (snapshot.numberOfSurvivingCells == 0u && snapshot.numberOfCellsBorn == 0u)
    {
        info.status = 1;
    }
    else if (info.numberOfSimulationTicksSurvived >= MAX_NUMBER_OF_TICKS_TO_SURVIVE)
    {
        info.status = 2;
    }
    else
    {
        info.status = 0;
    }

    gridInfoBuffer[0]             = info;
    gridInfoCopyBuffer[0]         = info;
    gridPermutationsCopyBuffer[0] = permutations;

    if (info.status > 0)
    {
        NextPermutation();
    }
}
