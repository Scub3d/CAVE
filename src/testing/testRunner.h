#pragma once

#include "../modes/applicationModeHandler.h"

#include <string>
#include <vector>

namespace Cave
{
	class TestRunner
	{
	public:
		explicit TestRunner(ModeServices& services);
		int RunTests(const std::string& testName);

	private:
		struct TickTiming { float computeMs; float renderMs; };
		struct TestResult { std::string name; bool passed; std::string summary; };

		ModeServices& _services;

		std::vector<TickTiming> RunSimulationTicks(int gridSize, int spawnSize, int ticks,
			const std::string& birth, const std::string& survival, int maxCellState,
			const std::string& neighborhood, bool enableComputeSkip);
		TestResult RunComputeSkipTest();
		TestResult RunSparseBenchmarkTest();
		TestResult RunEncodeBenchmarkTest();
		TestResult RunRenderConsistencyTest();
		TestResult RunLargeGridTest();
		TestResult RunCliRulesTest();
		TestResult RunGhostExchangeTest();
	};

} // namespace Cave
