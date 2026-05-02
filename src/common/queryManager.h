#pragma once

#include <mutex>
#include <unordered_map>
#include <vector>
#include <string>

#include <vulkan/vulkan.hpp>

namespace Cave
{

class QueryManager
{
private:
	std::mutex _mutex;
	std::unordered_map<std::string, uint32_t> _registry;
	std::unordered_map<std::string, float> _smoothedTimings;
	float _timestampPeriod = 1.0f;
	float _smoothingFactor = 0.1f;
	int _nextID = 0;

public:
	void SetTimestampPeriod(float period) { _timestampPeriod = period; }

	uint32_t RegisterQuery(const std::string &name)
	{
		std::lock_guard<std::mutex> lock(_mutex);
		if (!_registry.contains(name))
		{
			_registry[name] = _nextID++;
		}
		return _registry[name];
	}

	[[nodiscard]] uint32_t GetQueryId(const std::string &name)
	{
		std::lock_guard<std::mutex> lock(_mutex);
		if (_registry.contains(name))
		{
			return _registry.at(name);
		}
		return 0;
	}

	void WriteTimestamp(vk::CommandBuffer commandBuffer, vk::QueryPool queryPool,
	                    vk::PipelineStageFlagBits stage, const std::string &name)
	{
		uint32_t queryId = RegisterQuery(name);
		commandBuffer.writeTimestamp(stage, queryPool, queryId);
	}

	std::unordered_map<std::string, float> GetTimingsMs(const std::vector<uint64_t> &rawTimestamps)
	{
		std::lock_guard<std::mutex> lock(_mutex);
		std::unordered_map<std::string, float> timingsMs;

		for (auto &[name, id] : _registry)
		{
			if (name.ends_with("_start"))
			{
				auto endName = name.substr(0, name.size() - 5) + "end";
				if (_registry.contains(endName))
				{
					uint32_t startId = id;
					uint32_t endId = _registry[endName];
					if (startId < rawTimestamps.size() && endId < rawTimestamps.size())
					{
						uint64_t start = rawTimestamps[startId];
						uint64_t end = rawTimestamps[endId];
						if (end >= start && start > 0)
						{
							float ms = static_cast<float>(end - start) * _timestampPeriod / 1e6f;
							auto truncated = name.substr(0, name.size() - 6);
							timingsMs[truncated] = ms;

							// Exponential moving average for smoothed display
							if (_smoothedTimings.contains(truncated))
								_smoothedTimings[truncated] = _smoothedTimings[truncated] * (1.0f - _smoothingFactor) + ms * _smoothingFactor;
							else
								_smoothedTimings[truncated] = ms;
						}
					}
				}
			}
		}

		return timingsMs;
	}

	std::unordered_map<std::string, float> GetSmoothedTimings() const
	{
		return _smoothedTimings;
	}

	int GetNextID() { return _nextID; }
};

} // namespace Cave
