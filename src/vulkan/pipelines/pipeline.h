#pragma once

#include <memory>
#include <map>

#include "../deviceContext.h"
#include "../descriptors.h"

#include "../../common/structs.h"
#include "../../common/logger.h"

namespace Cave
{

class Pipeline
{
public:
	struct DescriptorOption
	{
		bool _multiple;
		uint32_t _value;
		std::vector<uint32_t> _values;

		DescriptorOption(uint32_t value) : _multiple(false), _value(value) {}
		DescriptorOption(std::vector<uint32_t> values) : _multiple(true), _values(std::move(values)) {}
		[[nodiscard]] uint32_t Get(size_t index) const
		{
			if (_multiple)
			{
				return _values[index];
			}

			return _value;
		}
	};

protected:
	DeviceContext &_deviceContext;
	vk::Pipeline _pipeline;
	vk::PipelineLayout _pipelineLayout;
	std::vector<vk::DescriptorSetLayoutBinding> _descriptorSetLayoutBindings;
	std::vector<vk::PushConstantRange> _pushConstantRanges;
	std::map<uint32_t, std::shared_ptr<Descriptor>> _descriptors;

	void BuildPipelineLayout();

	vk::PipelineShaderStageCreateInfo PopulatePipelineShaderStageCreateInfo(const vk::ShaderModule &shaderModule, const vk::ShaderStageFlagBits &shaderStageFlagBits)
	{
		LOG_DEBUG("Populating Vulkan Pipeline Shader Stage Create Info");
		return vk::PipelineShaderStageCreateInfo(
			vk::PipelineShaderStageCreateFlags(), // flags
			shaderStageFlagBits,				  // stage
			shaderModule,						  // module
			"main"								  // pName
		);
	}

public:
	explicit Pipeline(DeviceContext &deviceContext);
	virtual ~Pipeline();
	Pipeline(const Pipeline &) = delete;
	Pipeline(Pipeline &&) = delete;
	Pipeline &operator=(const Pipeline &) = delete;
	Pipeline &operator=(Pipeline &&) = delete;

	void AddDescriptorSet(uint32_t set, std::shared_ptr<Descriptor> descriptor);
	void AddPushConstant(vk::ShaderStageFlags stageFlags, uint32_t offset, uint32_t size);

	virtual void Build() = 0;
	virtual void Bind(const vk::CommandBuffer &commandBuffer, vk::PipelineBindPoint pipelineBindPoint, uint8_t currentFrame, DescriptorOption option);

	vk::Pipeline GetPipeline() const { return _pipeline; }
	vk::PipelineLayout GetPipelineLayout() { return _pipelineLayout; }
};

} // namespace Cave
