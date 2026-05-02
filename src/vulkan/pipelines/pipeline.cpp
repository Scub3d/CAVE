#include "pipeline.h"
#include "../../common/logger.h"

#include <utility>

namespace Cave
{

Pipeline::Pipeline(DeviceContext &deviceContext) : _deviceContext{deviceContext}
{
}

Pipeline::~Pipeline()
{
	if (_pipeline)
		_deviceContext.GetDevice().destroyPipeline(_pipeline);
	if (_pipelineLayout)
		_deviceContext.GetDevice().destroyPipelineLayout(_pipelineLayout);
}

void Pipeline::BuildPipelineLayout()
{
	LOG_DEBUG("Building vulkan pipeline layout");

	std::vector<vk::DescriptorSetLayout> descriptorSetLayouts;
	descriptorSetLayouts.reserve(_descriptors.size());

	for (auto &descriptorSet : _descriptors)
	{
		descriptorSetLayouts.push_back(descriptorSet.second->GetDescriptorSetLayout());
	}

	vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo = vk::PipelineLayoutCreateInfo(
		vk::PipelineLayoutCreateFlags(),					// flags
		static_cast<uint32_t>(descriptorSetLayouts.size()), // setLayoutCount
		descriptorSetLayouts.data(),						// pSetLayouts
		0,													// pushConstantRangeCount
		nullptr												// pPushConstantRanges
	);

	if (!_pushConstantRanges.empty())
	{
		pipelineLayoutCreateInfo.setPushConstantRangeCount(_pushConstantRanges.size());
		pipelineLayoutCreateInfo.setPPushConstantRanges(_pushConstantRanges.data());
	}

	try
	{
		_pipelineLayout = _deviceContext.GetDevice().createPipelineLayout(pipelineLayoutCreateInfo);
	}
	catch (vk::SystemError error)
	{
		LOG_ERROR("Failed to build vulkan pipeline layout");
	}
}

void Pipeline::Bind(const vk::CommandBuffer &commandBuffer, vk::PipelineBindPoint pipelineBindPoint, uint8_t currentFrame, DescriptorOption option)
{
	commandBuffer.bindPipeline(pipelineBindPoint, _pipeline);

	if (_descriptors.empty())
		return;

	std::vector<vk::DescriptorSet> descriptorSetsToBind;
	descriptorSetsToBind.reserve(_descriptors.size());
	auto descriptorSetIndex = 0;
	for (auto &descriptor : _descriptors)
	{
		descriptorSetsToBind.push_back(descriptor.second->GetDescriptorSet(currentFrame, option.Get(descriptorSetIndex++)));
	}

	commandBuffer.bindDescriptorSets(pipelineBindPoint, _pipelineLayout, 0, descriptorSetsToBind, nullptr);
}

void Pipeline::AddDescriptorSet(uint32_t set, std::shared_ptr<Descriptor> descriptor)
{
	_descriptors[set] = std::move(descriptor);
}

void Pipeline::AddPushConstant(vk::ShaderStageFlags stageFlags, uint32_t offset, uint32_t size)
{
	_pushConstantRanges.emplace_back(stageFlags, offset, size);
}

} // namespace Cave