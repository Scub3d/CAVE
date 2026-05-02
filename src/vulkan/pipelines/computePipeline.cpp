#include "computePipeline.h"
#include "../../common/logger.h"

namespace Cave
{

ComputePipeline::ComputePipeline(DeviceContext &deviceContext, std::shared_ptr<Shader> shader) : Pipeline(deviceContext), _shader(std::move(shader))
{
}

void ComputePipeline::Build()
{
	LOG_DEBUG("Building Vulkan Compute Pipeline");

	BuildPipelineLayout();

	_pipelineShaderStageCreateInfo = PopulatePipelineShaderStageCreateInfo(_shader->GetShaderModule(), vk::ShaderStageFlagBits::eCompute);

	PopulateComputePipelineCreateInfo();

	try
	{
		_pipeline = (_deviceContext.GetDevice().createComputePipeline(nullptr, _computePipelineCreateInfo)).value;
		LOG_INFO("Compute pipeline created: {}", _pipeline ? "OK" : "FAILED");
	}
	catch (vk::SystemError error)
	{
		LOG_ERROR("Failed To Create Vulkan Compute Pipeline: {}", error.what());
	}
}

void ComputePipeline::PopulateComputePipelineCreateInfo()
{
	LOG_DEBUG("Populating Vulkan Compute Pipeline Create Info");

	_computePipelineCreateInfo = vk::ComputePipelineCreateInfo(
		vk::PipelineCreateFlags(),		// flags
		_pipelineShaderStageCreateInfo, // stage
		_pipelineLayout,				// layout
		nullptr,						// basePipelineHandle
		-1								// basePipelineIndex
	);
}

} // namespace Cave
