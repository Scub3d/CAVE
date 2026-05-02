#pragma once

#include <vector>
#include <fstream>
#include <sstream>

#include "pipeline.h"
#include "../shader.h"
#include "../deviceContext.h"

namespace Cave
{

class ComputePipeline : public Pipeline
{
private:
	std::shared_ptr<Shader> _shader;

	vk::ComputePipelineCreateInfo _computePipelineCreateInfo;
	vk::PipelineShaderStageCreateInfo _pipelineShaderStageCreateInfo;

	void PopulateComputePipelineCreateInfo();

public:
	explicit ComputePipeline(DeviceContext &deviceContext, std::shared_ptr<Shader> shader);
	void Build() override;
};

} // namespace Cave