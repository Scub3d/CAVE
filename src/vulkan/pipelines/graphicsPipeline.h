#pragma once

#include <vector>
#include <fstream>
#include <sstream>

#include "pipeline.h"
#include "../shader.h"
#include "../deviceContext.h"
#include "../objMesh.h"
#include "../../common/structs.h"

namespace Cave
{
	class GraphicsPipeline : public Pipeline
	{
	private:
		std::shared_ptr<Shader> _vertexShader, _fragmentShader;

		std::vector<vk::VertexInputAttributeDescription> _vertexInputAttributeDescriptions;
		std::vector<vk::PipelineShaderStageCreateInfo> _pipelineShaderStageCreateInfo;

		vk::GraphicsPipelineCreateInfo _graphicsPipelineCreateInfo;

		std::vector<vk::VertexInputBindingDescription> _vertexInputBindingDescriptions;

		vk::PipelineShaderStageCreateInfo _vertexPipelineShaderStageCreateInfo, _fragmentPipelineShaderStageCreateInfo;

		vk::PipelineVertexInputStateCreateInfo _pipelineVertexInputStateCreateInfo;
		vk::PipelineInputAssemblyStateCreateInfo _pipelineInputAssemblyStateCreateInfo;
		vk::PipelineViewportStateCreateInfo _pipelineViewportStateCreateInfo;
		vk::PipelineRasterizationStateCreateInfo _pipelineRasterizationStateCreateInfo;
		vk::PipelineDepthStencilStateCreateInfo _pipelineDepthStencilStateCreateInfo;
		vk::PipelineMultisampleStateCreateInfo _pipelineMultisampleStateCreateInfo;
		vk::PipelineColorBlendAttachmentState _pipelineColorBlendAttachmentState;
		vk::PipelineColorBlendStateCreateInfo _pipelineColorBlendStateCreateInfo;
		vk::PipelineDynamicStateCreateInfo _dynamicStateCreateInfo;

		std::vector<vk::DynamicState> _dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};

		vk::CullModeFlagBits _cullMode = vk::CullModeFlagBits::eBack;
		bool _blendEnabled = true;

		vk::Format _dynamicColorFormat = vk::Format::eUndefined;
		vk::Format _dynamicDepthFormat = vk::Format::eUndefined;
		vk::PipelineRenderingCreateInfo _pipelineRenderingCreateInfo;

		void PopulatePipelineDepthStencilStateCreateInfo();
		void PopulatePipelineVertexInputStateCreateInfo();
		void PopulatePipelineInputAssemblyStateCreateInfo();
		void PopulatePipelineViewportStateCreateInfo();
		void PopulatePipelineRasterizationStateCreateInfo();
		void PopulatePipelineMultisampleStateCreateInfo();
		void PopulatePipelineColorBlendAttachmentState();
		void PopulatePipelineColorBlendStateCreateInfo();
		void PopulatePipelineDynamicStateCreateInfo();

	public:
		explicit GraphicsPipeline(DeviceContext &deviceContext, std::shared_ptr<Shader> vertexShader, std::shared_ptr<Shader> fragmentShader);
		void Build() override;

		// Call before Build() to configure vertex input for the simulation pipeline.
		void SetVertexInput(std::vector<vk::VertexInputBindingDescription> bindings,
		                    std::vector<vk::VertexInputAttributeDescription> attributes);

		// Call before Build() to set the color and depth formats for dynamic rendering.
		void SetDynamicRenderingInfo(vk::Format colorFormat, vk::Format depthFormat);

		// Call before Build() to override the default back-face culling.
		void SetCullMode(vk::CullModeFlagBits cullMode) { _cullMode = cullMode; }

		// Call before Build() to disable alpha blending. Required for shaders that encode
		// non-color data (e.g., distance) in the alpha channel — otherwise the default
		// eSrcAlpha/eOneMinusSrcAlpha blend treats that data as an opacity multiplier.
		void SetBlendEnabled(bool enabled) { _blendEnabled = enabled; }

		void AddDescriptorSetLayout(vk::DescriptorSetLayout descriptorSetLayout);

		vk::Pipeline GetPipeline() const { return _pipeline; }
		vk::PipelineLayout GetPipelineLayout() const { return _pipelineLayout; }
	};

} // namespace Cave
