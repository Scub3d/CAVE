#include "graphicsPipeline.h"
#include "../../common/logger.h"

namespace Cave
{

GraphicsPipeline::GraphicsPipeline(DeviceContext &deviceContext, std::shared_ptr<Shader> vertexShader, std::shared_ptr<Shader> fragmentShader) : Pipeline(deviceContext), _vertexShader{std::move(vertexShader)}, _fragmentShader{std::move(fragmentShader)}
{
}

void GraphicsPipeline::SetVertexInput(std::vector<vk::VertexInputBindingDescription> bindings,
                                      std::vector<vk::VertexInputAttributeDescription> attributes)
{
	_vertexInputBindingDescriptions   = std::move(bindings);
	_vertexInputAttributeDescriptions = std::move(attributes);
}

void GraphicsPipeline::SetDynamicRenderingInfo(vk::Format colorFormat, vk::Format depthFormat)
{
	_dynamicColorFormat  = colorFormat;
	_dynamicDepthFormat  = depthFormat;
}

void GraphicsPipeline::PopulatePipelineVertexInputStateCreateInfo()
{
	LOG_DEBUG("Populating Vulkan PipelineVertexInputStateCreateInfo");

	// If no custom vertex input was configured, fall back to GLTFVertex defaults.
	if (_vertexInputBindingDescriptions.empty())
	{
		_vertexInputBindingDescriptions = {
			vk::VertexInputBindingDescription(
				0,                            // binding
				sizeof(GLTFVertex),           // stride
				vk::VertexInputRate::eVertex  // inputRate
			)};
		_vertexInputAttributeDescriptions = {
			vk::VertexInputAttributeDescription(
				0,                                // location
				0,                                // binding
				vk::Format::eR32G32B32Sfloat,     // format
				offsetof(GLTFVertex, position)    // offset
			),
			vk::VertexInputAttributeDescription(
				1,                                // location
				0,                                // binding
				vk::Format::eR32G32B32Sfloat,     // format
				offsetof(GLTFVertex, normal)      // offset
			),
			vk::VertexInputAttributeDescription(
				2,                                // location
				0,                                // binding
				vk::Format::eR32G32B32Sfloat,     // format
				offsetof(GLTFVertex, uv)          // offset
			),
			vk::VertexInputAttributeDescription(
				3,                                // location
				0,                                // binding
				vk::Format::eR32G32B32Sfloat,     // format
				offsetof(GLTFVertex, color)       // offset
			)};
	}

	_pipelineVertexInputStateCreateInfo = vk::PipelineVertexInputStateCreateInfo(
		vk::PipelineVertexInputStateCreateFlags(),                          // flags
		static_cast<uint32_t>(_vertexInputBindingDescriptions.size()),      // vertexBindingDescriptionCount
		_vertexInputBindingDescriptions.data(),                             // pVertexBindingDescriptions
		static_cast<uint32_t>(_vertexInputAttributeDescriptions.size()),    // vertexAttributeDescriptionCount
		_vertexInputAttributeDescriptions.data()                            // pVertexAttributeDescriptions
	);
}

void GraphicsPipeline::PopulatePipelineDepthStencilStateCreateInfo()
{
	LOG_DEBUG("Populating Vulkan PipelineDepthStencilStateCreateInfo");
	vk::StencilOpState frontStencilOpState = vk::StencilOpState(
		vk::StencilOp::eKeep,  // failOp
		vk::StencilOp::eKeep,  // passOp
		vk::StencilOp::eKeep,  // depthFailOp
		vk::CompareOp::eNever, // compareOp
		0,					   // compareMask
		0,					   // writeMask
		0					   // reference
	);

	vk::StencilOpState backStencilOpState = vk::StencilOpState(
		vk::StencilOp::eKeep,  // failOp
		vk::StencilOp::eKeep,  // passOp
		vk::StencilOp::eKeep,  // depthFailOp
		vk::CompareOp::eNever, // compareOp
		0,					   // compareMask
		0,					   // writeMask
		0					   // reference
	);

	_pipelineDepthStencilStateCreateInfo = vk::PipelineDepthStencilStateCreateInfo(
		vk::PipelineDepthStencilStateCreateFlags(), // flags
		true,										// depthTestEnable
		true,										// depthWriteEnable
		vk::CompareOp::eLessOrEqual,				// depthCompareOp
		false,										// depthBoundsTestEnable
		false,										// stencilTestEnable
		frontStencilOpState,						// front
		backStencilOpState,							// back
		0.0f,										// minDepthBounds
		1.0f										// maxDepthBounds
	);
}

void GraphicsPipeline::PopulatePipelineInputAssemblyStateCreateInfo()
{
	LOG_DEBUG("Creating pipeline input assembly state create info");
	_pipelineInputAssemblyStateCreateInfo = vk::PipelineInputAssemblyStateCreateInfo(
		vk::PipelineInputAssemblyStateCreateFlags(), // flags
		vk::PrimitiveTopology::eTriangleList,		 // topology
		vk::False									 // primitiveRestartEnable
	);
}

void GraphicsPipeline::PopulatePipelineDynamicStateCreateInfo()
{
	LOG_DEBUG("Creating pipeline dynamic state create info");

	_dynamicStateCreateInfo = vk::PipelineDynamicStateCreateInfo(
		vk::PipelineDynamicStateCreateFlags(),		  // flags
		static_cast<uint32_t>(_dynamicStates.size()), // dynamicStateCount
		_dynamicStates.data(),						  // pDynamicStates
		nullptr										  // pNext
	);
}

void GraphicsPipeline::PopulatePipelineViewportStateCreateInfo()
{
	LOG_DEBUG("Creating pipeline viewport state create info");

	_pipelineViewportStateCreateInfo = vk::PipelineViewportStateCreateInfo(
		vk::PipelineViewportStateCreateFlags(), // flags
		1,                                      // viewportCount
		nullptr,                                // pViewports (dynamic state, ignored)
		1,                                      // scissorCount
		nullptr,                                // pScissors (dynamic state, ignored)
		nullptr                                 // pNext
	);
}

void GraphicsPipeline::PopulatePipelineRasterizationStateCreateInfo()
{
	LOG_DEBUG("Creating pipeline rasterization state create info");
	_pipelineRasterizationStateCreateInfo = vk::PipelineRasterizationStateCreateInfo(
		vk::PipelineRasterizationStateCreateFlags(), // flags
		vk::False,									 // depthClampEnable
		vk::False,									 // rasterizerDiscardEnable
		vk::PolygonMode::eFill,						 // polygonMode
		_cullMode,									 // cullMode
		vk::FrontFace::eCounterClockwise,			 // frontFace
		vk::False,									 // depthBiasEnable
		0.0f,										 // depthBiasConstantFactor
		0.0f,										 // depthBiasClamp
		0.0f,										 // depthBiasSlopeFactor
		1.0f										 // lineWidth
	);
}

void GraphicsPipeline::PopulatePipelineMultisampleStateCreateInfo()
{
	LOG_DEBUG("Creating pipeline multisample state create info");

	_pipelineMultisampleStateCreateInfo = vk::PipelineMultisampleStateCreateInfo(
		vk::PipelineMultisampleStateCreateFlags(), // flags
		vk::SampleCountFlagBits::e1,			   // rasterizationSamples
		vk::False,								   // sampleShadingEnable
		1.0f,									   // minSampleShading
		nullptr,								   // pSampleMask (nullptr = all bits set)
		vk::False,								   // alphaToCoverageEnable
		vk::False								   // alphaToOneEnable
	);
}

void GraphicsPipeline::PopulatePipelineColorBlendAttachmentState()
{
	LOG_DEBUG("Creating pipeline color blend attachment state");
	_pipelineColorBlendAttachmentState = vk::PipelineColorBlendAttachmentState(
		_blendEnabled ? vk::True : vk::False,	// blendEnable
		vk::BlendFactor::eSrcAlpha,			// srcColorBlendFactor
		vk::BlendFactor::eOneMinusSrcAlpha, // dstColorBlendFactor
		vk::BlendOp::eAdd,					// colorBlendOp
		vk::BlendFactor::eOne,				// srcAlphaBlendFactor
		vk::BlendFactor::eZero,				// dstAlphaBlendFactor
		vk::BlendOp::eAdd,					// alphaBlendOp
		(vk::ColorComponentFlagBits)15		// colorWriteMask
	);
}

void GraphicsPipeline::PopulatePipelineColorBlendStateCreateInfo()
{
	LOG_DEBUG("Creating pipeline color blend state create info");
	_pipelineColorBlendStateCreateInfo = vk::PipelineColorBlendStateCreateInfo(
		vk::PipelineColorBlendStateCreateFlags(),	 // flags
		vk::False,									 // logicOpEnable
		vk::LogicOp::eNoOp,							 // logicOp
		1,											 // attachmentCount
		&_pipelineColorBlendAttachmentState,		 // pAttachments
		std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f} // blendConstants
	);
}

void GraphicsPipeline::Build()
{
	LOG_DEBUG("Building Vulkan Graphics Pipeline");

	_vertexPipelineShaderStageCreateInfo = PopulatePipelineShaderStageCreateInfo(_vertexShader->GetShaderModule(), vk::ShaderStageFlagBits::eVertex);
	_fragmentPipelineShaderStageCreateInfo = PopulatePipelineShaderStageCreateInfo(_fragmentShader->GetShaderModule(), vk::ShaderStageFlagBits::eFragment);

	std::vector<vk::PipelineShaderStageCreateInfo> pipelineShaderStageCreateInfos = std::vector<vk::PipelineShaderStageCreateInfo>{
		_vertexPipelineShaderStageCreateInfo,
		_fragmentPipelineShaderStageCreateInfo};

	PopulatePipelineVertexInputStateCreateInfo();
	PopulatePipelineInputAssemblyStateCreateInfo();
	PopulatePipelineViewportStateCreateInfo();
	PopulatePipelineRasterizationStateCreateInfo();
	PopulatePipelineMultisampleStateCreateInfo();
	PopulatePipelineColorBlendAttachmentState();
	PopulatePipelineColorBlendStateCreateInfo();
	PopulatePipelineDepthStencilStateCreateInfo();
	PopulatePipelineDynamicStateCreateInfo();

	BuildPipelineLayout();

	_pipelineRenderingCreateInfo = vk::PipelineRenderingCreateInfo(
		0,                     // viewMask
		1,                     // colorAttachmentCount
		&_dynamicColorFormat,  // pColorAttachmentFormats
		_dynamicDepthFormat,   // depthAttachmentFormat
		vk::Format::eUndefined // stencilAttachmentFormat
	);

	_graphicsPipelineCreateInfo = vk::GraphicsPipelineCreateInfo(
		vk::PipelineCreateFlags(),                                       // flags
		static_cast<uint32_t>(pipelineShaderStageCreateInfos.size()),    // stageCount
		pipelineShaderStageCreateInfos.data(),                           // pStages
		&_pipelineVertexInputStateCreateInfo,                            // pVertexInputState
		&_pipelineInputAssemblyStateCreateInfo,                          // pInputAssemblyState
		nullptr,                                                         // pTessellationState
		&_pipelineViewportStateCreateInfo,                               // pViewportState
		&_pipelineRasterizationStateCreateInfo,                          // pRasterizationState
		&_pipelineMultisampleStateCreateInfo,                            // pMultisampleState
		&_pipelineDepthStencilStateCreateInfo,                           // pDepthStencilState
		&_pipelineColorBlendStateCreateInfo,                             // pColorBlendState
		&_dynamicStateCreateInfo,                                        // pDynamicState
		_pipelineLayout,                                                 // layout
		{},                                                              // renderPass
		0,                                                               // subpass
		nullptr,                                                         // basePipelineHandle
		-1,                                                              // basePipelineIndex
		&_pipelineRenderingCreateInfo                                    // pNext
	);

	try
	{
		_pipeline = (_deviceContext.GetDevice().createGraphicsPipeline(nullptr, _graphicsPipelineCreateInfo)).value;
		LOG_INFO("Graphics pipeline created: {}", _pipeline ? "OK" : "FAILED");
	}
	catch (vk::SystemError error)
	{
		LOG_ERROR("Failed to create graphics pipeline: {}", error.what());
	}
}

} // namespace Cave
