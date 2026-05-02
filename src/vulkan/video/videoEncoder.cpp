#include "videoEncoder.h"
#include "../../common/logger.h"
#include "../shader.h"
#include <set>

// Windows.h defines CreateSemaphore as a macro — undef before using DeviceContext::CreateSemaphore
#ifdef CreateSemaphore
#undef CreateSemaphore
#endif

namespace Cave
{
	VideoEncoder::VideoEncoder(DeviceContext &deviceContext, const std::vector<vk::Image> &inputImages, const std::vector<vk::ImageView> &inputImageViews, uint32_t width, uint32_t height, uint32_t fps)
		: _deviceContext{deviceContext}, _inputImages{inputImages}, _inputImageViews{inputImageViews}, _width{width}, _height{height}, _fps{fps}
	{
		QueueFamilies queueFamilies = _deviceContext.GetQueueFamilies();
		_encodeCommandPool = _deviceContext.CreateCommandPool(queueFamilies.VideoEncodeFamily.Index.value());

		CreateVideoSession();
		AllocateVideoSessionMemory();
		CreateVideoSessionParameters();
		ReadBitstreamHeader();
		AllocateOutputBitStream();
		AllocateReferenceImages();
		AllocateIntermediateImages();
		CreateOutputQueryPool();
		BuildDescriptorResources();
		CreateYCbCrConversionPipeline();

		_interQueueSemaphore = _deviceContext.CreateSemaphore();
		_interQueueSemaphore2 = _deviceContext.CreateSemaphore();
		_interQueueSemaphore3 = _deviceContext.CreateSemaphore();
		_encodeFinishedFence = _deviceContext.CreateFence();

		// Submit initial initialization commands and wait for finish
		vk::CommandBuffer commandBuffer = _deviceContext.CreateCommandBuffer(_encodeCommandPool);

		vk::CommandBufferBeginInfo commandBufferBeginInfo = vk::CommandBufferBeginInfo(
			vk::CommandBufferUsageFlagBits::eOneTimeSubmit // flags
		);

		commandBuffer.begin(commandBufferBeginInfo);

		InitRateControl(commandBuffer);
		TransitionImagesInitial(commandBuffer);

		commandBuffer.end();

		vk::SubmitInfo submitInfo = vk::SubmitInfo(
			{},			   // waitSemaphoreCount
			{},			   // pWaitSemaphores
			{},			   // pWaitDstStageMask
			1,			   // commandBufferCount
			&commandBuffer // pCommandBuffers
		);

		_deviceContext.GetDevice().resetFences(1, &_encodeFinishedFence);
		_deviceContext.GetVideoEncodeQueue().submit(submitInfo, _encodeFinishedFence);
		_deviceContext.GetDevice().waitForFences(1, &_encodeFinishedFence, vk::True, UINT64_MAX);
		_deviceContext.GetDevice().freeCommandBuffers(_encodeCommandPool, 1, &commandBuffer);

		_frameCount = 0;
		_initialized = true;
	}

	VideoEncoder::~VideoEncoder()
	{
		if (!_initialized)
		{
			return;
		}

		if (_running)
		{
			const char *data;
			size_t size;
			GetOutputVideoPacket(data, size);

			_deviceContext.GetDevice().freeCommandBuffers(_deviceContext.GetComputeCommandPool(), 1, &_computeCommandBuffer);
			_deviceContext.GetDevice().freeCommandBuffers(_encodeCommandPool, 1, &_encodeCommandBuffer);
		}

		auto &dldi = _deviceContext.GetDispatchLoaderDynamic();

		_deviceContext.GetDevice().destroyFence(_encodeFinishedFence);
		_deviceContext.GetDevice().destroySemaphore(_interQueueSemaphore);
		_deviceContext.GetDevice().destroySemaphore(_interQueueSemaphore2);
		_deviceContext.GetDevice().destroySemaphore(_interQueueSemaphore3);

		_deviceContext.GetDevice().destroyVideoSessionParametersKHR(_videoSessionParameters, nullptr, dldi);
		_deviceContext.GetDevice().destroyQueryPool(_queryPool, nullptr, dldi);

		_deviceContext.GetDevice().destroyPipeline(_computePipeline);
		_deviceContext.GetDevice().destroyPipelineLayout(_computePipelineLayout);
		_deviceContext.GetDevice().destroyDescriptorSetLayout(_computeDescriptorSetLayout);

		_deviceContext.GetDevice().destroyVideoSessionKHR(_videoSession, nullptr, dldi);
		_bitStreamHeader.clear();

		_deviceContext.GetDevice().destroyCommandPool(_encodeCommandPool);
		_initialized = false;
	}

	void VideoEncoder::QueueEncode(uint32_t currentImageIx)
	{
		ConvertRGBtoYCbCr(currentImageIx);
		EncodeVideoFrame();
		_running = true;
	}

	void VideoEncoder::FinishEncode(const char *&data, size_t &size)
	{
		if (!_running)
		{
			size = 0;
			return;
		}
		if (_bitStreamHeaderPending)
		{
			data = _bitStreamHeader.data();
			size = _bitStreamHeader.size();
			_bitStreamHeaderPending = false;
			return;
		}

		GetOutputVideoPacket(data, size);

		_deviceContext.GetDevice().freeCommandBuffers(_deviceContext.GetComputeCommandPool(), 1, &_computeCommandBuffer);
		_deviceContext.GetDevice().freeCommandBuffers(_encodeCommandPool, 1, &_encodeCommandBuffer);

		_frameCount++;
		_running = false;
	}

	void VideoEncoder::CreateVideoSession()
	{
		QueueFamilies queueFamilies = _deviceContext.GetQueueFamilies();
		auto &dldi = _deviceContext.GetDispatchLoaderDynamic();

		_encodeH264ProfileInfoExt = vk::VideoEncodeH264ProfileInfoKHR(
			STD_VIDEO_H264_PROFILE_IDC_MAIN // stdProfileIdc_
		);

		_videoProfile = vk::VideoProfileInfoKHR(
			vk::VideoCodecOperationFlagBitsKHR::eEncodeH264, // videoCodedOperation
			vk::VideoChromaSubsamplingFlagBitsKHR::e420,	 // chromaSubsampling
			vk::VideoComponentBitDepthFlagBitsKHR::e8,		 // lumaBitDepth
			vk::VideoComponentBitDepthFlagBitsKHR::e8,		 // chromaBitDepth
			&_encodeH264ProfileInfoExt						 // pNext
		);

		_videoProfileList = vk::VideoProfileListInfoKHR(
			1,			   // profileCount
			&_videoProfile // pProfiles
		);

		vk::VideoEncodeH264CapabilitiesKHR videoEncodeH264CapabilitiesKHR = vk::VideoEncodeH264CapabilitiesKHR();

		vk::VideoEncodeCapabilitiesKHR videoEncodeCapabilitiesKHR = vk::VideoEncodeCapabilitiesKHR(
			{},								// flags
			{},								// rateControlModes
			{},								// maxRateControlLayers
			{},								// maxBitRate
			{},								// maxQualityLevels
			{},								// encodeInputPictureGranularity
			{},								// supportedEncodeFeedbackFlags
			&videoEncodeH264CapabilitiesKHR // pNext
		);

		vk::VideoCapabilitiesKHR videoCapabilitiesKHR = vk::VideoCapabilitiesKHR(
			{},							// flags
			{},							// minBitstreamBufferOffsetAlignment
			{},							// minBitstreamBufferSizeAlignment
			{},							// pictureAccessGranularity
			{},							// miCodedExtent
			{},							// maxCodedExtent
			{},							// maxDpbSlots
			{},							// maxActiveReferencePictures
			{},							// stdHeaderVersion
			&videoEncodeCapabilitiesKHR // pNext
		);

		vk::Result result = _deviceContext.GetPhysicalDevice().getVideoCapabilitiesKHR(&_videoProfile, &videoCapabilitiesKHR, dldi);

		LOG_INFO("VideoEncoder caps: maxCodedExtent={}x{}, maxDpbSlots={}, requested={}x{}",
			videoCapabilitiesKHR.maxCodedExtent.width, videoCapabilitiesKHR.maxCodedExtent.height,
			videoCapabilitiesKHR.maxDpbSlots, _width, _height);

		_chosenRateControlMode = vk::VideoEncodeRateControlModeFlagBitsKHR::eDefault;

		if (videoEncodeCapabilitiesKHR.rateControlModes & vk::VideoEncodeRateControlModeFlagBitsKHR::eVbr)
		{
			_chosenRateControlMode = vk::VideoEncodeRateControlModeFlagBitsKHR::eVbr;
		}
		else if (videoEncodeCapabilitiesKHR.rateControlModes & vk::VideoEncodeRateControlModeFlagBitsKHR::eCbr)
		{
			_chosenRateControlMode = vk::VideoEncodeRateControlModeFlagBitsKHR::eCbr;
		}
		else if (videoEncodeCapabilitiesKHR.rateControlModes & vk::VideoEncodeRateControlModeFlagBitsKHR::eDisabled)
		{
			_chosenRateControlMode = vk::VideoEncodeRateControlModeFlagBitsKHR::eDisabled;
		}

		vk::PhysicalDeviceVideoEncodeQualityLevelInfoKHR qualityLevelInfo = vk::PhysicalDeviceVideoEncodeQualityLevelInfoKHR(
			&_videoProfile, // pVideoProfile
			0				// qualityLevel
		);

		vk::VideoEncodeH264QualityLevelPropertiesKHR h264QualityLevelProperties = vk::VideoEncodeH264QualityLevelPropertiesKHR();

		vk::VideoEncodeQualityLevelPropertiesKHR qualityLevelProperties = vk::VideoEncodeQualityLevelPropertiesKHR(
			{},							// preferredRateControlMode
			{},							// preferredRateControlLayerCount
			&h264QualityLevelProperties // pNext
		);

		result = _deviceContext.GetPhysicalDevice().getVideoEncodeQualityLevelPropertiesKHR(&qualityLevelInfo, &qualityLevelProperties, dldi);

		vk::PhysicalDeviceVideoFormatInfoKHR videoFormatInfo = vk::PhysicalDeviceVideoFormatInfoKHR(
			vk::ImageUsageFlagBits::eVideoEncodeSrcKHR | vk::ImageUsageFlagBits::eTransferDst, // imageUsage
			&_videoProfileList																   // pNext
		);

		std::vector<vk::VideoFormatPropertiesKHR> srcVideoFormatProperties = _deviceContext.GetPhysicalDevice().getVideoFormatPropertiesKHR(videoFormatInfo, dldi);

		_chosenSrcImageFormat = vk::Format::eUndefined;
		for (const auto &formatProperties : srcVideoFormatProperties)
		{
			if (formatProperties.format == vk::Format::eG8B8R82Plane420Unorm)
			{
				_chosenSrcImageFormat = formatProperties.format;
				break;
			}
		}

		if (_chosenSrcImageFormat == vk::Format::eUndefined)
			throw std::runtime_error("Error: no supported video encode source image format");

		videoFormatInfo.imageUsage = vk::ImageUsageFlagBits::eVideoEncodeDpbKHR;

		std::vector<vk::VideoFormatPropertiesKHR> dpbVideoFormatProperties = _deviceContext.GetPhysicalDevice().getVideoFormatPropertiesKHR(videoFormatInfo, dldi);

		if (dpbVideoFormatProperties.size() < 1)
			throw std::runtime_error("Error: no supported video encode DPB image format");

		_chosenDpbImageFormat = dpbVideoFormatProperties[0].format;

		vk::ExtensionProperties h264ExtensionVersion;
		std::strncpy(h264ExtensionVersion.extensionName, VK_STD_VULKAN_VIDEO_CODEC_H264_ENCODE_EXTENSION_NAME, VK_MAX_EXTENSION_NAME_SIZE - 1);
		h264ExtensionVersion.specVersion = VK_STD_VULKAN_VIDEO_CODEC_H264_ENCODE_SPEC_VERSION;

		vk::VideoSessionCreateInfoKHR videoSessionCreateInfoKHR = vk::VideoSessionCreateInfoKHR(
			queueFamilies.VideoEncodeFamily.Index.value(), // queueFamilyIndex
			{},											   // flags
			&_videoProfile,								   // pVideoProfile
			_chosenSrcImageFormat,						   // pictureFormat
			{_width, _height},							   // maxCodedExtent
			_chosenDpbImageFormat,						   // referencePictureFormat
			16,											   // maxDpbSlots
			16,											   // maxActiveReferencePictures
			&h264ExtensionVersion						   // pStdHeaderVersion
		);

		_deviceContext.GetDevice().createVideoSessionKHR(&videoSessionCreateInfoKHR, nullptr, &_videoSession, dldi);
	}

	uint32_t VideoEncoder::FindMemoryTypeIndex(uint32_t supportedMemoryIndices)
	{
		vk::MemoryPropertyFlags memoryPropertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached;
		LOG_DEBUG("Finding memory type index");
		vk::PhysicalDeviceMemoryProperties physicalDeviceMemoryProperties = _deviceContext.GetPhysicalDevice().getMemoryProperties();

		for (uint32_t memoryTypeIndex = 0; memoryTypeIndex < physicalDeviceMemoryProperties.memoryTypeCount; memoryTypeIndex++)
		{
			if (supportedMemoryIndices & (1 << memoryTypeIndex) && (physicalDeviceMemoryProperties.memoryTypes[memoryTypeIndex].propertyFlags & memoryPropertyFlags) == physicalDeviceMemoryProperties.memoryTypes[memoryTypeIndex].propertyFlags)
			{
				return memoryTypeIndex;
			}
		}

		throw std::runtime_error("Failed to find suitable memory type :(");
	}

	void VideoEncoder::AllocateVideoSessionMemory()
	{
		auto &dldi = _deviceContext.GetDispatchLoaderDynamic();
		std::vector<vk::VideoSessionMemoryRequirementsKHR> videoSessionMemoryRequirements = _deviceContext.GetDevice().getVideoSessionMemoryRequirementsKHR(_videoSession, dldi);

		if (videoSessionMemoryRequirements.size() == 0)
		{
			return;
		}

		std::vector<vk::BindVideoSessionMemoryInfoKHR> encodeSessionBindMemory(videoSessionMemoryRequirements.size());

		_videoSessionDeviceMemory = std::vector<vk::DeviceMemory>(videoSessionMemoryRequirements.size());

		for (uint32_t memoryIndex = 0; memoryIndex < videoSessionMemoryRequirements.size(); memoryIndex++)
		{
			uint32_t memoryTypeIndex = FindMemoryTypeIndex(videoSessionMemoryRequirements[memoryIndex].memoryRequirements.memoryTypeBits);

			vk::MemoryAllocateInfo memoryAllocateInfo = vk::MemoryAllocateInfo(
				videoSessionMemoryRequirements[memoryIndex].memoryRequirements.size, // allocationSize
				memoryTypeIndex														 // memoryTypeIndex
			);

			_videoSessionDeviceMemory[memoryIndex] = _deviceContext.GetDevice().allocateMemory(memoryAllocateInfo);

			encodeSessionBindMemory[memoryIndex] = vk::BindVideoSessionMemoryInfoKHR(
				videoSessionMemoryRequirements[memoryIndex].memoryBindIndex,		// memoryBindIndex
				_videoSessionDeviceMemory[memoryIndex],								// memory
				0,																	// memoryOffset
				videoSessionMemoryRequirements[memoryIndex].memoryRequirements.size // memorySize
			);
		}

		_deviceContext.GetDevice().bindVideoSessionMemoryKHR(_videoSession, encodeSessionBindMemory, dldi);
	}

	void VideoEncoder::CreateVideoSessionParameters()
	{
		auto &dldi = _deviceContext.GetDispatchLoaderDynamic();

		_vui = h264::GetStdVideoH264SequenceParameterSetVui(_fps);
		_sps = h264::GetStdVideoH264SequenceParameterSet(_width, _height, &_vui);
		_pps = h264::GetStdVideoH264PictureParameterSet();

		vk::VideoEncodeH264SessionParametersAddInfoKHR videoEncodeH264SessionParametersAddInfoKHR = vk::VideoEncodeH264SessionParametersAddInfoKHR(
			1,	   // stdSPSCount
			&_sps, // pStdSPSs
			1,	   // stdPPSCount
			&_pps  // pStdPPSs
		);

		vk::VideoEncodeH264SessionParametersCreateInfoKHR videoEncodeH264SessionParametersCreateInfoKHR = vk::VideoEncodeH264SessionParametersCreateInfoKHR(
			1,											// maxStdSPSCount
			1,											// maxStdPPSCount
			&videoEncodeH264SessionParametersAddInfoKHR // pParametersAddInfo
		);

		vk::VideoSessionParametersCreateInfoKHR videoSessionParametersCreateInfoKHR = vk::VideoSessionParametersCreateInfoKHR(
			{},											   // flags
			nullptr,									   // videoSessionParametersTemplate
			_videoSession,								   // videoSession
			&videoEncodeH264SessionParametersCreateInfoKHR // pNext
		);

		_videoSessionParameters = _deviceContext.GetDevice().createVideoSessionParametersKHR(videoSessionParametersCreateInfoKHR, nullptr, dldi);
	}

	void VideoEncoder::ReadBitstreamHeader()
	{
		auto &dldi = _deviceContext.GetDispatchLoaderDynamic();

		vk::VideoEncodeH264SessionParametersGetInfoKHR h264getInfo = vk::VideoEncodeH264SessionParametersGetInfoKHR(
			vk::True, // writeStdSPS
			vk::True, // writeStdPPS
			0,		  // stdSPSId
			0		  // stdPPSId
		);

		vk::VideoEncodeSessionParametersGetInfoKHR getInfo = vk::VideoEncodeSessionParametersGetInfoKHR(
			_videoSessionParameters, // videoSessionParameters
			&h264getInfo			 // pNext
		);

		vk::VideoEncodeH264SessionParametersFeedbackInfoKHR h264feedback = vk::VideoEncodeH264SessionParametersFeedbackInfoKHR();

		vk::VideoEncodeSessionParametersFeedbackInfoKHR feedback = vk::VideoEncodeSessionParametersFeedbackInfoKHR(
			{},			  // hasOverrides
			&h264feedback // pNext
		);

		size_t datalen = 1024;

		_deviceContext.GetDevice().getEncodedVideoSessionParametersKHR(&getInfo, nullptr, &datalen, nullptr, dldi);
		_bitStreamHeader.resize(datalen);
		_deviceContext.GetDevice().getEncodedVideoSessionParametersKHR(&getInfo, &feedback, &datalen, _bitStreamHeader.data(), dldi);

		_bitStreamHeaderPending = true;
	}

	void VideoEncoder::AllocateOutputBitStream()
	{
		// Size for the worst-case I-frame at the configured resolution. Pre-fix this was a
		// hardcoded 4 MB which silently truncated at 4K (a 4K H.264 keyframe can easily
		// exceed 10 MB). 4 bytes/pixel (1 byte/pixel uncompressed × 4× safety) gives
		// ~33 MB at 4K UHD, ~8 MB at 1080p, ~4 MB at 720p — all small enough to be free.
		uint64_t bitStreamSize = std::max<uint64_t>(4ULL * 1024 * 1024,
			static_cast<uint64_t>(_width) * static_cast<uint64_t>(_height) * 4ULL);
		LOG_INFO("VideoEncoder: bitstream buffer size {} MB for {}x{}", bitStreamSize / (1024 * 1024), _width, _height);

		_bitStreamBuffer = std::make_unique<Buffer>(_deviceContext, bitStreamSize,
													vk::BufferUsageFlagBits::eVideoEncodeDstKHR,
													VMA_MEMORY_USAGE_AUTO,
													VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
													false, 0, "Video Bitstream Buffer", &_videoProfileList);
		_bitStreamData = reinterpret_cast<char *>(_bitStreamBuffer->GetVmaAllocationInfo().pMappedData);
	}

	void VideoEncoder::AllocateReferenceImages()
	{
		QueueFamilies queueFamilies = _deviceContext.GetQueueFamilies();

		_dpbImages.resize(_inputImages.size());

		for (uint32_t imageIndex = 0; imageIndex < _inputImages.size(); imageIndex++)
		{
			_dpbImages[imageIndex] = std::make_unique<Image>(_deviceContext, vk::Format::eG8B8R82Plane420Unorm, 1, 1,
															 vk::Extent3D(_width, _height, 1), vk::ImageTiling::eOptimal,
															 vk::ImageUsageFlagBits::eVideoEncodeDpbKHR,
															 vk::MemoryPropertyFlagBits::eDeviceLocal,
															 vk::ImageCreateFlags(),
															 vk::ImageAspectFlagBits::eColor,
															 vk::ImageViewType::e2D,
															 vk::SharingMode::eExclusive,
															 std::vector<uint32_t>{queueFamilies.VideoEncodeFamily.Index.value()},
															 vk::ImageType::e2D,
															 &_videoProfileList);
		}
	}

	void VideoEncoder::AllocateIntermediateImages()
	{
		QueueFamilies queueFamilies = _deviceContext.GetQueueFamilies();
		std::vector<uint32_t> queueFamilyIndices;
		vk::SharingMode sharingMode;

		// The combined YCbCr image is written by the compute queue (copyImage from luma/chroma)
		// and read by the video encode queue. Include all three families.
		sharingMode = vk::SharingMode::eConcurrent;
		{
			std::set<uint32_t> uniqueFamilies = {
				queueFamilies.GraphicsFamily.Index.value(),
				queueFamilies.ComputeFamily.Index.value(),
				queueFamilies.VideoEncodeFamily.Index.value()};
			queueFamilyIndices.assign(uniqueFamilies.begin(), uniqueFamilies.end());
		}

		_yCbCrImage = std::make_unique<Image>(_deviceContext, vk::Format::eG8B8R82Plane420Unorm, 1, 1,
											  vk::Extent3D(_width, _height, 1), vk::ImageTiling::eOptimal,
											  vk::ImageUsageFlagBits::eVideoEncodeSrcKHR | vk::ImageUsageFlagBits::eTransferDst,
											  vk::MemoryPropertyFlagBits::eDeviceLocal,
											  vk::ImageCreateFlags(),
											  vk::ImageAspectFlagBits::eColor,
											  vk::ImageViewType::e2D,
											  sharingMode, queueFamilyIndices,
										  vk::ImageType::e2D, &_videoProfileList);

		// YCbCr images need concurrent sharing — written by compute shader, read by video encode
		std::vector<uint32_t> ycbcrQueueFamilies = {
			queueFamilies.ComputeFamily.Index.value(),
			queueFamilies.VideoEncodeFamily.Index.value()};

		_yCbCrImageLuma = std::make_unique<Image>(_deviceContext, vk::Format::eR8Unorm, 1, 1,
												  vk::Extent3D(_width, _height, 1), vk::ImageTiling::eOptimal,
												  vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eStorage,
												  vk::MemoryPropertyFlagBits::eDeviceLocal,
												  vk::ImageCreateFlags(),
												  vk::ImageAspectFlagBits::eColor,
												  vk::ImageViewType::e2D,
												  vk::SharingMode::eConcurrent, ycbcrQueueFamilies);

		_yCbCrImageChroma = std::make_unique<Image>(_deviceContext, vk::Format::eR8G8Unorm, 1, 1,
													vk::Extent3D(_width / 2, _height / 2, 1), vk::ImageTiling::eOptimal,
													vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eStorage,
													vk::MemoryPropertyFlagBits::eDeviceLocal,
													vk::ImageCreateFlags(),
													vk::ImageAspectFlagBits::eColor,
													vk::ImageViewType::e2D,
													vk::SharingMode::eConcurrent, ycbcrQueueFamilies);
	}

	void VideoEncoder::CreateOutputQueryPool()
	{
		vk::QueryPoolVideoEncodeFeedbackCreateInfoKHR queryPoolVideoEncodeFeedbackCreateInfo = vk::QueryPoolVideoEncodeFeedbackCreateInfoKHR(
			vk::VideoEncodeFeedbackFlagBitsKHR::eBitstreamBufferOffset | vk::VideoEncodeFeedbackFlagBitsKHR::eBitstreamBytesWritten, // encodeFeedbackFlags
			&_videoProfile																											 // pNext
		);

		vk::QueryPoolCreateInfo queryPoolCreateInfo = vk::QueryPoolCreateInfo(
			{},										// flags
			vk::QueryType::eVideoEncodeFeedbackKHR, // queryType
			1,										// queryCount
			{},										// pipelineStatistics
			&queryPoolVideoEncodeFeedbackCreateInfo // pNext
		);

		_queryPool = _deviceContext.GetDevice().createQueryPool(queryPoolCreateInfo);
	}

	void VideoEncoder::CreateYCbCrConversionPipeline()
	{
		std::vector<char> shaderCode = Shader::ReadFile("shaders/compiled/rgb2ycbcr.spv");

		vk::ShaderModuleCreateInfo shaderModuleCreateInfo = vk::ShaderModuleCreateInfo(
			vk::ShaderModuleCreateFlags(),
			shaderCode.size(),
			reinterpret_cast<const uint32_t *>(shaderCode.data()));

		vk::ShaderModule shaderModule = _deviceContext.GetDevice().createShaderModule(shaderModuleCreateInfo);

		vk::PipelineShaderStageCreateInfo shaderStageCreateInfo = vk::PipelineShaderStageCreateInfo(
			vk::PipelineShaderStageCreateFlags(),
			vk::ShaderStageFlagBits::eCompute,
			shaderModule,
			"main");

		vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo = vk::PipelineLayoutCreateInfo(
			vk::PipelineLayoutCreateFlags(),
			1,
			&_computeDescriptorSetLayout);

		_computePipelineLayout = _deviceContext.GetDevice().createPipelineLayout(pipelineLayoutCreateInfo);

		vk::ComputePipelineCreateInfo computePipelineCreateInfo = vk::ComputePipelineCreateInfo(
			vk::PipelineCreateFlags(),
			shaderStageCreateInfo,
			_computePipelineLayout);

		auto pipelineResult = _deviceContext.GetDevice().createComputePipeline(nullptr, computePipelineCreateInfo);
		_computePipeline = pipelineResult.value;

		_deviceContext.GetDevice().destroyShaderModule(shaderModule);
	}

	void VideoEncoder::InitRateControl(vk::CommandBuffer commandBuffer)
	{
		auto &dldi = _deviceContext.GetDispatchLoaderDynamic();

		vk::VideoBeginCodingInfoKHR encodeBeginInfo = vk::VideoBeginCodingInfoKHR(
			vk::VideoBeginCodingFlagsKHR(), // flags
			_videoSession,					// videoSession
			_videoSessionParameters			// videoSessionParameters
		);

		// Scale bitrate with pixel count: ~0.3 bits/pixel/frame for high quality H.264
		// 1080p@30fps → ~19 Mbps avg, 4K@30fps → ~75 Mbps avg.
		// Cap at 40 Mbps avg / 80 Mbps max so we stay under H.264 Level 5.1 Main's
		// 240 Mbps spec ceiling AND the typical NVENC encoder rate-control hardware
		// caps; uncapped 4K runs were ~99 Mbps max which the encoder rejected.
		uint64_t pixelCount = static_cast<uint64_t>(_width) * _height;
		uint64_t averageBitrate = std::min<uint64_t>(
			static_cast<uint64_t>(pixelCount * _fps * 0.3),
			40ULL * 1000ULL * 1000ULL);
		uint64_t maxBitrate = std::min<uint64_t>(averageBitrate * 2,
			80ULL * 1000ULL * 1000ULL);
		LOG_INFO("VideoEncoder: rate control avg={} Mbps, max={} Mbps for {}x{}@{}fps",
			averageBitrate / 1000000, maxBitrate / 1000000, _width, _height, _fps);

		_encodeRateControlLayerInfo = vk::VideoEncodeRateControlLayerInfoKHR(
			averageBitrate,					 // averageBitrate
			maxBitrate,						 // maxBitrate
			_fps,							 // frameRateNumerator
			1,								 // frameRateDenominator
			&_encodeH264RateControlLayerInfo // pNext
		);

		_encodeH264RateControlInfo = vk::VideoEncodeH264RateControlInfoKHR(
			vk::VideoEncodeH264RateControlFlagBitsKHR::eRegularGop | vk::VideoEncodeH264RateControlFlagBitsKHR::eReferencePatternFlat, // flags
			16,																														   // gopFrameCount
			16,																														   // idrPeriod
			0,																														   // consecutiveBFrameCount
			1																														   // temporalLayerCount
		);

		_encodeRateControlInfo = vk::VideoEncodeRateControlInfoKHR(
			vk::VideoEncodeRateControlFlagsKHR(), // flags
			_chosenRateControlMode,				  // rateControlMode
			1,									  // layerCount
			&_encodeRateControlLayerInfo,		  // pLayers
			200,								  // virtualBufferSizeInMs
			100,								  // initialVirtualBufferSizeInMs
			&_encodeH264RateControlInfo			  // pNext
		);

		vk::VideoCodingControlInfoKHR codingControlInfo = vk::VideoCodingControlInfoKHR(
			vk::VideoCodingControlFlagBitsKHR::eReset | vk::VideoCodingControlFlagBitsKHR::eEncodeRateControl, // flags
			&_encodeRateControlInfo																			   // pNext
		);

		if (_encodeRateControlInfo.rateControlMode & vk::VideoEncodeRateControlModeFlagBitsKHR::eCbr)
		{
			_encodeRateControlLayerInfo.averageBitrate = _encodeRateControlLayerInfo.maxBitrate;
		}

		if (_encodeRateControlInfo.rateControlMode & vk::VideoEncodeRateControlModeFlagBitsKHR::eDisabled || _encodeRateControlInfo.rateControlMode == vk::VideoEncodeRateControlModeFlagBitsKHR::eDefault)
		{
			_encodeH264RateControlInfo.temporalLayerCount = 0;
			_encodeRateControlInfo.layerCount = 0;
		}

		vk::VideoEndCodingInfoKHR encodeEndInfo = vk::VideoEndCodingInfoKHR();

		commandBuffer.beginVideoCodingKHR(&encodeBeginInfo, dldi);
		commandBuffer.controlVideoCodingKHR(&codingControlInfo, dldi);
		commandBuffer.endVideoCodingKHR(&encodeEndInfo, dldi);
	}

	void VideoEncoder::TransitionImagesInitial(vk::CommandBuffer commandBuffer)
	{
		auto &dldi = _deviceContext.GetDispatchLoaderDynamic();

		std::vector<vk::ImageMemoryBarrier2> barriers = std::vector<vk::ImageMemoryBarrier2>(_dpbImages.size());

		vk::ImageSubresourceRange imageSubresourceRange = vk::ImageSubresourceRange(
			vk::ImageAspectFlagBits::eColor, // aspectMask
			0,								 // baseMipLevel
			1,								 // levelCount
			0,								 // baseArrayLayer
			1								 // layerCount
		);

		for (uint32_t imageIndex = 0; imageIndex < _dpbImages.size(); imageIndex++)
		{
			barriers[imageIndex] = vk::ImageMemoryBarrier2(
				vk::PipelineStageFlagBits2::eBottomOfPipe, // srcStageMask
				{},										   // srcAccessMask
				vk::PipelineStageFlagBits2::eTopOfPipe,	   // dstStageMask
				{},										   // dstAccessMask
				vk::ImageLayout::eUndefined,			   // oldLayout
				vk::ImageLayout::eVideoEncodeDpbKHR,	   // newLayout
				{},										   // srcQueueFamilyIndex
				{},										   // dstQueueFamilyIndex
				*_dpbImages[imageIndex]->GetImage(),	   // image
				imageSubresourceRange					   // subresourceRange
			);
		}

		vk::DependencyInfoKHR dependencyInfo = vk::DependencyInfoKHR(
			{},										// dependencyFlags
			{},										// memoryBarrierCount
			{},										// pMemoryBarriers
			{},										// bufferMemoryBarrierCount
			{},										// pBufferMemoryBarriers
			static_cast<uint32_t>(barriers.size()), // imageMemoryBarrierCount
			barriers.data()							// pImageMemoryBarriers
		);

		commandBuffer.pipelineBarrier2KHR(dependencyInfo, dldi);
	}

	void VideoEncoder::ConvertRGBtoYCbCr(uint32_t currentImageIx)
	{
		auto &dldi = _deviceContext.GetDispatchLoaderDynamic();

		// begin command buffer for compute shader
		_computeCommandBuffer = _deviceContext.CreateCommandBuffer(_deviceContext.GetComputeCommandPool());

		vk::CommandBufferBeginInfo commandBufferBeginInfo = vk::CommandBufferBeginInfo(
			vk::CommandBufferUsageFlagBits::eOneTimeSubmit // flags
		);

		_computeCommandBuffer.begin(commandBufferBeginInfo);

		std::vector<vk::ImageMemoryBarrier2> barriers;

		vk::ImageSubresourceRange imageSubresourceRange = vk::ImageSubresourceRange(
			vk::ImageAspectFlagBits::eColor, // aspectMask
			0,								 // baseMipLevel
			1,								 // levelCount
			0,								 // baseArrayLayer
			1								 // layerCount
		);

		// transition YCbCr image (luma and chroma) to be shader target
		barriers.push_back(vk::ImageMemoryBarrier2(
			vk::PipelineStageFlagBits2::eNone,			// srcStageMask
			vk::AccessFlagBits2::eNone,					// srcAccessMask
			vk::PipelineStageFlagBits2::eComputeShader, // dstStageMask
			vk::AccessFlagBits2::eShaderStorageWrite,	// dstAccessMask
			vk::ImageLayout::eUndefined,				// oldLayout
			vk::ImageLayout::eGeneral,					// newLayout
			vk::QueueFamilyIgnored,						// srcQueueFamilyIndex
			vk::QueueFamilyIgnored,						// dstQueueFamilyIndex
			*_yCbCrImageLuma->GetImage(),				// image
			imageSubresourceRange						// subresourceRange
			));

		barriers.push_back(vk::ImageMemoryBarrier2(
			vk::PipelineStageFlagBits2::eNone,			// srcStageMask
			vk::AccessFlagBits2::eNone,					// srcAccessMask
			vk::PipelineStageFlagBits2::eComputeShader, // dstStageMask
			vk::AccessFlagBits2::eShaderStorageWrite,	// dstAccessMask
			vk::ImageLayout::eUndefined,				// oldLayout
			vk::ImageLayout::eGeneral,					// newLayout
			vk::QueueFamilyIgnored,						// srcQueueFamilyIndex
			vk::QueueFamilyIgnored,						// dstQueueFamilyIndex
			*_yCbCrImageChroma->GetImage(),				// image
			imageSubresourceRange						// subresourceRange
			));

		// transition source image to be shader source
		// The image arrives in TRANSFER_SRC_OPTIMAL from the render pass output.
		barriers.push_back(vk::ImageMemoryBarrier2(
			vk::PipelineStageFlagBits2::eTransfer,				// srcStageMask
			vk::AccessFlagBits2::eTransferRead,					// srcAccessMask
			vk::PipelineStageFlagBits2::eComputeShader,			// dstStageMask
			vk::AccessFlagBits2::eShaderStorageRead,			// dstAccessMask
			vk::ImageLayout::eTransferSrcOptimal,				// oldLayout
			vk::ImageLayout::eGeneral,							// newLayout
			vk::QueueFamilyIgnored,								// srcQueueFamilyIndex
			vk::QueueFamilyIgnored,								// dstQueueFamilyIndex
			_inputImages[currentImageIx],						// image
			imageSubresourceRange								// subresourceRange
			));

		vk::DependencyInfoKHR dependencyInfo = vk::DependencyInfoKHR(
			{},										// dependencyFlags
			{},										// memoryBarrierCount
			{},										// pMemoryBarriers
			{},										// bufferMemoryBarrierCount
			{},										// pBufferMemoryBarriers
			static_cast<uint32_t>(barriers.size()), // imageMemoryBarrierCount
			barriers.data()							// pImageMemoryBarriers
		);

		_computeCommandBuffer.pipelineBarrier2KHR(dependencyInfo, dldi);

		// run the RGB->YCbCr conversion shader
		_computeCommandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, _computePipeline);
		_computeCommandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, _computePipelineLayout, 0, _descriptorSets[currentImageIx], nullptr);
		_computeCommandBuffer.dispatch((_width + 15) / 16, (_height + 15) / 16, 1);

		barriers.clear();

		// transition the luma and chroma images to be copy source
		barriers.push_back(vk::ImageMemoryBarrier2(
			vk::PipelineStageFlagBits2::eComputeShader, // srcStageMask
			vk::AccessFlagBits2::eShaderStorageWrite,	// srcAccessMask
			vk::PipelineStageFlagBits2::eCopy,			// dstStageMask
			vk::AccessFlagBits2::eTransferRead,			// dstAccessMask
			vk::ImageLayout::eGeneral,					// oldLayout
			vk::ImageLayout::eTransferSrcOptimal,		// newLayout
			vk::QueueFamilyIgnored,						// srcQueueFamilyIndex
			vk::QueueFamilyIgnored,						// dstQueueFamilyIndex
			*_yCbCrImageLuma->GetImage(),				// image
			imageSubresourceRange						// subresourceRange
			));

		barriers.push_back(vk::ImageMemoryBarrier2(
			vk::PipelineStageFlagBits2::eComputeShader, // srcStageMask
			vk::AccessFlagBits2::eShaderStorageWrite,	// srcAccessMask
			vk::PipelineStageFlagBits2::eCopy,			// dstStageMask
			vk::AccessFlagBits2::eTransferRead,			// dstAccessMask
			vk::ImageLayout::eGeneral,					// oldLayout
			vk::ImageLayout::eTransferSrcOptimal,		// newLayout
			vk::QueueFamilyIgnored,						// srcQueueFamilyIndex
			vk::QueueFamilyIgnored,						// dstQueueFamilyIndex
			*_yCbCrImageChroma->GetImage(),				// image
			imageSubresourceRange						// subresourceRange
			));

		// transition the full YCbCr image as copy target
		imageSubresourceRange.aspectMask = vk::ImageAspectFlagBits::ePlane0 | vk::ImageAspectFlagBits::ePlane1;

		barriers.push_back(vk::ImageMemoryBarrier2(
			vk::PipelineStageFlagBits2::eNone,	  // srcStageMask
			vk::AccessFlagBits2::eNone,			  // srcAccessMask
			vk::PipelineStageFlagBits2::eCopy,	  // dstStageMask
			vk::AccessFlagBits2::eTransferWrite,  // dstAccessMask
			vk::ImageLayout::eUndefined,		  // oldLayout
			vk::ImageLayout::eTransferDstOptimal, // newLayout
			vk::QueueFamilyIgnored,				  // srcQueueFamilyIndex
			vk::QueueFamilyIgnored,				  // dstQueueFamilyIndex
			*_yCbCrImage->GetImage(),			  // image
			imageSubresourceRange				  // subresourceRange
			));

		dependencyInfo = vk::DependencyInfoKHR(
			{},										// dependencyFlags
			{},										// memoryBarrierCount
			{},										// pMemoryBarriers
			{},										// bufferMemoryBarrierCount
			{},										// pBufferMemoryBarriers
			static_cast<uint32_t>(barriers.size()), // imageMemoryBarrierCount
			barriers.data()							// pImageMemoryBarriers
		);

		_computeCommandBuffer.pipelineBarrier2KHR(dependencyInfo, dldi);

		// copy the full luma image into the 1st plane of the YCbCr image
		vk::ImageSubresourceLayers sourceSubresource = vk::ImageSubresourceLayers(
			vk::ImageAspectFlagBits::eColor, // aspectMask
			0,								 // mipLevel
			0,								 // baseArrayLayer
			1								 // layerCount
		);

		vk::ImageSubresourceLayers destinationSubresource = vk::ImageSubresourceLayers(
			vk::ImageAspectFlagBits::ePlane0, // aspectMask
			0,								  // mipLevel
			0,								  // baseArrayLayer
			1								  // layerCount
		);

		vk::ImageCopy regions = vk::ImageCopy(
			sourceSubresource,		// srcSubresource
			{0, 0, 0},				// srcOffset
			destinationSubresource, // dstSubresource
			{0, 0, 0},				// dstOffset
			{_width, _height, 1}	// extent
		);

		_computeCommandBuffer.copyImage(*_yCbCrImageLuma->GetImage(), vk::ImageLayout::eTransferSrcOptimal, *_yCbCrImage->GetImage(), vk::ImageLayout::eTransferDstOptimal, 1, &regions);

		// copy the full chroma image into the 2nd plane of the YCbCr image
		regions.dstSubresource.aspectMask = vk::ImageAspectFlagBits::ePlane1;
		regions.extent = vk::Extent3D(_width / 2, _height / 2, 1);

		_computeCommandBuffer.copyImage(*_yCbCrImageChroma->GetImage(), vk::ImageLayout::eTransferSrcOptimal, *_yCbCrImage->GetImage(), vk::ImageLayout::eTransferDstOptimal, 1, &regions);

		_computeCommandBuffer.end();

		vk::PipelineStageFlags dstStageMasks[] = {vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands};
		vk::Semaphore signalSemaphores[] = {_interQueueSemaphore, _interQueueSemaphore3};
		vk::Semaphore waitSemaphores[] = {_interQueueSemaphore2, _interQueueSemaphore3};

		vk::SubmitInfo submitInfo = vk::SubmitInfo(
			{},						// waitSemaphoreCount
			{},						// pWaitSemaphores
			{},						// pWaitDstStageMask
			1,						// commandBufferCount
			&_computeCommandBuffer, // pCommandBuffers
			2,						// signalSemaphoreCount
			signalSemaphores		// pSignalSemaphores
		);

		if (_frameCount != 0)
		{
			submitInfo.waitSemaphoreCount = 2;
			submitInfo.pWaitSemaphores = waitSemaphores;
			submitInfo.pWaitDstStageMask = dstStageMasks;
		}

		_deviceContext.GetComputeQueue().submit(submitInfo);
	}

	void VideoEncoder::EncodeVideoFrame()
	{
		auto &dldi = _deviceContext.GetDispatchLoaderDynamic();

		const uint32_t GOP_LENGTH = 16;
		const uint32_t gopFrameCount = _frameCount % GOP_LENGTH;
		uint32_t frameCount = gopFrameCount;
		const uint32_t querySlotId = 0;

		// begin command buffer for video encode
		_encodeCommandBuffer = _deviceContext.CreateCommandBuffer(_encodeCommandPool);

		vk::CommandBufferBeginInfo commandBufferBeginInfo = vk::CommandBufferBeginInfo(
			vk::CommandBufferUsageFlagBits::eOneTimeSubmit // flags
		);

		_encodeCommandBuffer.begin(commandBufferBeginInfo);
		_encodeCommandBuffer.resetQueryPool(_queryPool, querySlotId, 1);

		// start a video encode session
		vk::VideoPictureResourceInfoKHR dpbPicResource = vk::VideoPictureResourceInfoKHR(
			{0, 0},										   // codedOffset
			{_width, _height},							   // codedExtent
			0,											   // baseArrayLayer
			*_dpbImages[gopFrameCount & 1]->GetImageView() // iamgeViewBinding
		);

		vk::VideoPictureResourceInfoKHR refPicResource = vk::VideoPictureResourceInfoKHR(
			{0, 0},											  // codedOffset
			{_width, _height},								  // codedExtent
			0,												  // baseArrayLayer
			*_dpbImages[!(gopFrameCount & 1)]->GetImageView() // iamgeViewBinding
		);

		const uint32_t MaxPicOrderCntLsb = 1 << (_sps.log2_max_pic_order_cnt_lsb_minus4 + 4);

		StdVideoEncodeH264ReferenceInfo dpbRefInfo = {};
		dpbRefInfo.FrameNum = gopFrameCount;
		dpbRefInfo.PicOrderCnt = (dpbRefInfo.FrameNum * 2) % MaxPicOrderCntLsb;
		dpbRefInfo.primary_pic_type = dpbRefInfo.FrameNum == 0 ? STD_VIDEO_H264_PICTURE_TYPE_IDR : STD_VIDEO_H264_PICTURE_TYPE_P;

		vk::VideoEncodeH264DpbSlotInfoKHR dpbSlotInfo = vk::VideoEncodeH264DpbSlotInfoKHR(
			&dpbRefInfo // pStdReferenceInfo
		);

		StdVideoEncodeH264ReferenceInfo refRefInfo = {};
		refRefInfo.FrameNum = gopFrameCount - 1;
		refRefInfo.PicOrderCnt = (refRefInfo.FrameNum * 2) % MaxPicOrderCntLsb;
		refRefInfo.primary_pic_type = refRefInfo.FrameNum == 0 ? STD_VIDEO_H264_PICTURE_TYPE_IDR : STD_VIDEO_H264_PICTURE_TYPE_P;

		vk::VideoEncodeH264DpbSlotInfoKHR refSlotInfo = vk::VideoEncodeH264DpbSlotInfoKHR(
			&refRefInfo // pStdReferenceInfo
		);

		std::vector<vk::VideoReferenceSlotInfoKHR> referenceSlots = std::vector<vk::VideoReferenceSlotInfoKHR>(2);

		referenceSlots[0] = vk::VideoReferenceSlotInfoKHR(
			-1,				 // slotIndex
			&dpbPicResource, // pPictureResource
			&dpbSlotInfo	 // pNext
		);

		referenceSlots[1] = vk::VideoReferenceSlotInfoKHR(
			!(gopFrameCount & 1), // slotIndex
			&refPicResource,	  // pPictureResource
			&refSlotInfo		  // pNext
		);

		vk::VideoBeginCodingInfoKHR encodeBeginInfo = vk::VideoBeginCodingInfoKHR(
			vk::VideoBeginCodingFlagsKHR(), // flags
			_videoSession,					// videoSession
			_videoSessionParameters,		// videoSessionParameters
			gopFrameCount == 0 ? 1 : 2,		// referenceSlotCount
			referenceSlots.data(),			// pReferenceSlots
			&_encodeRateControlInfo			// pNext
		);

		_encodeCommandBuffer.beginVideoCodingKHR(&encodeBeginInfo, dldi);

		// transition the YCbCr image to be a video encode source
		vk::ImageSubresourceRange imageSubresourceRange = vk::ImageSubresourceRange(
			vk::ImageAspectFlagBits::ePlane0 | vk::ImageAspectFlagBits::ePlane1, // aspectMask
			0,																	 // baseMipLevel
			1,																	 // levelCount
			0,																	 // baseArrayLayer
			1																	 // layerCount
		);

		vk::ImageMemoryBarrier2 imageMemoryBarrier = vk::ImageMemoryBarrier2(
			vk::PipelineStageFlagBits2::eCopy,			 // srcStageMask
			vk::AccessFlagBits2::eTransferWrite,		 // srcAccessMask
			vk::PipelineStageFlagBits2::eVideoEncodeKHR, // dstStageMask
			vk::AccessFlagBits2::eVideoEncodeReadKHR,	 // dstAccessMask
			vk::ImageLayout::eTransferDstOptimal,		 // oldLayout
			vk::ImageLayout::eVideoEncodeSrcKHR,		 // newLayout
			{},											 // srcQueueFamilyIndex
			{},											 // dstQueueFamilyIndex
			*_yCbCrImage->GetImage(),					 // image
			imageSubresourceRange						 // subresourceRange
		);

		vk::DependencyInfoKHR dependencyInfo = vk::DependencyInfoKHR(
			{},					// dependencyFlags
			{},					// memoryBarrierCount
			{},					// pMemoryBarriers
			{},					// bufferMemoryBarrierCount
			{},					// pBufferMemoryBarriers
			1,					// imageMemoryBarrierCount
			&imageMemoryBarrier // pImageMemoryBarriers
		);

		_encodeCommandBuffer.pipelineBarrier2KHR(dependencyInfo, dldi);

		// set the YCbCr image as input picture for the encoder
		vk::VideoPictureResourceInfoKHR inputPicResource = vk::VideoPictureResourceInfoKHR(
			{0, 0},						 // codedOffset
			{_width, _height},			 // codedExtent
			0,							 // baseArrayLayer
			*_yCbCrImage->GetImageView() // imageViewBinding
		);

		// set all the frame parameters
		bool useConstantQp = (_chosenRateControlMode & vk::VideoEncodeRateControlModeFlagBitsKHR::eDisabled) == vk::VideoEncodeRateControlModeFlagBitsKHR::eDisabled;
		h264::FrameInfo frameInfo = h264::FrameInfo(frameCount, _width, _height, _sps, _pps, gopFrameCount, useConstantQp);

		vk::VideoEncodeH264PictureInfoKHR *encodeH264FrameInfo = frameInfo.GetEncodeH264FrameInfo();

		referenceSlots[0].slotIndex = gopFrameCount & 1;

		vk::VideoEncodeInfoKHR videoEncodeInfo = vk::VideoEncodeInfoKHR(
			vk::VideoEncodeFlagsKHR(),	   // flags
			_bitStreamBuffer->GetBuffer(), // dstBuffer
			0,							   // dstBufferOffset
			4 * 1024 * 1024,			   // dstBufferRange
			inputPicResource,			   // srcPictureResource
			&referenceSlots[0],			   // pSetupReferenceSlot
			{},							   // referenceSlotCount
			{},							   // pReferenceSlots
			{},							   // precedingExternallyEncodedBytes
			encodeH264FrameInfo			   // pNext
		);

		if (gopFrameCount > 0)
		{
			videoEncodeInfo.referenceSlotCount = 1;
			videoEncodeInfo.pReferenceSlots = &referenceSlots[1];
		}

		_encodeCommandBuffer.beginQuery(_queryPool, querySlotId, vk::QueryControlFlags());
		_encodeCommandBuffer.encodeVideoKHR(&videoEncodeInfo, dldi);
		_encodeCommandBuffer.endQuery(_queryPool, querySlotId);

		vk::VideoEndCodingInfoKHR encodeEndInfo = vk::VideoEndCodingInfoKHR();
		_encodeCommandBuffer.endVideoCodingKHR(&encodeEndInfo, dldi);

		_encodeCommandBuffer.end();

		vk::PipelineStageFlags dstStageMask = vk::PipelineStageFlagBits::eAllCommands;
		vk::SubmitInfo submitInfo = vk::SubmitInfo(
			1,					   // waitSemaphoreCount
			&_interQueueSemaphore, // pWaitSemaphores
			&dstStageMask,		   // pWaitDstStageMask
			1,					   // commandBufferCount
			&_encodeCommandBuffer, // pCommandBuffers
			1,					   // signalSemaphoreCount
			&_interQueueSemaphore2 // pSignalSemaphores
		);

		_deviceContext.GetDevice().resetFences(1, &_encodeFinishedFence);
		_deviceContext.GetVideoEncodeQueue().submit(submitInfo, _encodeFinishedFence);
	}

	uint32_t alignup(uint32_t size, uint32_t alignment)
	{
		return (size + alignment - 1) & ~(alignment - 1);
	}

	void VideoEncoder::GetOutputVideoPacket(const char *&data, size_t &size)
	{
		_deviceContext.GetDevice().waitForFences(1, &_encodeFinishedFence, vk::True, UINT64_MAX);

		VideoEncodeStatus encodeResult;
		memset(&encodeResult, 0, sizeof(encodeResult));

		encodeResult = _deviceContext.GetDevice().getQueryPoolResult<VideoEncodeStatus>(_queryPool, 0, 1, sizeof(VideoEncodeStatus), vk::QueryResultFlagBits::eWithStatusKHR | vk::QueryResultFlagBits::eWait).value;

		vk::DeviceSize nonCoherentAtomSize = _deviceContext.GetPhysicalDevice().getProperties().limits.nonCoherentAtomSize;
		auto alignedSize = alignup(encodeResult.bitstreamSize, nonCoherentAtomSize);

		vmaInvalidateAllocation(_deviceContext.GetVmaAllocator(), _bitStreamBuffer->GetVmaAllocation(), encodeResult.bitstreamStartOffset, alignedSize);

		data = _bitStreamData + encodeResult.bitstreamStartOffset;
		size = encodeResult.bitstreamSize;
	}

	void VideoEncoder::BuildDescriptorResources()
	{
		// Create descriptor set layout with 3 storage image bindings
		std::vector<vk::DescriptorSetLayoutBinding> layoutBindings = {
			{0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
			{1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
			{2, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute}};

		vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = vk::DescriptorSetLayoutCreateInfo(
			vk::DescriptorSetLayoutCreateFlags(),
			static_cast<uint32_t>(layoutBindings.size()),
			layoutBindings.data());

		_computeDescriptorSetLayout = _deviceContext.GetDevice().createDescriptorSetLayout(descriptorSetLayoutCreateInfo);

		// Allocate one descriptor set per input image
		std::vector<vk::DescriptorSetLayout> layouts(_inputImages.size(), _computeDescriptorSetLayout);

		vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = vk::DescriptorSetAllocateInfo(
			_deviceContext.GetDescriptorPool(),
			static_cast<uint32_t>(layouts.size()),
			layouts.data());

		_descriptorSets = _deviceContext.GetDevice().allocateDescriptorSets(descriptorSetAllocateInfo);

		// Update descriptor sets — luma and chroma are shared across all sets, only input image differs
		for (uint32_t imageIndex = 0; imageIndex < _inputImages.size(); imageIndex++)
		{
			vk::DescriptorImageInfo imageInfo0 = vk::DescriptorImageInfo({}, _inputImageViews[imageIndex], vk::ImageLayout::eGeneral);
			vk::DescriptorImageInfo imageInfo1 = vk::DescriptorImageInfo({}, *_yCbCrImageLuma->GetImageView(), vk::ImageLayout::eGeneral);
			vk::DescriptorImageInfo imageInfo2 = vk::DescriptorImageInfo({}, *_yCbCrImageChroma->GetImageView(), vk::ImageLayout::eGeneral);

			std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
				vk::WriteDescriptorSet(_descriptorSets[imageIndex], 0, 0, 1, vk::DescriptorType::eStorageImage, &imageInfo0),
				vk::WriteDescriptorSet(_descriptorSets[imageIndex], 1, 0, 1, vk::DescriptorType::eStorageImage, &imageInfo1),
				vk::WriteDescriptorSet(_descriptorSets[imageIndex], 2, 0, 1, vk::DescriptorType::eStorageImage, &imageInfo2)};

			_deviceContext.GetDevice().updateDescriptorSets(writeDescriptorSets, nullptr);
		}
	}
}
