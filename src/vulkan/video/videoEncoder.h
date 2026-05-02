#pragma once

#include "../pipelines/computePipeline.h"

#include <memory>
#include <vector>

#include "../deviceContext.h"

#include "../image.h"
#include "../buffer.h"

#include "h264parameterset.h"

namespace Cave
{
	class VideoEncoder
	{
	private:
		DeviceContext &_deviceContext;

		uint32_t _width, _height;

		vk::DescriptorSetLayout _computeDescriptorSetLayout;
		vk::Pipeline _computePipeline;
		vk::PipelineLayout _computePipelineLayout;

		vk::CommandPool _encodeCommandPool;
		vk::CommandBuffer _computeCommandBuffer, _encodeCommandBuffer;

		std::vector<vk::DescriptorSet> _descriptorSets;

		static const uint32_t NUMBER_OF_FRAMES_TO_WRITE = 300;

		bool _initialized{false};
		bool _running{false};

		vk::VideoSessionKHR _videoSession;
		StdVideoH264SequenceParameterSetVui _vui;
		StdVideoH264SequenceParameterSet _sps;
		StdVideoH264PictureParameterSet _pps;
		vk::VideoSessionParametersKHR _videoSessionParameters;
		vk::VideoEncodeH264ProfileInfoKHR _encodeH264ProfileInfoExt;
		vk::VideoProfileInfoKHR _videoProfile;
		vk::VideoProfileListInfoKHR _videoProfileList;

		vk::VideoEncodeRateControlModeFlagBitsKHR _chosenRateControlMode;

		vk::VideoEncodeH264RateControlLayerInfoKHR _encodeH264RateControlLayerInfo;
		vk::VideoEncodeRateControlLayerInfoKHR _encodeRateControlLayerInfo;
		vk::VideoEncodeH264RateControlInfoKHR _encodeH264RateControlInfo;
		vk::VideoEncodeRateControlInfoKHR _encodeRateControlInfo;

		std::vector<vk::DeviceMemory> _videoSessionDeviceMemory;

		vk::Format _chosenSrcImageFormat;
		vk::Format _chosenDpbImageFormat;

		vk::Semaphore _interQueueSemaphore;
		vk::Semaphore _interQueueSemaphore2;
		vk::Semaphore _interQueueSemaphore3;

		vk::QueryPool _queryPool;

		std::unique_ptr<Buffer> _bitStreamBuffer;

		std::vector<char> _bitStreamHeader;
		bool _bitStreamHeaderPending;

		char *_bitStreamData;

		std::vector<vk::Image> _inputImages;
		std::vector<vk::ImageView> _inputImageViews;

		std::unique_ptr<Image> _yCbCrImage;
		std::unique_ptr<Image> _yCbCrImageLuma;
		std::unique_ptr<Image> _yCbCrImageChroma;
		std::vector<std::unique_ptr<Image>> _dpbImages;

		uint32_t _frameCount;

		vk::Fence _encodeFinishedFence;

		uint32_t _fps;

		void CreateVideoSession();
		void AllocateVideoSessionMemory();
		void CreateVideoSessionParameters();
		void ReadBitstreamHeader();
		void AllocateOutputBitStream();
		void AllocateReferenceImages();
		void AllocateIntermediateImages();
		void CreateOutputQueryPool();
		void CreateYCbCrConversionPipeline();
		void InitRateControl(vk::CommandBuffer cmdBuf);
		void TransitionImagesInitial(vk::CommandBuffer commandBuffer);
		void ConvertRGBtoYCbCr(uint32_t currentImageIx);
		void EncodeVideoFrame();
		void GetOutputVideoPacket(const char *&data, size_t &size);
		void BuildDescriptorResources();

		uint32_t FindMemoryTypeIndex(uint32_t supportedMemoryIndices);

	public:
		VideoEncoder(DeviceContext &deviceContext, const std::vector<vk::Image> &inputImages, const std::vector<vk::ImageView> &inputImageViews, uint32_t width, uint32_t height, uint32_t fps);
		~VideoEncoder();

		VideoEncoder(const VideoEncoder &) = delete;
		VideoEncoder &operator=(const VideoEncoder &) = delete;

		void QueueEncode(uint32_t currentImageIx);
		void FinishEncode(const char *&data, size_t &size);
	};
}
