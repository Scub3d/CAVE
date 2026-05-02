#pragma once

#include <cassert>
#include <cmath>

#pragma once

#define NOMINMAX
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_funcs.hpp>
#include <vulkan/vulkan_structs.hpp>
#include <vulkan/vulkan_video.hpp>

#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#define WIN32_LEAN_AND_MEAN 1
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#endif

#include <GLFW/glfw3.h>

#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <map>
#include <optional>
#include <set>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <array>
#include <memory>
#include <unordered_map>
#include <string>

#include <vk_video/vulkan_video_codec_h264std.h>
#include <vk_video/vulkan_video_codecs_common.h>

namespace h264
{
	// adapted from Nvidia sample code

	static const uint32_t H264MbSizeAlignment = 16;

	template <typename sizeType>
	static sizeType AlignSize(sizeType size, sizeType alignment)
	{
		assert((alignment & (alignment - 1)) == 0);
		return (size + alignment - 1) & ~(alignment - 1);
	}

	static StdVideoH264SequenceParameterSetVui GetStdVideoH264SequenceParameterSetVui(uint32_t fps)
	{
		StdVideoH264SpsVuiFlags vuiFlags = {};
		vuiFlags.timing_info_present_flag = 1u;
		vuiFlags.fixed_frame_rate_flag = 1u;

		StdVideoH264SequenceParameterSetVui vui = {};
		vui.flags = vuiFlags;
		vui.num_units_in_tick = 1;
		vui.time_scale = fps * 2; // 2 fields

		return vui;
	}

	static StdVideoH264SequenceParameterSet GetStdVideoH264SequenceParameterSet(uint32_t width, uint32_t height, StdVideoH264SequenceParameterSetVui *pVui)
	{
		StdVideoH264SpsFlags spsFlags = {};
		spsFlags.direct_8x8_inference_flag = 1u;
		spsFlags.frame_mbs_only_flag = 1u;
		spsFlags.vui_parameters_present_flag = (pVui == NULL) ? 0u : 1u;

		const uint32_t mbAlignedWidth = AlignSize(width, H264MbSizeAlignment);
		const uint32_t mbAlignedHeight = AlignSize(height, H264MbSizeAlignment);

		// Pick the minimum H.264 level that fits the frame's macroblock count.
		// Level 4.1 caps at 8192 MBs (1920x1088) so 4K and larger require ≥5.1.
		// MB-per-frame thresholds from H.264 Annex A; decoders handle smaller frames
		// at higher levels just fine, so over-picking is safe.
		uint32_t macroblockCount = (mbAlignedWidth / H264MbSizeAlignment) * (mbAlignedHeight / H264MbSizeAlignment);
		StdVideoH264LevelIdc levelIdc = STD_VIDEO_H264_LEVEL_IDC_4_1;
		if (macroblockCount > 8192)   levelIdc = STD_VIDEO_H264_LEVEL_IDC_4_2;  // > 1920x1088
		if (macroblockCount > 8704)   levelIdc = STD_VIDEO_H264_LEVEL_IDC_5_0;  // > Level 4.2 cap
		if (macroblockCount > 22080)  levelIdc = STD_VIDEO_H264_LEVEL_IDC_5_1;  // > 1920x1472 (covers 4K UHD = 32400 MBs)
		if (macroblockCount > 36864)  levelIdc = STD_VIDEO_H264_LEVEL_IDC_6_0;  // > 4096x2304 (covers up to 8K)

		StdVideoH264SequenceParameterSet sps = {};
		sps.profile_idc = STD_VIDEO_H264_PROFILE_IDC_MAIN;
		sps.level_idc = levelIdc;
		sps.seq_parameter_set_id = 0u;
		sps.chroma_format_idc = STD_VIDEO_H264_CHROMA_FORMAT_IDC_420;
		sps.bit_depth_luma_minus8 = 0u;
		sps.bit_depth_chroma_minus8 = 0u;
		sps.log2_max_frame_num_minus4 = 0u;
		sps.pic_order_cnt_type = STD_VIDEO_H264_POC_TYPE_0;
		sps.max_num_ref_frames = 1u;
		sps.pic_width_in_mbs_minus1 = mbAlignedWidth / H264MbSizeAlignment - 1;
		sps.pic_height_in_map_units_minus1 = mbAlignedHeight / H264MbSizeAlignment - 1;
		sps.flags = spsFlags;
		sps.pSequenceParameterSetVui = pVui;
		sps.frame_crop_right_offset = mbAlignedWidth - width;
		sps.frame_crop_bottom_offset = mbAlignedHeight - height;

		// This allows for picture order count values in the range [0, 255].
		sps.log2_max_pic_order_cnt_lsb_minus4 = 4u;

		if (sps.frame_crop_right_offset || sps.frame_crop_bottom_offset)
		{
			sps.flags.frame_cropping_flag = true;

			if (sps.chroma_format_idc == STD_VIDEO_H264_CHROMA_FORMAT_IDC_420)
			{
				sps.frame_crop_right_offset >>= 1;
				sps.frame_crop_bottom_offset >>= 1;
			}
		}

		return sps;
	}

	static StdVideoH264PictureParameterSet GetStdVideoH264PictureParameterSet(void)
	{
		StdVideoH264PpsFlags ppsFlags = {};
		// ppsFlags.transform_8x8_mode_flag = 1u;
		ppsFlags.transform_8x8_mode_flag = 0u;
		ppsFlags.constrained_intra_pred_flag = 0u;
		ppsFlags.deblocking_filter_control_present_flag = 1u;
		ppsFlags.entropy_coding_mode_flag = 1u;

		StdVideoH264PictureParameterSet pps = {};
		pps.seq_parameter_set_id = 0u;
		pps.pic_parameter_set_id = 0u;
		pps.num_ref_idx_l0_default_active_minus1 = 0u;
		pps.flags = ppsFlags;

		return pps;
	}

	class FrameInfo
	{
	private:
		StdVideoEncodeH264SliceHeaderFlags _sliceHeaderFlags = {};
		StdVideoEncodeH264SliceHeader _sliceHeader = {};
		vk::VideoEncodeH264NaluSliceInfoKHR _sliceInfo;
		StdVideoEncodeH264PictureInfoFlags _pictureInfoFlags = {};
		StdVideoEncodeH264PictureInfo _stdPictureInfo = {};
		vk::VideoEncodeH264PictureInfoKHR _encodeH264FrameInfo;
		StdVideoEncodeH264ReferenceListsInfo _referenceLists = {};

	public:
		FrameInfo(uint32_t frameCount, uint32_t width, uint32_t height, StdVideoH264SequenceParameterSet sps, StdVideoH264PictureParameterSet pps, uint32_t gopFrameCount, bool useConstantQp)
		{
			bool isI = gopFrameCount == 0;
			const uint32_t MaxPicOrderCntLsb = 1 << (sps.log2_max_pic_order_cnt_lsb_minus4 + 4);

			_sliceHeaderFlags.direct_spatial_mv_pred_flag = 1;
			_sliceHeaderFlags.num_ref_idx_active_override_flag = 0;

			_sliceHeader.flags = _sliceHeaderFlags;
			_sliceHeader.slice_type = isI ? STD_VIDEO_H264_SLICE_TYPE_I : STD_VIDEO_H264_SLICE_TYPE_P;
			_sliceHeader.cabac_init_idc = (StdVideoH264CabacInitIdc)0;
			_sliceHeader.disable_deblocking_filter_idc = (StdVideoH264DisableDeblockingFilterIdc)0;
			_sliceHeader.slice_alpha_c0_offset_div2 = 0;
			_sliceHeader.slice_beta_offset_div2 = 0;

			uint32_t picWidthInMbs = sps.pic_width_in_mbs_minus1 + 1;
			uint32_t picHeightInMbs = sps.pic_height_in_map_units_minus1 + 1;
			uint32_t iPicSizeInMbs = picWidthInMbs * picHeightInMbs;

			_sliceInfo = vk::VideoEncodeH264NaluSliceInfoKHR(
				useConstantQp ? pps.pic_init_qp_minus26 + 26 : 0, // constantQp
				&_sliceHeader									  // pStdSliceHeader
																  // pNext
			);

			_pictureInfoFlags.IdrPicFlag = isI ? 1 : 0; // every I frame is an IDR frame
			_pictureInfoFlags.is_reference = 1;
			_pictureInfoFlags.adaptive_ref_pic_marking_mode_flag = 0;
			_pictureInfoFlags.no_output_of_prior_pics_flag = isI ? 1 : 0;

			_stdPictureInfo.flags = _pictureInfoFlags;
			_stdPictureInfo.seq_parameter_set_id = 0;
			_stdPictureInfo.pic_parameter_set_id = pps.pic_parameter_set_id;
			_stdPictureInfo.idr_pic_id = 0;
			_stdPictureInfo.primary_pic_type = isI ? STD_VIDEO_H264_PICTURE_TYPE_IDR : STD_VIDEO_H264_PICTURE_TYPE_P;
			// _stdPictureInfo.temporal_id = 1;

			// frame_num is incremented for each reference frame transmitted.
			// In our case, only the first frame (which is IDR) is a reference
			// frame with frame_num == 0, and all others have frame_num == 1.
			_stdPictureInfo.frame_num = frameCount;

			// POC is incremented by 2 for each coded frame.
			_stdPictureInfo.PicOrderCnt = (frameCount * 2) % MaxPicOrderCntLsb;
			_referenceLists.num_ref_idx_l0_active_minus1 = 0;
			_referenceLists.num_ref_idx_l1_active_minus1 = 0;
			std::fill_n(_referenceLists.RefPicList0, STD_VIDEO_H264_MAX_NUM_LIST_REF, STD_VIDEO_H264_NO_REFERENCE_PICTURE);
			std::fill_n(_referenceLists.RefPicList1, STD_VIDEO_H264_MAX_NUM_LIST_REF, STD_VIDEO_H264_NO_REFERENCE_PICTURE);
			if (!isI)
			{
				_referenceLists.RefPicList0[0] = !(gopFrameCount & 1);
			}
			_stdPictureInfo.pRefLists = &_referenceLists;

			_encodeH264FrameInfo = vk::VideoEncodeH264PictureInfoKHR(
				1,				 // naluSliceEntryCount
				&_sliceInfo,	 // pNaluSliceEntries
				&_stdPictureInfo // pStdPictureInfo
								 // generatePrefixNalu
								 // pNext
			);
		}

		inline vk::VideoEncodeH264PictureInfoKHR *GetEncodeH264FrameInfo() { return &_encodeH264FrameInfo; };
	};
};