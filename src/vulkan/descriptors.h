#pragma once

#include "buffer.h"
#include "image.h"
#include "deviceContext.h"

#include <unordered_map>

namespace Cave
{

class Descriptor : public std::enable_shared_from_this<Descriptor>
{
public: // Structs
	struct DescriptorBinding
	{
		vk::DescriptorType type;
		vk::DescriptorSetLayoutBinding layoutBinding;

		// buffer info
		std::shared_ptr<Buffer> buffer;
		vk::DescriptorBufferInfo bufferInfo;

		// image info
		std::shared_ptr<Image> image;
		vk::DescriptorImageInfo imageInfo;
	};

private:
	struct ArrayDescriptorBinding
	{
		vk::DescriptorType type;
		vk::DescriptorSetLayoutBinding layoutBinding;            // descriptorCount = buffers.size()
		std::vector<std::shared_ptr<Buffer>> buffers;
		std::vector<vk::DescriptorBufferInfo> bufferInfos;
	};

	DeviceContext &_deviceContext;
	vk::DescriptorSetLayout _descriptorSetLayout;
	std::vector<vk::DescriptorSet> _descriptorSets;
	size_t _maxOptions = 1;
	std::unordered_map<uint32_t, std::vector<DescriptorBinding>> _bindings;
	// Array-mode bindings (descriptor arrays, indexed at runtime in the shader).
	// Mutually exclusive at a given binding number with `_bindings`.
	std::unordered_map<uint32_t, ArrayDescriptorBinding> _arrayBindings;

public:
	explicit Descriptor(DeviceContext &deviceContext);
	~Descriptor();

	void BindBufferToDescriptorSet(uint32_t binding, vk::DescriptorType type, vk::ShaderStageFlags stage, std::shared_ptr<Buffer> buffer);
	void BindImageToDescriptorSet(uint32_t binding, vk::DescriptorType descriptor, vk::ShaderStageFlagBits stage, std::shared_ptr<Image> image);
	void BindImageToDescriptorSet(uint32_t binding, vk::DescriptorType descriptor, vk::ShaderStageFlagBits stage, vk::ImageView imageView);

	// Binds an array of buffers at a single binding slot — the shader sees
	// `RWStructuredBuffer<T> name[N]` and indexes by `name[i]` at runtime.
	// Requires `shaderStorageBufferArrayDynamicIndexing` (enabled at device creation).
	// Used by search mode to fold per-slot dispatches into one 2D dispatch with
	// `gl_GlobalInvocationID.y` selecting the slot. Other consumers should keep
	// using the single-buffer `BindBufferToDescriptorSet`.
	void BindBufferArrayToDescriptorSet(uint32_t binding, vk::DescriptorType type, vk::ShaderStageFlags stage,
		const std::vector<std::shared_ptr<Buffer>>& buffers);

	void Build();

	vk::DescriptorSet GetDescriptorSet(uint8_t currentFrame, uint8_t option) const;
	vk::DescriptorSet GetDescriptorSet(uint32_t index) const;

	vk::DescriptorSetLayout GetDescriptorSetLayout() const { return _descriptorSetLayout; }
};

} // namespace Cave