#include "descriptors.h"
#include "../common/logger.h"

namespace Cave
{

Descriptor::Descriptor(DeviceContext &deviceContext) : _deviceContext{deviceContext}
{
}

Descriptor::~Descriptor()
{
	vk::Device device = _deviceContext.GetDevice();
	if (!_descriptorSets.empty())
	{
		device.freeDescriptorSets(_deviceContext.GetDescriptorPool(), _descriptorSets);
		_descriptorSets.clear();
	}
	if (_descriptorSetLayout)
	{
		device.destroyDescriptorSetLayout(_descriptorSetLayout);
		_descriptorSetLayout = nullptr;
	}
}

void Descriptor::Build()
{
	LOG_DEBUG("Building Vulkan Descriptor");

	std::vector<vk::DescriptorSetLayoutBinding> layoutBindings;
	for (auto &[_, options] : _bindings)
	{
		layoutBindings.push_back(options[0].layoutBinding);
		if (options.size() != 1)
		{
			if (_maxOptions == 1)
			{
				_maxOptions = options.size();
			}
			else if (_maxOptions != options.size())
			{
				throw std::runtime_error("Inconsistent number of alternative buffers " + std::to_string(options[0].layoutBinding.binding));
			}
		}
	}
	// Array-mode bindings contribute one entry each (descriptorCount = N at that binding).
	for (auto &[_, arrayBinding] : _arrayBindings)
	{
		layoutBindings.push_back(arrayBinding.layoutBinding);
	}

	vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = vk::DescriptorSetLayoutCreateInfo(
		{},                    // flags
		layoutBindings.size(), // bindingCount
		layoutBindings.data()  // pBindings
	);
	_descriptorSetLayout = _deviceContext.GetDevice().createDescriptorSetLayout(descriptorSetLayoutCreateInfo);

	std::vector<vk::DescriptorSetLayout> layouts(_deviceContext.GetFramesInFlight() * _maxOptions, _descriptorSetLayout);
	vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = vk::DescriptorSetAllocateInfo(
		_deviceContext.GetDescriptorPool(), // descriptorPool
		layouts.size(),                     // descriptorSetCount
		layouts.data()                      // pSetLayouts
	);
	_descriptorSets = _deviceContext.GetDevice().allocateDescriptorSets(descriptorSetAllocateInfo);

	for (int frameIndex = 0; frameIndex < _deviceContext.GetFramesInFlight(); frameIndex++)
	{
		std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
		for (auto &binding : _bindings)
		{
			for (auto optionIndex = 0; optionIndex < _maxOptions; optionIndex++)
			{
				if (binding.second.size() == 1)
				{
					if (binding.second[0].buffer != nullptr)
					{
						binding.second[0].buffer->BoundToDescriptor(static_cast<std::weak_ptr<Descriptor>>(shared_from_this()), frameIndex * _maxOptions + optionIndex, binding.first, binding.second[0].type);
						writeDescriptorSets.emplace_back(
							_descriptorSets[frameIndex * _maxOptions + optionIndex], // dstSet
							binding.first,                                           // dstBinding
							0,                                                       // dstArrayElement
							1,                                                       // descriptorCount
							binding.second[0].type,                                  // descriptorType
							nullptr,                                                 // pImageInfo
							&binding.second[0].bufferInfo                            // pBufferInfo
						);
					}
					else
					{
						writeDescriptorSets.emplace_back(
							_descriptorSets[frameIndex * _maxOptions + optionIndex], // dstSet
							binding.first,                                           // dstBinding
							0,                                                       // dstArrayElement
							1,                                                       // descriptorCount
							binding.second[0].type,                                  // descriptorType
							&binding.second[0].imageInfo,                            // pImageInfo
							nullptr                                                  // pBufferInfo
						);
					}
				}
				else
				{
					if (binding.second.at(optionIndex).buffer != nullptr)
					{
						binding.second.at(optionIndex).buffer->BoundToDescriptor(static_cast<std::weak_ptr<Descriptor>>(shared_from_this()), frameIndex * _maxOptions + optionIndex, binding.first, binding.second.at(optionIndex).type);
						writeDescriptorSets.emplace_back(
							_descriptorSets[frameIndex * _maxOptions + optionIndex], // dstSet
							binding.first,                                           // dstBinding
							0,                                                       // dstArrayElement
							1,                                                       // descriptorCount
							binding.second.at(optionIndex).type,                     // descriptorType
							nullptr,                                                 // pImageInfo
							&binding.second.at(optionIndex).bufferInfo               // pBufferInfo
						);
					}
					else
					{
						writeDescriptorSets.emplace_back(
							_descriptorSets[frameIndex * _maxOptions + optionIndex], // dstSet
							binding.first,                                           // dstBinding
							0,                                                       // dstArrayElement
							1,                                                       // descriptorCount
							binding.second.at(optionIndex).type,                     // descriptorType
							&binding.second.at(optionIndex).imageInfo,               // pImageInfo
							nullptr                                                  // pBufferInfo
						);
					}
				}
			}
		}

		// Array-mode bindings: write the entire array at this binding once per
		// descriptor set. dstArrayElement=0, descriptorCount=N, pBufferInfo points
		// to N contiguous VkDescriptorBufferInfos.
		std::vector<vk::WriteDescriptorSet> arrayWrites;
		for (auto &[_, arrayBinding] : _arrayBindings)
		{
			for (auto optionIndex = 0; optionIndex < _maxOptions; optionIndex++)
			{
				vk::DescriptorSet dstSet = _descriptorSets[frameIndex * _maxOptions + optionIndex];
				arrayWrites.emplace_back(
					dstSet,                                                               // dstSet
					arrayBinding.layoutBinding.binding,                                   // dstBinding
					0,                                                                    // dstArrayElement
					static_cast<uint32_t>(arrayBinding.bufferInfos.size()),               // descriptorCount
					arrayBinding.type,                                                    // descriptorType
					nullptr,                                                              // pImageInfo
					arrayBinding.bufferInfos.data()                                       // pBufferInfo
				);
				for (size_t i = 0; i < arrayBinding.buffers.size(); ++i)
				{
					if (arrayBinding.buffers[i])
					{
						arrayBinding.buffers[i]->BoundToDescriptor(
							static_cast<std::weak_ptr<Descriptor>>(shared_from_this()),
							frameIndex * _maxOptions + optionIndex,
							arrayBinding.layoutBinding.binding,
							arrayBinding.type);
					}
				}
			}
		}
		if (!arrayWrites.empty())
			_deviceContext.GetDevice().updateDescriptorSets(arrayWrites, nullptr);

		_deviceContext.GetDevice().updateDescriptorSets(writeDescriptorSets, nullptr);
	}
}

void Descriptor::BindBufferArrayToDescriptorSet(uint32_t binding, vk::DescriptorType type, vk::ShaderStageFlags stage,
	const std::vector<std::shared_ptr<Buffer>>& buffers)
{
	LOG_DEBUG("Binding Vulkan Buffer Array (count={}) To Vulkan Descriptor Set", buffers.size());
	if (buffers.empty())
		throw std::runtime_error("BindBufferArrayToDescriptorSet: empty buffer array at binding " + std::to_string(binding));

	const vk::DescriptorSetLayoutBinding layoutBinding = vk::DescriptorSetLayoutBinding(
		binding,                                  // binding
		type,                                     // descriptorType
		static_cast<uint32_t>(buffers.size()),    // descriptorCount = N
		stage                                     // stageFlags
	);

	if (_bindings.contains(binding))
	{
		throw std::runtime_error("Binding " + std::to_string(binding) +
			" already used by single-buffer BindBufferToDescriptorSet; cannot also be array-bound");
	}
	if (_arrayBindings.contains(binding))
	{
		throw std::runtime_error("Binding " + std::to_string(binding) + " already array-bound");
	}

	ArrayDescriptorBinding arrayBinding;
	arrayBinding.type = type;
	arrayBinding.layoutBinding = layoutBinding;
	arrayBinding.buffers = buffers;
	arrayBinding.bufferInfos.reserve(buffers.size());
	for (const auto& buffer : buffers)
	{
		arrayBinding.bufferInfos.emplace_back(
			buffer->GetBuffer(),  // buffer
			0,                    // offset
			buffer->GetSize()     // range
		);
	}
	_arrayBindings.emplace(binding, std::move(arrayBinding));
}

void Descriptor::BindBufferToDescriptorSet(uint32_t binding, vk::DescriptorType type, vk::ShaderStageFlags stage, std::shared_ptr<Buffer> buffer)
{
	LOG_DEBUG("Binding Vulkan Buffer To Vulkan Descriptor Set");
	const vk::DescriptorSetLayoutBinding layoutBinding = vk::DescriptorSetLayoutBinding(
		binding, // binding
		type,    // descriptorType
		1,       // descriptorCount
		stage    // stageFlags
	);

	const auto &bindingVector = _bindings[binding];
	if (!bindingVector.empty() && bindingVector[bindingVector.size() - 1].layoutBinding != layoutBinding)
	{
		throw std::runtime_error("Binding " + std::to_string(binding) + " already exists with different layout binding");
	}

	vk::DescriptorBufferInfo descriptorBufferInfo = vk::DescriptorBufferInfo(
		buffer->GetBuffer(), // buffer
		0,                   // offset
		buffer->GetSize()    // range
	);

	_bindings[binding].push_back(
		DescriptorBinding{
			type,
			layoutBinding,
			buffer,
			descriptorBufferInfo});
}

void Descriptor::BindImageToDescriptorSet(uint32_t binding, vk::DescriptorType descriptor, vk::ShaderStageFlagBits stage, std::shared_ptr<Image> image)
{
	LOG_DEBUG("Binding Vulkan Image To Vulkan Descriptor Set");

	const vk::DescriptorSetLayoutBinding layoutBinding = vk::DescriptorSetLayoutBinding(
		binding,    // binding
		descriptor, // descriptorType
		1,          // descriptorCount
		stage       // stageFlags
	);

	const auto &bindingVector = _bindings[binding];
	if (!bindingVector.empty() && bindingVector[bindingVector.size() - 1].layoutBinding != layoutBinding)
	{
		throw std::runtime_error("Binding " + std::to_string(binding) + " already exists with different layout binding");
	}

	vk::DescriptorImageInfo descriptorImageInfo = vk::DescriptorImageInfo(
		{},                        // sampler
		*image->GetImageView(),    // imageView
		vk::ImageLayout::eGeneral  // imageLayout
	);

	_bindings[binding].push_back(
		DescriptorBinding{
			descriptor,
			layoutBinding,
			nullptr,
			{},
			image,
			descriptorImageInfo});
}

// void Descriptor::BindImageToDescriptorSet(uint32_t binding, vk::DescriptorType descriptor, vk::ShaderStageFlagBits stage, vk::ImageView imageView) {
//	LOG_DEBUG("Binding Vulkan Image To Vulkan Descriptor Set");
//
//	const vk::DescriptorSetLayoutBinding layoutBinding{ binding, descriptor, 1, stage };
//
//	const auto& bindingVector = _bindings[binding];
//	if (!bindingVector.empty() && bindingVector[bindingVector.size() - 1].layoutBinding != layoutBinding) {
//		throw std::runtime_error("Binding " + std::to_string(binding) + " already exists with different layout binding");
//	}
//
//	_bindings[binding].push_back(
//		DescriptorBinding{
//			descriptor,
//			layoutBinding,
//			nullptr,
//			{},
//			nullptr,
//			vk::DescriptorImageInfo({}, imageView, vk::ImageLayout::eGeneral)
//		}
//	);
// }

vk::DescriptorSet Descriptor::GetDescriptorSet(uint8_t currentFrame, uint8_t option) const
{
	if (option >= _maxOptions)
	{
		throw std::runtime_error("Invalid option " + std::to_string(option));
	}
	return _descriptorSets[currentFrame * _maxOptions + option];
}

vk::DescriptorSet Descriptor::GetDescriptorSet(uint32_t index) const
{
	return _descriptorSets[index];
}

} // namespace Cave