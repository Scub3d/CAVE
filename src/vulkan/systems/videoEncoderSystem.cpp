#include "videoEncoderSystem.h"
#include "../../common/logger.h"
#include "../../common/structs.h"

namespace Cave
{
    VideoEncoderSystem::VideoEncoderSystem(VulkanInstance &vulkanInstance, DeviceContext &deviceContext,
                                           VulkanSimulationRenderer &simulationRenderer,
                                           uint32_t width, uint32_t height, uint32_t fps)
        : _deviceContext{deviceContext}, _simulationRenderer{simulationRenderer},
          _width{width}, _height{height}, _fps{fps}
    {
        _graphicsCommandPool = _deviceContext.GetGraphicsCommandPool();

        uint32_t framesInFlight = _deviceContext.GetFramesInFlight();
        _renderInFlightFences.resize(framesInFlight);
        for (uint32_t frameIndex = 0; frameIndex < framesInFlight; frameIndex++)
        {
            _renderInFlightFences[frameIndex] = _deviceContext.CreateFence();
        }

        BuildBoundingBoxMesh();
        BuildCameraBuffers();
        BuildColorAttachments();
        BuildRenderDescriptors();
        BuildRenderPipeline();
        BuildRenderCommandBuffers();

        // VideoEncoder receives the color attachment handles it will read from after render.
        std::vector<vk::Image> inputImages;
        std::vector<vk::ImageView> inputImageViews;
        for (uint32_t frameIndex = 0; frameIndex < framesInFlight; frameIndex++)
        {
            inputImages.push_back(*_colorAttachments[frameIndex]->GetImage());
            inputImageViews.push_back(*_colorAttachments[frameIndex]->GetImageView());
        }
        _videoEncoder = std::make_unique<VideoEncoder>(_deviceContext, inputImages, inputImageViews, _width, _height, _fps);

        // Overlay targets the same color/depth attachments we render/blit into. Both formats must
        // match exactly for ImGui's pipeline to validate inside our renderpass. Depth format is
        // populated by BuildColorAttachments above. The BlitAndEncodeFrame path doesn't use depth
        // (its overlay-only pass attaches color only), but the RenderAndEncodeFrame path does.
        _videoOverlay = std::make_unique<VideoOverlay>(
            vulkanInstance, _deviceContext,
            _deviceContext.GetColorFormat(), _depthFormat,
            vk::Extent2D{_width, _height}, framesInFlight);
    }

    VideoEncoderSystem::~VideoEncoderSystem()
    {
        for (vk::Fence renderFence : _renderInFlightFences)
        {
            _deviceContext.GetDevice().destroyFence(renderFence);
        }
    }

    void VideoEncoderSystem::BuildBoundingBoxMesh()
    {
        // Unit cube vertices [-1, 1] in each axis
        std::vector<Vertex> vertices = {
            {{-1.0f, -1.0f, -1.0f, 1.0f}},
            {{ 1.0f, -1.0f, -1.0f, 1.0f}},
            {{ 1.0f,  1.0f, -1.0f, 1.0f}},
            {{-1.0f,  1.0f, -1.0f, 1.0f}},
            {{-1.0f, -1.0f,  1.0f, 1.0f}},
            {{ 1.0f, -1.0f,  1.0f, 1.0f}},
            {{ 1.0f,  1.0f,  1.0f, 1.0f}},
            {{-1.0f,  1.0f,  1.0f, 1.0f}},
        };

        std::vector<uint32_t> indices = {
            0, 1, 2, 2, 3, 0, // -Z face
            4, 6, 5, 6, 4, 7, // +Z face
            0, 4, 5, 5, 1, 0, // -Y face
            2, 6, 7, 7, 3, 2, // +Y face
            0, 3, 7, 7, 4, 0, // -X face
            1, 5, 6, 6, 2, 1, // +X face
        };

        _boundingBoxIndexCount = static_cast<uint32_t>(indices.size());

        _boundingBoxVertexBuffer = std::make_unique<Buffer>(
            _deviceContext,
            sizeof(Vertex) * vertices.size(),
            vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_GPU_ONLY, 0, false, 0, "Video Encoder Bounding Box Vertices");
        _boundingBoxVertexBuffer->Upload(vertices.data(), static_cast<uint32_t>(sizeof(Vertex) * vertices.size()));

        _boundingBoxIndexBuffer = std::make_unique<Buffer>(
            _deviceContext,
            sizeof(uint32_t) * indices.size(),
            vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_GPU_ONLY, 0, false, 0, "Video Encoder Bounding Box Indices");
        _boundingBoxIndexBuffer->Upload(indices.data(), static_cast<uint32_t>(sizeof(uint32_t) * indices.size()));
    }

    void VideoEncoderSystem::BuildCameraBuffers()
    {
        uint32_t framesInFlight = _deviceContext.GetFramesInFlight();
        _cameraUniformBuffers.resize(framesInFlight);
        for (uint32_t frameIndex = 0; frameIndex < framesInFlight; frameIndex++)
        {
            _cameraUniformBuffers[frameIndex] = Buffer::Uniform(_deviceContext, sizeof(CameraData));
        }
    }

    void VideoEncoderSystem::BuildColorAttachments()
    {
        uint32_t framesInFlight = _deviceContext.GetFramesInFlight();
        vk::Extent3D attachmentExtent(_width, _height, 1);
        std::vector<uint32_t> queueFamilyIndices = std::vector<uint32_t>();

        _colorAttachments.resize(framesInFlight);
        for (uint32_t frameIndex = 0; frameIndex < framesInFlight; frameIndex++)
        {
            _colorAttachments[frameIndex] = std::make_unique<Image>(
                _deviceContext,
                _deviceContext.GetColorFormat(),
                1, 1, attachmentExtent,
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
                vk::MemoryPropertyFlagBits::eDeviceLocal,
                vk::ImageCreateFlags(),
                vk::ImageAspectFlagBits::eColor,
                vk::ImageViewType::e2D,
                vk::SharingMode::eConcurrent,
                std::vector<uint32_t>{
                    _deviceContext.GetQueueFamilies().GraphicsFamily.Index.value(),
                    _deviceContext.GetQueueFamilies().ComputeFamily.Index.value()});
        }

        _depthFormat = Image::FindSupportedFormat(
            _deviceContext.GetPhysicalDevice(),
            {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
            vk::ImageTiling::eOptimal,
            vk::FormatFeatureFlagBits::eDepthStencilAttachment);

        _depthAttachment = std::make_unique<Image>(
            _deviceContext,
            _depthFormat,
            1, 1, attachmentExtent,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eDepthStencilAttachment,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            vk::ImageCreateFlags(),
            vk::ImageAspectFlagBits::eDepth,
            vk::ImageViewType::e2D,
            vk::SharingMode::eExclusive,
            queueFamilyIndices);
    }

    void VideoEncoderSystem::BuildRenderDescriptors()
    {
        _renderDescriptor = std::make_shared<Descriptor>(_deviceContext);
        uint32_t framesInFlight = _deviceContext.GetFramesInFlight();
        for (uint32_t frameIndex = 0; frameIndex < framesInFlight; frameIndex++)
        {
            // binding 0 — camera UBO
            _renderDescriptor->BindBufferToDescriptorSet(0, vk::DescriptorType::eUniformBuffer,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                _cameraUniformBuffers[frameIndex]);

            uint32_t nextFrame = (frameIndex + 1) % framesInFlight;

            if (_simulationRenderer.UsesImages())
            {
                // binding 1 — cell state image (read in fragment shader)
                _renderDescriptor->BindImageToDescriptorSet(1, vk::DescriptorType::eStorageImage,
                    vk::ShaderStageFlagBits::eFragment, _simulationRenderer.GetCellStateImage(nextFrame));

                // binding 2 — skip grid image (read in fragment shader)
                _renderDescriptor->BindImageToDescriptorSet(2, vk::DescriptorType::eStorageImage,
                    vk::ShaderStageFlagBits::eFragment, _simulationRenderer.GetSkipGridImage(nextFrame));
            }
            else
            {
                // binding 1 — cell state buffer (read in fragment shader)
                _renderDescriptor->BindBufferToDescriptorSet(1, vk::DescriptorType::eStorageBuffer,
                    vk::ShaderStageFlagBits::eFragment, _simulationRenderer.GetCellStateBuffer(nextFrame));

                // binding 2 — skip grid buffer (read in fragment shader)
                _renderDescriptor->BindBufferToDescriptorSet(2, vk::DescriptorType::eStorageBuffer,
                    vk::ShaderStageFlagBits::eFragment, _simulationRenderer.GetSkipGridBuffer(nextFrame));
            }
        }
        _renderDescriptor->Build();
    }

    void VideoEncoderSystem::BuildRenderPipeline()
    {
        auto vertexShader   = std::make_shared<Shader>(_deviceContext, _simulationRenderer.GetRayMarchVertexShaderModule());
        auto fragmentShader = std::make_shared<Shader>(_deviceContext, _simulationRenderer.GetRayMarchFragmentShaderModule());
        _graphicsPipeline = std::make_unique<GraphicsPipeline>(_deviceContext, vertexShader, fragmentShader);

        // Vertex input: single binding for the bounding box cube (vec4 position)
        _graphicsPipeline->SetVertexInput(
            {
                vk::VertexInputBindingDescription(
                    0,                              // binding
                    sizeof(Vertex),                 // stride
                    vk::VertexInputRate::eVertex    // inputRate
                ),
            },
            {
                vk::VertexInputAttributeDescription(
                    0,                                  // location
                    0,                                  // binding
                    vk::Format::eR32G32B32A32Sfloat,    // format
                    offsetof(Vertex, position)          // offset
                ),
            });

        vk::Format depthFormat = Image::FindSupportedFormat(
            _deviceContext.GetPhysicalDevice(),
            {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
            vk::ImageTiling::eOptimal,
            vk::FormatFeatureFlagBits::eDepthStencilAttachment);

        _graphicsPipeline->SetCullMode(vk::CullModeFlagBits::eNone);
        _graphicsPipeline->SetBlendEnabled(false);
        _graphicsPipeline->SetDynamicRenderingInfo(_deviceContext.GetColorFormat(), depthFormat);
        _graphicsPipeline->AddDescriptorSet(0, _renderDescriptor);
        _graphicsPipeline->AddPushConstant(
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            0, sizeof(RayMarchPushConstants));
        _graphicsPipeline->Build();
    }

    void VideoEncoderSystem::BuildRenderCommandBuffers()
    {
        uint32_t framesInFlight = _deviceContext.GetFramesInFlight();
        _graphicsCommandBuffers.resize(framesInFlight);
        for (uint32_t frameIndex = 0; frameIndex < framesInFlight; frameIndex++)
        {
            _graphicsCommandBuffers[frameIndex] = _deviceContext.CreateCommandBuffer(_graphicsCommandPool);
        }
    }

    void VideoEncoderSystem::UpdateCameraBuffer(uint32_t frameIndex, const CameraData &cameraData)
    {
        _cameraUniformBuffers[frameIndex]->Upload(&cameraData, sizeof(CameraData));
    }

    void VideoEncoderSystem::RenderAndEncodeFrame(uint32_t frameIndex, vk::Semaphore computeCompletedSemaphore,
                                                     uint64_t computeCompletedSemaphoreWaitValue, const CameraData &cameraData,
                                                     const RayMarchPushConstants &rayMarchPushConstants,
                                                     std::shared_ptr<QueryManager> queryManager, vk::QueryPool queryPool)
    {
        vk::Device device = _deviceContext.GetDevice();

        auto waitResult = device.waitForFences(_renderInFlightFences[frameIndex], vk::True, UINT64_MAX);
        if (waitResult != vk::Result::eSuccess)
        {
            LOG_ERROR("VideoEncoderSystem: timed out waiting for render fence");
            return;
        }
        device.resetFences(_renderInFlightFences[frameIndex]);

        UpdateCameraBuffer(frameIndex, cameraData);

        vk::CommandBuffer graphicsCommandBuffer = _graphicsCommandBuffers[frameIndex];
        graphicsCommandBuffer.reset();

        vk::CommandBufferBeginInfo commandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        graphicsCommandBuffer.begin(commandBufferBeginInfo);

        vk::ImageSubresourceRange colorRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
        vk::ImageSubresourceRange depthRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1);

        // Transition color: undefined → color attachment
        vk::ImageMemoryBarrier colorBarrierIn(
            {}, vk::AccessFlagBits::eColorAttachmentWrite,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            *_colorAttachments[frameIndex]->GetImage(), colorRange);
        graphicsCommandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput,
            {}, {}, {}, colorBarrierIn);

        // Transition depth: undefined → depth stencil
        vk::ImageMemoryBarrier depthBarrierIn(
            {}, vk::AccessFlagBits::eDepthStencilAttachmentWrite,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            *_depthAttachment->GetImage(), depthRange);
        graphicsCommandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eEarlyFragmentTests,
            {}, {}, {}, depthBarrierIn);

        // Begin dynamic rendering
        vk::ClearValue colorClear(vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}});
        vk::ClearValue depthClear(vk::ClearDepthStencilValue{1.0f, 0});

        vk::RenderingAttachmentInfo colorAttachmentInfo(
            *_colorAttachments[frameIndex]->GetImageView(), vk::ImageLayout::eColorAttachmentOptimal,
            vk::ResolveModeFlagBits::eNone, {}, {},
            vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, colorClear);

        vk::RenderingAttachmentInfo depthAttachmentInfo(
            *_depthAttachment->GetImageView(), vk::ImageLayout::eDepthStencilAttachmentOptimal,
            vk::ResolveModeFlagBits::eNone, {}, {},
            vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare, depthClear);

        vk::Extent2D renderExtent(_width, _height);
        vk::RenderingInfo renderingInfo(
            {}, vk::Rect2D{{0, 0}, renderExtent}, 1, 0,
            colorAttachmentInfo, &depthAttachmentInfo, nullptr);

        graphicsCommandBuffer.beginRendering(renderingInfo);

        // Viewport with negative height for Y-flip (matches RenderSystem)
        vk::Viewport viewport(
            0.0f, static_cast<float>(_height),
            static_cast<float>(_width), -static_cast<float>(_height),
            0.0f, 1.0f);
        vk::Rect2D scissor({0, 0}, renderExtent);
        graphicsCommandBuffer.setViewport(0, viewport);
        graphicsCommandBuffer.setScissor(0, scissor);

        // Bind pipeline + camera descriptor
        _graphicsPipeline->Bind(graphicsCommandBuffer, vk::PipelineBindPoint::eGraphics,
                                static_cast<uint8_t>(frameIndex), Pipeline::DescriptorOption{frameIndex});

        // Push ray march constants
        graphicsCommandBuffer.pushConstants<RayMarchPushConstants>(
            _graphicsPipeline->GetPipelineLayout(),                                     // layout
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,      // stageFlags
            0,                                                                           // offset
            rayMarchPushConstants                                                        // values
        );

        // Bind bounding box mesh and draw
        vk::Buffer vertexBuffers[] = {_boundingBoxVertexBuffer->GetBuffer()};
        vk::DeviceSize offsets[] = {0};
        graphicsCommandBuffer.bindVertexBuffers(0, 1, vertexBuffers, offsets);
        graphicsCommandBuffer.bindIndexBuffer(_boundingBoxIndexBuffer->GetBuffer(), 0, vk::IndexType::eUint32);
        if (queryManager) queryManager->WriteTimestamp(graphicsCommandBuffer, queryPool, vk::PipelineStageFlagBits::eTopOfPipe, "encodeRender_start");
        graphicsCommandBuffer.drawIndexed(_boundingBoxIndexCount, 1, 0, 0, 0);
        if (queryManager) queryManager->WriteTimestamp(graphicsCommandBuffer, queryPool, vk::PipelineStageFlagBits::eBottomOfPipe, "encodeRender_end");

        // Draw overlay on top of the rendered scene, inside the same dynamic rendering pass.
        if (_overlayEnabled && _videoOverlay)
        {
            _videoOverlay->RecordDrawCommands(graphicsCommandBuffer);
        }

        graphicsCommandBuffer.endRendering();

        // Transition color: color attachment → transfer src (for encoder to read)
        vk::ImageMemoryBarrier colorBarrierOut(
            vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eTransferRead,
            vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            *_colorAttachments[frameIndex]->GetImage(), colorRange);
        graphicsCommandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eTransfer,
            {}, {}, {}, colorBarrierOut);

        graphicsCommandBuffer.end();

        vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eFragmentShader;

        vk::TimelineSemaphoreSubmitInfo timelineSemaphoreSubmitInfo = vk::TimelineSemaphoreSubmitInfo(
            1,                                              // waitSemaphoreValueCount
            &computeCompletedSemaphoreWaitValue,            // pWaitSemaphoreValues
            0,                                              // signalSemaphoreValueCount
            nullptr                                         // pSignalSemaphoreValues
        );

        vk::SubmitInfo submitInfo = vk::SubmitInfo(
            1,                            // waitSemaphoreCount
            &computeCompletedSemaphore,   // pWaitSemaphores
            &waitStage,                   // pWaitDstStageMask
            1,                            // commandBufferCount
            &graphicsCommandBuffer,       // pCommandBuffers
            0,                            // signalSemaphoreCount
            nullptr                       // pSignalSemaphores
        );
        submitInfo.pNext = &timelineSemaphoreSubmitInfo;

        _deviceContext.GetGraphicsQueue().submit(submitInfo, _renderInFlightFences[frameIndex]);

        // Block until this frame's render is complete before handing the image to the encoder.
        device.waitForFences(_renderInFlightFences[frameIndex], vk::True, UINT64_MAX);

        _videoEncoder->QueueEncode(frameIndex);

        // Immediately collect the encoded frame data. This frees command buffers
        // and resets semaphore state so the next frame can proceed cleanly.
        const char *bitstreamData = nullptr;
        size_t bitstreamSize = 0;

        // First frame: collect the H.264 header
        _videoEncoder->FinishEncode(bitstreamData, bitstreamSize);
        WriteBitstreamChunk(bitstreamData, bitstreamSize);

        // Collect the frame's encoded packet
        _videoEncoder->FinishEncode(bitstreamData, bitstreamSize);
        WriteBitstreamChunk(bitstreamData, bitstreamSize);
    }

    void VideoEncoderSystem::BlitAndEncodeFrame(uint32_t frameIndex, vk::Image sourceImage, vk::Extent2D sourceExtent)
    {
        vk::Device device = _deviceContext.GetDevice();

        auto waitResult = device.waitForFences(_renderInFlightFences[frameIndex], vk::True, UINT64_MAX);
        if (waitResult != vk::Result::eSuccess)
        {
            LOG_ERROR("VideoEncoderSystem::BlitAndEncodeFrame: timed out waiting for render fence");
            return;
        }
        device.resetFences(_renderInFlightFences[frameIndex]);

        vk::CommandBuffer commandBuffer = _graphicsCommandBuffers[frameIndex];
        commandBuffer.reset();
        commandBuffer.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

        vk::ImageSubresourceRange colorRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);

        // Transition encoder's color attachment: undefined → transfer dst
        vk::ImageMemoryBarrier dstBarrier(
            {},                                                       // srcAccessMask
            vk::AccessFlagBits::eTransferWrite,                       // dstAccessMask
            vk::ImageLayout::eUndefined,                              // oldLayout
            vk::ImageLayout::eTransferDstOptimal,                     // newLayout
            VK_QUEUE_FAMILY_IGNORED,                                  // srcQueueFamilyIndex
            VK_QUEUE_FAMILY_IGNORED,                                  // dstQueueFamilyIndex
            *_colorAttachments[frameIndex]->GetImage(),               // image
            colorRange                                                // subresourceRange
        );
        commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
            {}, {}, {}, dstBarrier);

        // Blit source (compositor output, eTransferSrcOptimal) → encoder color attachment
        // vkCmdBlitImage handles format conversion (R16G16B16A16Sfloat → B8G8R8A8Srgb)
        vk::ImageBlit blitRegion = vk::ImageBlit(
            vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1), // srcSubresource
            std::array<vk::Offset3D, 2>{
                vk::Offset3D(0, 0, 0),
                vk::Offset3D(static_cast<int32_t>(sourceExtent.width),
                             static_cast<int32_t>(sourceExtent.height), 1)},      // srcOffsets
            vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1), // dstSubresource
            std::array<vk::Offset3D, 2>{
                vk::Offset3D(0, 0, 0),
                vk::Offset3D(static_cast<int32_t>(_width),
                             static_cast<int32_t>(_height), 1)}                   // dstOffsets
        );
        commandBuffer.blitImage(
            sourceImage, vk::ImageLayout::eTransferSrcOptimal,
            *_colorAttachments[frameIndex]->GetImage(), vk::ImageLayout::eTransferDstOptimal,
            blitRegion, vk::Filter::eLinear);

        if (_overlayEnabled && _videoOverlay)
        {
            // Transition: transfer dst → color attachment (preserve blitted content via loadOp=eLoad)
            vk::ImageMemoryBarrier toColorBarrier(
                vk::AccessFlagBits::eTransferWrite,                       // srcAccessMask
                vk::AccessFlagBits::eColorAttachmentWrite,                // dstAccessMask
                vk::ImageLayout::eTransferDstOptimal,                     // oldLayout
                vk::ImageLayout::eColorAttachmentOptimal,                 // newLayout
                VK_QUEUE_FAMILY_IGNORED,                                  // srcQueueFamilyIndex
                VK_QUEUE_FAMILY_IGNORED,                                  // dstQueueFamilyIndex
                *_colorAttachments[frameIndex]->GetImage(),               // image
                colorRange                                                // subresourceRange
            );
            commandBuffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eColorAttachmentOutput,
                {}, {}, {}, toColorBarrier);

            // Transition depth attachment to DepthStencilAttachmentOptimal. The overlay doesn't
            // read or write depth, but the ImGui pipeline was created with depthAttachmentFormat
            // matching this image's format, so we MUST attach a compatible depth view to satisfy
            // VUID-vkCmdDrawIndexed-dynamicRenderingUnusedAttachments-08914.
            vk::ImageSubresourceRange depthRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1);
            vk::ImageMemoryBarrier depthBarrier(
                {},                                                       // srcAccessMask
                vk::AccessFlagBits::eDepthStencilAttachmentWrite,         // dstAccessMask
                vk::ImageLayout::eUndefined,                              // oldLayout (discard prior contents)
                vk::ImageLayout::eDepthStencilAttachmentOptimal,          // newLayout
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                *_depthAttachment->GetImage(), depthRange);
            commandBuffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eEarlyFragmentTests,
                {}, {}, {}, depthBarrier);

            // Begin a tiny dynamic render pass to draw the overlay on top of the blitted frame.
            vk::RenderingAttachmentInfo overlayColorAttachmentInfo(
                *_colorAttachments[frameIndex]->GetImageView(),                    // imageView
                vk::ImageLayout::eColorAttachmentOptimal,                          // imageLayout
                vk::ResolveModeFlagBits::eNone, {}, {},                            // resolve
                vk::AttachmentLoadOp::eLoad,                                       // loadOp
                vk::AttachmentStoreOp::eStore,                                     // storeOp
                vk::ClearValue{}                                                   // clearValue (unused with eLoad)
            );
            vk::ClearValue depthClear(vk::ClearDepthStencilValue{1.0f, 0});
            vk::RenderingAttachmentInfo overlayDepthAttachmentInfo(
                *_depthAttachment->GetImageView(), vk::ImageLayout::eDepthStencilAttachmentOptimal,
                vk::ResolveModeFlagBits::eNone, {}, {},
                vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare, depthClear);
            vk::Extent2D renderExtent(_width, _height);
            vk::RenderingInfo overlayRenderingInfo(
                {}, vk::Rect2D{{0, 0}, renderExtent}, 1, 0,
                overlayColorAttachmentInfo, &overlayDepthAttachmentInfo, nullptr);

            commandBuffer.beginRendering(overlayRenderingInfo);

            vk::Viewport overlayViewport(
                0.0f, 0.0f,
                static_cast<float>(_width), static_cast<float>(_height),
                0.0f, 1.0f);
            vk::Rect2D overlayScissor({0, 0}, renderExtent);
            commandBuffer.setViewport(0, overlayViewport);
            commandBuffer.setScissor(0, overlayScissor);

            _videoOverlay->RecordDrawCommands(commandBuffer);

            commandBuffer.endRendering();

            // Transition: color attachment → transfer src (for encoder readback)
            vk::ImageMemoryBarrier toTransferSrcBarrier(
                vk::AccessFlagBits::eColorAttachmentWrite,                // srcAccessMask
                vk::AccessFlagBits::eTransferRead,                        // dstAccessMask
                vk::ImageLayout::eColorAttachmentOptimal,                 // oldLayout
                vk::ImageLayout::eTransferSrcOptimal,                     // newLayout
                VK_QUEUE_FAMILY_IGNORED,                                  // srcQueueFamilyIndex
                VK_QUEUE_FAMILY_IGNORED,                                  // dstQueueFamilyIndex
                *_colorAttachments[frameIndex]->GetImage(),               // image
                colorRange                                                // subresourceRange
            );
            commandBuffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eTransfer,
                {}, {}, {}, toTransferSrcBarrier);
        }
        else
        {
            // Transition encoder's color attachment: transfer dst → transfer src (for encoder readback)
            vk::ImageMemoryBarrier srcBarrier(
                vk::AccessFlagBits::eTransferWrite,                       // srcAccessMask
                vk::AccessFlagBits::eTransferRead,                        // dstAccessMask
                vk::ImageLayout::eTransferDstOptimal,                     // oldLayout
                vk::ImageLayout::eTransferSrcOptimal,                     // newLayout
                VK_QUEUE_FAMILY_IGNORED,                                  // srcQueueFamilyIndex
                VK_QUEUE_FAMILY_IGNORED,                                  // dstQueueFamilyIndex
                *_colorAttachments[frameIndex]->GetImage(),               // image
                colorRange                                                // subresourceRange
            );
            commandBuffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
                {}, {}, {}, srcBarrier);
        }

        commandBuffer.end();

        vk::SubmitInfo submitInfo = vk::SubmitInfo(
            0,                // waitSemaphoreCount
            nullptr,          // pWaitSemaphores
            nullptr,          // pWaitDstStageMask
            1,                // commandBufferCount
            &commandBuffer,   // pCommandBuffers
            0,                // signalSemaphoreCount
            nullptr           // pSignalSemaphores
        );
        _deviceContext.GetGraphicsQueue().submit(submitInfo, _renderInFlightFences[frameIndex]);

        device.waitForFences(_renderInFlightFences[frameIndex], vk::True, UINT64_MAX);

        // Encode the blitted frame
        _videoEncoder->QueueEncode(frameIndex);

        const char* bitstreamData = nullptr;
        size_t bitstreamSize = 0;

        _videoEncoder->FinishEncode(bitstreamData, bitstreamSize);
        WriteBitstreamChunk(bitstreamData, bitstreamSize);

        _videoEncoder->FinishEncode(bitstreamData, bitstreamSize);
        WriteBitstreamChunk(bitstreamData, bitstreamSize);
    }

    void VideoEncoderSystem::WriteBitstreamChunk(const char* data, size_t size)
    {
        if (size == 0)
            return;
        if (_bitstreamStreamingEnabled)
        {
            _bitstreamFile.write(data, static_cast<std::streamsize>(size));
            _bitstreamBytesWritten += size;
        }
        else
        {
            _accumulatedBitstream.insert(_accumulatedBitstream.end(), data, data + size);
        }
    }

    void VideoEncoderSystem::OpenBitstreamForJob(const std::string& outputPath, bool streaming)
    {
        // Defensive: if a prior job's file is still open (e.g. caller skipped Finish), close it.
        if (_bitstreamFile.is_open())
            _bitstreamFile.close();

        _bitstreamFilePath = outputPath;
        _bitstreamStreamingEnabled = streaming;
        _bitstreamBytesWritten = 0;
        _accumulatedBitstream.clear();

        if (!streaming)
            return;

        _bitstreamFile.open(outputPath, std::ios::binary | std::ios::trunc);
        if (!_bitstreamFile.is_open())
        {
            LOG_ERROR("VideoEncoderSystem::OpenBitstreamForJob: failed to open {} for streaming. Falling back to in-RAM accumulation.", outputPath);
            _bitstreamStreamingEnabled = false;
        }
    }

    void VideoEncoderSystem::ResetForNewJob()
    {
        _deviceContext.GetDevice().waitIdle();

        // Rebuild descriptors (cell state images and skip grid may have been recreated)
        _renderDescriptor.reset();
        BuildRenderDescriptors();

        // Rebuild graphics pipeline with updated ray march shaders
        _graphicsPipeline.reset();
        BuildRenderPipeline();

        // Create fresh H.264 encoder session for the new bitstream
        uint32_t framesInFlight = _deviceContext.GetFramesInFlight();
        std::vector<vk::Image> inputImages;
        std::vector<vk::ImageView> inputImageViews;
        for (uint32_t frameIndex = 0; frameIndex < framesInFlight; frameIndex++)
        {
            inputImages.push_back(*_colorAttachments[frameIndex]->GetImage());
            inputImageViews.push_back(*_colorAttachments[frameIndex]->GetImageView());
        }
        _videoEncoder.reset();
        _videoEncoder = std::make_unique<VideoEncoder>(_deviceContext, inputImages, inputImageViews, _width, _height, _fps);

        // _accumulatedBitstream / _bitstreamFile are owned by OpenBitstreamForJob's
        // lifecycle (called by the mode immediately after ResetForNewJob). Don't touch
        // them here — clearing would race the path setup the mode is about to do.
    }

    void VideoEncoderSystem::Finish(std::vector<char> &outBitstream)
    {
        if (_bitstreamStreamingEnabled)
        {
            if (_bitstreamFile.is_open())
            {
                _bitstreamFile.flush();
                _bitstreamFile.close();
            }
            outBitstream.clear();
        }
        else
        {
            outBitstream = std::move(_accumulatedBitstream);
        }
    }
}
