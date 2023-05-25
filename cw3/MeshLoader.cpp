#include "MeshLoader.hpp"
#include <limits>

#include <cstring> // for std::memcpy()

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/to_string.hpp"
#include "glm/vec4.hpp"
namespace lut = labutils;

IndexedMesh create_indexed_mesh(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator, BakedModel const& model, std::uint32_t meshIndex)
{

	BakedMeshData mesh = model.meshes[meshIndex];
	
	//See if this is a foliage mesh
	std::uint32_t materialId = model.meshes[meshIndex].materialId;
	std::uint32_t alphaId = model.materials[materialId].alphaMaskTextureId;
	std::uint32_t normalId = model.materials[materialId].normalMapTextureId;

	bool isAlpha = false;
	bool isNormalMap = false;
	if (alphaId != 0xffffffff)// if this is a foliage mesh
	{
		isAlpha = true;
	}

	if (normalId != 0xffffffff)
	{
		isNormalMap = true;
	}


	lut::Buffer vertexPosGPU = lut::create_buffer(
		aAllocator,
		mesh.positions.size() * sizeof(glm::vec3),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);

	lut::Buffer texCoordGPU = lut::create_buffer(
		aAllocator,
		mesh.texcoords.size() * sizeof(glm::vec2),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);

	lut::Buffer normalGPU = lut::create_buffer(
		aAllocator,
		mesh.normals.size() * sizeof(glm::vec3),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);

	lut::Buffer indicesGPU = lut::create_buffer(
		aAllocator,
		mesh.indices.size() * sizeof(std::uint32_t),
		VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);



	//===========================Staging buffer initialize==================================
	lut::Buffer posStaging = lut::create_buffer(
		aAllocator,
		mesh.positions.size() * sizeof(glm::vec3),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_MEMORY_USAGE_CPU_TO_GPU
	);

	lut::Buffer texCoordStaging = lut::create_buffer(
		aAllocator,
		mesh.texcoords.size() * sizeof(glm::vec2),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_MEMORY_USAGE_CPU_TO_GPU
	);

	lut::Buffer normalStaging = lut::create_buffer(
		aAllocator,
		mesh.normals.size() * sizeof(glm::vec3),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_MEMORY_USAGE_CPU_TO_GPU
	);

	lut::Buffer indicesStaging = lut::create_buffer(
		aAllocator,
		mesh.indices.size() * sizeof(std::uint32_t),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_MEMORY_USAGE_CPU_TO_GPU
	);


	void* posPtr = nullptr;
	if (auto const res = vmaMapMemory(aAllocator.allocator, posStaging.allocation, &posPtr); VK_SUCCESS != res)
	{
		throw lut::Error("Mapping memory for writing\n"
			"vmaMapMemory() returned %s", lut::to_string(res).c_str());
	}
	std::memcpy(posPtr, mesh.positions.data(), mesh.positions.size() * sizeof(glm::vec3));
	vmaUnmapMemory(aAllocator.allocator, posStaging.allocation);


	void* texPtr = nullptr;
	if (auto const res = vmaMapMemory(aAllocator.allocator, texCoordStaging.allocation, &texPtr); VK_SUCCESS != res)
	{
		throw lut::Error("Mapping memory for writing\n"
			"vmaMapMemory() returned %s", lut::to_string(res).c_str());
	}
	std::memcpy(texPtr, mesh.texcoords.data(), mesh.texcoords.size() * sizeof(glm::vec2));
	vmaUnmapMemory(aAllocator.allocator, texCoordStaging.allocation);


	void* norPtr = nullptr;
	if (auto const res = vmaMapMemory(aAllocator.allocator, normalStaging.allocation, &norPtr); VK_SUCCESS != res)
	{
		throw lut::Error("Mapping memory for writing\n"
			"vmaMapMemory() returned %s", lut::to_string(res).c_str());
	}
	std::memcpy(norPtr, mesh.normals.data(), mesh.normals.size() * sizeof(glm::vec3));
	vmaUnmapMemory(aAllocator.allocator, normalStaging.allocation);



	void* indicePtr = nullptr;
	if (auto const res = vmaMapMemory(aAllocator.allocator, indicesStaging.allocation, &indicePtr); VK_SUCCESS != res)
	{
		throw lut::Error("Mapping memory for writing\n"
			"vmaMapMemory() returned %s", lut::to_string(res).c_str());
	}
	std::memcpy(indicePtr, mesh.indices.data(), mesh.indices.size() * sizeof(std::uint32_t));
	vmaUnmapMemory(aAllocator.allocator, indicesStaging.allocation);


	// We need to ensure that the Vulkan resources are alive until all the
//  transfers have completed. For simplicity, we will just wait for the
//  operations to complete with a fence. A more complex solution might want
//  to queue transfers, let these take place in the background while
//  performing other tasks.

	lut::Fence uploadComplete = lut::create_fence(aContext);

	// Queue data uploads from staging buffers to the final buffers
	// This uses a separate command pool for simplicity.
	lut::CommandPool uploadPool = create_command_pool(aContext);
	VkCommandBuffer uploadCmd = alloc_command_buffer(aContext, uploadPool.handle);
	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = 0;
	beginInfo.pInheritanceInfo = nullptr;

	if (auto const res = vkBeginCommandBuffer(uploadCmd, &beginInfo); VK_SUCCESS != res)
	{
		throw lut::Error("Beginning command buffer recording\n"
			"vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
	}

	VkBufferCopy pcopy{};
	pcopy.size = mesh.positions.size() * sizeof(glm::vec3);
	vkCmdCopyBuffer(uploadCmd, posStaging.buffer, vertexPosGPU.buffer, 1, &pcopy);
	lut::buffer_barrier(uploadCmd,
		vertexPosGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);

	VkBufferCopy tcopy{};
	tcopy.size = mesh.texcoords.size() * sizeof(glm::vec2);
	vkCmdCopyBuffer(uploadCmd, texCoordStaging.buffer, texCoordGPU.buffer, 1, &tcopy);
	lut::buffer_barrier(uploadCmd,
		texCoordGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);

	VkBufferCopy ncopy{};
	ncopy.size = mesh.normals.size() * sizeof(glm::vec3);
	vkCmdCopyBuffer(uploadCmd, normalStaging.buffer, normalGPU.buffer, 1, &ncopy);
	lut::buffer_barrier(uploadCmd,
		normalGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);

	VkBufferCopy icopy{};
	icopy.size = mesh.indices.size() * sizeof(std::uint32_t);
	vkCmdCopyBuffer(uploadCmd, indicesStaging.buffer, indicesGPU.buffer, 1, &icopy);
	lut::buffer_barrier(uploadCmd,
		indicesGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);



	if (auto const res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res)
	{
		throw lut::Error("Ending command buffer recording\n"
			"vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
	}

	// Submit transfer commands
	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &uploadCmd;
	if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, uploadComplete.handle); VK_SUCCESS != res)
	{
		throw lut::Error("Submitting commands\n"
			"vkQueueSubmit() returned %s", lut::to_string(res).c_str());
	}
	// Wait for commands to finish before we destroy the temporary resources
	// required for the transfers (staging buffers, command pool, ...)
	//
	// The code doesn’t destory the resources implicitly – the resources are
	// destroyed by the destructors of the labutils wrappers for the various
	// objects once we leave the function’s scope.
	if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
	{
		throw lut::Error("Waiting for upload to complete\n"
			"vkWaitForFences() returned %s", lut::to_string(res).c_str());
	}


	return IndexedMesh{
		std::move(vertexPosGPU),
		std::move(texCoordGPU),
		std::move(normalGPU),
		std::move(indicesGPU),
		mesh.materialId,
		static_cast<uint32_t> (mesh.indices.size()),
		isAlpha,
		isNormalMap
	};
}


screenImage create_screen_image(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator)
{

	//position

	std::vector<glm::vec2> positions = {
		{0.0f,0.0f},{0.0f,1.0f},{1.0f,0.0f},{1.0f,1.0f}
	};

	std::vector<glm::vec2> texcoords = {
	{0.0f,0.0f},{1.0f,0.0f},{0.0f,1.0f},{1.0f,1.0f}
	};


	std::vector<uint32_t> indices = {
	1, 0, 2,
	2, 3, 0
	};



	lut::Buffer vertexPosGPU = lut::create_buffer(
		aAllocator,
		positions.size() * sizeof(glm::vec2),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);

	lut::Buffer texCoordGPU = lut::create_buffer(
		aAllocator,
		texcoords.size() * sizeof(glm::vec2),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);

	lut::Buffer indicesGPU = lut::create_buffer(
		aAllocator,
		indices.size() * sizeof(std::uint32_t),
		VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);


	//===========================Staging buffer initialize==================================
	lut::Buffer posStaging = lut::create_buffer(
		aAllocator,
		positions.size() * sizeof(glm::vec2),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_MEMORY_USAGE_CPU_TO_GPU
	);

	lut::Buffer texCoordStaging = lut::create_buffer(
		aAllocator,
		texcoords.size() * sizeof(glm::vec2),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_MEMORY_USAGE_CPU_TO_GPU
	);


	lut::Buffer indicesStaging = lut::create_buffer(
		aAllocator,
		indices.size() * sizeof(std::uint32_t),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_MEMORY_USAGE_CPU_TO_GPU
	);


	void* posPtr = nullptr;
	if (auto const res = vmaMapMemory(aAllocator.allocator, posStaging.allocation, &posPtr); VK_SUCCESS != res)
	{
		throw lut::Error("Mapping memory for writing\n"
			"vmaMapMemory() returned %s", lut::to_string(res).c_str());
	}
	std::memcpy(posPtr, positions.data(), positions.size() * sizeof(glm::vec2));
	vmaUnmapMemory(aAllocator.allocator, posStaging.allocation);


	void* texPtr = nullptr;
	if (auto const res = vmaMapMemory(aAllocator.allocator, texCoordStaging.allocation, &texPtr); VK_SUCCESS != res)
	{
		throw lut::Error("Mapping memory for writing\n"
			"vmaMapMemory() returned %s", lut::to_string(res).c_str());
	}
	std::memcpy(texPtr, texcoords.data(), texcoords.size() * sizeof(glm::vec2));
	vmaUnmapMemory(aAllocator.allocator, texCoordStaging.allocation);

	void* indicePtr = nullptr;
	if (auto const res = vmaMapMemory(aAllocator.allocator, indicesStaging.allocation, &indicePtr); VK_SUCCESS != res)
	{
		throw lut::Error("Mapping memory for writing\n"
			"vmaMapMemory() returned %s", lut::to_string(res).c_str());
	}
	std::memcpy(indicePtr, indices.data(), indices.size() * sizeof(std::uint32_t));
	vmaUnmapMemory(aAllocator.allocator, indicesStaging.allocation);

	// We need to ensure that the Vulkan resources are alive until all the
//  transfers have completed. For simplicity, we will just wait for the
//  operations to complete with a fence. A more complex solution might want
//  to queue transfers, let these take place in the background while
//  performing other tasks.

	lut::Fence uploadComplete = lut::create_fence(aContext);

	// Queue data uploads from staging buffers to the final buffers
	// This uses a separate command pool for simplicity.
	lut::CommandPool uploadPool = create_command_pool(aContext);
	VkCommandBuffer uploadCmd = alloc_command_buffer(aContext, uploadPool.handle);
	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = 0;
	beginInfo.pInheritanceInfo = nullptr;

	if (auto const res = vkBeginCommandBuffer(uploadCmd, &beginInfo); VK_SUCCESS != res)
	{
		throw lut::Error("Beginning command buffer recording\n"
			"vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
	}

	VkBufferCopy pcopy{};
	pcopy.size = positions.size() * sizeof(glm::vec2);
	vkCmdCopyBuffer(uploadCmd, posStaging.buffer, vertexPosGPU.buffer, 1, &pcopy);
	lut::buffer_barrier(uploadCmd,
		vertexPosGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);

	VkBufferCopy tcopy{};
	tcopy.size = texcoords.size() * sizeof(glm::vec2);
	vkCmdCopyBuffer(uploadCmd, texCoordStaging.buffer, texCoordGPU.buffer, 1, &tcopy);
	lut::buffer_barrier(uploadCmd,
		texCoordGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);

	VkBufferCopy icopy{};
	icopy.size = indices.size() * sizeof(std::uint32_t);
	vkCmdCopyBuffer(uploadCmd, indicesStaging.buffer, indicesGPU.buffer, 1, &icopy);
	lut::buffer_barrier(uploadCmd,
		indicesGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);


	if (auto const res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res)
	{
		throw lut::Error("Ending command buffer recording\n"
			"vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
	}

	// Submit transfer commands
	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &uploadCmd;
	if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, uploadComplete.handle); VK_SUCCESS != res)
	{
		throw lut::Error("Submitting commands\n"
			"vkQueueSubmit() returned %s", lut::to_string(res).c_str());
	}
	// Wait for commands to finish before we destroy the temporary resources
	// required for the transfers (staging buffers, command pool, ...)
	//
	// The code doesn’t destory the resources implicitly – the resources are
	// destroyed by the destructors of the labutils wrappers for the various
	// objects once we leave the function’s scope.
	if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
	{
		throw lut::Error("Waiting for upload to complete\n"
			"vkWaitForFences() returned %s", lut::to_string(res).c_str());
	}


	return screenImage{
		std::move(vertexPosGPU),
		std::move(texCoordGPU),
		std::move(indicesGPU),
		static_cast<uint32_t> (indices.size())
	};
}