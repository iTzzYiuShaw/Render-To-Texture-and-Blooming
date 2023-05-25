#pragma once

#include "../labutils/vulkan_context.hpp"

#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp" 

#include "baked_model.hpp"


struct IndexedMesh
{
	std::uint32_t materialId;
	std::uint32_t indexSize;
	bool isAlphaMask;
	bool isNormalMap;

	labutils::Buffer pos;
	labutils::Buffer texcoords;
	labutils::Buffer normals;
	labutils::Buffer indices;


	//Default constructor
	IndexedMesh(labutils::Buffer pPos, labutils::Buffer pTexCoord, labutils::Buffer pNormal,
		labutils::Buffer pIndices, std::uint32_t pMaterialId, std::uint32_t pIndexSize,bool isAlphaMask, bool isNormalMap)
		:pos(std::move(pPos)),texcoords(std::move(pTexCoord)),normals(std::move(pNormal)),
		indices(std::move(pIndices)),materialId(pMaterialId),indexSize(pIndexSize), isAlphaMask(isAlphaMask),isNormalMap(isNormalMap)
	{}


	IndexedMesh(IndexedMesh&& other)noexcept :
		pos(std::move(other.pos)), texcoords(std::move(other.texcoords)), normals(std::move(other.normals)),
		indices(std::move(other.indices)), materialId(other.materialId), indexSize(other.indexSize), isAlphaMask(other.isAlphaMask), isNormalMap(other.isNormalMap)
	{}
};

IndexedMesh create_indexed_mesh(labutils::VulkanContext const&, labutils::Allocator const&, BakedModel const&,std::uint32_t meshIndex);

struct screenImage
{
	labutils::Buffer pos;
	labutils::Buffer texcoords;
	labutils::Buffer indices;
	std::uint32_t indexSize;

	screenImage(labutils::Buffer pPos, labutils::Buffer pTexCoord, labutils::Buffer pIndices, std::uint32_t pIndexSize) :
		pos(std::move(pPos)), texcoords(std::move(pTexCoord)), indices(std::move(pIndices)), indexSize(pIndexSize) {}


	screenImage(screenImage&& other)noexcept :
		pos(std::move(other.pos)), texcoords(std::move(other.texcoords)), indices(std::move(other.indices)), indexSize(other.indexSize)
	{}
};
screenImage create_screen_image(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator);