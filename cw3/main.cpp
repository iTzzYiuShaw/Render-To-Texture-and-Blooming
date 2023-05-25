#include "glm/fwd.hpp"


#include <tuple>
#include <chrono>
#include <limits>
#include <vector>
#include <stdexcept>

#include <cstdio>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <math.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <volk/volk.h>
#if !defined(GLM_FORCE_RADIANS)
#	define GLM_FORCE_RADIANS
#endif
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../labutils/to_string.hpp"
#include "../labutils/vulkan_window.hpp"

#include "../labutils/angle.hpp"
using namespace labutils::literals;

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/vkimage.hpp"
#include "../labutils/vkobject.hpp"
#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp" 
namespace lut = labutils;

#include "baked_model.hpp"
#include "MeshLoader.hpp"

#include <chrono>
#include<vulkan/vulkan.h>


namespace
{
	using Clock_ = std::chrono::steady_clock;
	using Secondsf_ = std::chrono::duration<float, std::ratio<1>>;

	namespace cfg
	{
		// Compiled shader code for the graphics pipeline(s)
		// See sources in cw1/shaders/*. 
#		define SHADERDIR_ "D:/Working/MSc Game Engineering/Vulkan/A3/cw3/assets/cw3/shaders/"
		constexpr char const* kVertShaderPath = SHADERDIR_ "default.vert.spv";
		constexpr char const* kFragShaderPath = SHADERDIR_ "default.frag.spv";

		constexpr char const* kPostVertShaderPath = SHADERDIR_ "fullscreen.vert.spv";
		constexpr char const* kPostFragShaderPath = SHADERDIR_ "fullscreen.frag.spv";

		constexpr char const* kVertDensityShaderPath = SHADERDIR_ "defaultDensity.vert.spv";
		constexpr char const* kGeomDensityShaderPath = SHADERDIR_ "defaultDensity.geom.spv";
		constexpr char const* kFragDensityShaderPath = SHADERDIR_ "defaultDensity.frag.spv";

		constexpr char const* kPBRVertShaderPath = SHADERDIR_ "default.vert.spv";
		constexpr char const* kPBRFragShaderPath = SHADERDIR_ "default.frag.spv";

		constexpr char const* kBrightVertShaderPath = SHADERDIR_ "bright.vert.spv";
		constexpr char const* kBrightFragShaderPath = SHADERDIR_ "bright.frag.spv";

		constexpr char const* kVerticalVertShaderPath = SHADERDIR_ "vertical.vert.spv";
		constexpr char const* kVerticalFragShaderPath = SHADERDIR_ "vertical.frag.spv";

		constexpr char const* kHorizontalVertShaderPath = SHADERDIR_ "horizontal.vert.spv";
		constexpr char const* kHorizontalFragShaderPath = SHADERDIR_ "horizontal.frag.spv";

		constexpr char const* kPostprocessVertShaderPath = SHADERDIR_ "postprocess.vert.spv";
		constexpr char const* kPostprocessFragShaderPath = SHADERDIR_ "postprocess.frag.spv";


#		undef SHADERDIR_

#		define MODELDIR_ "assets/cw1/"
		constexpr char const* kObjectPath = MODELDIR_ "sponza_with_ship.obj";
		constexpr char const* kMateriaPath = MODELDIR_ "sponza_with_ship.mtl";
#		undef MODELDIR_

#		define TEXTUREDIR_ "assets/cw1/"
		constexpr char const* kFloorTexture = TEXTUREDIR_ "asphalt.png";
#		undef TEXTUREDIR_


		constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;


		// General rule: with a standard 24 bit or 32 bit float depth buffer,
		// you can support a 1:1000 ratio between the near and far plane with
		// minimal depth fighting. Larger ratios will introduce more depth
		// fighting problems; smaller ratios will increase the depth buffer's
		// resolution but will also limit the view distance.
		constexpr float kCameraNear = 0.1f;
		constexpr float kCameraFar = 100.f;
		constexpr auto kCameraFov = 60.0_degf;


		constexpr float kCameraBaseSpeed = 1.7f; // units/second 
		constexpr float kCameraFastMult = 5.f; // speed multiplier 
		constexpr float kCameraSlowMult = 0.05f; // speed multiplier 
		constexpr float kCameraMouseSensitivity = 0.001f; // radians per pixel 
	}

	// GLFW callbacks
	void glfw_callback_key_press(GLFWwindow*, int, int, int, int);
	void glfw_callback_button(GLFWwindow*, int, int, int);
	void glfw_callback_motion(GLFWwindow*, double, double);

	enum class EInputState
	{
		forward,
		backward,
		strafeLeft,
		strafeRight,
		levitate,
		sink,
		fast,
		slow,
		mousing,
		max,
		lightForward,
		lightBackward,
		lightStrafeLeft,
		lightStrafeRight
	};

	struct UserState
	{
		bool inputMap[std::size_t(EInputState::max)] = {};
		float mouseX = 0.f, mouseY = 0.f;
		float previousX = 0.f, previousY = 0.f;
		bool wasMousing = false;
		glm::mat4 camera2world = glm::identity<glm::mat4>();
	};


	namespace glsl
	{


		struct SceneUniform
		{
			// Note: need to be careful about the packing/alignment here! 
			glm::mat4 camera;
			glm::mat4 projection;
			glm::mat4 projCam;
			glm::vec3 cameraPos;
		};

		struct MaterialUniform
		{
			glm::vec4 baseColor;
			glm::vec4 emissiveColor;
			glm::vec2 rouAndMetal;
		};

		struct LightSource
		{
			glm::vec4 position;
			glm::vec4 color;
			float intensity;
		};

		struct GaussianUniform
		{
			float data[22];
		};

		static_assert(sizeof(SceneUniform) <= 65536, "SceneUniform must be less than 65536 bytes for vkCmdUpdateBuffer");
		static_assert(sizeof(SceneUniform) % 4 == 0, "SceneUniform size must be a multiple of 4 bytes");

	}

	// Local types/structures:

	// Local functions:
	void calculateGaussianUniform(lut::VulkanWindow const&, glsl::GaussianUniform& gVerticalUniform, glsl::GaussianUniform& gHorizontalUniform);
	float Gaussian(float distance, float factor);

	void glfw_callback_key_press(GLFWwindow*, int, int, int, int);
	//Task1&2
	lut::RenderPass create_render_pass(lut::VulkanWindow const&);
	lut::RenderPass create_postRender_pass(lut::VulkanWindow const&);
	//Task3
	lut::RenderPass create_filter_PBR_render_pass(lut::VulkanWindow const& aWindow);
	lut::RenderPass create_postProcessing_render_pass(lut::VulkanWindow const& aWindow);

	lut::DescriptorSetLayout create_scene_descriptor_layout(lut::VulkanWindow const&);
	lut::DescriptorSetLayout create_mipmap_descriptor_layout(lut::VulkanWindow const&);
	lut::DescriptorSetLayout create_lightSource_descriptor_layout(lut::VulkanWindow const&);
	lut::DescriptorSetLayout create_object_descriptor_layout(lut::VulkanWindow const&);
	lut::DescriptorSetLayout create_intermediateImage_descriptor_layout(lut::VulkanWindow const&);
	lut::DescriptorSetLayout create_material_descriptor_layout(lut::VulkanWindow const&);
	lut::DescriptorSetLayout create_vGaussian_descriptor_layout(lut::VulkanWindow const&);
	lut::DescriptorSetLayout create_hGaussian_descriptor_layout(lut::VulkanWindow const&);

	//task 3
	lut::DescriptorSetLayout create_bright_descriptor_layout(lut::VulkanWindow const&);
	lut::DescriptorSetLayout create_vertical_descriptor_layout(lut::VulkanWindow const&);
	lut::DescriptorSetLayout create_horizontal_descriptor_layout(lut::VulkanWindow const&);
	lut::DescriptorSetLayout create_PBR_descriptor_layout(lut::VulkanWindow const&);




	lut::PipelineLayout create_pipeline_layout(lut::VulkanContext const&, VkDescriptorSetLayout const&, VkDescriptorSetLayout aObjectLayout, VkDescriptorSetLayout  const& aMipmapLayout, VkDescriptorSetLayout const& aLightSource
	);
	lut::PipelineLayout create_postPipeline_layout(lut::VulkanContext const&, VkDescriptorSetLayout const&);

	//Task 3

	lut::PipelineLayout create_bright_PBR_pipeline_layout(lut::VulkanContext const& aContext, VkDescriptorSetLayout const& aSceneLayout, VkDescriptorSetLayout aMaterialLayout, VkDescriptorSetLayout  const& aTexturelayout,
		VkDescriptorSetLayout const& aLightSource);

	lut::PipelineLayout create_vertical_pipeline_layout(lut::VulkanContext const& aContext, VkDescriptorSetLayout const& aBrightLayout, VkDescriptorSetLayout vGaussianLayout);
	lut::PipelineLayout create_horizontal_pipeline_layout(lut::VulkanContext const& aContext, VkDescriptorSetLayout const& aVerticalLayout, VkDescriptorSetLayout hGaussianLayout);

	lut::PipelineLayout create_postprocess_pipeline_layout(lut::VulkanContext const& aContext, VkDescriptorSetLayout const& aGaussianLayout, VkDescriptorSetLayout const& aPBRLayout);


	lut::Pipeline create_piepline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);
	lut::Pipeline create_alpha_pipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);
	lut::Pipeline create_post_pipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);

	lut::Pipeline create_bright_PBR_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout,
		char const* kVertShaderPath, char const* kFragShaderPath,uint32_t subpassIndex);

	lut::Pipeline create_filter_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout,
		char const* kVertShaderPath, char const* kFragShaderPath, uint32_t subpassIndex);


	void create_swapchain_framebuffers
		(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, std::vector<lut::Framebuffer>& aFramebuffers,
			VkImageView aDepthView, VkImageView aInterImageView);

	void create_NewSwapchain_framebuffers(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, std::vector<lut::Framebuffer>& aFramebuffers);

	void create_framebuffer_R1(lut::VulkanWindow const& aWindow,
		VkRenderPass aRenderPass, lut::Framebuffer& aFramebuffers, VkImageView aBrightView, VkImageView aVerticalView,
		VkImageView aHorizontalView, VkImageView aPBRView, VkImageView aDepthView);


	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const&, lut::Allocator const&);
	std::tuple<lut::Image, lut::ImageView> create_offlineimage_view_buffer(lut::VulkanContext const&, VkFormat, labutils::VulkanWindow const&, labutils::Allocator const&);
	lut::Framebuffer create_interImage_framebuffer(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkImageView depthView, VkImageView interImageView);
	void update_scene_uniforms(
		glsl::SceneUniform&,
		std::uint32_t aFramebufferWidth,
		std::uint32_t aFramebufferHeight,
		UserState const& userState
	);

	void update_user_state(UserState&, float aElapsedTime);
	void record_commands(
		VkCommandBuffer,
		VkRenderPass,
		VkFramebuffer,
		VkPipeline pipeTexture,
		VkPipeline pipeAlpha,
		VkPipeline pipePost,
		VkExtent2D const&,
		std::vector<IndexedMesh>* indexedMesh,
		VkBuffer aSceneUBO,
		glsl::SceneUniform
		const& aSceneUniform,
		VkBuffer aLightUBO,
		glsl::LightSource const& aLightUniform,
		VkPipelineLayout,
		VkDescriptorSet aSceneDescriptors,
		VkDescriptorSet lightDescriptors,
		VkDescriptorSet interImageDescriptor,
		BakedModel const& bakedModel,
		glsl::MaterialUniform& aMaterialUniform,
		std::vector<VkDescriptorSet*>* materialDescriptor,
		std::vector<VkBuffer*>* materialBuffer,
		VkFramebuffer interImageBuffer,
		screenImage const& fullImage,
		VkPipelineLayout postPipeLayout,
		lut::VulkanWindow const&,
		std::uint32_t imageIndex,
		std::vector<VkDescriptorSet*>* textureDescriptorsSet,

		//Task3
		VkPipeline brightPipe,
		VkPipeline verticalpipe,
		VkPipeline horizontalPipe,
		VkPipeline postprocessPipe,
		VkRenderPass filterPass,
		VkRenderPass postProcessPass,

		lut::Framebuffer const& firstFrameBuffer,
		VkDescriptorSet brightDescriptors,
		VkDescriptorSet verticalDescriptors,
		VkDescriptorSet horizontalDescriptors,
		VkDescriptorSet PBRDescriptors,

		VkPipelineLayout birght_PBR_layout,
		VkPipelineLayout verticalPipeLayout,
		VkPipelineLayout horizontalPipeLayout,
		VkPipelineLayout postProcessPipeLayout,
		glsl::GaussianUniform& vGaussianUniform,
		VkBuffer vGaussianUBO,
		VkDescriptorSet vGaussianDescriptors,

		glsl::GaussianUniform& hGaussianUniform,
		VkBuffer hGaussianUBO,
		VkDescriptorSet hGaussianDescriptors
	);
	void submit_commands(
		lut::VulkanContext const&,
		VkCommandBuffer,
		VkFence,
		VkSemaphore,
		VkSemaphore
	);

	void present_results(
		VkQueue,
		VkSwapchainKHR,
		std::uint32_t aImageIndex,
		VkSemaphore,
		bool& aNeedToRecreateSwapchain
	);
}
int main() try
{
	//TODO-implement me.

	// Create our Vulkan Window
	lut::VulkanWindow window = lut::make_vulkan_window();

	UserState state{};
	glfwSetWindowUserPointer(window.window, &state);

	// Configure the GLFW window
	glfwSetKeyCallback(window.window, &glfw_callback_key_press);
	glfwSetMouseButtonCallback(window.window, &glfw_callback_button);
	glfwSetCursorPosCallback(window.window, &glfw_callback_motion);



	lut::Allocator allocator = lut::create_allocator(window);
	lut::CommandPool cpool = lut::create_command_pool(window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
	lut::DescriptorPool dpool = lut::create_descriptor_pool(window);


	//scene Uniform descriptor
	lut::DescriptorSetLayout sceneLayout = create_scene_descriptor_layout(window);
	lut::DescriptorSetLayout objectLayout = create_object_descriptor_layout(window);
	lut::DescriptorSetLayout lightLayout = create_lightSource_descriptor_layout(window);
	lut::DescriptorSetLayout mipmapLayout = create_mipmap_descriptor_layout(window);	//Color descriptor
	lut::DescriptorSetLayout interImageLayout = create_intermediateImage_descriptor_layout(window);	//Intermediate image descriptor
	lut::DescriptorSetLayout materialLayout = create_material_descriptor_layout(window);

	//verticalGassian
	lut::DescriptorSetLayout vGaussianLayout = create_vGaussian_descriptor_layout(window);
	lut::DescriptorSetLayout hGaussianLayout = create_vGaussian_descriptor_layout(window);


	//HorizontalGaussian

	//Task 3
	lut::DescriptorSetLayout brightLayout = create_intermediateImage_descriptor_layout(window);
	lut::DescriptorSetLayout PBR_layout = create_intermediateImage_descriptor_layout(window);
	lut::DescriptorSetLayout verticalLayout = create_intermediateImage_descriptor_layout(window);
	lut::DescriptorSetLayout horizontalLayout = create_intermediateImage_descriptor_layout(window);



	// Intialize render passes
	//Task 1&2
	lut::RenderPass renderPass = create_render_pass(window);
	lut::RenderPass postRenderPass = create_postRender_pass(window);

	//Task 3
	lut::RenderPass filterPass = create_filter_PBR_render_pass(window);//FInding bright, vertical, horizontal, PBR
	lut::RenderPass postProcessPass = create_postProcessing_render_pass(window); //Fullscreen postProcessing

	//Initialize pipeline layouts
	//Task 1
	lut::PipelineLayout pipeLayout = create_pipeline_layout(window, sceneLayout.handle, materialLayout.handle, objectLayout.handle, lightLayout.handle);
	lut::PipelineLayout postPipeLayout = create_postPipeline_layout(window,interImageLayout.handle);

	//Task 3 
	lut::PipelineLayout bright_PBR_layout = create_bright_PBR_pipeline_layout(window, sceneLayout.handle, materialLayout.handle, objectLayout.handle, lightLayout.handle);
	lut::PipelineLayout verticalPipeLayout = create_vertical_pipeline_layout(window, brightLayout.handle,vGaussianLayout.handle);
	lut::PipelineLayout horizontalPipeLayout = create_horizontal_pipeline_layout(window, verticalLayout.handle, hGaussianLayout.handle);
	lut::PipelineLayout postProcessPipelayout = create_postprocess_pipeline_layout(window, horizontalLayout.handle, PBR_layout.handle);


	//Pipe line
	lut::Pipeline pipe = create_piepline(window, renderPass.handle, pipeLayout.handle);
	lut::Pipeline alphaPipe = create_alpha_pipeline(window, renderPass.handle, pipeLayout.handle);
	lut::Pipeline postPipeLine = create_post_pipeline(window, renderPass.handle, postPipeLayout.handle);

	//Task3
	lut::Pipeline brightPipeline = create_bright_PBR_pipeline(window, filterPass.handle, bright_PBR_layout.handle,cfg::kBrightVertShaderPath,cfg::kBrightFragShaderPath,0);
	lut::Pipeline verticalPipeLine = create_filter_pipeline(window, filterPass.handle, verticalPipeLayout.handle, cfg::kVerticalVertShaderPath, cfg::kVerticalFragShaderPath,1);//No vertexinput
	lut::Pipeline horizontalPipeline = create_filter_pipeline(window, filterPass.handle, horizontalPipeLayout.handle, cfg::kHorizontalVertShaderPath, cfg::kHorizontalFragShaderPath, 2);//No vertexinput
	lut::Pipeline postprocessPipeline = create_filter_pipeline(window, postProcessPass.handle, postProcessPipelayout.handle, cfg::kPostprocessVertShaderPath, cfg::kPostprocessFragShaderPath, 0);//No vertexinput


	//Samling sampler---------------
	lut::Sampler defalutSampler = lut::create_default_sampler(window);
	lut::CommandPool loadCmdPool = lut::create_command_pool(window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);

	//Task 3 create ImageView: Bright-Vertical-Horizontal(filter result); PBR (actual scene result)
	auto [brightBuffer, brightView] = create_offlineimage_view_buffer(window, VK_FORMAT_R8G8B8A8_SRGB, window, allocator);
	auto [verticalBuffer, verticalView] = create_offlineimage_view_buffer(window, VK_FORMAT_R8G8B8A8_SRGB, window, allocator);
	auto [horizontalBuffer, horizontalView] = create_offlineimage_view_buffer(window, VK_FORMAT_R8G8B8A8_SRGB, window, allocator);
	auto [PBRBuffer, PBRView] = create_offlineimage_view_buffer(window, VK_FORMAT_R8G8B8A8_SRGB, window, allocator);
	auto [depthBuffer, depthBufferView] = create_depth_buffer(window, allocator);
	//Task 3 create descriptor sets: Bright-Vertical-Horizontal(filter result); PBR (actual scene result)

	VkDescriptorSet brightDescriptor = lut::alloc_desc_set(window, dpool.handle,
		brightLayout.handle);

	{
		VkWriteDescriptorSet desc[1]{};
		VkDescriptorImageInfo textureInfo[1]{};

		//Base color
		textureInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		textureInfo[0].imageView = brightView.handle;
		textureInfo[0].sampler = defalutSampler.handle;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = brightDescriptor;
		desc[0].dstBinding = 0;
		desc[0].dstArrayElement = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		desc[0].descriptorCount = 1;
		desc[0].pImageInfo = &textureInfo[0];

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}
	VkDescriptorSet verticalDescriptor = lut::alloc_desc_set(window, dpool.handle,
		verticalLayout.handle);

	{
		VkWriteDescriptorSet desc[1]{};
		VkDescriptorImageInfo textureInfo[1]{};

		//Base color
		textureInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		textureInfo[0].imageView = verticalView.handle;
		textureInfo[0].sampler = defalutSampler.handle;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = verticalDescriptor;
		desc[0].dstBinding = 0;
		desc[0].dstArrayElement = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		desc[0].descriptorCount = 1;
		desc[0].pImageInfo = &textureInfo[0];

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

	VkDescriptorSet horizontalDescriptor = lut::alloc_desc_set(window, dpool.handle,
		horizontalLayout.handle);

	{
		VkWriteDescriptorSet desc[1]{};
		VkDescriptorImageInfo textureInfo[1]{};

		//Base color
		textureInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		textureInfo[0].imageView = horizontalView.handle;
		textureInfo[0].sampler = defalutSampler.handle;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = horizontalDescriptor;
		desc[0].dstBinding = 0;
		desc[0].dstArrayElement = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		desc[0].descriptorCount = 1;
		desc[0].pImageInfo = &textureInfo[0];

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}
	VkDescriptorSet PBRDescriptor = lut::alloc_desc_set(window, dpool.handle,
		PBR_layout.handle);

	{
		VkWriteDescriptorSet desc[1]{};
		VkDescriptorImageInfo textureInfo[1]{};

		//Base color
		textureInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		textureInfo[0].imageView = PBRView.handle;
		textureInfo[0].sampler = defalutSampler.handle;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = PBRDescriptor;
		desc[0].dstBinding = 0;
		desc[0].dstArrayElement = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		desc[0].descriptorCount = 1;
		desc[0].pImageInfo = &textureInfo[0];

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

	//Task 3: creates 2 frame buffer for 2 render passes
	lut::Framebuffer firstFrameBuffer;
	create_framebuffer_R1(window, filterPass.handle, firstFrameBuffer, brightView.handle, verticalView.handle, horizontalView.handle, PBRView.handle, depthBufferView.handle);

	std::vector<lut::Framebuffer> framebuffers;
	create_NewSwapchain_framebuffers(window, postProcessPass.handle, framebuffers);



	//Create offline 1 imageviews
	auto [interImageBuffer, interImageView] = create_offlineimage_view_buffer(window, VK_FORMAT_R8G8B8A8_SRGB, window, allocator);
	VkDescriptorSet interImagesDescriptor =lut::alloc_desc_set(window, dpool.handle,
		interImageLayout.handle);

	{
		VkWriteDescriptorSet desc[1]{};
		VkDescriptorImageInfo textureInfo[1]{};

		//Base color
		textureInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		textureInfo[0].imageView = interImageView.handle;
		textureInfo[0].sampler = defalutSampler.handle;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = interImagesDescriptor;
		desc[0].dstBinding = 0;
		desc[0].dstArrayElement = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		desc[0].descriptorCount = 1;
		desc[0].pImageInfo = &textureInfo[0];

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

	lut::Framebuffer interImageFrameBuffer = create_interImage_framebuffer(window, renderPass.handle, depthBufferView.handle,interImageView.handle);

	std::vector<VkCommandBuffer> cbuffers;
	std::vector<lut::Fence> cbfences;

	for (std::size_t i = 0; i < framebuffers.size(); ++i)
	{
		cbuffers.emplace_back(lut::alloc_command_buffer(window, cpool.handle));
		cbfences.emplace_back(lut::create_fence(window, VK_FENCE_CREATE_SIGNALED_BIT));
	}


	//Load model and meshes----------------------------------------------------------------------
	BakedModel bakedModel = load_baked_model("assets/cw3/ship.comp5822mesh");
	std::vector<IndexedMesh>* indexedMesh = new std::vector<IndexedMesh>;
	for (int i = 0; i < bakedModel.meshes.size(); i++)
	{
		auto mesh = bakedModel.meshes[i];
		IndexedMesh temp = create_indexed_mesh(window, allocator, bakedModel, i);
		indexedMesh->emplace_back(std::move(temp));
	}



	//Texture loading

	std::vector<VkDescriptorSet*>* textureDescriptorsSet = new std::vector<VkDescriptorSet*>;

	std::vector<lut::Image> imageSet;
	std::vector<lut::ImageView>imageViewSet;

	for (int i = 0; i < indexedMesh->size(); i++)//changed
	{

		std::uint32_t materialId = ((*indexedMesh)[i].materialId);

		//Sampling base color
		std::uint32_t baseColorId = bakedModel.materials[materialId].baseColorTextureId;
		const char* baseColorPath = bakedModel.textures[baseColorId].path.c_str();

		imageSet.push_back(std::move((lut::load_image_texture2d(baseColorPath, window, loadCmdPool.handle, allocator))));
		imageViewSet.push_back(std::move((lut::create_image_view_texture2d(window, imageSet[3 * i].image, VK_FORMAT_R8G8B8A8_SRGB))));

		//Sampling roughness
		std::uint32_t roughnessId = bakedModel.materials[materialId].roughnessTextureId;
		const char* roughnessPath = bakedModel.textures[roughnessId].path.c_str();

		imageSet.push_back(std::move((lut::load_image_texture2d(roughnessPath, window, loadCmdPool.handle, allocator))));
		imageViewSet.push_back(std::move((lut::create_image_view_texture2d(window, imageSet[3 * i + 1].image, VK_FORMAT_R8G8B8A8_SRGB))));

		//Sampling metalness
		std::uint32_t metalnessId = bakedModel.materials[materialId].metalnessTextureId;
		const char* metalnessPath = bakedModel.textures[metalnessId].path.c_str();

		imageSet.push_back(std::move((lut::load_image_texture2d(metalnessPath, window, loadCmdPool.handle, allocator))));
		imageViewSet.push_back(std::move((lut::create_image_view_texture2d(window, imageSet[3 * i + 2].image, VK_FORMAT_R8G8B8A8_SRGB))));


		VkDescriptorSet* textureDescriptors = new VkDescriptorSet;
		*textureDescriptors = lut::alloc_desc_set(window, dpool.handle,
			objectLayout.handle);

		{
			VkWriteDescriptorSet desc[3]{};
			VkDescriptorImageInfo textureInfo[3]{};

			//Base color
			textureInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			textureInfo[0].imageView = imageViewSet[3 * i].handle;
			textureInfo[0].sampler = defalutSampler.handle;

			desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc[0].dstSet = *textureDescriptors;
			desc[0].dstBinding = 0;
			desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			desc[0].descriptorCount = 1;
			desc[0].pImageInfo = &textureInfo[0];

			//Roughness
			textureInfo[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			textureInfo[1].imageView = imageViewSet[3 * i + 1].handle;
			textureInfo[1].sampler = defalutSampler.handle;

			desc[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc[1].dstSet = *textureDescriptors;
			desc[1].dstBinding = 1;
			desc[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			desc[1].descriptorCount = 1;
			desc[1].pImageInfo = &textureInfo[1];

			//Metalness
			textureInfo[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			textureInfo[2].imageView = imageViewSet[3 * i + 2].handle;
			textureInfo[2].sampler = defalutSampler.handle;

			desc[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc[2].dstSet = *textureDescriptors;
			desc[2].dstBinding = 2;
			desc[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			desc[2].descriptorCount = 1;
			desc[2].pImageInfo = &textureInfo[2];

			vkUpdateDescriptorSets(window.device, 3, desc, 0, nullptr);
		}
		textureDescriptorsSet->push_back(textureDescriptors);
	}


	//Fullscreen image object
	screenImage fullImage = create_screen_image(window, allocator);

	//lut::CommandPool loadCmdPool = lut::create_command_pool(window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);

	//Scene uniform----------------------------------------------------------------------
	lut::Buffer sceneUBO = lut::create_buffer(allocator, sizeof(glsl::SceneUniform),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY
	);
	// allocate descriptor set for uniform buffer
	VkDescriptorSet sceneDescriptors = lut::alloc_desc_set(window, dpool.handle, sceneLayout.handle);
	{
		VkWriteDescriptorSet desc[1]{};
		VkDescriptorBufferInfo sceneUboInfo{};
		sceneUboInfo.buffer = sceneUBO.buffer;
		sceneUboInfo.range = VK_WHOLE_SIZE;
		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = sceneDescriptors;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		desc[0].descriptorCount = 1;
		desc[0].pBufferInfo = &sceneUboInfo;
		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}



	//Light uniform----------------------------------------------------------------------

	lut::Buffer lightUBO = lut::create_buffer(allocator, sizeof(glsl::LightSource),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	VkDescriptorSet lightDescriptors = lut::alloc_desc_set(window, dpool.handle, lightLayout.handle);
	{
		VkWriteDescriptorSet desc[1]{};
		VkDescriptorBufferInfo lightUboInfo{};
		lightUboInfo.buffer = lightUBO.buffer;
		lightUboInfo.range = VK_WHOLE_SIZE;
		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = lightDescriptors;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		desc[0].descriptorCount = 1;
		desc[0].pBufferInfo = &lightUboInfo;
		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}




	//Gaussian vertical uniform

	lut::Buffer vGaussianUBO = lut::create_buffer(allocator, sizeof(glsl::GaussianUniform),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	VkDescriptorSet vGaussianDescriptors = lut::alloc_desc_set(window, dpool.handle, vGaussianLayout.handle);
	{
		VkWriteDescriptorSet desc[1]{};
		VkDescriptorBufferInfo vGaussianUboInfo{};
		vGaussianUboInfo.buffer = vGaussianUBO.buffer;
		vGaussianUboInfo.range = VK_WHOLE_SIZE;
		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = vGaussianDescriptors;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		desc[0].descriptorCount = 1;
		desc[0].pBufferInfo = &vGaussianUboInfo;
		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

	//Gaussian horizontal uniform
	lut::Buffer hGaussianUBO = lut::create_buffer(allocator, sizeof(glsl::GaussianUniform),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	VkDescriptorSet hGaussianDescriptors = lut::alloc_desc_set(window, dpool.handle, hGaussianLayout.handle);
	{
		VkWriteDescriptorSet desc[1]{};
		VkDescriptorBufferInfo hGaussianUboInfo{};
		hGaussianUboInfo.buffer = hGaussianUBO.buffer;
		hGaussianUboInfo.range = VK_WHOLE_SIZE;
		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = hGaussianDescriptors;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		desc[0].descriptorCount = 1;
		desc[0].pBufferInfo = &hGaussianUboInfo;
		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}




	//Material uniform----------------------------------------------------------------------

	std::vector<VkBuffer*>* materialUBOs = new std::vector<VkBuffer*>;
	std::vector<VkDescriptorSet*>* materialDescriptorSets = new std::vector<VkDescriptorSet*>;
	std::vector<lut::Buffer*>* bufferVec = new std::vector<lut::Buffer*>;
	for (size_t i = 0; i < indexedMesh->size(); ++i) {
		// 创建每个mesh的uniform buffer]

		lut::Buffer* ubo = new lut::Buffer;

		*ubo = (lut::create_buffer(allocator, sizeof(glsl::MaterialUniform),
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU));

		bufferVec->push_back(std::move(ubo));

		materialUBOs->push_back(&(ubo->buffer));

		
		VkDescriptorSet* dset = new VkDescriptorSet;
			
		*dset = lut::alloc_desc_set(window, dpool.handle, materialLayout.handle);

		materialDescriptorSets->push_back(dset);

		std::uint64_t id = (*indexedMesh)[i].materialId;
		VkWriteDescriptorSet desc{};
		VkDescriptorBufferInfo materialUboInfo{};
		materialUboInfo.buffer = *(*materialUBOs)[i];
		materialUboInfo.range = VK_WHOLE_SIZE;
		desc.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc.dstSet = *(*materialDescriptorSets)[i];
		desc.dstBinding = 0;
		desc.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		desc.descriptorCount = 1;
		desc.pBufferInfo = &materialUboInfo;
		vkUpdateDescriptorSets(window.device, 1, &desc, 0, nullptr);


		glsl::MaterialUniform materialUniform; 
		materialUniform.baseColor = glm::vec4(bakedModel.materials[id].baseColor,1.0f);
		materialUniform.emissiveColor = glm::vec4(bakedModel.materials[id].emissiveColor,1.0f);
		materialUniform.rouAndMetal = glm::vec2(bakedModel.materials[id].roughness, bakedModel.materials[id].metalness);

		void* data;
		vmaMapMemory(allocator.allocator, ubo->allocation, &data);
		memcpy(data, &materialUniform, sizeof(glsl::MaterialUniform));
		vmaUnmapMemory(allocator.allocator, ubo->allocation);
	}


	lut::Semaphore imageAvailable = lut::create_semaphore(window);
	lut::Semaphore renderFinished = lut::create_semaphore(window);

	glsl::SceneUniform sceneUniforms{};
	glsl::MaterialUniform materialUniform{};
	glsl::LightSource lightSourceUniforms{ glm::vec4(4.0f, 10.0f, 0.0f,0.0f), glm::vec4(1.0f, 1.0f, 1.0f,0.0f), 1.0f };;

	glsl::GaussianUniform vGaussianUniform{};
	glsl::GaussianUniform hGaussianUniform{};
	calculateGaussianUniform(window, vGaussianUniform, hGaussianUniform);
	// Application main loop
	bool recreateSwapchain = false;
	auto previousClock = Clock_::now();

	while (!glfwWindowShouldClose(window.window))
	{

		glfwPollEvents(); // or: glfwWaitEvents()

		// Recreate swap chain?
		if (recreateSwapchain)
		{
			//TODO: re-create swapchain and associated resources!
			vkDeviceWaitIdle(window.device);

			// Recreate them 
			auto const changes = recreate_swapchain(window);

			if (changes.changedFormat)
				renderPass = create_render_pass(window);


			if (changes.changedSize)
			{
				pipe = create_piepline(window, renderPass.handle, pipeLayout.handle);
				postPipeLine = create_post_pipeline(window, renderPass.handle, postPipeLayout.handle);
				//pipe = create_density_pipeline(window, renderPass.handle, pipeLayout.handle);

				//Task 3
				brightPipeline = create_bright_PBR_pipeline(window, filterPass.handle, bright_PBR_layout.handle, cfg::kBrightVertShaderPath, cfg::kBrightFragShaderPath, 0);
				verticalPipeLine = create_filter_pipeline(window, filterPass.handle, verticalPipeLayout.handle, cfg::kVerticalVertShaderPath, cfg::kVerticalFragShaderPath, 1);//No vertexinput
				horizontalPipeline = create_filter_pipeline(window, filterPass.handle, horizontalPipeLayout.handle, cfg::kHorizontalVertShaderPath, cfg::kHorizontalFragShaderPath, 2);//No vertexinput
				postprocessPipeline = create_filter_pipeline(window, postProcessPass.handle, postProcessPipelayout.handle, cfg::kPostprocessVertShaderPath, cfg::kPostprocessFragShaderPath, 0);//No vertexinput
			}

			if (changes.changedSize)
			{
				std::tie(depthBuffer, depthBufferView) = create_depth_buffer(window, allocator);
				std::tie(interImageBuffer, interImageView) = create_offlineimage_view_buffer(window, VK_FORMAT_R8G8B8A8_SRGB, window, allocator);
				std::tie(brightBuffer, brightView) = create_offlineimage_view_buffer(window, VK_FORMAT_R8G8B8A8_SRGB, window, allocator);
				std::tie(verticalBuffer, verticalView) = create_offlineimage_view_buffer(window, VK_FORMAT_R8G8B8A8_SRGB, window, allocator);
				std::tie(horizontalBuffer, horizontalView) = create_offlineimage_view_buffer(window, VK_FORMAT_R8G8B8A8_SRGB, window, allocator);
				std::tie(PBRBuffer, PBRView) = create_offlineimage_view_buffer(window, VK_FORMAT_R8G8B8A8_SRGB, window, allocator);
				std::tie(depthBuffer, depthBufferView) = create_depth_buffer(window, allocator);

				{
					VkWriteDescriptorSet desc[1]{};
					VkDescriptorImageInfo textureInfo[1]{};

					//Base color
					textureInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
					textureInfo[0].imageView = interImageView.handle;
					textureInfo[0].sampler = defalutSampler.handle;

					desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					desc[0].dstSet = interImagesDescriptor;
					desc[0].dstBinding = 0;
					desc[0].dstArrayElement = 0;
					desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
					desc[0].descriptorCount = 1;
					desc[0].pImageInfo = &textureInfo[0];

					constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
					vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
				}

				{
					VkWriteDescriptorSet desc[1]{};
					VkDescriptorImageInfo textureInfo[1]{};

					//bright color
					textureInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
					textureInfo[0].imageView = brightView.handle;
					textureInfo[0].sampler = defalutSampler.handle;

					desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					desc[0].dstSet = brightDescriptor;
					desc[0].dstBinding = 0;
					desc[0].dstArrayElement = 0;
					desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
					desc[0].descriptorCount = 1;
					desc[0].pImageInfo = &textureInfo[0];

					constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
					vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
				}
				{
					VkWriteDescriptorSet desc[1]{};
					VkDescriptorImageInfo textureInfo[1]{};

					//vertical color
					textureInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
					textureInfo[0].imageView = verticalView.handle;
					textureInfo[0].sampler = defalutSampler.handle;

					desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					desc[0].dstSet = verticalDescriptor;
					desc[0].dstBinding = 0;
					desc[0].dstArrayElement = 0;
					desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
					desc[0].descriptorCount = 1;
					desc[0].pImageInfo = &textureInfo[0];

					constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
					vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);

				}
				{
					VkWriteDescriptorSet desc[1]{};
					VkDescriptorImageInfo textureInfo[1]{};

					//horizontal color
					textureInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
					textureInfo[0].imageView = horizontalView.handle;
					textureInfo[0].sampler = defalutSampler.handle;

					desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					desc[0].dstSet = horizontalDescriptor;
					desc[0].dstBinding = 0;
					desc[0].dstArrayElement = 0;
					desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
					desc[0].descriptorCount = 1;
					desc[0].pImageInfo = &textureInfo[0];

					constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
					vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);

				}

				{
					VkWriteDescriptorSet desc[1]{};
					VkDescriptorImageInfo textureInfo[1]{};

					//PBR color
					textureInfo[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
					textureInfo[0].imageView = PBRView.handle;
					textureInfo[0].sampler = defalutSampler.handle;

					desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					desc[0].dstSet = PBRDescriptor;
					desc[0].dstBinding = 0;
					desc[0].dstArrayElement = 0;
					desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
					desc[0].descriptorCount = 1;
					desc[0].pImageInfo = &textureInfo[0];

					constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
					vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
				}
			}
				
			framebuffers.clear();
			create_NewSwapchain_framebuffers(window, postProcessPass.handle, framebuffers);
			create_framebuffer_R1(window, filterPass.handle, firstFrameBuffer, brightView.handle, verticalView.handle, horizontalView.handle, PBRView.handle, depthBufferView.handle);
			calculateGaussianUniform(window, vGaussianUniform, hGaussianUniform);
			recreateSwapchain = false;
			continue;
		}

		//TODO: acquire swapchain image.
		std::uint32_t imageIndex = 0;
		auto const acquireRes = vkAcquireNextImageKHR(
			window.device,
			window.swapchain,
			std::numeric_limits<std::uint64_t>::max(),
			imageAvailable.handle,
			VK_NULL_HANDLE, &imageIndex);

		if (VK_SUBOPTIMAL_KHR == acquireRes || VK_ERROR_OUT_OF_DATE_KHR == acquireRes)
		{
			recreateSwapchain = true;
			continue;
		}

		if (VK_SUCCESS != acquireRes)
		{
			throw lut::Error("Unable to acquire enxt swapchain image\n"
				"vkAcquireNextImageKHR() returned %s", lut::to_string(acquireRes).c_str()
			);
		}

		//TODO: wait for command buffer to be available

		assert(std::size_t(imageIndex) < cbfences.size());

		if (auto const res = vkWaitForFences(window.device, 1, &cbfences[imageIndex].handle, VK_TRUE,
			std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to wait for command buffer fence %u\n"
				"vkWaitForFences() returned %s", imageIndex, lut::to_string(res).c_str());
		}

		if (auto const res = vkResetFences(window.device, 1, &cbfences[imageIndex].handle); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to reset command buffer fence %u\n" "vkResetFences() returned %s", imageIndex, lut::to_string(res).c_str());
		}

		//TODO: record and submit commands

		assert(std::size_t(imageIndex) < cbuffers.size());
		assert(std::size_t(imageIndex) < framebuffers.size());

		record_commands(
			cbuffers[imageIndex],
			renderPass.handle,
			framebuffers[imageIndex].handle,
			pipe.handle,
			alphaPipe.handle,
			postPipeLine.handle,
			window.swapchainExtent,
			indexedMesh,
			sceneUBO.buffer,
			sceneUniforms,
			lightUBO.buffer,
			lightSourceUniforms,
			pipeLayout.handle,
			sceneDescriptors,
			lightDescriptors,
			interImagesDescriptor,
			bakedModel,
			materialUniform,
			materialDescriptorSets,
			materialUBOs,
			interImageFrameBuffer.handle,
			fullImage,
			postPipeLayout.handle,
			window,
			imageIndex,
			textureDescriptorsSet,

			//Task 3
			brightPipeline.handle,
			verticalPipeLine.handle,
			horizontalPipeline.handle,
			postprocessPipeline.handle,

			filterPass.handle,
			postProcessPass.handle,
			firstFrameBuffer,
			brightDescriptor,
			verticalDescriptor,
			horizontalDescriptor,
			PBRDescriptor,

			bright_PBR_layout.handle,
			verticalPipeLayout.handle,
			horizontalPipeLayout.handle,
			postProcessPipelayout.handle,
			vGaussianUniform,
			vGaussianUBO.buffer,
			vGaussianDescriptors,
			hGaussianUniform,
			hGaussianUBO.buffer,
			hGaussianDescriptors
		);

		submit_commands(
			window,
			cbuffers[imageIndex],
			cbfences[imageIndex].handle,
			imageAvailable.handle,
			renderFinished.handle
		);

		present_results(
			window.presentQueue,
			window.swapchain,
			imageIndex,
			renderFinished.handle,
			recreateSwapchain);


		auto const now = Clock_::now();
		auto const dt = std::chrono::duration_cast<Secondsf_>(now - previousClock).count();
		previousClock = now;

		update_user_state(state, dt);
		update_scene_uniforms(sceneUniforms, window.swapchainExtent.width, window.swapchainExtent.height, state);
	}

	// Cleanup takes place automatically in the destructors, but we sill need
	// to ensure that all Vulkan commands have finished before that.
	vkDeviceWaitIdle(window.device);

	for (auto des : *materialDescriptorSets)
	{
		delete des;
	}
	delete materialDescriptorSets;

	for (auto vkb : *bufferVec)
	{
		delete vkb;
	}
	delete materialUBOs;


	for (auto des : *textureDescriptorsSet)
	{
		delete des;
	}

	delete textureDescriptorsSet;
	delete bufferVec;
	delete indexedMesh;
	return 0;
}
catch (std::exception const& eErr)
{
	std::fprintf(stderr, "\n");
	std::fprintf(stderr, "Error: %s\n", eErr.what());
	return 1;
}



//Key response event
namespace
{
	void glfw_callback_key_press(GLFWwindow* aWindow, int aKey, int /*aScanCode*/, int aAction, int /*aModifierFlags*/)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWindow));
		assert(state);

		bool const isReleased = (GLFW_RELEASE == aAction);

		switch (aKey)
		{
		case GLFW_KEY_W:
			state->inputMap[std::size_t(EInputState::forward)] = !isReleased;
			break;
		case GLFW_KEY_S:
			state->inputMap[std::size_t(EInputState::backward)] = !isReleased;
			break;
		case GLFW_KEY_A:
			state->inputMap[std::size_t(EInputState::strafeLeft)] = !isReleased;
			break;
		case GLFW_KEY_D:
			state->inputMap[std::size_t(EInputState::strafeRight)] = !isReleased;
			break;
		case GLFW_KEY_E:
			state->inputMap[std::size_t(EInputState::levitate)] = !isReleased;
			break;
		case GLFW_KEY_Q:
			state->inputMap[std::size_t(EInputState::sink)] = !isReleased;
			break;

		case GLFW_KEY_LEFT_SHIFT:
			state->inputMap[std::size_t(EInputState::fast)] = !isReleased;
			break;
		case GLFW_KEY_RIGHT_SHIFT:
			state->inputMap[std::size_t(EInputState::fast)] = !isReleased;
			break;

		case GLFW_KEY_LEFT_CONTROL:
			state->inputMap[std::size_t(EInputState::slow)] = !isReleased;
			break;
		case GLFW_KEY_RIGHT_CONTROL:
			state->inputMap[std::size_t(EInputState::slow)] = !isReleased;
			break;
		case GLFW_KEY_LEFT:
			state->inputMap[std::size_t(EInputState::lightStrafeLeft)] = !isReleased;
			break;
		case GLFW_KEY_RIGHT:
			state->inputMap[std::size_t(EInputState::lightStrafeRight)] = !isReleased;
			break;
		case GLFW_KEY_UP:
			state->inputMap[std::size_t(EInputState::lightForward)] = !isReleased;
			break;
		case GLFW_KEY_DOWN:
			state->inputMap[std::size_t(EInputState::lightBackward)] = !isReleased;
			break;
		default:
			;
		}
	}

	void glfw_callback_button(GLFWwindow* aWin, int aBut, int aAct, int)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWin));
		assert(state);

		if (GLFW_MOUSE_BUTTON_RIGHT == aBut && GLFW_PRESS == aAct)
		{
			auto& flag = state->inputMap[std::size_t(EInputState::mousing)];

			flag = !flag;

			if (flag)
				glfwSetInputMode(aWin, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

			else
				glfwSetInputMode(aWin, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

		}
	}

	void glfw_callback_motion(GLFWwindow* aWin, double aX, double aY)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWin));
		assert(state);

		state->mouseX = float(aX);
		state->mouseY = float(aY);
	}
}

namespace
{
	void update_scene_uniforms(glsl::SceneUniform& aSceneUniforms, std::uint32_t aFramebufferWidth, std::uint32_t aFramebufferHeight, UserState const& userState)
	{
		float const aspect = aFramebufferWidth / float(aFramebufferHeight);

		aSceneUniforms.projection = glm::perspectiveRH_ZO(
			lut::Radians(cfg::kCameraFov).value(),
			aspect,
			cfg::kCameraNear,
			cfg::kCameraFar
		);
		aSceneUniforms.projection[1][1] *= -1.f; // mirror Y axis 

		//aSceneUniforms.camera = glm::translate(glm::vec3(0.f, -0.3f, -1.f));
		aSceneUniforms.camera = glm::inverse(userState.camera2world);

		aSceneUniforms.projCam = aSceneUniforms.projection * aSceneUniforms.camera;

		aSceneUniforms.cameraPos = glm::vec3(userState.camera2world[3]);

	}

	void update_user_state(UserState& aState, float aElapsedTime)
	{
		auto& cam = aState.camera2world;

		if (aState.inputMap[std::size_t(EInputState::mousing)])
		{
			// Only update the rotation on the second frame of mouse 7
			// navigation. This ensures that the previousX and Y variables are 8
			// initialized to sensible values. 9
			if (aState.wasMousing)
			{
				auto const sens = cfg::kCameraMouseSensitivity;
				auto const dx = sens * (aState.mouseX - aState.previousX);
				auto const dy = sens * (aState.mouseY - aState.previousY);

				cam = cam * glm::rotate(-dy, glm::vec3(1.f, 0.f, 0.f));
				cam = cam * glm::rotate(-dx, glm::vec3(0.f, 1.f, 0.f));
			}

			aState.previousX = aState.mouseX;
			aState.previousY = aState.mouseY;
			aState.wasMousing = true;
		}
		else
		{
			aState.wasMousing = false;
		}

		auto const move = aElapsedTime * cfg::kCameraBaseSpeed *
			(aState.inputMap[std::size_t(EInputState::fast)] ? cfg::kCameraFastMult : 1.f) *
			(aState.inputMap[std::size_t(EInputState::slow)] ? cfg::kCameraSlowMult : 1.f)
			;

		if (aState.inputMap[std::size_t(EInputState::forward)])
			cam = cam * glm::translate(glm::vec3(0.f, 0.f, -move));
		if (aState.inputMap[std::size_t(EInputState::backward)])
			cam = cam * glm::translate(glm::vec3(0.f, 0.f, +move));

		if (aState.inputMap[std::size_t(EInputState::strafeLeft)])
			cam = cam * glm::translate(glm::vec3(-move, 0.f, 0.f));
		if (aState.inputMap[std::size_t(EInputState::strafeRight)])
			cam = cam * glm::translate(glm::vec3(+move, 0.f, 0.f));

		if (aState.inputMap[std::size_t(EInputState::levitate)])
			cam = cam * glm::translate(glm::vec3(0.f, +move, 0.f));
		if (aState.inputMap[std::size_t(EInputState::sink)])
			cam = cam * glm::translate(glm::vec3(0.f, -move, 0.f));

	}

}



//Vulkan logic
namespace
{
	//Task 1 & 2
	lut::RenderPass create_render_pass(lut::VulkanWindow const& aWindow)
	{

		VkAttachmentDescription attachments[3]{};
		//off screen color attachment
		attachments[0].format = VK_FORMAT_R8G8B8A8_SRGB; //changed! 
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; //changed! 

		//Depth attachment
		attachments[1].format = cfg::kDepthFormat;
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		//swap chain images attachment
		attachments[2].format = aWindow.swapchainFormat;
		attachments[2].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[2].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;



		//Subpass 1
		VkAttachmentReference colorAttachmentRef0{};
		colorAttachmentRef0.attachment = 0;
		colorAttachmentRef0.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachmentRef0{};
		depthAttachmentRef0.attachment = 1;
		depthAttachmentRef0.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;


		VkAttachmentReference inputAttachmentRef1{};
		inputAttachmentRef1.attachment = 0;
		inputAttachmentRef1.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		VkAttachmentReference colorAttachmentRef1{};
		colorAttachmentRef1.attachment = 2;
		colorAttachmentRef1.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;


		//Subpasses, subrendering procedures
		VkSubpassDescription subpasses[2]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 1;
		subpasses[0].pColorAttachments = &colorAttachmentRef0;
		subpasses[0].pDepthStencilAttachment = &depthAttachmentRef0;


		subpasses[1].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[1].colorAttachmentCount = 1;
		subpasses[1].pColorAttachments = &colorAttachmentRef1;
		subpasses[1].inputAttachmentCount = 1;
		subpasses[1].pInputAttachments = &inputAttachmentRef1;


		// Subpass dependency
		VkSubpassDependency dependency = {};
		dependency.srcSubpass = 0;
		dependency.dstSubpass = 1;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependency.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;


		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 3;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 2;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 1; //changed! 

		VkSubpassDependency dependencies[] = { dependency };
		passInfo.pDependencies = dependencies; //changed! 

		VkRenderPass rpass = VK_NULL_HANDLE;
		if (auto const res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create render pass\n" "vkCreateRenderPass() returned %s", lut::to_string(res).c_str());
		}

		return lut::RenderPass(aWindow.device, rpass);

	}

	lut::RenderPass create_postRender_pass(lut::VulkanWindow const& aWindow)
	{
		VkAttachmentDescription attachments[1]{};
		attachments[0].format = aWindow.swapchainFormat; //changed! 
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; //changed! 

		VkAttachmentReference subpassAttachments[1]{};
		subpassAttachments[0].attachment = 0; // this refers to attachments[0] 
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;


		//Subpasses, subrendering procedures
		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 1;
		subpasses[0].pColorAttachments = subpassAttachments;

		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 1;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 0; //changed! 
		passInfo.pDependencies = nullptr; //changed! 
		VkRenderPass rpass = VK_NULL_HANDLE;

		if (auto const res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create render pass\n" "vkCreateRenderPass() returned %s", lut::to_string(res).c_str());
		}

		return lut::RenderPass(aWindow.device, rpass);
	}

	//Task 3
	lut::RenderPass create_filter_PBR_render_pass(lut::VulkanWindow const& aWindow)
	{
		//Bright light
		VkAttachmentDescription attachments[5]{};
		//off screen color attachment
		//Finding brightlight
		attachments[0].format = VK_FORMAT_R8G8B8A8_SRGB; //changed! 
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; //changed! 

		//Vertical
		attachments[1].format = VK_FORMAT_R8G8B8A8_SRGB; //changed! 
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; //changed! 

		//Horizontal
		attachments[2].format = VK_FORMAT_R8G8B8A8_SRGB; //changed! 
		attachments[2].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[2].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; //changed! 


		//PBR attachment
		attachments[3].format = VK_FORMAT_R8G8B8A8_SRGB; //changed! 
		attachments[3].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[3].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[3].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[3].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[3].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; //changed! 

		//Depth attachment
		attachments[4].format = cfg::kDepthFormat;
		attachments[4].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[4].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[4].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[4].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[4].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;



		
		//One
		VkAttachmentReference colorAttachmentRef0[] = {
		{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
		{3, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}
		};



		VkAttachmentReference colorAttachmentRef1{};
		colorAttachmentRef1.attachment = 1;
		colorAttachmentRef1.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference inputAttachmentRef1{};
		inputAttachmentRef1.attachment = 0;
		inputAttachmentRef1.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;



		VkAttachmentReference colorAttachmentRef2{};
		colorAttachmentRef2.attachment = 2;
		colorAttachmentRef2.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference inputAttachmentRef2{};
		inputAttachmentRef2.attachment = 1;
		inputAttachmentRef2.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;



		VkAttachmentReference depthAttachmentRef0{};
		depthAttachmentRef0.attachment = 4;
		depthAttachmentRef0.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;


		//Subpasses, subrendering procedures
		VkSubpassDescription subpasses[3]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 2;
		subpasses[0].pColorAttachments = colorAttachmentRef0;
		subpasses[0].pDepthStencilAttachment = &depthAttachmentRef0;


		subpasses[1].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[1].colorAttachmentCount = 1;
		subpasses[1].pColorAttachments = &colorAttachmentRef1;
		subpasses[1].pDepthStencilAttachment = &depthAttachmentRef0;
		subpasses[1].pInputAttachments = &inputAttachmentRef1;
		subpasses[1].inputAttachmentCount = 1;


		subpasses[2].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[2].colorAttachmentCount = 1;
		subpasses[2].pColorAttachments = &colorAttachmentRef2;
		subpasses[2].pDepthStencilAttachment = &depthAttachmentRef0;
		subpasses[2].pInputAttachments = &inputAttachmentRef2;
		subpasses[2].inputAttachmentCount = 1;



		//Create dependency
		VkSubpassDependency dependency = {};
		dependency.srcSubpass = 0;
		dependency.dstSubpass = 1;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependency.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		VkSubpassDependency dependency1 = {};
		dependency1.srcSubpass = 1;
		dependency1.dstSubpass = 2;
		dependency1.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency1.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependency1.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependency1.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	


		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 5;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 3;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 2; //changed! 

		VkSubpassDependency dependencies[] = { dependency,dependency1};
		passInfo.pDependencies = dependencies; //changed! 

		VkRenderPass rpass = VK_NULL_HANDLE;
		if (auto const res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create render pass\n" "vkCreateRenderPass() returned %s", lut::to_string(res).c_str());
		}

		return lut::RenderPass(aWindow.device, rpass);
	}


	lut::RenderPass create_actualScene_render_pass(lut::VulkanWindow const& aWindow)
	{
		//Bright light
		VkAttachmentDescription attachments[2]{};

		//off screen color attachment
		//Finding brightlight
		attachments[0].format = VK_FORMAT_R8G8B8A8_SRGB; //changed! 
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; //changed! 

		//Depth attachment
		attachments[1].format = cfg::kDepthFormat;
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;



		VkAttachmentReference colorAttachmentRef0{};
		colorAttachmentRef0.attachment = 0;
		colorAttachmentRef0.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;


		VkAttachmentReference depthAttachmentRef0{};
		depthAttachmentRef0.attachment = 1;
		depthAttachmentRef0.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;


		//Subpasses, subrendering procedures
		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 1;
		subpasses[0].pColorAttachments = &colorAttachmentRef0;
		subpasses[0].pDepthStencilAttachment = &depthAttachmentRef0;


		//Create dependency
		VkSubpassDependency dependencies[1]{};
		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;




		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 2;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 1; //changed! 

		passInfo.pDependencies = dependencies; //changed! 

		VkRenderPass rpass = VK_NULL_HANDLE;
		if (auto const res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create render pass\n" "vkCreateRenderPass() returned %s", lut::to_string(res).c_str());
		}

		return lut::RenderPass(aWindow.device, rpass);
	}


	lut::RenderPass create_postProcessing_render_pass(lut::VulkanWindow const& aWindow)
	{
		VkAttachmentDescription attachments[1]{};
		attachments[0].format = aWindow.swapchainFormat; //changed! 
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; //changed! 


		VkAttachmentReference subpassAttachments[1]{};
		subpassAttachments[0].attachment = 0; // this refers to attachments[0] 
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;


		//Subpasses, subrendering procedures
		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 1;
		subpasses[0].pColorAttachments = subpassAttachments;


		VkSubpassDependency dependencies[1]{};
		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;


		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = sizeof(attachments) / sizeof(attachments[0]);
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 1;
		passInfo.pDependencies = dependencies;

		VkRenderPass rpass = VK_NULL_HANDLE;
		if (auto const res = vkCreateRenderPass(aWindow.device, &passInfo,
			nullptr, &rpass); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create render pass for texture\n"
				"vkCreateRenderPass() returned %s", lut::to_string(res).c_str());
		}

		return lut::RenderPass(aWindow.device, rpass);
	}


	lut::PipelineLayout create_pipeline_layout(lut::VulkanContext const& aContext, VkDescriptorSetLayout const& aSceneLayout, VkDescriptorSetLayout aMaterialLayout, VkDescriptorSetLayout  const& aTexturelayout,
		VkDescriptorSetLayout const& aLightSource
	)
	{
		VkDescriptorSetLayout layouts[] = {
			aSceneLayout,// set 0
			aMaterialLayout, //set 1
			aTexturelayout, // set 2
			aLightSource
		};

		VkPushConstantRange pushConstantRange{};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;  
		pushConstantRange.offset = 0;  
		pushConstantRange.size = sizeof(int) + sizeof(int); 

		//create a pipeline layout object(VkPipelineLayout),
		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]);
		layoutInfo.pSetLayouts = layouts;
		layoutInfo.pushConstantRangeCount = 1;
		layoutInfo.pPushConstantRanges = &pushConstantRange;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create pipeline layout\n""vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str());
		}
		return lut::PipelineLayout(aContext.device, layout);
	}

	lut::PipelineLayout create_bright_PBR_pipeline_layout(lut::VulkanContext const& aContext, VkDescriptorSetLayout const& aSceneLayout, VkDescriptorSetLayout aMaterialLayout, VkDescriptorSetLayout  const& aTexturelayout,
		VkDescriptorSetLayout const& aLightSource) 
	{
		VkDescriptorSetLayout layouts[] = {
		aSceneLayout,// set 0
		aMaterialLayout, //set 1
		aTexturelayout, // set 2
		aLightSource
		};

		VkPushConstantRange pushConstantRange{};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(int) + sizeof(int);

		//create a pipeline layout object(VkPipelineLayout),
		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]);
		layoutInfo.pSetLayouts = layouts;
		layoutInfo.pushConstantRangeCount = 1;
		layoutInfo.pPushConstantRanges = &pushConstantRange;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create pipeline layout\n""vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str());
		}
		return lut::PipelineLayout(aContext.device, layout);
	}

	lut::PipelineLayout create_postPipeline_layout(lut::VulkanContext const& aContext, VkDescriptorSetLayout const& aIntermidiateDescriptor)
	{

		VkDescriptorSetLayout layouts[] = { // Order must match the set = N in the shaders 
			aIntermidiateDescriptor
		};

		// Pipeline layout for Render pass B
		VkPipelineLayoutCreateInfo layoutInfoB{};
		layoutInfoB.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfoB.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]);
		layoutInfoB.pSetLayouts = layouts;
		layoutInfoB.pushConstantRangeCount = 0;
		layoutInfoB.pPushConstantRanges = nullptr;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfoB, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create pipeline layout\n""vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str());
		}
		return lut::PipelineLayout(aContext.device, layout);
	}

	lut::PipelineLayout create_vertical_pipeline_layout(lut::VulkanContext const& aContext, VkDescriptorSetLayout const& aBrightLayout, VkDescriptorSetLayout vGaussianLayout)
	{
		VkDescriptorSetLayout layouts[] = { // Order must match the set = N in the shaders 
			aBrightLayout,
			vGaussianLayout
		};

		// Pipeline layout for Render pass B
		VkPipelineLayoutCreateInfo layoutInfoB{};
		layoutInfoB.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfoB.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]);
		layoutInfoB.pSetLayouts = layouts;
		layoutInfoB.pushConstantRangeCount = 0;
		layoutInfoB.pPushConstantRanges = nullptr;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfoB, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create pipeline layout\n""vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str());
		}
		return lut::PipelineLayout(aContext.device, layout);
	}
	lut::PipelineLayout create_horizontal_pipeline_layout(lut::VulkanContext const& aContext, VkDescriptorSetLayout const& aVerticalLayout, VkDescriptorSetLayout hGaussianLayout)
	{
		VkDescriptorSetLayout layouts[] = { // Order must match the set = N in the shaders 
			aVerticalLayout,
			hGaussianLayout
		};

		// Pipeline layout for Render pass B
		VkPipelineLayoutCreateInfo layoutInfoB{};
		layoutInfoB.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfoB.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]);
		layoutInfoB.pSetLayouts = layouts;
		layoutInfoB.pushConstantRangeCount = 0;
		layoutInfoB.pPushConstantRanges = nullptr;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfoB, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create pipeline layout\n""vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str());
		}
		return lut::PipelineLayout(aContext.device, layout);
	}
	lut::PipelineLayout create_postprocess_pipeline_layout(lut::VulkanContext const& aContext, VkDescriptorSetLayout const& aGaussianLayout, VkDescriptorSetLayout const& aPBRLayout)
	{
		VkDescriptorSetLayout layouts[] = { // Order must match the set = N in the shaders 
			aGaussianLayout,
			aPBRLayout
		};

		// Pipeline layout for Render pass B
		VkPipelineLayoutCreateInfo layoutInfoB{};
		layoutInfoB.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfoB.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]);
		layoutInfoB.pSetLayouts = layouts;
		layoutInfoB.pushConstantRangeCount = 0;
		layoutInfoB.pPushConstantRanges = nullptr;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfoB, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create pipeline layout\n""vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str());
		}
		return lut::PipelineLayout(aContext.device, layout);
	}




	lut::Pipeline create_piepline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{

		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kFragShaderPath);

		//There are 2 stages: VertexShader -> Fragment shader
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";


		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;

		VkVertexInputBindingDescription vertexInputs[3]{};
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(float) * 3;
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[1].binding = 1;
		vertexInputs[1].stride = sizeof(float) * 2;
		vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[2].binding = 2;
		vertexInputs[2].stride = sizeof(float) * 3;
		vertexInputs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription vertexAttributes[3]{};
		vertexAttributes[0].binding = 0; // must match binding above 
		vertexAttributes[0].location = 0; // must match shader 
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;

		vertexAttributes[1].binding = 1; // must match binding above 
		vertexAttributes[1].location = 1; // must match shader 
		vertexAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
		vertexAttributes[1].offset = 0;

		vertexAttributes[2].binding = 2; // must match binding above 
		vertexAttributes[2].location = 2; // must match shader 
		vertexAttributes[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[2].offset = 0;


		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 3; // number of vertexInputs above 
		inputInfo.pVertexBindingDescriptions = vertexInputs;
		inputInfo.vertexAttributeDescriptionCount = 3; // number of vertexAttributes above 
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;


		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		VkViewport viewPort{};
		viewPort.x = 0.f;
		viewPort.y = 0.f;
		viewPort.width = float(aWindow.swapchainExtent.width);
		viewPort.height = float(aWindow.swapchainExtent.height);
		viewPort.minDepth = 0.f;
		viewPort.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0,0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width,aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewPort;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		//Rasterization State
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.f; // required. 

		//Multisample State
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeInfo.stageCount = 2; // vertex + fragment stages 
		pipeInfo.pStages = stages;
		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr; // no tessellation 
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr; // no dynamic states 
		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0; // first subpass of aRenderPass 

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n" "vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}
	lut::Pipeline create_alpha_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{
		// Load shader modules 
		// For this example, we only use the vertex and fragment shaders.
		// Other shader stages (geometry, tessellation) aren’t used here, and as such we omit them.
		// Load the 
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kFragShaderPath);


		//There are 2 stages: VertexShader -> Fragment shader
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";


		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_FALSE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;

		VkVertexInputBindingDescription vertexInputs[3]{};
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(float) * 3;
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[1].binding = 1;
		vertexInputs[1].stride = sizeof(float) * 2;
		vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[2].binding = 2;
		vertexInputs[2].stride = sizeof(float) * 3;
		vertexInputs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;


		/**The vertex shader expects two inputs, the position and the color.
		Consequently, these are described with two VkVertexInputAttributeDescription instances*/
		VkVertexInputAttributeDescription vertexAttributes[3]{};
		vertexAttributes[0].binding = 0; // must match binding above 
		vertexAttributes[0].location = 0; // must match shader 
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;

		vertexAttributes[1].binding = 1; // must match binding above 
		vertexAttributes[1].location = 1; // must match shader 
		vertexAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
		vertexAttributes[1].offset = 0;

		vertexAttributes[2].binding = 2; // must match binding above 
		vertexAttributes[2].location = 2; // must match shader 
		vertexAttributes[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[2].offset = 0;





		//Vertex Input state
		//we specify what buffers vertices are sourced from, and what vertex attributes in our shaders these correspond to
		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 3; // number of vertexInputs above 
		inputInfo.pVertexBindingDescriptions = vertexInputs;
		inputInfo.vertexAttributeDescriptionCount = 3; // number of vertexAttributes above 
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;


		//VkPipelineVertexInputStateCreateInfo inputInfo{};
		//inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		//inputInfo.vertexBindingDescriptionCount = 3; // number of vertexInputs above 
		//inputInfo.pVertexBindingDescriptions = vertexInputs;
		//inputInfo.vertexAttributeDescriptionCount = 3; // number of vertexAttributes above 
		//inputInfo.pVertexAttributeDescriptions = vertexAttributes;


		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		//Tessellation State
		//For this exercise, we can leave it as nullptr
		//--To be implemented
		//VkPipelineTessellationDomainOriginStateCreateInfo tessellationInfo{};

		//Viewport State create info:
		//1: Initialize viewPort;
		//2: Initialize scissor;
		//3： createInfo
		VkViewport viewPort{};
		viewPort.x = 0.f;
		viewPort.y = 0.f;
		viewPort.width = float(aWindow.swapchainExtent.width);
		viewPort.height = float(aWindow.swapchainExtent.height);
		viewPort.minDepth = 0.f;
		viewPort.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0,0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width,aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewPort;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		//Rasterization State
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.f; // required. 

		//Multisample State
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		//Depth/Stencil State levae it as nullptr
		//To be implemented:----
		//VkPipelineDepthStencilStateCreateInfo depthStencilInfo{ };


		//Color Blend State
		// Define blend state 1
		// We define one blend state per color attachment - this example uses a 
		// single color attachment, so we only need one. Right now, we don’t do any 
		// blending, so we can ignore most of the members. 

		//1: Initialize blendStates
		//2: Create colorblendStateInfo
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_TRUE; // New! Used to be VK FALSE. 
		blendStates[0].colorBlendOp = VK_BLEND_OP_ADD; // New! 
		blendStates[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA; // New! 
		blendStates[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; // New! 
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		blendStates[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		blendStates[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		blendStates[0].alphaBlendOp = VK_BLEND_OP_ADD;


		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		//Dynamic States
		//Exercise 2 does not use any dynamic state, so the pDynamicState member is left at nullptr.
		//To be implemented:
		//VkPipelineDynamicStateCreateInfo dynamicInfo{};


		//Create the pipeLine
		//Some states that we won't use will be set to nullptr
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeInfo.stageCount = 2; // vertex + fragment stages 
		pipeInfo.pStages = stages;
		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr; // no tessellation 
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo; // no depth or stencil buffers 
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr; // no dynamic states 
		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0; // first subpass of aRenderPass 

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n" "vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}

	lut::Pipeline create_post_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{

		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kPostVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kPostFragShaderPath);

		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";


		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_FALSE;
		depthInfo.depthWriteEnable = VK_FALSE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;

		VkVertexInputBindingDescription vertexInputs[2]{};
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(float) * 2;
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[1].binding = 1;
		vertexInputs[1].stride = sizeof(float) * 2;
		vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription vertexAttributes[2]{};
		vertexAttributes[0].binding = 0; // must match binding above 
		vertexAttributes[0].location = 0; // must match shader 
		vertexAttributes[0].format = VK_FORMAT_R32G32_SFLOAT;
		vertexAttributes[0].offset = 0;

		vertexAttributes[1].binding = 1; // must match binding above 
		vertexAttributes[1].location = 1; // must match shader 
		vertexAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
		vertexAttributes[1].offset = 0;


		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 2; // number of vertexInputs above 
		inputInfo.pVertexBindingDescriptions = vertexInputs;
		inputInfo.vertexAttributeDescriptionCount = 2; // number of vertexAttributes above 
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;


		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		VkViewport viewPort{};
		viewPort.x = 0.f;
		viewPort.y = 0.f;
		viewPort.width = float(aWindow.swapchainExtent.width);
		viewPort.height = float(aWindow.swapchainExtent.height);
		viewPort.minDepth = 0.f;
		viewPort.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0,0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width,aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewPort;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		//Rasterization State
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.f; // required. 

		//Multisample State
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeInfo.stageCount = 2; // vertex + fragment stages 
		pipeInfo.pStages = stages;
		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr; // no tessellation 
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo; // no depth or stencil buffers 
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr; // no dynamic states 
		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 1; // second subpass of aRenderPass 

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n" "vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}


	lut::Pipeline create_bright_PBR_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout, 
		char const* kVertShaderPath, char const* kFragShaderPath, uint32_t subpassIndex)
	{
		lut::ShaderModule vert = lut::load_shader_module(aWindow, kVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, kFragShaderPath);

		//There are 2 stages: VertexShader -> Fragment shader
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";


		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;

		VkVertexInputBindingDescription vertexInputs[3]{};
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(float) * 3;
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[1].binding = 1;
		vertexInputs[1].stride = sizeof(float) * 2;
		vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[2].binding = 2;
		vertexInputs[2].stride = sizeof(float) * 3;
		vertexInputs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription vertexAttributes[3]{};
		vertexAttributes[0].binding = 0; // must match binding above 
		vertexAttributes[0].location = 0; // must match shader 
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;

		vertexAttributes[1].binding = 1; // must match binding above 
		vertexAttributes[1].location = 1; // must match shader 
		vertexAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
		vertexAttributes[1].offset = 0;

		vertexAttributes[2].binding = 2; // must match binding above 
		vertexAttributes[2].location = 2; // must match shader 
		vertexAttributes[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[2].offset = 0;


		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 3; // number of vertexInputs above 
		inputInfo.pVertexBindingDescriptions = vertexInputs;
		inputInfo.vertexAttributeDescriptionCount = 3; // number of vertexAttributes above 
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;


		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		VkViewport viewPort{};
		viewPort.x = 0.f;
		viewPort.y = 0.f;
		viewPort.width = float(aWindow.swapchainExtent.width);
		viewPort.height = float(aWindow.swapchainExtent.height);
		viewPort.minDepth = 0.f;
		viewPort.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0,0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width,aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewPort;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		//Rasterization State
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.f; // required. 

		//Multisample State
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState blendStates[2]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		blendStates[1].blendEnable = VK_FALSE;
		blendStates[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 2;
		blendInfo.pAttachments = blendStates;

		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeInfo.stageCount = 2; // vertex + fragment stages 
		pipeInfo.pStages = stages;
		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr; // no tessellation 
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr; // no dynamic states 
		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = subpassIndex; // first subpass of aRenderPass 

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n" "vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}


	lut::Pipeline create_filter_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout,
		char const* kVertShaderPath, char const* kFragShaderPath, uint32_t subpassIndex)
	{
		// Load shader modules
		lut::ShaderModule vert = lut::load_shader_module(aWindow, kVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, kFragShaderPath);

		// Define shader stages in the pipeline
		// Two stages, 1. Vertex shader 2. Fragment shader
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 0;
		inputInfo.pVertexBindingDescriptions = nullptr;
		inputInfo.vertexAttributeDescriptionCount = 0;
		inputInfo.pVertexAttributeDescriptions = nullptr;

		// Define which primitive (point, line, triangle,...)
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		// Define viewport and scissor regions
		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0, 0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width,
			aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		// Define rasterization options
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_FRONT_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.0f;

		// Define multisampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// Define blend state
		// i.e. which color channels to write
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorBlendOp = VK_BLEND_OP_ADD;
		blendStates[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		blendStates[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
			VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		// Depth Testing
		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.0f;
		depthInfo.maxDepthBounds = 1.0f;

		// Create pipeline
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		pipeInfo.stageCount = 2; // Vertex and fragment stages
		pipeInfo.pStages = stages;

		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr;
		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = subpassIndex;

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device,
			VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n"
				"vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}



	void create_swapchain_framebuffers(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, std::vector<lut::Framebuffer>& aFramebuffers, 
		VkImageView aDepthView,VkImageView aInterImageView)
	{
		assert(aFramebuffers.empty());

		for (std::size_t i = 0; i < aWindow.swapViews.size(); ++i)
		{
			VkImageView attachments[3] = {
				aInterImageView,
				aDepthView,           // depth ImageView
				aWindow.swapViews[i]
				// 与swapchain关联的ImageView，格式是aWindow.swapchainFormat
			};

			VkFramebufferCreateInfo fbInfo{};
			fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbInfo.flags = 0; // normal framebuffer 
			fbInfo.renderPass = aRenderPass;
			fbInfo.attachmentCount = 3;
			fbInfo.pAttachments = attachments;
			fbInfo.width = aWindow.swapchainExtent.width;
			fbInfo.height = aWindow.swapchainExtent.height;
			fbInfo.layers = 1;

			VkFramebuffer fb = VK_NULL_HANDLE;
			if (auto const res = vkCreateFramebuffer(aWindow.device, &fbInfo, nullptr, &fb); VK_SUCCESS != res)
			{

				throw lut::Error("Unable to create framebuffer for swap chain image %zu\n" "vkCreateFramebuffer() returned %s", i, lut::to_string(res).c_str());

			}

			aFramebuffers.emplace_back(lut::Framebuffer(aWindow.device, fb));
		}


		assert(aWindow.swapViews.size() == aFramebuffers.size());
	}

	


	void create_NewSwapchain_framebuffers(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, std::vector<lut::Framebuffer>& aFramebuffers)
	{
		assert(aFramebuffers.empty());

		for (std::size_t i = 0; i < aWindow.swapViews.size(); ++i)
		{
			VkImageView attachments[1] = {
				aWindow.swapViews[i]
			};

			VkFramebufferCreateInfo fbInfo{};
			fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbInfo.flags = 0; // normal framebuffer 
			fbInfo.renderPass = aRenderPass;
			fbInfo.attachmentCount = 1;
			fbInfo.pAttachments = attachments;
			fbInfo.width = aWindow.swapchainExtent.width;
			fbInfo.height = aWindow.swapchainExtent.height;
			fbInfo.layers = 1;

			VkFramebuffer fb = VK_NULL_HANDLE;
			if (auto const res = vkCreateFramebuffer(aWindow.device, &fbInfo, nullptr, &fb); VK_SUCCESS != res)
			{

				throw lut::Error("Unable to create framebuffer for swap chain image %zu\n" "vkCreateFramebuffer() returned %s", i, lut::to_string(res).c_str());

			}
			aFramebuffers.emplace_back(lut::Framebuffer(aWindow.device, fb));
		}

		assert(aWindow.swapViews.size() == aFramebuffers.size());
	}
	
	
	
	
	lut::Framebuffer create_interImage_framebuffer(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass,VkImageView depthView, VkImageView interImageView)
	{

		
		VkImageView attachments[3] = {
			interImageView,      // 中间渲染阶段的ImageView，格式是aWindow.swapchainFormat
			depthView,           // 深度附件的ImageView
			aWindow.swapViews[0]           // 深度附件的ImageView
		};


		VkFramebufferCreateInfo framebufferInfo{};
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferInfo.renderPass = aRenderPass; 
		framebufferInfo.attachmentCount = 3;
		framebufferInfo.pAttachments = attachments;
		framebufferInfo.width = aWindow.swapchainExtent.width;
		framebufferInfo.height = aWindow.swapchainExtent.height;
		framebufferInfo.layers = 1;

		VkFramebuffer framebuffer;
		if (auto const res = vkCreateFramebuffer(aWindow.device, &framebufferInfo, nullptr, &framebuffer); res != VK_SUCCESS) {
			throw lut::Error("Unable to create framebuffer\n"
				"vkCreateFramebuffer() returned %s", lut::to_string(res).c_str()
			);
		}
		return lut::Framebuffer(aWindow.device, framebuffer);
	}


	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const& aWindow, lut::Allocator const& aAllocator)
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = cfg::kDepthFormat;
		imageInfo.extent.width = aWindow.swapchainExtent.width;
		imageInfo.extent.height = aWindow.swapchainExtent.height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;
		if (const auto res = vmaCreateImage(aAllocator.allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to allocate depth buffer image.\n"
				"vmaCreateImage() returned %s", lut::to_string(res).c_str()
			);
		}

		lut::Image depthImage(aAllocator.allocator, image, allocation);

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = depthImage.image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = cfg::kDepthFormat;
		viewInfo.components = VkComponentMapping{};
		viewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_DEPTH_BIT,
			0,1,
			0,1
		};

		VkImageView view = VK_NULL_HANDLE;
		if (const auto res = vkCreateImageView(aWindow.device, &viewInfo, nullptr, &view); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create image view\n"
				"vkCreateImageView() returned %s", lut::to_string(res).c_str()
			);
		}

		return { std::move(depthImage),lut::ImageView(aWindow.device,view) };
	}

	std::tuple<lut::Image, lut::ImageView> create_offlineimage_view_buffer(lut::VulkanContext const& aContext, VkFormat aFormat, lut::VulkanWindow const& aWindow, lut::Allocator const& aAllocator)
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
		imageInfo.extent.width = aWindow.swapchainExtent.width;
		imageInfo.extent.height = aWindow.swapchainExtent.height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT| VK_IMAGE_USAGE_SAMPLED_BIT;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;
		if (const auto res = vmaCreateImage(aAllocator.allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to allocate depth buffer image.\n"
				"vmaCreateImage() returned %s", lut::to_string(res).c_str()
			);
		}
		lut::Image interImage(aAllocator.allocator, image, allocation);

		VkImageViewCreateInfo imageViewInfo{};
		imageViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		imageViewInfo.image = image;
		imageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageViewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
		imageViewInfo.components = {
			VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
		imageViewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_COLOR_BIT,
			0,1,
			0,1
		};

		VkImageView view = VK_NULL_HANDLE;
		if (const auto res = vkCreateImageView(aWindow.device, &imageViewInfo, nullptr, &view); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create image view\n"
				"vkCreateImageView() returned %s", lut::to_string(res).c_str()
			);
		}

		return {std::move(interImage),lut::ImageView(aWindow.device,std::move(view)) };
	}

	void record_commands(
		VkCommandBuffer aCmdBuff,
		VkRenderPass aRenderPass,
		VkFramebuffer aFramebuffer,
		VkPipeline aGraphicsPipe,
		VkPipeline aAlphaPipe,
		VkPipeline aPostPipe,
		VkExtent2D const& aImageExtent,
		std::vector<IndexedMesh>* indexedMesh,
		VkBuffer aSceneUBO,
		glsl::SceneUniform
		const& aSceneUniform,
		VkBuffer aLightUBO,
		glsl::LightSource const& aLightUniform,
		VkPipelineLayout aGraphicsLayout,
		VkDescriptorSet aSceneDescriptors,
		VkDescriptorSet lightDescriptors,
		VkDescriptorSet interImageDescriptors,
		BakedModel const& bakedModel,
		glsl::MaterialUniform& aMaterialUniform,
		std::vector<VkDescriptorSet*>* materialDescriptor,
		std::vector<VkBuffer*>* materialBuffer,
		VkFramebuffer interImageBuffer,
		screenImage const& fullImage,
		VkPipelineLayout postPipeLayout,
		lut::VulkanWindow const& aWindow,
		std::uint32_t imageIndex,
		std::vector<VkDescriptorSet*>* textureDescriptorsSet,
		//Task3
		VkPipeline brightPipe,
		VkPipeline verticalpipe,
		VkPipeline horizontalPipe,
		VkPipeline postprocessPipe,

		VkRenderPass filterPass,
		VkRenderPass postProcessPass,

		lut::Framebuffer const& firstFrameBuffer,
		VkDescriptorSet brightDescriptors,
		VkDescriptorSet verticalDescriptors,
		VkDescriptorSet horizontalDescriptors,
		VkDescriptorSet PBRDescriptors,

		VkPipelineLayout bright_PBR_layout,
		VkPipelineLayout verticalPipeLayout,
		VkPipelineLayout horizontalPipeLayout,
		VkPipelineLayout postProcessPipeLayout,

		glsl::GaussianUniform& vGaussianUniform,
		VkBuffer vGaussianUBO,
		VkDescriptorSet vGaussianDescriptors,

		glsl::GaussianUniform& hGaussianUniform,
		VkBuffer hGaussianUBO,
		VkDescriptorSet hGaussianDescriptors
	)
	{
		// Begin recording commands 
		VkCommandBufferBeginInfo begInfo{};
		begInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		begInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(aCmdBuff, &begInfo); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to begin recording command buffer\n""vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}


		// Upload scene uniforms
		lut::buffer_barrier(aCmdBuff, aSceneUBO,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT);

		vkCmdUpdateBuffer(aCmdBuff, aSceneUBO, 0, sizeof(glsl::SceneUniform), &aSceneUniform);

		lut::buffer_barrier(aCmdBuff,
			aSceneUBO,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT
		);

		// Upload light uniforms
		lut::buffer_barrier(aCmdBuff, aLightUBO,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT);

		vkCmdUpdateBuffer(aCmdBuff, aLightUBO, 0, sizeof(glsl::LightSource), &aLightUniform);

		lut::buffer_barrier(aCmdBuff,
			aLightUBO,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

		// Upload vGaussian uniforms
		lut::buffer_barrier(aCmdBuff, vGaussianUBO,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT);

		vkCmdUpdateBuffer(aCmdBuff, vGaussianUBO, 0, sizeof(glsl::GaussianUniform), &vGaussianUniform);

		lut::buffer_barrier(aCmdBuff,
			vGaussianUBO,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

		// Upload hGaussian uniforms
		lut::buffer_barrier(aCmdBuff, hGaussianUBO,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT);

		vkCmdUpdateBuffer(aCmdBuff, hGaussianUBO, 0, sizeof(glsl::GaussianUniform), &hGaussianUniform);

		lut::buffer_barrier(aCmdBuff,
			hGaussianUBO,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

		// Begin render pass 
		VkClearValue clearValues[5]{};
		clearValues[0].color.float32[0] = 0.0f; // Clear to a dark gray background. 
		clearValues[0].color.float32[1] = 0.0f; // If we were debugging, this would potentially 
		clearValues[0].color.float32[2] = 0.0f; // help us see whether the render pass took 
		clearValues[0].color.float32[3] = 1.f; // place, even if nothing else was drawn. 

		clearValues[1].color.float32[0] = 0.0f; // Clear to a dark gray background. 
		clearValues[1].color.float32[1] = 0.0f; // If we were debugging, this would potentially 
		clearValues[1].color.float32[2] = 0.0f; // help us see whether the render pass took 
		clearValues[1].color.float32[3] = 1.f;

		clearValues[2].color.float32[0] = 0.0f; // Clear to a dark gray background. 
		clearValues[2].color.float32[1] = 0.0f; // If we were debugging, this would potentially 
		clearValues[2].color.float32[2] = 0.0f; // help us see whether the render pass took 
		clearValues[2].color.float32[3] = 1.f;


		clearValues[3].color.float32[0] = 0.0f; // Clear to a dark gray background. 
		clearValues[3].color.float32[1] = 0.0f; // If we were debugging, this would potentially 
		clearValues[3].color.float32[2] = 0.0f; // help us see whether the render pass took 
		clearValues[3].color.float32[3] = 1.f;

		clearValues[4].depthStencil.depth = 1.f;

		VkRenderPassBeginInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		passInfo.renderPass = filterPass;
		passInfo.framebuffer = firstFrameBuffer.handle;
		passInfo.renderArea.offset = VkOffset2D{ 0, 0 };
		passInfo.renderArea.extent = VkExtent2D{ aImageExtent.width, aImageExtent.height };
		passInfo.clearValueCount = 5;
		passInfo.pClearValues = clearValues;


		vkCmdBeginRenderPass(aCmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);

		//Finding the brightest part
		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, brightPipe);
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, bright_PBR_layout, 0, 1, &aSceneDescriptors, 0, nullptr);
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, bright_PBR_layout, 3, 1, &lightDescriptors, 0, nullptr);

		//Bind vertex input for indexed mesh
		for (int i = 0; i < indexedMesh->size(); i++)
		{

			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, bright_PBR_layout, 1, 1, (*materialDescriptor)[i], 0, nullptr);

			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, bright_PBR_layout, 2, 1, (*textureDescriptorsSet)[i], 0, nullptr);

			VkBuffer buffers[3] = { (*indexedMesh)[i].pos.buffer,(*indexedMesh)[i].texcoords.buffer,(*indexedMesh)[i].normals.buffer };
			VkDeviceSize offsets[3]{};
			vkCmdBindVertexBuffers(aCmdBuff, 0, 3, buffers, offsets);

			vkCmdBindIndexBuffer(aCmdBuff, (*indexedMesh)[i].indices.buffer, 0, VK_INDEX_TYPE_UINT32);

			int isAlpha = 0;
			int isNormalMap = 0;
			if ((*indexedMesh)[i].isNormalMap)
			{
				isNormalMap = 1;
			}
			vkCmdPushConstants(aCmdBuff, aGraphicsLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(int), &isAlpha);
			vkCmdPushConstants(aCmdBuff, aGraphicsLayout, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(int), sizeof(int), &isNormalMap);
			vkCmdDrawIndexed(aCmdBuff, (*indexedMesh)[i].indexSize, 1, 0, 0, 0);
		}


		//Vertical Gaussian
		vkCmdNextSubpass(aCmdBuff, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, verticalpipe);

		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, verticalPipeLayout, 0, 1, &brightDescriptors, 0, nullptr);
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, verticalPipeLayout, 1, 1, &vGaussianDescriptors, 0, nullptr);

		vkCmdDraw(aCmdBuff, 6, 1, 0, 0);


		//Horizontal
		vkCmdNextSubpass(aCmdBuff, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, horizontalPipe);

		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, horizontalPipeLayout, 0, 1, &verticalDescriptors, 0, nullptr);
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, horizontalPipeLayout, 1, 1, &hGaussianDescriptors, 0, nullptr);

		vkCmdDraw(aCmdBuff, 6, 1, 0, 0);


		////PBR actual scene
		//vkCmdNextSubpass(aCmdBuff, VK_SUBPASS_CONTENTS_INLINE);

		//vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, PBRPipe);

		//vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, bright_PBR_layout, 0, 1, &aSceneDescriptors, 0, nullptr);
		//vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, bright_PBR_layout, 3, 1, &lightDescriptors, 0, nullptr);

		////Bind vertex input for indexed mesh
		//for (int i = 0; i < indexedMesh->size(); i++)
		//{

		//	vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, bright_PBR_layout, 1, 1, (*materialDescriptor)[i], 0, nullptr);

		//	vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, bright_PBR_layout, 2, 1, (*textureDescriptorsSet)[i], 0, nullptr);

		//	VkBuffer buffers[3] = { (*indexedMesh)[i].pos.buffer,(*indexedMesh)[i].texcoords.buffer,(*indexedMesh)[i].normals.buffer };
		//	VkDeviceSize offsets[3]{};
		//	vkCmdBindVertexBuffers(aCmdBuff, 0, 3, buffers, offsets);

		//	vkCmdBindIndexBuffer(aCmdBuff, (*indexedMesh)[i].indices.buffer, 0, VK_INDEX_TYPE_UINT32);

		//	int isAlpha = 0;
		//	int isNormalMap = 0;
		//	if ((*indexedMesh)[i].isNormalMap)
		//	{
		//		isNormalMap = 1;
		//	}
		//	vkCmdPushConstants(aCmdBuff, aGraphicsLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(int), &isAlpha);
		//	vkCmdPushConstants(aCmdBuff, aGraphicsLayout, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(int), sizeof(int), &isNormalMap);
		//	vkCmdDrawIndexed(aCmdBuff, (*indexedMesh)[i].indexSize, 1, 0, 0, 0);
		//}

		// End the render pass 
		vkCmdEndRenderPass(aCmdBuff);

		passInfo.framebuffer = aFramebuffer;
		passInfo.renderPass = postProcessPass;

		VkClearValue clearValues1[1]{};
		clearValues1[0].color.float32[0] = 0.2f; // Clear to a dark gray background. 
		clearValues1[0].color.float32[1] = 0.2f; // If we were debugging, this would potentially 
		clearValues1[0].color.float32[2] = 0.2f; // help us see whether the render pass took 
		clearValues1[0].color.float32[3] = 1.f; // place, even if nothing else was drawn. 

		VkRenderPassBeginInfo passInfo1{};
		passInfo1.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		passInfo1.renderPass = postProcessPass;
		passInfo1.framebuffer = aFramebuffer;
		passInfo1.renderArea.offset = VkOffset2D{ 0, 0 };
		passInfo1.renderArea.extent = VkExtent2D{ aImageExtent.width, aImageExtent.height };
		passInfo1.clearValueCount = 1;
		passInfo1.pClearValues = clearValues1;


		vkCmdBeginRenderPass(aCmdBuff, &passInfo1, VK_SUBPASS_CONTENTS_INLINE);

		//Finding the brightest part
		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, postprocessPipe);

		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, postProcessPipeLayout, 0, 1, &horizontalDescriptors, 0, nullptr);
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, postProcessPipeLayout, 1, 1, &PBRDescriptors, 0, nullptr);

		vkCmdDraw(aCmdBuff, 6, 1, 0, 0);

		vkCmdEndRenderPass(aCmdBuff);
		// End command recording 
		if (auto const res = vkEndCommandBuffer(aCmdBuff); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to end recording command buffer\n" "vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

	}

	void submit_commands(lut::VulkanContext const& aContext, VkCommandBuffer aCmdBuff, VkFence aFence, VkSemaphore aWaitSemaphore, VkSemaphore aSignalSemaphore)
	{

		//We must wait for the imageAvailable semaphore to become signalled, indicating that the swapchain image is ready,
		//before we draw to the image
		VkPipelineStageFlags waitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;

		submitInfo.pCommandBuffers = &aCmdBuff;
		submitInfo.waitSemaphoreCount = 1;

		//Wait for imageAvailable semaphore to become signalled
		submitInfo.pWaitSemaphores = &aWaitSemaphore;
		//Indicates which stage of the pipeline should wait
		submitInfo.pWaitDstStageMask = &waitPipelineStages;

		//we want to signal the renderFinished semaphore when commands have finished, to indicate that 
		//the rendered image is ready for presentation
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &aSignalSemaphore;
		if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, aFence); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to submit command buffer to queue\n"
				"vkQueueSubmit() returned %s", lut::to_string(res).c_str()
			);
		}
	}

	lut::DescriptorSetLayout create_scene_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0; // number must match the index of the corresponding 
		// binding = N declaration in the shader(s)! 
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create descriptor set layout\n" "vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());

		}

		return lut::DescriptorSetLayout(aWindow.device, layout);


	}


	lut::DescriptorSetLayout create_lightSource_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0; // this must match the shaders 
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create descriptor set layout\n""vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());

		}

		return lut::DescriptorSetLayout(aWindow.device, layout);

	}

	lut::DescriptorSetLayout create_material_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0; // this must match the shaders 
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create descriptor set layout\n""vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());

		}

		return lut::DescriptorSetLayout(aWindow.device, layout);

	}

	lut::DescriptorSetLayout create_mipmap_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0; // number must match the index of the corresponding 
		// binding = N declaration in the shader(s)! 
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n" "vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());

		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	lut::DescriptorSetLayout create_object_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[3]{};
		bindings[0].binding = 0; // this must match the shaders 
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		bindings[1].binding = 1; // this must match the shaders 
		bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[1].descriptorCount = 1;
		bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		bindings[2].binding = 2; // this must match the shaders 
		bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[2].descriptorCount = 1;
		bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;



		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create descriptor set layout\n""vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());

		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	lut::DescriptorSetLayout create_intermediateImage_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		// Descriptor set layout binding for the intermediate texture
		VkDescriptorSetLayoutBinding intermediateTextureBinding{};
		intermediateTextureBinding.binding = 0; 
		intermediateTextureBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		intermediateTextureBinding.descriptorCount = 1; 
		intermediateTextureBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT; 

		// Descriptor set layout for the intermediate texture
		VkDescriptorSetLayoutCreateInfo intermediateTextureLayoutInfo{};
		intermediateTextureLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		intermediateTextureLayoutInfo.bindingCount = 1; // Only one binding in this layout
		intermediateTextureLayoutInfo.pBindings = &intermediateTextureBinding;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &intermediateTextureLayoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create descriptor set layout\n""vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());

		}
		return lut::DescriptorSetLayout(aWindow.device, layout);
	}
	
	lut::DescriptorSetLayout create_vGaussian_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0; // this must match the shaders 
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create descriptor set layout\n""vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());

		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}



	void create_framebuffer_R1(lut::VulkanWindow const& aWindow,
		VkRenderPass aRenderPass, lut::Framebuffer& aFramebuffers, VkImageView aBrightView, VkImageView aVerticalView,
		VkImageView aHorizontalView, VkImageView aPBRView, VkImageView aDepthView)
	{
		VkImageView attachments[5] = {
			aBrightView,
			aVerticalView,
			aHorizontalView,
			aPBRView,
			aDepthView
		};

		VkFramebufferCreateInfo fbInfo{};
		fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		fbInfo.flags = 0;
		fbInfo.renderPass = aRenderPass;
		fbInfo.attachmentCount = 5;
		fbInfo.pAttachments = attachments;
		fbInfo.width = aWindow.swapchainExtent.width;
		fbInfo.height = aWindow.swapchainExtent.height;
		fbInfo.layers = 1;

		VkFramebuffer fb = VK_NULL_HANDLE;
		if (auto const res = vkCreateFramebuffer(aWindow.device, &fbInfo,
			nullptr, &fb); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create framebuffer for swap chain\n"
				"vkCreateFramebuffer() returned %s",
				lut::to_string(res).c_str());
		}

		aFramebuffers = lut::Framebuffer(aWindow.device, fb);
	}

	void create_framebuffer_R2(lut::VulkanWindow const& aWindow,
		VkRenderPass aRenderPass, lut::Framebuffer& aFramebuffers, VkImageView aBrightView, VkImageView aDepthView)
	{
		VkImageView attachments[2] = {
			aBrightView,
			aDepthView
		};

		VkFramebufferCreateInfo fbInfo{};
		fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		fbInfo.flags = 0;
		fbInfo.renderPass = aRenderPass;
		fbInfo.attachmentCount = 5;
		fbInfo.pAttachments = attachments;
		fbInfo.width = aWindow.swapchainExtent.width;
		fbInfo.height = aWindow.swapchainExtent.height;
		fbInfo.layers = 1;

		VkFramebuffer fb = VK_NULL_HANDLE;
		if (auto const res = vkCreateFramebuffer(aWindow.device, &fbInfo,
			nullptr, &fb); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create framebuffer for swap chain\n"
				"vkCreateFramebuffer() returned %s",
				lut::to_string(res).c_str());
		}

		aFramebuffers = lut::Framebuffer(aWindow.device, fb);
	}


	//Task 3
	lut::DescriptorSetLayout create_bright_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		// Descriptor set layout binding for the intermediate texture
		VkDescriptorSetLayoutBinding intermediateTextureBinding{};
		intermediateTextureBinding.binding = 0;
		intermediateTextureBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		intermediateTextureBinding.descriptorCount = 1;
		intermediateTextureBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		// Descriptor set layout for the intermediate texture
		VkDescriptorSetLayoutCreateInfo intermediateTextureLayoutInfo{};
		intermediateTextureLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		intermediateTextureLayoutInfo.bindingCount = 1; // Only one binding in this layout
		intermediateTextureLayoutInfo.pBindings = &intermediateTextureBinding;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &intermediateTextureLayoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create descriptor set layout\n""vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());

		}
		return lut::DescriptorSetLayout(aWindow.device, layout);
	}
	lut::DescriptorSetLayout create_vertical_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		// Descriptor set layout binding for the intermediate texture
		VkDescriptorSetLayoutBinding intermediateTextureBinding{};
		intermediateTextureBinding.binding = 0;
		intermediateTextureBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		intermediateTextureBinding.descriptorCount = 1;
		intermediateTextureBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		// Descriptor set layout for the intermediate texture
		VkDescriptorSetLayoutCreateInfo intermediateTextureLayoutInfo{};
		intermediateTextureLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		intermediateTextureLayoutInfo.bindingCount = 1; // Only one binding in this layout
		intermediateTextureLayoutInfo.pBindings = &intermediateTextureBinding;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &intermediateTextureLayoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create descriptor set layout\n""vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());

		}
		return lut::DescriptorSetLayout(aWindow.device, layout);
	}
	lut::DescriptorSetLayout create_horizontal_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		// Descriptor set layout binding for the intermediate texture
		VkDescriptorSetLayoutBinding intermediateTextureBinding{};
		intermediateTextureBinding.binding = 0;
		intermediateTextureBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		intermediateTextureBinding.descriptorCount = 1;
		intermediateTextureBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		// Descriptor set layout for the intermediate texture
		VkDescriptorSetLayoutCreateInfo intermediateTextureLayoutInfo{};
		intermediateTextureLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		intermediateTextureLayoutInfo.bindingCount = 1; // Only one binding in this layout
		intermediateTextureLayoutInfo.pBindings = &intermediateTextureBinding;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &intermediateTextureLayoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create descriptor set layout\n""vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());

		}
		return lut::DescriptorSetLayout(aWindow.device, layout);
	}
	lut::DescriptorSetLayout create_PBR_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		// Descriptor set layout binding for the intermediate texture
		VkDescriptorSetLayoutBinding intermediateTextureBinding{};
		intermediateTextureBinding.binding = 0;
		intermediateTextureBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		intermediateTextureBinding.descriptorCount = 1;
		intermediateTextureBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		// Descriptor set layout for the intermediate texture
		VkDescriptorSetLayoutCreateInfo intermediateTextureLayoutInfo{};
		intermediateTextureLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		intermediateTextureLayoutInfo.bindingCount = 1; // Only one binding in this layout
		intermediateTextureLayoutInfo.pBindings = &intermediateTextureBinding;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &intermediateTextureLayoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create descriptor set layout\n""vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());

		}
		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	void present_results(VkQueue aPresentQueue, VkSwapchainKHR aSwapchain, std::uint32_t aImageIndex, VkSemaphore aRenderFinished, bool& aNeedToRecreateSwapchain)
	{

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;

		//we pass the renderFinished semaphore to pWaitSemaphores, to indicate that presentation should only occur once the semaphore is signalled
		presentInfo.pWaitSemaphores = &aRenderFinished;
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &aSwapchain;
		presentInfo.pImageIndices = &aImageIndex;
		presentInfo.pResults = nullptr;

		// Wait for the rendering to be finished before presenting the results
		vkQueueWaitIdle(aPresentQueue);
		auto const presentRes = vkQueuePresentKHR(aPresentQueue, &presentInfo);

		if (VK_SUBOPTIMAL_KHR == presentRes || VK_ERROR_OUT_OF_DATE_KHR == presentRes)
		{
			aNeedToRecreateSwapchain = true;
		}
		else if (VK_SUCCESS != presentRes)
		{
			throw lut::Error("Unable present swapchain image %u\n" "vkQueuePresentKHR() returned %s", aImageIndex, lut::to_string(presentRes).c_str());
		}
	}

	void calculateGaussianUniform(lut::VulkanWindow const& aWindow, glsl::GaussianUniform& gUniformVertical, glsl::GaussianUniform& gUniformHorizontal)
	{
		std::uint32_t height = aWindow.swapchainExtent.height;
		std::uint32_t width = aWindow.swapchainExtent.width;

		float xOffset = 1.0f / width;
		float yOffset = 1.0f / height;

		float verticalAmount = 0.0f;
		float horizontalAmout = 0.0f;

		for (int i = 0; i < 22; i++)
		{
			float verticalDistance = i;
			float horizontalDistance = i;

			gUniformVertical.data[i] = Gaussian(verticalDistance, 9);
			gUniformHorizontal.data[i] = Gaussian(horizontalDistance, 9);

			verticalAmount += gUniformVertical.data[i];
			horizontalAmout += gUniformHorizontal.data[i];
		}

		//Normalization
		for (int i = 0; i < 22; i++)
		{
			gUniformVertical.data[i] /= verticalAmount;
			gUniformHorizontal.data[i] /= horizontalAmout;
		}

	}

	float Gaussian(float distance, float factor)
	{
		float result = (1.0f / (std::sqrt(2 * 3.1415926 * std::pow(factor, 2)))) * std::exp(-1.0f * (std::pow(distance, 2)) / (std::pow(factor, 2) * 2));
		return result;
	}
}







//EOF vim:syntax=cpp:foldmethod=marker:ts=4:noexpandtab: 
