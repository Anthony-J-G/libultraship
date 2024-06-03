#include "gfx_vulkan.h"

#ifdef ENABLE_VULKAN

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#include <map>
#include <unordered_map>

#ifndef _LANGUAGE_C
#define _LANGUAGE_C
#endif

#ifdef __MINGW32__
#define FOR_WINDOWS 1
#else
#define FOR_WINDOWS 0
#endif

#ifdef _MSC_VER
#include <SDL2/SDL.h>
// #define GL_GLEXT_PROTOTYPES 1
#include <GL/glew.h>
#elif FOR_WINDOWS
#include <GL/glew.h>
#include "SDL.h"
#define GL_GLEXT_PROTOTYPES 1
#include "SDL_vulkan.h"
#elif __APPLE__
#include <SDL2/SDL.h>
#include <GL/glew.h>
#elif USE_OPENGLES
#include <SDL2/SDL.h>
#include <GLES3/gl3.h>
#else
#include <SDL2/SDL.h>
#define GL_GLEXT_PROTOTYPES 1
#include <SDL2/SDL_vulkan.h>
#endif

#include "gfx_cc.h"
#include "gfx_rendering_api.h"
#include "window/gui/Gui.h"
#include "window/Window.h"
#include "gfx_pc.h"
#include <public/bridge/consolevariablebridge.h>

#include <array>
#include <vector>
#include <set>
#include <iostream>
#include <cstdint>   // Necessary for uint32_t
#include <limits>    // Necessary for std::numeric_limits
#include <algorithm> // Necessary for std::clamp
#include <fstream>
#include <filesystem>
#include "Context.h"

#include <SDL_vulkan.h>
#include <vulkan/vulkan.h>
#include <assert.h>

#include "spdlog/spdlog.h"

using namespace std;
#define MAX_FRAMES_IN_FLIGHT 2


struct ShaderProgram {
};

std::map<std::pair<uint64_t, uint32_t>, struct ShaderProgram> shader_program_pool;
FilteringMode current_filter_mode = FILTER_NONE;

static VkInstance g_Instance = VK_NULL_HANDLE;
static std::vector<VkPhysicalDevice> g_PhysDevices;
static VkPhysicalDevice g_PrimaryPhysDevice = VK_NULL_HANDLE;
static VkDevice g_LogicDevice = VK_NULL_HANDLE;
static VkSurfaceKHR g_Surface = VK_NULL_HANDLE;
static VkQueue g_GraphicsQueue = VK_NULL_HANDLE;
static VkQueue g_PresentQueue = VK_NULL_HANDLE;
static VkSwapchainKHR g_SwapChain = VK_NULL_HANDLE;
static std::vector<VkImage> g_SwapChainImages;
static VkFormat g_SwapChainImgFmt;
static VkExtent2D g_SwapChainExtent;
static std::vector<VkImageView> g_SwapChainImgViews;
static VkPipelineLayout g_PipelineLayout = VK_NULL_HANDLE;
static VkRenderPass g_RenderPass = VK_NULL_HANDLE;
static VkPipeline g_GraphicsPipeline = VK_NULL_HANDLE;
static std::vector<VkFramebuffer> g_SwapChainFramebuffers;
static VkCommandPool g_CommandPool;
static std::vector<VkCommandBuffer> g_CommandBuffers;
static std::vector<VkSemaphore> g_ImageAvailableSemaphores;
static std::vector<VkSemaphore> g_RenderFinishedSemaphores;
static std::vector<VkFence> g_InFlightFences;
static uint32_t g_CurrentFrame = 0;
static bool g_FramebufferResized = false;
static bool g_WindowMinimizedResized = false;
static SDL_Window* g_WindowReference;

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};


void InitializeWindow(SDL_Window* wndref) {
    g_WindowReference = wndref;
}


bool ValidateVulkan() {
    // TODO
    return true;
}


QueueFamilyIndices findQueueFamilies(VkPhysicalDevice gpu) {
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &queueFamilyCount, queueFamilies.data());

    int i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphicsFamily = i;
        }

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(gpu, i, g_Surface, &presentSupport);

        if (presentSupport) {
            indices.presentFamily = i;
        }

        if (indices.isComplete()) {
            break;
        }

        i++;
    }

    return indices;
}

SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
    SwapChainSupportDetails details = {};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_PrimaryPhysDevice, g_Surface, &details.capabilities);

    uint32_t formatCount;
    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(g_PrimaryPhysDevice, g_Surface, &formatCount, nullptr);

    vkGetPhysicalDeviceSurfacePresentModesKHR(g_PrimaryPhysDevice, g_Surface, &presentModeCount, nullptr);

    if (formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(g_PrimaryPhysDevice, g_Surface, &formatCount, details.formats.data());
    }
    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(g_PrimaryPhysDevice, g_Surface, &presentModeCount,
                                                  details.presentModes.data());
    }

    return details;
}

VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
    for (const auto& availableFormat : availableFormats) {
        // Try to return SRGB format if available
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
            availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }
    }

    // Otherwise return the default format
    return availableFormats[0];
}

VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
    for (const auto& availablePresentMode : availablePresentModes) {
        // If Triple Buffering is availble, return it
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return VK_PRESENT_MODE_MAILBOX_KHR;
        }
    }

    // Otherwise return VSync
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    } else {
        int width, height;
        SDL_Vulkan_GetDrawableSize(
            g_WindowReference, &width, &height
        );

        VkExtent2D actualExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };

        actualExtent.width =
            std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height =
            std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        return actualExtent;
    }
}

static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        std::cout << "failed to open file " << filename << " @ " << std::filesystem::current_path() << "!\n";
        exit(1);
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}

VkShaderModule createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo shader_module_create_info = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    shader_module_create_info.codeSize = code.size();
    shader_module_create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    VkResult res = vkCreateShaderModule(g_LogicDevice, &shader_module_create_info, nullptr, &shaderModule);
    if (res != VK_SUCCESS) {
        std::cout << "Instance Failure\n";
        // spdlog::error("Vulkan Instance Creation Error: Vulkan returned code .");
        exit(1);
    }

    return shaderModule;
}

void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
    }

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = g_RenderPass;
    renderPassInfo.framebuffer = g_SwapChainFramebuffers[imageIndex];
    renderPassInfo.renderArea.offset = { 0, 0 };
    renderPassInfo.renderArea.extent = g_SwapChainExtent;

    VkClearValue clearColor = { { { 0.0f, 0.0f, 0.0f, 1.0f } } };
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, g_GraphicsPipeline);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)g_SwapChainExtent.width;
    viewport.height = (float)g_SwapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = { 0, 0 };
    scissor.extent = g_SwapChainExtent;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    vkCmdDraw(commandBuffer, 3, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffer);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
    }
}

static const char* gfx_vulkan_get_name() {
    return "Vulkan";
}

static int gfx_vulkan_get_max_texture_size() {
    return 0;
}

static struct GfxClipParameters gfx_vulkan_get_clip_parameters(void) {
    return { false, false };
}

static void gfx_vulkan_unload_shader(struct ShaderProgram* old_prg) {
}

static void gfx_vulkan_load_shader(struct ShaderProgram* new_prg) {
}

static struct ShaderProgram* gfx_vulkan_create_and_load_new_shader(uint64_t shader_id0, uint32_t shader_id1) {
    return &shader_program_pool[make_pair(shader_id0, shader_id1)];
}

static struct ShaderProgram* gfx_vulkan_lookup_shader(uint64_t shader_id0, uint32_t shader_id1) {
    return &shader_program_pool[make_pair(shader_id0, shader_id1)];
}

static void gfx_vulkan_shader_get_info(struct ShaderProgram* prg, uint8_t* num_inputs, bool used_textures[2]) {
}

static GLuint gfx_vulkan_new_texture(void) {
    GLuint ret;
    glGenTextures(1, &ret);
    return ret;
}

static void gfx_vulkan_delete_texture(uint32_t texID) {
    glDeleteTextures(1, &texID);
}

static void gfx_vulkan_select_texture(int tile, GLuint texture_id) {
    glActiveTexture(GL_TEXTURE0 + tile);
    glBindTexture(GL_TEXTURE_2D, texture_id);
}

static void gfx_vulkan_upload_texture(const uint8_t* rgba32_buf, uint32_t width, uint32_t height) {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba32_buf);
}


static uint32_t gfx_cm_to_vulkan(uint32_t val) {  
    return 0;
}

static void gfx_vulkan_set_sampler_parameters(int tile, bool linear_filter, uint32_t cms, uint32_t cmt) {
}

static void gfx_vulkan_set_depth_test_and_mask(bool depth_test, bool z_upd) {
}

static void gfx_vulkan_set_zmode_decal(bool zmode_decal) {
}

static void gfx_vulkan_set_viewport(int x, int y, int width, int height) {
    glViewport(x, y, width, height);
}

static void gfx_vulkan_set_scissor(int x, int y, int width, int height) {
    glScissor(x, y, width, height);
}

static void gfx_vulkan_set_use_alpha(bool use_alpha) {
}

static void gfx_vulkan_draw_triangles(float buf_vbo[], size_t buf_vbo_len, size_t buf_vbo_num_tris) {
}

static void gfx_vulkan_on_resize(void) {
}

static void gfx_vulkan_start_frame(void) {
}

static void gfx_vulkan_end_frame(void) {
    glFlush();
}

static void gfx_vulkan_finish_render(void) {
}

static void gfx_vulkan_update_framebuffer_parameters(int fb_id, uint32_t width, uint32_t height, uint32_t msaa_level,
                                                     bool vulkan_invert_y, bool render_target, bool has_depth_buffer,
                                                     bool can_extract_depth) {
}


void gfx_vulkan_start_draw_to_framebuffer(int fb_id, float noise_scale) {
}

void gfx_vulkan_clear_framebuffer() {
}

void gfx_vulkan_resolve_msaa_color_buffer(int fb_id_target, int fb_id_source) {
}

void* gfx_vulkan_get_framebuffer_texture_id(int fb_id) {
    return (void*)(uintptr_t)0;
}

void gfx_vulkan_select_texture_fb(int fb_id) {
}

void gfx_vulkan_copy_framebuffer(int fb_dst_id, int fb_src_id, int srcX0, int srcY0, int srcX1, int srcY1, int dstX0,
                                 int dstY0, int dstX1, int dstY1) {
}

void gfx_vulkan_read_framebuffer_to_cpu(int fb_id, uint32_t width, uint32_t height, uint16_t* rgba16_buf) {
}

static std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff>
gfx_vulkan_get_pixel_depth(int fb_id, const std::set<std::pair<float, float>>& coordinates) {
    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff> res;
    return res;
}

void gfx_vulkan_set_texture_filter(FilteringMode mode) {
}

FilteringMode gfx_vulkan_get_texture_filter(void) {
    return current_filter_mode;
}

void gfx_vulkan_make_instance() {
    VkApplicationInfo app_info = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app_info.apiVersion = VK_API_VERSION_1_3;

    const std::array<const char*, 2> instance_extensions = { "VK_KHR_surface", "VK_KHR_win32_surface" };
    const std::array<const char*, 1> enabled_layers = { "VK_LAYER_KHRONOS_validation" };

    VkInstanceCreateInfo create_info = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    create_info.pApplicationInfo = &app_info;
    create_info.enabledExtensionCount = instance_extensions.size();
    create_info.ppEnabledExtensionNames = instance_extensions.data();
    create_info.enabledLayerCount = enabled_layers.size();
    create_info.ppEnabledLayerNames = enabled_layers.data();

    VkResult res = vkCreateInstance(&create_info, nullptr, &g_Instance);
    if (res != VK_SUCCESS) {
        std::cout << "Instance Failure\n";
        // spdlog::error("Vulkan Instance Creation Error: Vulkan returned code .");
        exit(1);
    }
}

void gfx_vulkan_define_phys_device() {
    uint32_t count;
    vkEnumeratePhysicalDevices(g_Instance, &count, nullptr);
    if (count <= 0) {
        std::cout << "No Available Devices Found Failure\n";
        exit(1);
    }

    g_PhysDevices = std::vector<VkPhysicalDevice>(count);
    vkEnumeratePhysicalDevices(g_Instance, &count, g_PhysDevices.data());

    g_PrimaryPhysDevice = g_PhysDevices[0];
}

void gfx_vulkan_make_surface() {
    if (!SDL_Vulkan_CreateSurface(g_WindowReference, g_Instance, &g_Surface)) {
        std::cout << "SDL could not create Vulkan surface: %s\n", SDL_GetError();
        exit(1);
    }
}

void gfx_vulkan_make_logic_device() {
    QueueFamilyIndices indices = findQueueFamilies(g_PrimaryPhysDevice);
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

    float priority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &priority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkDeviceQueueCreateInfo queue_create_info = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    queue_create_info.queueFamilyIndex = 0;
    queue_create_info.pQueuePriorities = &priority;
    queue_create_info.queueCount = static_cast<uint32_t>(queueCreateInfos.size());

    const std::array<const char*, 1> enabled_extensions = { "VK_KHR_swapchain" };

    VkDeviceCreateInfo device_create_info = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    device_create_info.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    device_create_info.pQueueCreateInfos = queueCreateInfos.data();
    device_create_info.enabledExtensionCount = 1;
    device_create_info.ppEnabledExtensionNames = enabled_extensions.data();

    VkResult res = vkCreateDevice(g_PrimaryPhysDevice, &device_create_info, nullptr, &g_LogicDevice);
    if (res != VK_SUCCESS) {
        std::cout << "Logic Device Failure\n";
        exit(1);
    }
    vkGetDeviceQueue(g_LogicDevice, indices.graphicsFamily.value(), 0, &g_GraphicsQueue);
    vkGetDeviceQueue(g_LogicDevice, indices.presentFamily.value(), 0, &g_PresentQueue);
}

void gfx_vulkan_make_swapchain() {
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(g_PrimaryPhysDevice);

    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1; // Avoid waiting on the driver's internal ops
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR swap_chain_create_info = { .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    swap_chain_create_info.surface = g_Surface;
    swap_chain_create_info.minImageCount = imageCount;
    swap_chain_create_info.imageFormat = surfaceFormat.format;
    swap_chain_create_info.imageColorSpace = surfaceFormat.colorSpace;
    swap_chain_create_info.imageExtent = extent;
    swap_chain_create_info.imageArrayLayers = 1;
    swap_chain_create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    QueueFamilyIndices indices = findQueueFamilies(g_PrimaryPhysDevice);
    uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

    if (indices.graphicsFamily != indices.presentFamily) {
        swap_chain_create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        swap_chain_create_info.queueFamilyIndexCount = 2;
        swap_chain_create_info.pQueueFamilyIndices = queueFamilyIndices;
    } else {
        swap_chain_create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        swap_chain_create_info.queueFamilyIndexCount = 0;     // Optional
        swap_chain_create_info.pQueueFamilyIndices = nullptr; // Optional
    }

    swap_chain_create_info.preTransform = swapChainSupport.capabilities.currentTransform;
    swap_chain_create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swap_chain_create_info.presentMode = presentMode;
    swap_chain_create_info.clipped = VK_TRUE;
    swap_chain_create_info.oldSwapchain = VK_NULL_HANDLE;

    VkResult res = vkCreateSwapchainKHR(g_LogicDevice, &swap_chain_create_info, nullptr, &g_SwapChain);
    if (res != VK_SUCCESS) {
        std::cout << "Instance Failure\n";
        // spdlog::error("Vulkan Instance Creation Error: Vulkan returned code .");
        exit(1);
    }

    vkGetSwapchainImagesKHR(g_LogicDevice, g_SwapChain, &imageCount, nullptr);
    g_SwapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(g_LogicDevice, g_SwapChain, &imageCount, g_SwapChainImages.data());

    g_SwapChainImgFmt = surfaceFormat.format;
    g_SwapChainExtent = extent;
}

void gfx_vulkan_make_img_views() {
    g_SwapChainImgViews.resize(g_SwapChainImages.size());

    for (size_t i = 0; i < g_SwapChainImages.size(); i++) {
        VkImageViewCreateInfo image_view_create_info = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        image_view_create_info.image = g_SwapChainImages[i];
        image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        image_view_create_info.format = g_SwapChainImgFmt;
        image_view_create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        image_view_create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        image_view_create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        image_view_create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        image_view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        image_view_create_info.subresourceRange.baseMipLevel = 0;
        image_view_create_info.subresourceRange.levelCount = 1;
        image_view_create_info.subresourceRange.baseArrayLayer = 0;
        image_view_create_info.subresourceRange.layerCount = 1;

        VkResult res = vkCreateImageView(g_LogicDevice, &image_view_create_info, nullptr, &g_SwapChainImgViews[i]);
        if (res != VK_SUCCESS) {
            std::cout << "Image View Creation Failure\n";
            // spdlog::error("Vulkan Instance Creation Error: Vulkan returned code .");
            exit(1);
        }
    }
}

void gfx_vulkan_make_render_pass() {
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = g_SwapChainImgFmt;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;

    if (vkCreateRenderPass(g_LogicDevice, &renderPassInfo, nullptr, &g_RenderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }
}

void gfx_vulkan_make_gfxpipe() {
    auto vertShaderCode = readFile("out/vert.spv");
    auto fragShaderCode = readFile("out/frag.spv");

    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

    VkPipelineVertexInputStateCreateInfo vertex_input_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO
    };
    vertex_input_create_info.vertexBindingDescriptionCount = 0;
    vertex_input_create_info.pVertexBindingDescriptions = nullptr; // Optional
    vertex_input_create_info.vertexAttributeDescriptionCount = 0;
    vertex_input_create_info.pVertexAttributeDescriptions = nullptr; // Optional

    VkPipelineInputAssemblyStateCreateInfo input_assembly_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO
    };
    input_assembly_create_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_assembly_create_info.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewport_state_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO
    };
    viewport_state_create_info.viewportCount = 1;
    viewport_state_create_info.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer_state_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO
    };
    rasterizer_state_create_info.depthClampEnable = VK_FALSE;
    rasterizer_state_create_info.rasterizerDiscardEnable = VK_FALSE;
    rasterizer_state_create_info.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer_state_create_info.lineWidth = 1.0f;
    rasterizer_state_create_info.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer_state_create_info.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer_state_create_info.depthBiasEnable = VK_FALSE;
    rasterizer_state_create_info.depthBiasConstantFactor = 0.0f; // Optional
    rasterizer_state_create_info.depthBiasClamp = 0.0f;          // Optional
    rasterizer_state_create_info.depthBiasSlopeFactor = 0.0f;    // Optional

    VkPipelineMultisampleStateCreateInfo multisampling_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO
    };
    multisampling_create_info.sampleShadingEnable = VK_FALSE;
    multisampling_create_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling_create_info.minSampleShading = 1.0f;          // Optional
    multisampling_create_info.pSampleMask = nullptr;            // Optional
    multisampling_create_info.alphaToCoverageEnable = VK_FALSE; // Optional
    multisampling_create_info.alphaToOneEnable = VK_FALSE;      // Optional

    VkPipelineColorBlendAttachmentState color_blend_attach_state = {};
    color_blend_attach_state.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    color_blend_attach_state.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo color_blend_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO
    };
    color_blend_create_info.logicOpEnable = VK_FALSE;
    color_blend_create_info.logicOp = VK_LOGIC_OP_COPY; // Optional
    color_blend_create_info.attachmentCount = 1;
    color_blend_create_info.pAttachments = &color_blend_attach_state;
    color_blend_create_info.blendConstants[0] = 0.0f; // Optional
    color_blend_create_info.blendConstants[1] = 0.0f; // Optional
    color_blend_create_info.blendConstants[2] = 0.0f; // Optional
    color_blend_create_info.blendConstants[3] = 0.0f; // Optional

    std::vector<VkDynamicState> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamic_state_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO
    };
    dynamic_state_create_info.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamic_state_create_info.pDynamicStates = dynamicStates.data();

    VkPipelineLayoutCreateInfo pipeline_layout_create_info = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipeline_layout_create_info.setLayoutCount = 0;            // Optional
    pipeline_layout_create_info.pSetLayouts = nullptr;         // Optional
    pipeline_layout_create_info.pushConstantRangeCount = 0;    // Optional
    pipeline_layout_create_info.pPushConstantRanges = nullptr; // Optional

    VkResult res = vkCreatePipelineLayout(g_LogicDevice, &pipeline_layout_create_info, nullptr, &g_PipelineLayout);
    if (res != VK_SUCCESS) {
        std::cout << "Image View Creation Failure\n";
        // spdlog::error("Vulkan Instance Creation Error: Vulkan returned code .");
        exit(1);
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertex_input_create_info;
    pipelineInfo.pInputAssemblyState = &input_assembly_create_info;
    pipelineInfo.pViewportState = &viewport_state_create_info;
    pipelineInfo.pRasterizationState = &rasterizer_state_create_info;
    pipelineInfo.pMultisampleState = &multisampling_create_info;
    pipelineInfo.pColorBlendState = &color_blend_create_info;
    pipelineInfo.pDynamicState = &dynamic_state_create_info;
    pipelineInfo.layout = g_PipelineLayout;
    pipelineInfo.renderPass = g_RenderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

    if (vkCreateGraphicsPipelines(g_LogicDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &g_GraphicsPipeline) !=
        VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    vkDestroyShaderModule(g_LogicDevice, fragShaderModule, nullptr);
    vkDestroyShaderModule(g_LogicDevice, vertShaderModule, nullptr);
}

int gfx_vulkan_create_framebuffer() {
    g_SwapChainFramebuffers.resize(g_SwapChainImgViews.size());

    for (size_t i = 0; i < g_SwapChainImgViews.size(); i++) {
        VkImageView attachments[] = { g_SwapChainImgViews[i] };

        VkFramebufferCreateInfo framebufferInfo = { .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
        framebufferInfo.renderPass = g_RenderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = g_SwapChainExtent.width;
        framebufferInfo.height = g_SwapChainExtent.height;
        framebufferInfo.layers = 1;

        VkResult res = vkCreateFramebuffer(g_LogicDevice, &framebufferInfo, nullptr, &g_SwapChainFramebuffers[i]);
        if (res != VK_SUCCESS) {
            std::cout << "Image View Creation Failure\n";
            // spdlog::error("Vulkan Instance Creation Error: Vulkan returned code .");
            exit(1);
        }
    }
}

void gfx_vulkan_make_cmd_pool() {
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(g_PrimaryPhysDevice);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

    VkResult res = vkCreateCommandPool(g_LogicDevice, &poolInfo, nullptr, &g_CommandPool);
    if (res != VK_SUCCESS) {
        std::cout << "Image View Creation Failure\n";
        // spdlog::error("Vulkan Instance Creation Error: Vulkan returned code .");
        exit(1);
    }
}

void gfx_vulkan_make_cmd_buffers() {
    g_CommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocInfo = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocInfo.commandPool = g_CommandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t)g_CommandBuffers.size();

    VkResult res = vkAllocateCommandBuffers(g_LogicDevice, &allocInfo, g_CommandBuffers.data());
    if (res != VK_SUCCESS) {
        std::cout << "Command Buffer Failure\n";
        // spdlog::error("Vulkan Instance Creation Error: Vulkan returned code .");
        exit(1);
    }
}

void gfx_vulkan_make_sync_objs() {
    // Create objects to help the CPU and GPU stay in sync (Semaphores stall GPU; Fences stall CPU)
    g_ImageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    g_RenderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    g_InFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };

    VkFenceCreateInfo fenceInfo = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // Initialize the fence as Signalled to avoid First Frame Hanging

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateSemaphore(g_LogicDevice, &semaphoreInfo, nullptr, &g_ImageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(g_LogicDevice, &semaphoreInfo, nullptr, &g_RenderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(g_LogicDevice, &fenceInfo, nullptr, &g_InFlightFences[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create synchronization objects for a frame!");
        }
    }
}

void InvalidateSwapChain() {
    for (size_t i = 0; i < g_SwapChainFramebuffers.size(); i++) {
        vkDestroyFramebuffer(g_LogicDevice, g_SwapChainFramebuffers[i], nullptr);
    }

    for (size_t i = 0; i < g_SwapChainImgViews.size(); i++) {
        vkDestroyImageView(g_LogicDevice, g_SwapChainImgViews[i], nullptr);
    }

    vkDestroySwapchainKHR(g_LogicDevice, g_SwapChain, nullptr);
}

void RebuildSwapChain() {
    int width = 0, height = 0;
    SDL_Vulkan_GetDrawableSize(g_WindowReference, &width, &height);
    while (width == 0 || height == 0) { // Handle Minimized Window
        SDL_Vulkan_GetDrawableSize(g_WindowReference, &width, &height);
    }
    vkDeviceWaitIdle(g_LogicDevice);

    InvalidateSwapChain();

    gfx_vulkan_make_swapchain();
    gfx_vulkan_make_img_views();
    gfx_vulkan_create_framebuffer();
}

void AwaitIdle() {
    vkDeviceWaitIdle(g_LogicDevice);
}

void DrawFrame() {

    vkWaitForFences(g_LogicDevice, 1, &g_InFlightFences[g_CurrentFrame], VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(g_LogicDevice, g_SwapChain, UINT64_MAX,
                                            g_ImageAvailableSemaphores[g_CurrentFrame], VK_NULL_HANDLE, &imageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        RebuildSwapChain();
        return;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    // Only reset the fence if we are submitting work (to avoid Deadlock)
    vkResetFences(g_LogicDevice, 1, &g_InFlightFences[g_CurrentFrame]);

    vkResetCommandBuffer(g_CommandBuffers[g_CurrentFrame], /*VkCommandBufferResetFlagBits*/ 0);
    recordCommandBuffer(g_CommandBuffers[g_CurrentFrame], imageIndex);

    VkSubmitInfo submitInfo = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO };

    VkSemaphore waitSemaphores[] = { g_ImageAvailableSemaphores[g_CurrentFrame] };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &g_CommandBuffers[g_CurrentFrame];

    VkSemaphore signalSemaphores[] = { g_RenderFinishedSemaphores[g_CurrentFrame] };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(g_GraphicsQueue, 1, &submitInfo, g_InFlightFences[g_CurrentFrame]) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
    }

    VkPresentInfoKHR presentInfo = { .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    VkSwapchainKHR swapChains[] = { g_SwapChain };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;

    presentInfo.pImageIndices = &imageIndex;

    result = vkQueuePresentKHR(g_PresentQueue, &presentInfo);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
        RebuildSwapChain();
    } else if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to present swap chain image!");
    }

    g_CurrentFrame = (g_CurrentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void CreateImage() {
    VkImageCreateInfo image_create_info = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    image_create_info.imageType = VK_IMAGE_TYPE_2D;
    image_create_info.format = VK_FORMAT_R8G8B8A8_UNORM;
    image_create_info.extent.width = 512;
    image_create_info.extent.height = 512;
    image_create_info.extent.depth = 1;
    image_create_info.arrayLayers = 1;
    image_create_info.mipLevels = 1;
    image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;

    VkImage image;
    VkResult res = vkCreateImage(g_LogicDevice, &image_create_info, nullptr, &image);
    if (res != VK_SUCCESS) {
        std::cout << "Create Image Failure\n";
        // Debug Failure Here
    }
}


static void gfx_vulkan_init(void) {
    gfx_vulkan_make_instance();
    gfx_vulkan_make_surface();
    gfx_vulkan_define_phys_device();
    gfx_vulkan_make_logic_device();
    gfx_vulkan_make_swapchain();
    gfx_vulkan_make_img_views();
    gfx_vulkan_make_render_pass();
    gfx_vulkan_make_gfxpipe();
    gfx_vulkan_create_framebuffer();
    gfx_vulkan_make_cmd_pool();
    gfx_vulkan_make_cmd_buffers();
    gfx_vulkan_make_sync_objs();
}


struct GfxRenderingAPI gfx_vulkan_api = {   gfx_vulkan_get_name,
                                            gfx_vulkan_get_max_texture_size,
                                            gfx_vulkan_get_clip_parameters,
                                            gfx_vulkan_unload_shader,
                                            gfx_vulkan_load_shader,
                                            gfx_vulkan_create_and_load_new_shader,
                                            gfx_vulkan_lookup_shader,
                                            gfx_vulkan_shader_get_info,
                                            gfx_vulkan_new_texture,
                                            gfx_vulkan_select_texture,
                                            gfx_vulkan_upload_texture,
                                            gfx_vulkan_set_sampler_parameters,
                                            gfx_vulkan_set_depth_test_and_mask,
                                            gfx_vulkan_set_zmode_decal,
                                            gfx_vulkan_set_viewport,
                                            gfx_vulkan_set_scissor,
                                            gfx_vulkan_set_use_alpha,
                                            gfx_vulkan_draw_triangles,
                                            gfx_vulkan_init,
                                            gfx_vulkan_on_resize,
                                            gfx_vulkan_start_frame,
                                            gfx_vulkan_end_frame,
                                            gfx_vulkan_finish_render,
                                            gfx_vulkan_create_framebuffer,
                                            gfx_vulkan_update_framebuffer_parameters,
                                            gfx_vulkan_start_draw_to_framebuffer,
                                            gfx_vulkan_copy_framebuffer,
                                            gfx_vulkan_clear_framebuffer,
                                            gfx_vulkan_read_framebuffer_to_cpu,
                                            gfx_vulkan_resolve_msaa_color_buffer,
                                            gfx_vulkan_get_pixel_depth,
                                            gfx_vulkan_get_framebuffer_texture_id,
                                            gfx_vulkan_select_texture_fb,
                                            gfx_vulkan_delete_texture,
                                            gfx_vulkan_set_texture_filter,
                                            gfx_vulkan_get_texture_filter };
#endif // ENABLE_VULKAN