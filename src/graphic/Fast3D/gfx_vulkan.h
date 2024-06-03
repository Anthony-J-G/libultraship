#ifndef GFX_VULKAN_H
#define GFX_VULKAN_H

#include "gfx_rendering_api.h"

struct SDL_Window;

extern struct GfxRenderingAPI gfx_vulkan_api;

/**
 * @brief Checks if the Vulkan API is properly initialized and functional.
 *
 * This function performs a series of checks to ensure that the Vulkan
 * API is available and can be used for rendering. It includes checking
 * the presence of Vulkan-capable GPU, Vulkan instance creation, and
 * extension support.
 *
 * @return true if Vulkan is properly initialized and functional.
 * @return false if there are issues with Vulkan initialization.
 */
bool ValidateVulkan(); // TODO: Implement me pls :3

/**
 *  NOTE : This is SUPER hacky and bad, but I'm doing this for the singular purpose of making
 *	sure that while the Vulkan implementation is still being put together it remains as modular
 *	as possible. This means instead of doing things the "proper" way and modifying the SDL 
 *	implementation too much, this hacky bit is necessary.
**/ 
void InitializeWindow(SDL_Window* wndref);

#endif