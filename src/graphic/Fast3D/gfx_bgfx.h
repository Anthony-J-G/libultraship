#pragma once

#include "gfx_rendering_api.h"

extern struct GfxRenderingAPI gfx_bgfx_api;

struct SDL_SysWMinfo;
struct SDL_Window;

void InitializeBgfxWindowInfo(SDL_SysWMinfo windowinfo, SDL_Window* windowref);
