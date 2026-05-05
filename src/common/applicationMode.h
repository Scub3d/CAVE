#pragma once

namespace Cave
{
	enum class ApplicationMode
	{
		None,           // No mode active — mode selector screen
		Rendering,      // Real-time cellular automata visualization
		Search,         // Headless parallel simulation search
		VideoEncoding,  // Record simulation to video file
		LookingGlass    // Hologram output to Looking Glass Portrait via Vulkan-OpenGL interop
	};
}
