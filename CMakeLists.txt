cmake_minimum_required(VERSION 3.24)
project(Cave)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Find Vulkan (SDK must be installed)
find_package(Vulkan REQUIRED)

# 2. FetchContent setup
include(FetchContent)

# --- Header-only / lightweight repos ----------------------------------------

FetchContent_Declare(
	glfw
	GIT_REPOSITORY "https://github.com/glfw/glfw"
	GIT_TAG 3.3.8
	GIT_SHALLOW ON
)

FetchContent_Declare(
	glm
	GIT_REPOSITORY "https://github.com/g-truc/glm"
	GIT_TAG 0af55ccecd98d4e5a8d1fad7de25ba429d60e863 # refs/tags/1.0.1
	GIT_SHALLOW ON
)

FetchContent_Declare(
	spdlog
	GIT_REPOSITORY "https://github.com/gabime/spdlog"
	GIT_TAG v1.17.0
	GIT_SHALLOW ON
)

FetchContent_Declare(
	json
	URL "https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz"
)

FetchContent_Declare(
	cxxopts
	GIT_REPOSITORY "https://github.com/jarro2783/cxxopts"
	GIT_TAG v3.2.1
	GIT_SHALLOW ON
)

FetchContent_Declare(
	imgui
	GIT_REPOSITORY "https://github.com/ocornut/imgui"
	GIT_TAG v1.91.8
	GIT_SHALLOW ON
)

# stb has no version tags; pin a commit hash for reproducibility.
FetchContent_Declare(
	stb
	GIT_REPOSITORY "https://github.com/nothings/stb"
	GIT_TAG 31c1ad37456438565541f4919958214b6e762fb4
)

FetchContent_Declare(
	tinyobjloader
	GIT_REPOSITORY "https://github.com/tinyobjloader/tinyobjloader"
	GIT_TAG v2.0.0rc13
	GIT_SHALLOW ON
)

# Set dep options BEFORE MakeAvailable
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
set(CXXOPTS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CXXOPTS_BUILD_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(glfw glm spdlog json cxxopts imgui stb tinyobjloader)

# --- ImGui (no upstream CMakeLists; build a STATIC lib manually) ------------

add_library(imgui_lib STATIC
	${imgui_SOURCE_DIR}/imgui.cpp
	${imgui_SOURCE_DIR}/imgui_demo.cpp
	${imgui_SOURCE_DIR}/imgui_draw.cpp
	${imgui_SOURCE_DIR}/imgui_tables.cpp
	${imgui_SOURCE_DIR}/imgui_widgets.cpp
	${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
	${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui_lib PUBLIC
	${imgui_SOURCE_DIR}
	${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui_lib PUBLIC Vulkan::Vulkan glfw)

# --- stb (interface target wrapping the include dir) ------------------------

add_library(stb INTERFACE)
target_include_directories(stb INTERFACE ${stb_SOURCE_DIR})

# --- Slang (FetchContent'd from upstream releases) --------------------------

if(WIN32)
	FetchContent_Declare(
		slang_release
		URL https://github.com/shader-slang/slang/releases/download/v2026.5.2/slang-2026.5.2-windows-x86_64.zip
		URL_HASH SHA256=8751a5c897dacc3d05bb5bb063abf4200b2bd91052e893abd48cd941219a1549
		DOWNLOAD_EXTRACT_TIMESTAMP TRUE
	)
elseif(UNIX AND NOT APPLE)
	FetchContent_Declare(
		slang_release
		URL https://github.com/shader-slang/slang/releases/download/v2026.5.2/slang-2026.5.2-linux-x86_64.tar.gz
		# URL_HASH SHA256=  # TODO: populate when first building on Linux
		DOWNLOAD_EXTRACT_TIMESTAMP TRUE
	)
endif()
FetchContent_MakeAvailable(slang_release)
set(SLANG_DIR "${slang_release_SOURCE_DIR}")

add_library(slang SHARED IMPORTED)
if(WIN32)
	set_target_properties(slang PROPERTIES
		IMPORTED_LOCATION "${SLANG_DIR}/bin/slang.dll"
		IMPORTED_IMPLIB   "${SLANG_DIR}/lib/slang.lib"
		INTERFACE_INCLUDE_DIRECTORIES "${SLANG_DIR}/include"
	)
else()
	set_target_properties(slang PROPERTIES
		IMPORTED_LOCATION "${SLANG_DIR}/lib/libslang.so"
		INTERFACE_INCLUDE_DIRECTORIES "${SLANG_DIR}/include"
	)
endif()

# 3. Source discovery
file(GLOB_RECURSE PROJECT_SOURCES "src/*.cpp")

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
	list(APPEND PROJECT_SOURCES "main.cpp")
endif()

# 4. Create target
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

# 5. Includes & linking
target_include_directories(${PROJECT_NAME} PRIVATE
	src
	${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
	Vulkan::Vulkan
	glfw
	glm
	spdlog
	nlohmann_json::nlohmann_json
	cxxopts
	imgui_lib
	stb
	tinyobjloader
	slang
)

# 6. Asset management
#
# Ensure runtime-output directories exist in the source tree. `data/`, `videos/`, and
# `images/` are created at configure time (idempotent — `MAKE_DIRECTORY` is a no-op if
# they already exist) so a fresh clone has the paths the runtime writes to. The folders
# are gitignored; this guarantees they're present without checking in placeholder files.
foreach(RUNTIME_OUTPUT_DIR data videos images)
	file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${RUNTIME_OUTPUT_DIR}")
endforeach()

# Copy asset directories the runtime needs next to the executable after each build.
# On multi-config generators (MSVC) the exe lands in e.g. build/Debug/, so we use
# a post-build step that resolves $<TARGET_FILE_DIR:…> at build time.
#
# Only `shaders/` (loaded at runtime by SlangCompiler + GLSL fragment loader + the
# rgb2ycbcr.spv encoder shader) and `models/` (cube.obj/.mtl, ED.obj/.mtl loaded by
# Simulation::ctor) are needed when running from the build dir. `data/`/`videos/`/
# `images/` are written/read using paths relative to the working directory; running
# Cave.exe from the repo root reads/writes them in-place.
foreach(DIR shaders models)
	if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${DIR}")
		add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_directory
				"${CMAKE_CURRENT_SOURCE_DIR}/${DIR}"
				"$<TARGET_FILE_DIR:${PROJECT_NAME}>/${DIR}"
			COMMENT "Copying ${DIR}/ to output directory"
		)
	endif()
endforeach()

# Copy Slang runtime libraries to output directory
if(WIN32)
	foreach(DLL slang.dll slang-compiler.dll slang-glslang.dll slang-glsl-module.dll slang-rt.dll slang-llvm.dll gfx.dll)
		if(EXISTS "${SLANG_DIR}/bin/${DLL}")
			add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
				COMMAND ${CMAKE_COMMAND} -E copy_if_different
					"${SLANG_DIR}/bin/${DLL}"
					"$<TARGET_FILE_DIR:${PROJECT_NAME}>"
				COMMENT "Copying Slang ${DLL}"
			)
		endif()
	endforeach()
else()
	# Linux .so files live in lib/. Glob at configure time so we pick up
	# whatever shared libs the release ships (libslang.so, libslang-rt.so, etc.)
	# plus any versioned symlinks/files the loader needs alongside the binary.
	file(GLOB SLANG_RUNTIME_LIBS "${SLANG_DIR}/lib/*.so" "${SLANG_DIR}/lib/*.so.*")
	foreach(LIB ${SLANG_RUNTIME_LIBS})
		add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
				"${LIB}"
				"$<TARGET_FILE_DIR:${PROJECT_NAME}>"
			COMMENT "Copying Slang ${LIB}"
		)
	endforeach()
endif()
