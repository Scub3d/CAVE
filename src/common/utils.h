#pragma once

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <array>

#if defined(_MSC_VER)
#define DEBUG_BREAK __debugbreak()
#else
#include <signal.h>
#define DEBUG_BREAK raise(SIGTRAP)
#endif

namespace Cave
{

inline void DebugBreak()
{
	std::cout << "Breakpoint here to debug." << std::endl;
	DEBUG_BREAK;
}

inline std::array<float, 4> ConvertHexToArray(uint32_t color)
{
	float red   = ((color >> 16) & 0xFF) / 255.0;
	float green = ((color >> 8) & 0xFF) / 255.0;
	float blue  = ((color) & 0xFF) / 255.0;

	return std::array<float, 4>{red, green, blue, 1.f};
}

inline std::vector<char> ReadFile(const std::string &filepath)
{
	std::vector<char> result;
	std::ifstream file(filepath, std::ios::ate | std::ios::binary);

	if (!file.is_open())
	{
		return result;
		//		std::stringstream message;
		//		message << "Failed to load \"" << filepath << "\"";
		//		Logging::Logger::GetLogger()->Print(message.str());
	}

	size_t fileSize = (size_t)file.tellg();
	result.resize(fileSize);

	file.seekg(0);
	file.read(result.data(), fileSize);

	file.close();

	return result;
}

inline bool IsStringInVector(std::vector<const char *> list, const char *name)
{
	bool found = false;
	for (auto &item : list)
	{
		if (strcmp(name, item) == 0)
		{
			found = true;
			break;
		}
	}
	return found;
}

template <typename T>
inline bool BitwiseCheck(const T &value, const T &checkValue)
{
	return ((value & checkValue) == checkValue);
}

template <typename T>
T Align(T value, T alignment)
{
	return (value + (alignment - 1)) & ~(alignment - 1);
};

inline std::string GetEnv(const std::string &variable)
{
	const char *value = std::getenv(variable.c_str());
	// It's invalid to assign nullptr to std::string
	return value != nullptr ? std::string(value) : std::string("");
}

inline void SetEnv(const std::string &variable, const std::string &value)
{
#if defined(_MSC_VER)
	_putenv_s(variable.c_str(), value.c_str());
#else
	setenv(variable.c_str(), value.c_str(), 1);
#endif
}

inline std::string ReadTextFile(const std::string &filepath)
{
	std::ifstream stream(filepath, std::fstream::in);
	std::string output;
	if (!stream.is_open())
	{
		std::cout << "Could not read file " << filepath.c_str() << ". File does not exist." << std::endl;
		return "";
	}
	std::string line;
	while (!stream.eof())
	{
		std::getline(stream, line);
		output.append(line + "\n");
	}
	stream.close();
	return output;
}

inline std::vector<char> ReadBinaryFile(const std::string &filepath)
{
	std::ifstream stream(filepath, std::fstream::in | std::fstream::binary | std::fstream::ate);
	if (!stream.is_open())
	{
		std::cout << "Could not read file " << filepath.c_str() << ". File does not exist." << std::endl;
		return {};
	}
	std::streamoff size = stream.tellg();
	std::vector<char> output(static_cast<size_t>(size));
	stream.seekg(0, std::fstream::beg);
	stream.read(output.data(), size);
	stream.close();
	return output;
}

#if defined(__ANDROID__)
#include <android/asset_manager.h>
inline std::string ReadTextFile(const std::string &filepath, AAssetManager *assetManager)
{
	AAsset *file = AAssetManager_open(assetManager, filepath.c_str(), AASSET_MODE_BUFFER);
	size_t fileLength = AAsset_getLength(file);
	std::string text;
	text.resize(fileLength);
	AAsset_read(file, (void *)text.data(), fileLength);
	AAsset_close(file);
	return text;
}

inline std::vector<char> ReadBinaryFile(const std::string &filepath, AAssetManager *assetManager)
{
	AAsset *file = AAssetManager_open(assetManager, filepath.c_str(), AASSET_MODE_BUFFER);
	size_t fileLength = AAsset_getLength(file);
	std::vector<char> binary(fileLength);
	AAsset_read(file, (void *)binary.data(), fileLength);
	AAsset_close(file);
	return binary;
}
#endif

} // namespace Cave