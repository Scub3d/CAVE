#include "engine.h"
#include "common/logger.h"
#include "common/startupConfig.h"
#include "testing/testRunner.h"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <cxxopts.hpp>
#include <atomic>
#include <algorithm>

using namespace Cave;

// Global validation error counter — incremented by Vulkan debug callback
std::atomic<int> g_validationErrorCount{0};

int main(int argc, char **argv)
{
    spdlog::set_level(spdlog::level::debug);

    cxxopts::Options options("Cave", "GPU-accelerated 3D cellular automata simulation engine");

    options.add_options()
        ("mode", "Application mode: render, search, video", cxxopts::value<std::string>()->default_value(""))
        ("birth", "Birth rules (e.g., \"1,4-6\")", cxxopts::value<std::string>()->default_value(""))
        ("survival", "Survival rules (e.g., \"4\")", cxxopts::value<std::string>()->default_value(""))
        ("max-cs", "Max cell state", cxxopts::value<int>()->default_value("5"))
        ("min-cs", "Min max cell state (search)", cxxopts::value<int>()->default_value("3"))
        ("max-cs-range", "Max max cell state (search)", cxxopts::value<int>()->default_value("10"))
        ("neighborhood", "Neighborhood types: F, E, C, W (e.g., \"FE\")", cxxopts::value<std::string>()->default_value("F"))
        ("grid", "Grid size (uniform X=Y=Z)", cxxopts::value<int>()->default_value("49"))
        ("spawn", "Spawn size (uniform X=Y=Z)", cxxopts::value<int>()->default_value("25"))
        ("spawn-mode", "Spawn fill: random or filled", cxxopts::value<std::string>()->default_value("filled"))
        ("spawn-density", "Fill density for random spawn [0,1]", cxxopts::value<float>()->default_value("0.5"))
        ("ticks", "Max ticks to survive (search/video)", cxxopts::value<int>()->default_value("100"))
        ("grid-sweep", "Grid-size sweep min:max:step for search mode (e.g. 30:60:10)", cxxopts::value<std::string>()->default_value(""))
        ("density-sweep", "Spawn-density sweep min:max:step in [0,1] (e.g. 0.25:1:0.25)", cxxopts::value<std::string>()->default_value(""))
        ("seed-runs", "Number of distinct per-cell random seeds per chunk (search mode)", cxxopts::value<int>()->default_value("1"))
        ("max-rule-bits", "Cap bits set per rule bitmask in search (0 = unbounded; K=3 covers Conway-class rules)", cxxopts::value<int>()->default_value("0"))
        ("gpu-vram-budget-gb", "Per-GPU VRAM budget in GiB for the chunk scheduler (search mode)", cxxopts::value<float>()->default_value("40"))
        ("chunks-per-config", "Number of partition copies per (grid × density × seed) chunk; partitions split the survival rule range upfront for parallelism (search mode)", cxxopts::value<int>()->default_value("1"))
        ("search-workgroup-size", "Threads per workgroup for the simulation compute kernel (search mode). Valid: 32, 64, 128, 256, 512", cxxopts::value<int>()->default_value("64"))
        ("json", "JSON file path (video import)", cxxopts::value<std::string>()->default_value(""))
        ("encode-all", "Encode all permutations from JSON", cxxopts::value<bool>()->default_value("false"))
        ("index", "Selected permutation index from JSON", cxxopts::value<int>()->default_value("-1"))
        ("width", "Video width", cxxopts::value<int>()->default_value("1920"))
        ("height", "Video height", cxxopts::value<int>()->default_value("1080"))
        ("fps", "Video FPS", cxxopts::value<int>()->default_value("30"))
        ("duration", "Video duration in ticks", cxxopts::value<int>()->default_value("300"))
        ("output", "Output folder", cxxopts::value<std::string>()->default_value(""))
        ("orbit-speed", "Camera orbit angular velocity (radians/tick)", cxxopts::value<float>()->default_value("-1"))
        ("orbit-angle", "Fix camera orbit angle (radians). When set, orbit-speed defaults to 0.", cxxopts::value<float>()->default_value("1e30"))
        ("orbit-elevation", "Fix camera orbit elevation (radians).", cxxopts::value<float>()->default_value("1e30"))
        ("orbit-radius", "Camera orbit radius (world units).", cxxopts::value<float>()->default_value("-1"))
        ("no-camera-zoom", "Disable video-mode zoom-out animation (fix camera at full-grid view)", cxxopts::value<bool>()->default_value("false"))
        ("frames", "Exit after N simulation ticks (-1 = unlimited)", cxxopts::value<int>()->default_value("-1"))
        ("capture-frame", "Capture screenshot at tick N", cxxopts::value<int>()->default_value("-1"))
        ("capture-output", "Path for screenshot output", cxxopts::value<std::string>()->default_value("data/capture.png"))
        ("log-file", "Enable logging to data/cave.log", cxxopts::value<bool>()->default_value("false"))
        ("benchmark", "Log GPU timing benchmark each second", cxxopts::value<bool>()->default_value("false"))
        ("compute-skip", "Enable compute skip optimization (default: true)", cxxopts::value<bool>()->default_value("true"))
        ("headless", "Disable GUI rendering (compute-only mode)", cxxopts::value<bool>()->default_value("false"))
        ("test", "Run built-in test (all, compute-skip, sparse-benchmark, render-consistency, large-grid, cli-rules)", cxxopts::value<std::string>()->default_value(""))
        ("list-gpus", "List available GPUs and exit", cxxopts::value<bool>()->default_value("false"))
        ("gpu", "Force specific GPU by index (use --list-gpus to see indices)", cxxopts::value<int>()->default_value("-1"))
        ("dual-gpu", "Enable dual-GPU domain decomposition", cxxopts::value<bool>()->default_value("false"))
        ("storage-mode", "Cell state storage: buffer or image", cxxopts::value<std::string>()->default_value("image"))
        ("shape", "Cell shape: cube or ed (elongated dodecahedron)", cxxopts::value<std::string>()->default_value("cube"))
        ("h,help", "Print usage");

    cxxopts::ParseResult result;
    try
    {
        result = options.parse(argc, argv);
    }
    catch (const std::exception& e)
    {
        fprintf(stderr, "Argument parse error: %s\n", e.what());
        return 2;
    }

    if (result.count("help"))
    {
        LOG_INFO("{}", options.help());
        return 0;
    }

    StartupConfig config{};
    std::string modeStr = result["mode"].as<std::string>();

    if (!modeStr.empty())
    {
        config.autoStart = true;

        if (modeStr == "render")
            config.mode = ApplicationMode::Rendering;
        else if (modeStr == "search")
            config.mode = ApplicationMode::Search;
        else if (modeStr == "video")
            config.mode = ApplicationMode::VideoEncoding;
        else
        {
            LOG_ERROR("Unknown mode: {}. Use render, search, or video.", modeStr);
            return 1;
        }

        config.birthRules = result["birth"].as<std::string>();
        config.survivalRules = result["survival"].as<std::string>();
        config.maxCellState = result["max-cs"].as<int>();
        config.minMaxCellState = result["min-cs"].as<int>();
        config.maxMaxCellState = result["max-cs-range"].as<int>();
        config.neighborhood = result["neighborhood"].as<std::string>();
        {
            bool hasFace   = config.neighborhood.find('F') != std::string::npos;
            bool hasEdge   = config.neighborhood.find('E') != std::string::npos;
            bool hasCorner = config.neighborhood.find('C') != std::string::npos;
            if (!hasFace && !hasEdge && !hasCorner)
            {
                LOG_ERROR("--neighborhood must include at least one of F / E / C (got \"{}\")", config.neighborhood);
                return 1;
            }
        }
        config.gridSize = result["grid"].as<int>();
        config.spawnSize = result["spawn"].as<int>();
        {
            std::string spawnModeStr = result["spawn-mode"].as<std::string>();
            if (spawnModeStr == "random")
                config.spawnMode = 0;
            else if (spawnModeStr == "filled")
                config.spawnMode = 1;
            else
            {
                LOG_ERROR("Unknown spawn-mode: {}. Use random or filled.", spawnModeStr);
                return 1;
            }
        }
        config.spawnRandomDensity = std::clamp(result["spawn-density"].as<float>(), 0.0f, 1.0f);
        config.maxTicks = result["ticks"].as<int>();
        {
            std::string gridSweepStr = result["grid-sweep"].as<std::string>();
            if (!gridSweepStr.empty())
            {
                // Parse "min:max:step" — three ints separated by colons. Colons avoid
                // comma collisions with cxxopts list-value handling on some versions.
                int parsedMin = 0, parsedMax = 0, parsedStep = 0;
                int matched = sscanf(gridSweepStr.c_str(), "%d:%d:%d", &parsedMin, &parsedMax, &parsedStep);
                if (matched != 3 || parsedMin <= 0 || parsedMax < parsedMin || parsedStep <= 0)
                {
                    LOG_ERROR("--grid-sweep must be min:max:step with positive ints and max >= min (got \"{}\")", gridSweepStr);
                    return 1;
                }
                config.searchGridSizeSweepEnabled = true;
                config.searchGridSizeSweepMin = parsedMin;
                config.searchGridSizeSweepMax = parsedMax;
                config.searchGridSizeSweepStep = parsedStep;
            }
        }
        {
            std::string densitySweepStr = result["density-sweep"].as<std::string>();
            if (!densitySweepStr.empty())
            {
                float parsedMin = 0.0f, parsedMax = 0.0f, parsedStep = 0.0f;
                int matched = sscanf(densitySweepStr.c_str(), "%f:%f:%f", &parsedMin, &parsedMax, &parsedStep);
                if (matched != 3 || parsedMin <= 0.0f || parsedMax < parsedMin || parsedStep <= 0.0f
                    || parsedMin > 1.0f || parsedMax > 1.0f)
                {
                    LOG_ERROR("--density-sweep must be min:max:step in (0,1] with max >= min (got \"{}\")", densitySweepStr);
                    return 1;
                }
                config.searchDensitySweepEnabled = true;
                config.searchDensitySweepMin = parsedMin;
                config.searchDensitySweepMax = parsedMax;
                config.searchDensitySweepStep = parsedStep;
            }
        }
        config.searchSeedSweepCount = std::max(1, result["seed-runs"].as<int>());
        config.searchMaxRuleBits = std::max(0, result["max-rule-bits"].as<int>());
        config.searchGpuVramBudgetGb = std::max(0.5f, result["gpu-vram-budget-gb"].as<float>());
        config.chunksPerConfig = std::max(1, result["chunks-per-config"].as<int>());
        {
            int wgs = result["search-workgroup-size"].as<int>();
            if (wgs != 32 && wgs != 64 && wgs != 128 && wgs != 256 && wgs != 512)
            {
                LOG_ERROR("--search-workgroup-size must be one of {{32, 64, 128, 256, 512}} (got {})", wgs);
                return 1;
            }
            config.searchWorkgroupSize = wgs;
        }
        config.jsonPath = result["json"].as<std::string>();
        config.encodeAll = result["encode-all"].as<bool>();
        config.selectedIndex = result["index"].as<int>();
        config.videoWidth = result["width"].as<int>();
        config.videoHeight = result["height"].as<int>();
        config.videoFps = result["fps"].as<int>();
        config.videoDurationTicks = result["duration"].as<int>();
        config.orbitSpeed = result["orbit-speed"].as<float>();
        config.orbitAngle = result["orbit-angle"].as<float>();
        config.orbitElevation = result["orbit-elevation"].as<float>();
        config.orbitRadius = result["orbit-radius"].as<float>();
        config.disableCameraZoom = result["no-camera-zoom"].as<bool>();

        std::string outputStr = result["output"].as<std::string>();
        if (!outputStr.empty())
        {
            if (config.mode == ApplicationMode::Search)
                config.searchOutputFolder = outputStr;
            else if (config.mode == ApplicationMode::VideoEncoding)
                config.videoOutputFolder = outputStr;
        }
    }

    // Automation flags (available even without --mode)
    config.maxFrames = result["frames"].as<int>();
    config.captureFrame = result["capture-frame"].as<int>();
    config.captureOutputPath = result["capture-output"].as<std::string>();
    config.logToFile = result["log-file"].as<bool>();
    config.benchmark = result["benchmark"].as<bool>();
    config.enableComputeSkip = result["compute-skip"].as<bool>();
    config.headless = result["headless"].as<bool>();
    config.testName = result["test"].as<std::string>();
    config.forcedGpuIndex = result["gpu"].as<int>();
    config.dualGpu = result["dual-gpu"].as<bool>();
    config.storageMode = (result["storage-mode"].as<std::string>() == "image")
        ? CellStateStorageMode::Image : CellStateStorageMode::Buffer;
    {
        std::string shapeStr = result["shape"].as<std::string>();
        std::transform(shapeStr.begin(), shapeStr.end(), shapeStr.begin(),
            [](unsigned char c) { return std::tolower(c); });
        if (shapeStr == "cube")
            config.shape = 0;
        else if (shapeStr == "ed" || shapeStr == "elongateddodecahedron")
            config.shape = 1;
        else
        {
            LOG_ERROR("Unknown --shape: {}. Use 'cube' or 'ed'.", shapeStr);
            return 1;
        }
    }

    // Configure file logging when requested or when running in CLI mode
    if (config.logToFile || config.autoStart)
    {
        try
        {
            auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            auto fileSink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("data/cave.log", true);
            auto logger = std::make_shared<spdlog::logger>("cave", spdlog::sinks_init_list{consoleSink, fileSink});
            logger->set_level(spdlog::level::debug);
            spdlog::set_default_logger(logger);
            LOG_INFO("File logging enabled: data/cave.log");
        }
        catch (const spdlog::spdlog_ex& exception)
        {
            LOG_ERROR("Failed to initialize file logging: {}", exception.what());
        }
    }

    if (result["list-gpus"].as<bool>())
    {
        glfwInit();
        uint32_t count = 0;
        vkEnumerateInstanceVersion(&count);

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.apiVersion = VK_API_VERSION_1_3;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        VkInstance instance;
        vkCreateInstance(&createInfo, nullptr, &instance);

        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        LOG_INFO("Available GPUs ({}):", deviceCount);
        for (uint32_t i = 0; i < deviceCount; i++)
        {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(devices[i], &props);

            VkPhysicalDeviceMemoryProperties memProps;
            vkGetPhysicalDeviceMemoryProperties(devices[i], &memProps);

            uint64_t totalVram = 0;
            for (uint32_t h = 0; h < memProps.memoryHeapCount; h++)
            {
                if (memProps.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                    totalVram += memProps.memoryHeaps[h].size;
            }

            const char* typeStr = "Unknown";
            switch (props.deviceType)
            {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: typeStr = "Discrete"; break;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: typeStr = "Integrated"; break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: typeStr = "Virtual"; break;
            case VK_PHYSICAL_DEVICE_TYPE_CPU: typeStr = "CPU"; break;
            default: break;
            }

            LOG_INFO("  [{}] {} ({}) — {:.1f} GB VRAM, Vulkan {}.{}.{}",
                i, props.deviceName, typeStr,
                totalVram / (1024.0 * 1024.0 * 1024.0),
                VK_VERSION_MAJOR(props.apiVersion),
                VK_VERSION_MINOR(props.apiVersion),
                VK_VERSION_PATCH(props.apiVersion));
        }

        vkDestroyInstance(instance, nullptr);
        glfwTerminate();
        return 0;
    }

    Engine engine;

    if (!config.testName.empty())
    {
        engine.Init(&config);
        Cave::TestRunner testRunner(engine.GetModeServices());
        int testResult = testRunner.RunTests(config.testName);
        engine.Cleanup();

        int validationErrors = g_validationErrorCount.load();
        return (testResult != 0 || validationErrors > 0) ? 1 : 0;
    }

    try
    {
        engine.Init(&config);
        engine.Run();
        engine.Cleanup();
    }
    catch (const std::exception& e)
    {
        // Without this, an uncaught Vulkan exception (e.g. ErrorOutOfDeviceMemory at
        // huge grid sizes) triggers std::terminate and the process exits with the
        // useless code 3 — no log message reaches the user about WHY we died. With
        // this catch, the underlying error reason is already in the log via the
        // throw site, plus we exit with code 4 so callers can distinguish a real
        // failure from a clean "auto-exit complete" (code 0) or validation-error (1).
        LOG_FATAL("Engine aborted: {}", e.what());
        return 4;
    }

    int validationErrors = g_validationErrorCount.load();
    if (validationErrors > 0)
    {
        LOG_ERROR("Exiting with {} validation error(s)", validationErrors);
    }
    return validationErrors > 0 ? 1 : 0;
}
