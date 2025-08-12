// GpuDispatcherTest.cpp
// Intel UHD 620 ile compute, AMD Radeon 530 ile render (Win32 + Vulkan).
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstring>

// ---------------------------------------------
// Common Buffer Struct
// ---------------------------------------------
struct Buffer {
    VkBuffer       buf;
    VkDeviceMemory mem;
};

// ---------------------------------------------
// Sabitler
// ---------------------------------------------
const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;
const size_t   ELEMENT_COUNT = 256;
const VkDeviceSize BUFFER_SIZE = ELEMENT_COUNT * sizeof(float);
const int      MAX_FRAMES_IN_FLIGHT = 2;

// ---------------------------------------------
// Global Objeler
// ---------------------------------------------
// Compute-side buffers (Intel)
Buffer bufferA, bufferB, bufferC;
// Render-side buffer (AMD)
Buffer bufferC_amd;

// Win32 pencere
HINSTANCE    g_hInstance;
HWND         g_hWnd;

// Vulkan
VkInstance           instance;
VkSurfaceKHR         surface;

VkPhysicalDevice     intelGPU = VK_NULL_HANDLE;
VkPhysicalDevice     amdGPU = VK_NULL_HANDLE;

VkDevice             intelDevice;
VkQueue              intelQueue;
uint32_t             intelComputeFamily;

VkDevice             amdDevice;
VkQueue              amdQueue;
uint32_t             amdGraphicsFamily;

// Compute pipeline (Intel)
VkCommandPool            intelCommandPool;
VkCommandBuffer          intelCmdBuffer;
VkDescriptorSetLayout    computeDescSetLayout;
VkPipelineLayout         computePipelineLayout;
VkPipeline               computePipeline;
VkDescriptorPool         computeDescPool;
VkDescriptorSet          computeDescSet;

// Render pipeline (AMD)
VkSwapchainKHR           swapchain;
std::vector<VkImage>     swapImages;
VkFormat                 swapchainImageFormat;
VkExtent2D               swapchainExtent;
std::vector<VkImageView> swapImageViews;
VkRenderPass             renderPass;
VkPipelineLayout         graphicsPipelineLayout;
VkPipeline               graphicsPipeline;
std::vector<VkFramebuffer> swapFramebuffers;
VkCommandPool             amdCommandPool;
std::vector<VkCommandBuffer> amdCommandBuffers;
VkDescriptorSetLayout     graphicsDescSetLayout;
VkDescriptorPool          graphicsDescPool;
VkDescriptorSet           graphicsDescSet;
VkSemaphore               imageAvailableSem[MAX_FRAMES_IN_FLIGHT];
VkSemaphore               renderFinishedSem[MAX_FRAMES_IN_FLIGHT];
VkFence                   inFlightFences[MAX_FRAMES_IN_FLIGHT];
size_t                    currentFrame = 0;

// ---------------------------------------------
// Yardımcı Fonksiyonlar
// ---------------------------------------------
static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Failed to open file: " + filename);
    size_t size = (size_t)file.tellg();
    if (size == 0 || size % 4 != 0)
        throw std::runtime_error("Invalid SPIR-V size: " + filename);
    std::vector<char> buf(size);
    file.seekg(0);
    file.read(buf.data(), size);
    return buf;
}

uint32_t findMemoryType(VkPhysicalDevice phys, uint32_t typeFilter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    throw std::runtime_error("Suitable memory type not found");
}

VkShaderModule createShaderModule(VkDevice dev, const std::vector<char>& code) {
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size();
    ci.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule mod;
    if (vkCreateShaderModule(dev, &ci, nullptr, &mod) != VK_SUCCESS)
        throw std::runtime_error("ShaderModule creation failed");
    return mod;
}

void createBuffer(
    VkDevice dev,
    VkPhysicalDevice phys,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags props,
    Buffer& out)
{
    // 1) Buffer create
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(dev, &bi, nullptr, &out.buf) != VK_SUCCESS)
        throw std::runtime_error("Failed to create buffer");

    // 2) Memory requirements + alloc
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, out.buf, &mr);

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = findMemoryType(phys, mr.memoryTypeBits, props);

    if (vkAllocateMemory(dev, &ai, nullptr, &out.mem) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate buffer memory");

    // 3) Bind
    vkBindBufferMemory(dev, out.buf, out.mem, 0);
}

// ---------------------------------------------
// 1) Win32 Pencere
// ---------------------------------------------
LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == WM_DESTROY) PostQuitMessage(0);
    return DefWindowProc(hWnd, msg, wParam, lParam);
}

void initWin32Window() {
    g_hInstance = GetModuleHandle(nullptr);
    const wchar_t CLASS_NAME[] = L"VkWin32Class";
    WNDCLASS wc{};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = g_hInstance;
    wc.lpszClassName = CLASS_NAME;
    RegisterClass(&wc);

    RECT wr = { 0, 0, (LONG)WIDTH, (LONG)HEIGHT };
    AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW, FALSE);

    g_hWnd = CreateWindowEx(
        0, CLASS_NAME, L"Intel→AMD Multi-GPU",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        wr.right - wr.left, wr.bottom - wr.top,
        nullptr, nullptr, g_hInstance, nullptr);

    ShowWindow(g_hWnd, SW_SHOW);
}

// ---------------------------------------------
// 2) Vulkan Instance + Surface
// ---------------------------------------------
void createInstance() {
    VkApplicationInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.pApplicationName = "GPU Dispatcher Test";
    ai.apiVersion = VK_API_VERSION_1_2;

    const char* exts[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME
    };

    VkInstanceCreateInfo ii{};
    ii.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ii.pApplicationInfo = &ai;
    ii.enabledExtensionCount = _countof(exts);
    ii.ppEnabledExtensionNames = exts;

    if (vkCreateInstance(&ii, nullptr, &instance) != VK_SUCCESS)
        throw std::runtime_error("Failed to create Vulkan instance");
}

void createWin32Surface() {
    VkWin32SurfaceCreateInfoKHR sci{};
    sci.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    sci.hinstance = g_hInstance;
    sci.hwnd = g_hWnd;
    if (vkCreateWin32SurfaceKHR(instance, &sci, nullptr, &surface) != VK_SUCCESS)
        throw std::runtime_error("Failed to create Win32 surface");
}

// ---------------------------------------------
// 3) Fiziksel Aygıtları Seçme
// ---------------------------------------------
void pickPhysicalDevices() {
    uint32_t cnt = 0;
    vkEnumeratePhysicalDevices(instance, &cnt, nullptr);
    if (cnt == 0) throw std::runtime_error("No Vulkan GPUs found");

    std::vector<VkPhysicalDevice> devs(cnt);
    vkEnumeratePhysicalDevices(instance, &cnt, devs.data());

    for (auto& d : devs) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(d, &p);
        std::cout << p.deviceName
            << " (vendorID=0x" << std::hex << p.vendorID << std::dec << ")\n";
        if (p.vendorID == 0x8086) intelGPU = d;
        if (p.vendorID == 0x1002) amdGPU = d;
    }

    if (intelGPU == VK_NULL_HANDLE || amdGPU == VK_NULL_HANDLE)
        throw std::runtime_error("Intel or AMD GPU not found");
}

// ---------------------------------------------
// 4) Mantıksal Cihazlar + Kuyruklar
// ---------------------------------------------
uint32_t findQueueFamily(VkPhysicalDevice dev, VkQueueFlagBits flags, bool needSurface = false) {
    uint32_t cnt = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &cnt, nullptr);
    std::vector<VkQueueFamilyProperties> fams(cnt);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &cnt, fams.data());

    for (uint32_t i = 0; i < cnt; i++) {
        if ((fams[i].queueFlags & flags) == flags) {
            if (needSurface) {
                VkBool32 sup = VK_FALSE;
                vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface, &sup);
                if (!sup) continue;
            }
            return i;
        }
    }
    throw std::runtime_error("Queue family not found");
}

void createLogicalDevices() {
    float pr = 1.0f;

    // Intel (compute)
    intelComputeFamily = findQueueFamily(intelGPU, VK_QUEUE_COMPUTE_BIT);
    VkDeviceQueueCreateInfo icq{};
    icq.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    icq.queueFamilyIndex = intelComputeFamily;
    icq.queueCount = 1;
    icq.pQueuePriorities = &pr;
    VkDeviceCreateInfo idi{};
    idi.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    idi.queueCreateInfoCount = 1;
    idi.pQueueCreateInfos = &icq;
    if (vkCreateDevice(intelGPU, &idi, nullptr, &intelDevice) != VK_SUCCESS)
        throw std::runtime_error("Failed to create Intel device");
    vkGetDeviceQueue(intelDevice, intelComputeFamily, 0, &intelQueue);

    // AMD (graphics + swapchain)
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(amdGPU, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateDeviceExtensionProperties(amdGPU, nullptr, &extCount, exts.data());
    bool swapSupported = false;
    for (auto& e : exts) {
        if (strcmp(e.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) {
            swapSupported = true;
            break;
        }
    }
    if (!swapSupported)
        throw std::runtime_error("AMD GPU missing VK_KHR_swapchain");

    amdGraphicsFamily = findQueueFamily(amdGPU, VK_QUEUE_GRAPHICS_BIT, true);
    VkDeviceQueueCreateInfo acq{};
    acq.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    acq.queueFamilyIndex = amdGraphicsFamily;
    acq.queueCount = 1;
    acq.pQueuePriorities = &pr;
    const char* aexts[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    VkDeviceCreateInfo adi{};
    adi.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    adi.queueCreateInfoCount = 1;
    adi.pQueueCreateInfos = &acq;
    adi.enabledExtensionCount = 1;
    adi.ppEnabledExtensionNames = aexts;
    if (vkCreateDevice(amdGPU, &adi, nullptr, &amdDevice) != VK_SUCCESS)
        throw std::runtime_error("Failed to create AMD device");
    vkGetDeviceQueue(amdDevice, amdGraphicsFamily, 0, &amdQueue);
}

// ---------------------------------------------
// 5) Command Pool & Buffer
// ---------------------------------------------
VkCommandPool createPool(VkDevice dev, uint32_t family) {
    VkCommandPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.queueFamilyIndex = family;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool pool;
    if (vkCreateCommandPool(dev, &pci, nullptr, &pool) != VK_SUCCESS)
        throw std::runtime_error("Failed to create command pool");
    return pool;
}

void createCommandResources() {
    intelCommandPool = createPool(intelDevice, intelComputeFamily);
    amdCommandPool = createPool(amdDevice, amdGraphicsFamily);

    // Intel compute cmd buffer
    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = intelCommandPool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    vkAllocateCommandBuffers(intelDevice, &cai, &intelCmdBuffer);

    // AMD render cmd buffers
    amdCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    cai.commandPool = amdCommandPool;
    cai.commandBufferCount = (uint32_t)amdCommandBuffers.size();
    vkAllocateCommandBuffers(amdDevice, &cai, amdCommandBuffers.data());
}

// ---------------------------------------------
// 6) Compute Buffer Oluşturma & Veri Doldurma
// ---------------------------------------------
void initComputeBuffers() {
    createBuffer(
        intelDevice, intelGPU,
        BUFFER_SIZE,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        bufferA);
    createBuffer(
        intelDevice, intelGPU,
        BUFFER_SIZE,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        bufferB);
    createBuffer(
        intelDevice, intelGPU,
        BUFFER_SIZE,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        bufferC);

    // Fill A and B
    auto fill = [&](Buffer& buf, bool isA) {
        void* ptr = nullptr;
        vkMapMemory(intelDevice, buf.mem, 0, BUFFER_SIZE, 0, &ptr);
        float* f = static_cast<float*>(ptr);
        for (size_t i = 0; i < ELEMENT_COUNT; i++)
            f[i] = isA ? float(i) : float(i * 2);
        vkUnmapMemory(intelDevice, buf.mem);
        };
    fill(bufferA, true);
    fill(bufferB, false);
}

// ---------------------------------------------
// 7) Compute Pipeline & Dispatch
// ---------------------------------------------
void createComputePipeline() {
    VkDescriptorSetLayoutBinding b[3]{};
    for (int i = 0; i < 3; i++) {
        b[i].binding = i;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsli{};
    dsli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsli.bindingCount = 3;
    dsli.pBindings = b;
    vkCreateDescriptorSetLayout(intelDevice, &dsli, nullptr, &computeDescSetLayout);

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &computeDescSetLayout;
    vkCreatePipelineLayout(intelDevice, &pli, nullptr, &computePipelineLayout);

    auto code = readFile("compute.spv");
    VkShaderModule compMod = createShaderModule(intelDevice, code);
    VkPipelineShaderStageCreateInfo psi{};
    psi.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    psi.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    psi.module = compMod;
    psi.pName = "main";

    VkComputePipelineCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage = psi;
    cpi.layout = computePipelineLayout;
    vkCreateComputePipelines(intelDevice, VK_NULL_HANDLE, 1, &cpi, nullptr, &computePipeline);
    vkDestroyShaderModule(intelDevice, compMod, nullptr);

    // Descriptor pool + set
    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = 3;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    dpci.maxSets = 1;
    vkCreateDescriptorPool(intelDevice, &dpci, nullptr, &computeDescPool);

    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = computeDescPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &computeDescSetLayout;
    vkAllocateDescriptorSets(intelDevice, &dsai, &computeDescSet);

    VkDescriptorBufferInfo infos[3] = {
        { bufferA.buf, 0, VK_WHOLE_SIZE },
        { bufferB.buf, 0, VK_WHOLE_SIZE },
        { bufferC.buf, 0, VK_WHOLE_SIZE }
    };
    VkWriteDescriptorSet ws[3]{};
    for (int i = 0; i < 3; i++) {
        ws[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[i].dstSet = computeDescSet;
        ws[i].dstBinding = i;
        ws[i].descriptorCount = 1;
        ws[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ws[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(intelDevice, 3, ws, 0, nullptr);
}

void recordAndSubmitCompute() {
    vkResetCommandPool(intelDevice, intelCommandPool, 0);

    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(intelCmdBuffer, &bi);

    vkCmdBindPipeline(intelCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(
        intelCmdBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        computePipelineLayout,
        0, 1, &computeDescSet,
        0, nullptr);

    vkCmdDispatch(intelCmdBuffer, ELEMENT_COUNT / 16, 1, 1);
    vkEndCommandBuffer(intelCmdBuffer);

    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &intelCmdBuffer;
    vkQueueSubmit(intelQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(intelQueue);
}

// ---------------------------------------------
// New:  AMD Result Buffer Creation
// ---------------------------------------------
void createResultBuffer() {
    createBuffer(
        amdDevice, amdGPU, BUFFER_SIZE,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        bufferC_amd);
}

// ---------------------------------------------
// 8) Swapchain + ImageViews (AMD)
// ---------------------------------------------
void createSwapChain() {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(amdGPU, surface, &caps);

    swapchainExtent = { WIDTH, HEIGHT };
    uint32_t cnt = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && cnt > caps.maxImageCount)
        cnt = caps.maxImageCount;

    VkSwapchainCreateInfoKHR sci{};
    sci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    sci.surface = surface;
    sci.minImageCount = cnt;
    sci.imageFormat = VK_FORMAT_B8G8R8A8_SRGB;
    sci.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    sci.imageExtent = swapchainExtent;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    sci.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(amdDevice, &sci, nullptr, &swapchain) != VK_SUCCESS)
        throw std::runtime_error("Failed to create swapchain");

    uint32_t imgCount;
    vkGetSwapchainImagesKHR(amdDevice, swapchain, &imgCount, nullptr);
    swapImages.resize(imgCount);
    vkGetSwapchainImagesKHR(amdDevice, swapchain, &imgCount, swapImages.data());
    swapchainImageFormat = sci.imageFormat;
}

void createImageViews() {
    swapImageViews.resize(swapImages.size());
    for (size_t i = 0; i < swapImages.size(); i++) {
        VkImageViewCreateInfo ivci{};
        ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image = swapImages[i];
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format = swapchainImageFormat;
        ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ivci.subresourceRange.baseMipLevel = 0;
        ivci.subresourceRange.levelCount = 1;
        ivci.subresourceRange.baseArrayLayer = 0;
        ivci.subresourceRange.layerCount = 1;
        vkCreateImageView(amdDevice, &ivci, nullptr, &swapImageViews[i]);
    }
}

// ---------------------------------------------
// 9) RenderPass + Graphics Pipeline (AMD)
// ---------------------------------------------
void createRenderPass() {
    VkAttachmentDescription attach{};
    attach.format = swapchainImageFormat;
    attach.samples = VK_SAMPLE_COUNT_1_BIT;
    attach.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attach.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attach.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attach.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attach.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference ref{};
    ref.attachment = 0;
    ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &ref;

    VkRenderPassCreateInfo rpci{};
    rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpci.attachmentCount = 1;
    rpci.pAttachments = &attach;
    rpci.subpassCount = 1;
    rpci.pSubpasses = &sub;

    vkCreateRenderPass(amdDevice, &rpci, nullptr, &renderPass);
}

void createGraphicsPipeline() {
    // Shader modules
    auto vertCode = readFile("vert.spv");
    auto fragCode = readFile("frag_data.spv");
    VkShaderModule vertMod = createShaderModule(amdDevice, vertCode);
    VkShaderModule fragMod = createShaderModule(amdDevice, fragCode);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod;
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod;
    stages[1].pName = "main";

    // Vertex Input
    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    // Input Assembly
    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    // Viewport / Scissor
    VkViewport vp{};
    vp.x = 0.0f;
    vp.y = 0.0f;
    vp.width = (float)swapchainExtent.width;
    vp.height = (float)swapchainExtent.height;
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;

    VkRect2D sc{};
    sc.offset = { 0,0 };
    sc.extent = swapchainExtent;

    VkPipelineViewportStateCreateInfo vpci{};
    vpci.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vpci.viewportCount = 1;
    vpci.pViewports = &vp;
    vpci.scissorCount = 1;
    vpci.pScissors = &sc;

    // Rasterizer
    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rs.lineWidth = 1.0f;

    // Multisample
    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Color blend
    VkPipelineColorBlendAttachmentState cb{};
    cb.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT;
    cb.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo cbci{};
    cbci.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cbci.attachmentCount = 1;
    cbci.pAttachments = &cb;

    // Descriptor set layout (SSBO: binding=0)
    VkDescriptorSetLayoutBinding db{};
    db.binding = 0;
    db.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    db.descriptorCount = 1;
    db.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo dsci{};
    dsci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsci.bindingCount = 1;
    dsci.pBindings = &db;
    vkCreateDescriptorSetLayout(amdDevice, &dsci, nullptr, &graphicsDescSetLayout);

    // Push constant (width)
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(int);

    // Pipeline layout
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &graphicsDescSetLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    vkCreatePipelineLayout(amdDevice, &plci, nullptr, &graphicsPipelineLayout);

    // Graphics pipeline
    VkGraphicsPipelineCreateInfo gpci{};
    gpci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpci.stageCount = 2;
    gpci.pStages = stages;
    gpci.pVertexInputState = &vi;
    gpci.pInputAssemblyState = &ia;
    gpci.pViewportState = &vpci;
    gpci.pRasterizationState = &rs;
    gpci.pMultisampleState = &ms;
    gpci.pColorBlendState = &cbci;
    gpci.layout = graphicsPipelineLayout;
    gpci.renderPass = renderPass;
    gpci.subpass = 0;

    vkCreateGraphicsPipelines(amdDevice, VK_NULL_HANDLE, 1, &gpci, nullptr, &graphicsPipeline);

    // Cleanup shader modules
    vkDestroyShaderModule(amdDevice, vertMod, nullptr);
    vkDestroyShaderModule(amdDevice, fragMod, nullptr);
}

// ---------------------------------------------
// 10) Framebuffer & Descriptor (AMD)
// ---------------------------------------------
void createFramebuffers() {
    swapFramebuffers.resize(swapImageViews.size());
    for (size_t i = 0; i < swapImageViews.size(); i++) {
        VkFramebufferCreateInfo fbci{};
        fbci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbci.renderPass = renderPass;
        fbci.attachmentCount = 1;
        fbci.pAttachments = &swapImageViews[i];
        fbci.width = swapchainExtent.width;
        fbci.height = swapchainExtent.height;
        vkCreateFramebuffer(amdDevice, &fbci, nullptr, &swapFramebuffers[i]);
    }
}

void createGraphicsDescriptorPool() {
    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    dpci.maxSets = 1;
    vkCreateDescriptorPool(amdDevice, &dpci, nullptr, &graphicsDescPool);
}

void createGraphicsDescriptorSet() {
    VkDescriptorSetAllocateInfo dsai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool = graphicsDescPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &graphicsDescSetLayout;
    vkAllocateDescriptorSets(amdDevice, &dsai, &graphicsDescSet);

    VkDescriptorBufferInfo bi{};
    bi.buffer = bufferC_amd.buf;
    bi.offset = 0;
    bi.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    w.dstSet = graphicsDescSet;
    w.dstBinding = 0;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo = &bi;
    vkUpdateDescriptorSets(amdDevice, 1, &w, 0, nullptr);
}

// ---------------------------------------------
// 11) Senkronizasyon (AMD)
// ---------------------------------------------
void createSyncObjects() {
    VkSemaphoreCreateInfo sci{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VkFenceCreateInfo      fci{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkCreateSemaphore(amdDevice, &sci, nullptr, &imageAvailableSem[i]);
        vkCreateSemaphore(amdDevice, &sci, nullptr, &renderFinishedSem[i]);
        vkCreateFence(amdDevice, &fci, nullptr, &inFlightFences[i]);
    }
}

// ---------------------------------------------
// 12) Komut Kaydı & Sunum (AMD)
// ---------------------------------------------
void recordAMDCommandBuffer(uint32_t idx) {
    VkCommandBuffer cmd = amdCommandBuffers[idx];
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(cmd, &bi);

    VkRenderPassBeginInfo rpbi{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    rpbi.renderPass = renderPass;
    rpbi.framebuffer = swapFramebuffers[idx];
    rpbi.renderArea.offset = { 0,0 };
    rpbi.renderArea.extent = swapchainExtent;
    VkClearValue clearColor{ {{0.0f,0.0f,0.0f,1.0f}} };
    rpbi.clearValueCount = 1;
    rpbi.pClearValues = &clearColor;

    vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
    vkCmdBindDescriptorSets(
        cmd,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        graphicsPipelineLayout,
        0, 1, &graphicsDescSet,
        0, nullptr);

    // Push constant: width
    vkCmdPushConstants(
        cmd,
        graphicsPipelineLayout,
        VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(int),
        &swapchainExtent.width);

    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

void copyComputeToGraphics() {
    // 1) Read back from Intel bufferC
    void* intelPtr = nullptr;
    vkMapMemory(intelDevice, bufferC.mem, 0, BUFFER_SIZE, 0, &intelPtr);
    std::vector<float> hostData(ELEMENT_COUNT);
    std::memcpy(hostData.data(), intelPtr, BUFFER_SIZE);
    vkUnmapMemory(intelDevice, bufferC.mem);

    // 2) Write into AMD bufferC_amd
    void* amdPtr = nullptr;
    VkResult r = vkMapMemory(amdDevice, bufferC_amd.mem, 0, BUFFER_SIZE, 0, &amdPtr);
    if (r != VK_SUCCESS || amdPtr == nullptr) {
        throw std::runtime_error("Failed to map AMD buffer memory");
    }
    std::memcpy(amdPtr, hostData.data(), BUFFER_SIZE);
    vkUnmapMemory(amdDevice, bufferC_amd.mem);
}

void drawFrame() {
    vkWaitForFences(amdDevice, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
    vkResetFences(amdDevice, 1, &inFlightFences[currentFrame]);

    uint32_t imgIndex;
    vkAcquireNextImageKHR(
        amdDevice, swapchain, UINT64_MAX,
        imageAvailableSem[currentFrame], VK_NULL_HANDLE, &imgIndex);

    recordAMDCommandBuffer(imgIndex);

    VkSemaphore waitSem[] = { imageAvailableSem[currentFrame] };
    VkSemaphore sigSem[] = { renderFinishedSem[currentFrame] };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = waitSem;
    si.pWaitDstStageMask = waitStages;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &amdCommandBuffers[imgIndex];
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = sigSem;
    vkQueueSubmit(amdQueue, 1, &si, inFlightFences[currentFrame]);

    VkPresentInfoKHR pi{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = sigSem;
    pi.swapchainCount = 1;
    pi.pSwapchains = &swapchain;
    pi.pImageIndices = &imgIndex;
    vkQueuePresentKHR(amdQueue, &pi);

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

// ---------------------------------------------
// 13) Temizlik
// ---------------------------------------------
void cleanup() {
    vkDeviceWaitIdle(amdDevice);
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(amdDevice, imageAvailableSem[i], nullptr);
        vkDestroySemaphore(amdDevice, renderFinishedSem[i], nullptr);
        vkDestroyFence(amdDevice, inFlightFences[i], nullptr);
    }
    vkDestroyDescriptorPool(amdDevice, graphicsDescPool, nullptr);
    vkDestroyDescriptorSetLayout(amdDevice, graphicsDescSetLayout, nullptr);
    vkDestroyPipeline(amdDevice, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(amdDevice, graphicsPipelineLayout, nullptr);
    vkDestroyRenderPass(amdDevice, renderPass, nullptr);
    for (auto fb : swapFramebuffers) vkDestroyFramebuffer(amdDevice, fb, nullptr);
    for (auto iv : swapImageViews)   vkDestroyImageView(amdDevice, iv, nullptr);
    vkDestroySwapchainKHR(amdDevice, swapchain, nullptr);
    vkDestroyCommandPool(amdDevice, amdCommandPool, nullptr);
    vkDestroyDevice(amdDevice, nullptr);

    vkDestroyPipeline(intelDevice, computePipeline, nullptr);
    vkDestroyPipelineLayout(intelDevice, computePipelineLayout, nullptr);
    vkDestroyDescriptorPool(intelDevice, computeDescPool, nullptr);
    vkDestroyDescriptorSetLayout(intelDevice, computeDescSetLayout, nullptr);
    vkDestroyBuffer(intelDevice, bufferA.buf, nullptr);
    vkFreeMemory(intelDevice, bufferA.mem, nullptr);
    vkDestroyBuffer(intelDevice, bufferB.buf, nullptr);
    vkFreeMemory(intelDevice, bufferB.mem, nullptr);
    vkDestroyBuffer(intelDevice, bufferC.buf, nullptr);
    vkFreeMemory(intelDevice, bufferC.mem, nullptr);
    vkDestroyCommandPool(intelDevice, intelCommandPool, nullptr);
    vkDestroyDevice(intelDevice, nullptr);

    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);
    DestroyWindow(g_hWnd);
    UnregisterClass(L"VkWin32Class", g_hInstance);
}

// ---------------------------------------------
// main()
// ---------------------------------------------
int WINAPI wWinMain(HINSTANCE, HINSTANCE, LPWSTR, int)
{
    try {
        // 1) Win32 window & Vulkan setup
        initWin32Window();
        createInstance();
        createWin32Surface();
        pickPhysicalDevices();
        createLogicalDevices();
        createCommandResources();

        // 2) Compute on Intel
        initComputeBuffers();
        createComputePipeline();
        recordAndSubmitCompute();

        // 3) Allocate AMD-side result buffer
        createResultBuffer();

        // 4) Copy compute results Intel → host → AMD
        copyComputeToGraphics();

        // 5) Build the render graph
        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();        // creates graphicsDescSetLayout

        // 6) Prepare the descriptor set for fragment‐side SSBO
        createGraphicsDescriptorPool();
        createGraphicsDescriptorSet();

        // 7) Framebuffers & synchronization
        createFramebuffers();
        createSyncObjects();

        // 8) Main render loop
        MSG msg = {};
        while (msg.message != WM_QUIT) {
            if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
            else {
                drawFrame();
            }
        }

        // 9) Cleanup
        cleanup();
        return 0;
    }
    catch (const std::exception& e) {
        MessageBoxA(nullptr, e.what(), "Error", MB_OK | MB_ICONERROR);
        return -1;
    }
}
