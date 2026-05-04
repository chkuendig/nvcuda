/*
 * D3D11 → ZLUDA texture interop helper — implementation.
 *
 * See d3d11_interop.h for the high-level rationale. This file is
 * intentionally self-contained (no vulkan/vulkan.h dependency) to keep
 * nvcuda's build inputs minimal — we declare just the few Vulkan types and
 * constants we need. Adding the full Vulkan-Headers as a build dep would
 * also work, but inline keeps things simple.
 */

#include "config.h"
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "windef.h"
#include "winbase.h"
#include "wine/debug.h"
#define COBJMACROS
#include "initguid.h"
#include "d3d11.h"

#include "d3d11_interop.h"

WINE_DEFAULT_DEBUG_CHANNEL(nvcuda);

/* ============================================================================
 * Minimal Vulkan typedefs (just what we need)
 * ========================================================================== */

typedef int32_t  VkResult;
typedef uint32_t VkFlags;
typedef uint32_t VkBool32;
typedef uint64_t VkInstance;
typedef uint64_t VkPhysicalDevice;
typedef uint64_t VkDevice;
typedef uint64_t VkBuffer;
typedef uint64_t VkDeviceMemory;
typedef uint64_t VkImage;
typedef uint32_t VkImageLayout;

#define VK_SUCCESS                                            0

/* sType values */
#define VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO                  12
#define VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO                5
#define VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO         1000072002
#define VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO  1000072013
#define VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR              1000074002
#define VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO                   14   /* used by IDXGIVkInteropSurface::GetVulkanImageInfo */

/* VkBufferUsageFlagBits */
#define VK_BUFFER_USAGE_TRANSFER_SRC_BIT                      0x00000001
#define VK_BUFFER_USAGE_TRANSFER_DST_BIT                      0x00000002
#define VK_BUFFER_USAGE_STORAGE_BUFFER_BIT                    0x00000020

/* VkSharingMode */
#define VK_SHARING_MODE_EXCLUSIVE                             0

/* VkExternalMemoryHandleTypeFlagBits */
#define VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT        0x00000200

/* VkMemoryPropertyFlagBits */
#define VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT                   0x00000001

/* VkMemoryHeapFlagBits — unused but for completeness */
#define VK_MAX_MEMORY_TYPES                                   32

/* Used as flags = 0 throughout */
typedef uint32_t VkBufferCreateFlags;
typedef uint32_t VkBufferUsageFlags;
typedef uint32_t VkExternalMemoryHandleTypeFlags;
typedef uint32_t VkMemoryPropertyFlags;
typedef uint32_t VkMemoryHeapFlags;

typedef struct VkExternalMemoryBufferCreateInfo {
    uint32_t                          sType;
    const void                       *pNext;
    VkExternalMemoryHandleTypeFlags   handleTypes;
} VkExternalMemoryBufferCreateInfo;

typedef struct VkBufferCreateInfo {
    uint32_t              sType;
    const void           *pNext;
    VkBufferCreateFlags   flags;
    uint64_t              size;
    VkBufferUsageFlags    usage;
    int                   sharingMode;
    uint32_t              queueFamilyIndexCount;
    const uint32_t       *pQueueFamilyIndices;
} VkBufferCreateInfo;

typedef struct VkMemoryRequirements {
    uint64_t size;
    uint64_t alignment;
    uint32_t memoryTypeBits;
} VkMemoryRequirements;

typedef struct VkExportMemoryAllocateInfo {
    uint32_t                          sType;
    const void                       *pNext;
    VkExternalMemoryHandleTypeFlags   handleTypes;
} VkExportMemoryAllocateInfo;

typedef struct VkMemoryAllocateInfo {
    uint32_t      sType;
    const void   *pNext;
    uint64_t      allocationSize;
    uint32_t      memoryTypeIndex;
} VkMemoryAllocateInfo;

typedef struct VkMemoryType {
    VkMemoryPropertyFlags propertyFlags;
    uint32_t              heapIndex;
} VkMemoryType;

typedef struct VkMemoryHeap {
    uint64_t          size;
    VkMemoryHeapFlags flags;
} VkMemoryHeap;

typedef struct VkPhysicalDeviceMemoryProperties {
    uint32_t      memoryTypeCount;
    VkMemoryType  memoryTypes[VK_MAX_MEMORY_TYPES];
    uint32_t      memoryHeapCount;
    VkMemoryHeap  memoryHeaps[16]; /* VK_MAX_MEMORY_HEAPS */
} VkPhysicalDeviceMemoryProperties;

typedef struct VkMemoryGetFdInfoKHR {
    uint32_t                          sType;
    const void                       *pNext;
    VkDeviceMemory                    memory;
    VkExternalMemoryHandleTypeFlags   handleType;
} VkMemoryGetFdInfoKHR;

/* Function pointer types */
typedef void* (*PFN_vkGetInstanceProcAddr)(VkInstance, const char*);
typedef void* (*PFN_vkGetDeviceProcAddr)(VkDevice, const char*);
typedef VkResult (*PFN_vkCreateBuffer)(VkDevice, const VkBufferCreateInfo*, const void*, VkBuffer*);
typedef void (*PFN_vkDestroyBuffer)(VkDevice, VkBuffer, const void*);
typedef void (*PFN_vkGetBufferMemoryRequirements)(VkDevice, VkBuffer, VkMemoryRequirements*);
typedef VkResult (*PFN_vkAllocateMemory)(VkDevice, const VkMemoryAllocateInfo*, const void*, VkDeviceMemory*);
typedef void (*PFN_vkFreeMemory)(VkDevice, VkDeviceMemory, const void*);
typedef VkResult (*PFN_vkBindBufferMemory)(VkDevice, VkBuffer, VkDeviceMemory, uint64_t);
typedef VkResult (*PFN_vkGetMemoryFdKHR)(VkDevice, const VkMemoryGetFdInfoKHR*, int*);
typedef void (*PFN_vkGetPhysicalDeviceMemoryProperties)(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties*);

/* ============================================================================
 * DXVK COM interfaces
 *
 * We declare just enough of the IDXGIVkInteropSurface / IDXGIVkInteropDevice
 * vtables to call GetDevice / GetVulkanHandles / GetVulkanImageInfo. The full
 * interfaces have many more methods but we don't need them for register.
 * ========================================================================== */

DEFINE_GUID(IID_IDXGIVkInteropSurface, 0x5546cf8c, 0x77e7, 0x4341, 0xb0, 0x5d, 0x8d, 0x4d, 0x50, 0x00, 0xe7, 0x7d);
DEFINE_GUID(IID_IDXGIVkInteropDevice,  0xe2ef5fa5, 0xdc21, 0x4af7, 0x90, 0xc4, 0xf6, 0x7e, 0xf6, 0xa0, 0x93, 0x23);

typedef struct VkImageCreateInfo {
    uint32_t  sType;
    const void *pNext;
    uint32_t  flags;
    int       imageType;
    int       format;
    struct { uint32_t width, height, depth; } extent;
    uint32_t  mipLevels;
    uint32_t  arrayLayers;
    int       samples;
    int       tiling;
    uint32_t  usage;
    int       sharingMode;
    uint32_t  queueFamilyIndexCount;
    const uint32_t *pQueueFamilyIndices;
    int       initialLayout;
} VkImageCreateInfo;

typedef struct IDXGIVkInteropDevice IDXGIVkInteropDevice;
typedef struct IDXGIVkInteropSurface IDXGIVkInteropSurface;

typedef struct IDXGIVkInteropSurfaceVtbl {
    /* IUnknown */
    HRESULT (STDMETHODCALLTYPE *QueryInterface)(IDXGIVkInteropSurface*, REFIID, void**);
    ULONG   (STDMETHODCALLTYPE *AddRef)(IDXGIVkInteropSurface*);
    ULONG   (STDMETHODCALLTYPE *Release)(IDXGIVkInteropSurface*);
    /* IDXGIVkInteropSurface */
    HRESULT (STDMETHODCALLTYPE *GetDevice)(IDXGIVkInteropSurface*, IDXGIVkInteropDevice**);
    HRESULT (STDMETHODCALLTYPE *GetVulkanImageInfo)(IDXGIVkInteropSurface*, VkImage*, VkImageLayout*, VkImageCreateInfo*);
} IDXGIVkInteropSurfaceVtbl;
struct IDXGIVkInteropSurface { const IDXGIVkInteropSurfaceVtbl *lpVtbl; };

/* IDXGIVkInteropDevice — only need GetVulkanHandles for register; the layout
 * Map flow (Lock/Flush/Submit) needs more methods we'll add later. */
typedef struct IDXGIVkInteropDeviceVtbl {
    /* IUnknown */
    HRESULT (STDMETHODCALLTYPE *QueryInterface)(IDXGIVkInteropDevice*, REFIID, void**);
    ULONG   (STDMETHODCALLTYPE *AddRef)(IDXGIVkInteropDevice*);
    ULONG   (STDMETHODCALLTYPE *Release)(IDXGIVkInteropDevice*);
    /* IDXGIVkInteropDevice */
    void    (STDMETHODCALLTYPE *GetVulkanHandles)(IDXGIVkInteropDevice*, VkInstance*, VkPhysicalDevice*, VkDevice*);
    /* others omitted intentionally — Map flow will need them later */
} IDXGIVkInteropDeviceVtbl;
struct IDXGIVkInteropDevice { const IDXGIVkInteropDeviceVtbl *lpVtbl; };

/* ============================================================================
 * Module-level Vulkan handle cache
 *
 * We resolve Vulkan entry points lazily on the first call. The handles from
 * DXVK belong to the same Vulkan loader (one process, one libvulkan.so.1)
 * regardless of who dlopen'd it, so caching device-level fn pointers per
 * VkDevice is fine.
 * ========================================================================== */

static void *g_vulkan_handle = NULL;
static PFN_vkGetInstanceProcAddr               g_vkGetInstanceProcAddr = NULL;
static PFN_vkGetDeviceProcAddr                 g_vkGetDeviceProcAddr = NULL;
static VkDevice                                g_vk_device = 0;
static VkPhysicalDevice                        g_vk_phys_device = 0;
static PFN_vkCreateBuffer                      g_vkCreateBuffer = NULL;
static PFN_vkDestroyBuffer                     g_vkDestroyBuffer = NULL;
static PFN_vkGetBufferMemoryRequirements       g_vkGetBufferMemoryRequirements = NULL;
static PFN_vkAllocateMemory                    g_vkAllocateMemory = NULL;
static PFN_vkFreeMemory                        g_vkFreeMemory = NULL;
static PFN_vkBindBufferMemory                  g_vkBindBufferMemory = NULL;
static PFN_vkGetMemoryFdKHR                    g_vkGetMemoryFdKHR = NULL;
static PFN_vkGetPhysicalDeviceMemoryProperties g_vkGetPhysicalDeviceMemoryProperties = NULL;
static VkPhysicalDeviceMemoryProperties        g_vk_mem_props;

static int load_vulkan_loader(void)
{
    if (g_vulkan_handle) return 0;
    g_vulkan_handle = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!g_vulkan_handle) {
        ERR("[ZLUDA] dlopen(libvulkan.so.1) failed: %s\n", dlerror());
        return -1;
    }
    g_vkGetInstanceProcAddr = dlsym(g_vulkan_handle, "vkGetInstanceProcAddr");
    g_vkGetDeviceProcAddr   = dlsym(g_vulkan_handle, "vkGetDeviceProcAddr");
    if (!g_vkGetInstanceProcAddr || !g_vkGetDeviceProcAddr) {
        ERR("[ZLUDA] dlsym vkGet*ProcAddr failed: %s\n", dlerror());
        return -1;
    }
    return 0;
}

#define RESOLVE_DEV(name) \
    do { \
        g_##name = (PFN_##name)g_vkGetDeviceProcAddr(dev, #name); \
        if (!g_##name) { ERR("[ZLUDA] vkGetDeviceProcAddr(\"%s\") = NULL\n", #name); return -1; } \
    } while (0)

#define RESOLVE_INST(name) \
    do { \
        g_##name = (PFN_##name)g_vkGetInstanceProcAddr(inst, #name); \
        if (!g_##name) { ERR("[ZLUDA] vkGetInstanceProcAddr(\"%s\") = NULL\n", #name); return -1; } \
    } while (0)

static int load_vulkan_fns(VkInstance inst, VkPhysicalDevice phys, VkDevice dev)
{
    if (g_vk_device == dev && g_vk_phys_device == phys) return 0;
    if (load_vulkan_loader() != 0) return -1;

    RESOLVE_DEV(vkCreateBuffer);
    RESOLVE_DEV(vkDestroyBuffer);
    RESOLVE_DEV(vkGetBufferMemoryRequirements);
    RESOLVE_DEV(vkAllocateMemory);
    RESOLVE_DEV(vkFreeMemory);
    RESOLVE_DEV(vkBindBufferMemory);
    RESOLVE_DEV(vkGetMemoryFdKHR);
    RESOLVE_INST(vkGetPhysicalDeviceMemoryProperties);

    g_vk_device      = dev;
    g_vk_phys_device = phys;
    memset(&g_vk_mem_props, 0, sizeof(g_vk_mem_props));
    g_vkGetPhysicalDeviceMemoryProperties(phys, &g_vk_mem_props);
    return 0;
}

/* Pick a memory type that's both DEVICE_LOCAL (so CUDA reads are fast on the
 * GPU) and supported by the buffer's memoryTypeBits. We don't need
 * HOST_VISIBLE because CUDA accesses go through the GPU. */
static int find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags want)
{
    for (uint32_t i = 0; i < g_vk_mem_props.memoryTypeCount; ++i) {
        if (!(type_bits & (1u << i))) continue;
        if ((g_vk_mem_props.memoryTypes[i].propertyFlags & want) != want) continue;
        return (int)i;
    }
    return -1;
}

/* ============================================================================
 * Public entry point
 * ========================================================================== */

#define DXGI_FORMAT_R32G32B32A32_FLOAT 2

int zluda_d3d11_setup_external_memory(
    ID3D11Resource         *pD3DResource,
    unsigned                Flags,
    ZludaD3D11ImportedDesc *pDesc)
{
    if (!pD3DResource || !pDesc) return ZLUDA_D3D11_ERR_NOT_SUPPORTED;
    memset(pDesc, 0, sizeof(*pDesc));
    pDesc->fd = -1;
    pDesc->d3d_resource = pD3DResource;
    pDesc->register_flags = Flags;

    /* Only Tex2D supported in this iteration. The P0 probe has already logged
     * the descriptor type at FIXME level if it was something else; this is
     * the silent rejection step. */
    ID3D11Texture2D *pTex2D = NULL;
    if (FAILED(ID3D11Resource_QueryInterface(pD3DResource, &IID_ID3D11Texture2D, (void**)&pTex2D))) {
        WARN("[ZLUDA] zluda_d3d11_setup_external_memory: not a Tex2D\n");
        return ZLUDA_D3D11_ERR_NOT_SUPPORTED;
    }
    D3D11_TEXTURE2D_DESC tex_desc;
    ID3D11Texture2D_GetDesc(pTex2D, &tex_desc);
    pDesc->width       = tex_desc.Width;
    pDesc->height      = tex_desc.Height;
    pDesc->dxgi_format = tex_desc.Format;
    pDesc->bind_flags  = tex_desc.BindFlags;
    ID3D11Texture2D_Release(pTex2D);

    if (tex_desc.Format != DXGI_FORMAT_R32G32B32A32_FLOAT
        || tex_desc.MipLevels != 1
        || tex_desc.ArraySize != 1
        || tex_desc.SampleDesc.Count != 1) {
        FIXME("[ZLUDA] setup_external_memory: descriptor not yet supported "
              "(fmt=%u mips=%u arr=%u samp=%u). Only Tex2D R32G32B32A32_FLOAT "
              "mips=1 arr=1 samp=1 is implemented; see "
              "project_d3d11_p0_resource_descriptor.md.\n",
              tex_desc.Format, tex_desc.MipLevels, tex_desc.ArraySize,
              tex_desc.SampleDesc.Count);
        return ZLUDA_D3D11_ERR_NOT_SUPPORTED;
    }

    /* Query DXVK interop interfaces */
    IDXGIVkInteropSurface *pInteropSurf = NULL;
    if (FAILED(ID3D11Resource_QueryInterface(pD3DResource,
            &IID_IDXGIVkInteropSurface, (void**)&pInteropSurf))) {
        ERR("[ZLUDA] QueryInterface(IDXGIVkInteropSurface) failed — DXVK missing or wrong version\n");
        return ZLUDA_D3D11_ERR_DXVK_INTEROP_MISSING;
    }
    IDXGIVkInteropDevice *pInteropDev = NULL;
    if (FAILED(pInteropSurf->lpVtbl->GetDevice(pInteropSurf, &pInteropDev))) {
        ERR("[ZLUDA] IDXGIVkInteropSurface::GetDevice failed\n");
        pInteropSurf->lpVtbl->Release(pInteropSurf);
        return ZLUDA_D3D11_ERR_DXVK_INTEROP_MISSING;
    }
    VkInstance       vk_instance     = 0;
    VkPhysicalDevice vk_phys         = 0;
    VkDevice         vk_dev          = 0;
    pInteropDev->lpVtbl->GetVulkanHandles(pInteropDev, &vk_instance, &vk_phys, &vk_dev);
    pInteropDev->lpVtbl->Release(pInteropDev);
    pInteropSurf->lpVtbl->Release(pInteropSurf);

    if (!vk_instance || !vk_phys || !vk_dev) {
        ERR("[ZLUDA] GetVulkanHandles returned NULL handle (inst=%p phys=%p dev=%p)\n",
            (void*)(uintptr_t)vk_instance, (void*)(uintptr_t)vk_phys, (void*)(uintptr_t)vk_dev);
        return ZLUDA_D3D11_ERR_DXVK_INTEROP_MISSING;
    }

    if (load_vulkan_fns(vk_instance, vk_phys, vk_dev) != 0) {
        return ZLUDA_D3D11_ERR_VULKAN_LOAD;
    }

    /* Allocate a LINEAR-tiled VkBuffer with external-memory bits. We use a
     * buffer rather than an image because:
     *   - HIP's hipExternalMemoryGetMappedBuffer wants a flat dev_ptr, not
     *     a hipArray.
     *   - Batman is observed to call cuGraphicsResourceGetMappedPointer_v2
     *     (linear pointer path) — see project_d3d11_p0_resource_descriptor.md.
     *   - LINEAR tiling is the only option ROCm's external-memory import
     *     path supports robustly. */
    const uint64_t pixel_size = 16; /* R32G32B32A32_FLOAT = 4 floats * 4 bytes */
    const uint64_t pitch      = (uint64_t)tex_desc.Width * pixel_size;
    const uint64_t alloc_size = pitch * tex_desc.Height;

    VkExternalMemoryBufferCreateInfo ext_buf_info = {
        .sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .pNext       = NULL,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    };
    VkBufferCreateInfo buf_info = {
        .sType                 = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext                 = &ext_buf_info,
        .flags                 = 0,
        .size                  = alloc_size,
        .usage                 = VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                               | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                               | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices   = NULL,
    };
    VkBuffer vk_buf = 0;
    VkResult vr = g_vkCreateBuffer(vk_dev, &buf_info, NULL, &vk_buf);
    if (vr != VK_SUCCESS) {
        ERR("[ZLUDA] vkCreateBuffer failed: %d\n", vr);
        return ZLUDA_D3D11_ERR_VULKAN_CALL;
    }

    VkMemoryRequirements req = { 0 };
    g_vkGetBufferMemoryRequirements(vk_dev, vk_buf, &req);

    int mem_type = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mem_type < 0) {
        ERR("[ZLUDA] no DEVICE_LOCAL memory type found (typeBits=0x%x)\n", req.memoryTypeBits);
        g_vkDestroyBuffer(vk_dev, vk_buf, NULL);
        return ZLUDA_D3D11_ERR_NO_MEMORY_TYPE;
    }

    VkExportMemoryAllocateInfo export_info = {
        .sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        .pNext       = NULL,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    };
    VkMemoryAllocateInfo alloc_info = {
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext           = &export_info,
        .allocationSize  = req.size,
        .memoryTypeIndex = (uint32_t)mem_type,
    };
    VkDeviceMemory vk_mem = 0;
    vr = g_vkAllocateMemory(vk_dev, &alloc_info, NULL, &vk_mem);
    if (vr != VK_SUCCESS) {
        ERR("[ZLUDA] vkAllocateMemory failed: %d\n", vr);
        g_vkDestroyBuffer(vk_dev, vk_buf, NULL);
        return ZLUDA_D3D11_ERR_VULKAN_CALL;
    }

    vr = g_vkBindBufferMemory(vk_dev, vk_buf, vk_mem, 0);
    if (vr != VK_SUCCESS) {
        ERR("[ZLUDA] vkBindBufferMemory failed: %d\n", vr);
        g_vkFreeMemory(vk_dev, vk_mem, NULL);
        g_vkDestroyBuffer(vk_dev, vk_buf, NULL);
        return ZLUDA_D3D11_ERR_VULKAN_CALL;
    }

    int dma_buf_fd = -1;
    VkMemoryGetFdInfoKHR fd_info = {
        .sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
        .pNext      = NULL,
        .memory     = vk_mem,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    };
    vr = g_vkGetMemoryFdKHR(vk_dev, &fd_info, &dma_buf_fd);
    if (vr != VK_SUCCESS || dma_buf_fd < 0) {
        ERR("[ZLUDA] vkGetMemoryFdKHR failed: vr=%d fd=%d\n", vr, dma_buf_fd);
        g_vkFreeMemory(vk_dev, vk_mem, NULL);
        g_vkDestroyBuffer(vk_dev, vk_buf, NULL);
        return ZLUDA_D3D11_ERR_VULKAN_CALL;
    }

    /* From here on the kernel-allocated dma_buf has its own refcount; we can
     * destroy our VkBuffer + VkDeviceMemory and the fd remains valid until
     * libcuda.so closes it (via hipDestroyExternalMemory).
     *
     * Keeping the VkBuffer around indefinitely just to keep the memory alive
     * would tie up a Vulkan-side handle for every CUgraphicsResource which
     * is unnecessary now that the dma_buf takes a reference on export. */
    g_vkDestroyBuffer(vk_dev, vk_buf, NULL);
    g_vkFreeMemory(vk_dev, vk_mem, NULL);

    pDesc->fd    = dma_buf_fd;
    pDesc->size  = req.size;
    pDesc->pitch = (uint32_t)pitch;

    TRACE("[ZLUDA] zluda_d3d11_setup_external_memory: %ux%u %u-byte fd=%d size=%llu pitch=%u\n",
          pDesc->width, pDesc->height, pDesc->dxgi_format,
          pDesc->fd, (unsigned long long)pDesc->size, pDesc->pitch);
    return ZLUDA_D3D11_OK;
}
