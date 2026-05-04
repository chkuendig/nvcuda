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
#include <errno.h>
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
#define VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT       0x00000001
#define VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT        0x00000200

/* sType for vkGetMemoryWin32HandleKHR — required because we use the Wine-PE
 * translation path that exposes only VK_KHR_external_memory_win32 even though
 * the underlying implementation is dma_buf-based. */
#define VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR    1000073003

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

typedef struct VkMemoryGetWin32HandleInfoKHR {
    uint32_t                          sType;
    const void                       *pNext;
    VkDeviceMemory                    memory;
    VkExternalMemoryHandleTypeFlags   handleType;
} VkMemoryGetWin32HandleInfoKHR;

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
typedef VkResult (*PFN_vkGetMemoryWin32HandleKHR)(VkDevice, const VkMemoryGetWin32HandleInfoKHR*, HANDLE*);
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

static HMODULE g_vulkan_module = NULL;
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
static PFN_vkGetMemoryWin32HandleKHR           g_vkGetMemoryWin32HandleKHR = NULL;
static PFN_vkGetPhysicalDeviceMemoryProperties g_vkGetPhysicalDeviceMemoryProperties = NULL;

/* Wine internals — used to translate the NT HANDLE returned by
 * vkGetMemoryWin32HandleKHR to a Linux fd that hipImportExternalMemory can
 * consume. Wine substitutes VK_KHR_external_memory_fd with
 * VK_KHR_external_memory_win32 in the PE-visible extension list (see
 * wine/dlls/winevulkan/vulkan.c), so we must use the win32 API, then unwrap.
 *
 * wine_server_handle_to_fd lives in ntdll.so (Linux side); since this code
 * is winegcc-compiled (MS-ABI by default) we annotate the typedef with
 * sysv_abi so the calling convention matches. */
typedef int __attribute__((sysv_abi)) (*PFN_wine_server_handle_to_fd)(HANDLE, unsigned int, int*, unsigned int*);
static void *g_ntdll_so_handle = NULL;
static PFN_wine_server_handle_to_fd g_wine_server_handle_to_fd = NULL;
static VkPhysicalDeviceMemoryProperties        g_vk_mem_props;

/* Load Wine's PE-side vulkan-1.dll, NOT the Linux libvulkan.so.1.
 *
 * Subtle but critical: DXVK is also PE-side and links vulkan via Wine's
 * vulkan-1.dll thunk, which wraps every dispatchable handle (VkInstance,
 * VkDevice, VkQueue, VkCommandBuffer) with a PE-side struct that hides the
 * underlying Linux handle. Calling vkGetDeviceProcAddr on the WRAPPED
 * VkDevice through the Linux loader returns NULL because the Linux loader
 * sees an unrecognised handle layout. We must use the same PE thunk DXVK
 * uses so the dispatcher unwraps correctly.
 *
 * Equally subtle — vkGetDeviceProcAddr MUST be obtained via
 * vkGetInstanceProcAddr(NULL, "vkGetDeviceProcAddr") rather than
 * GetProcAddress directly, because the latter returns the raw export which
 * doesn't have the loader's per-device dispatch wrapping. Wine's
 * vulkan-1.dll only statically exports the core functions; KHR extensions
 * (eg vkGetMemoryFdKHR) only come back through the bootstrapped
 * vkGetDeviceProcAddr — calling the raw GetProcAddress'd version with our
 * device handle SIGSEGVs. Confirmed via iter5 [ZLUDA-vk] step trace. */
static int load_vulkan_loader(void)
{
    if (g_vulkan_module) return 0;
    /* Prefer winevulkan.dll over vulkan-1.dll: vulkan-1.dll is just a thin
     * forwarding stub that exports a limited set of entry points whereas
     * winevulkan.dll has the actual implementation and exports every
     * Vulkan function (including KHR/EXT extensions) directly. DXVK itself
     * loads vkGetInstanceProcAddr from winevulkan.dll for the same reason
     * (see DXVK info log: "Vulkan: Found vkGetInstanceProcAddr in
     * winevulkan.dll"). Fall back to vulkan-1.dll just in case. */
    g_vulkan_module = LoadLibraryA("winevulkan.dll");
    if (!g_vulkan_module) g_vulkan_module = LoadLibraryA("vulkan-1.dll");
    if (!g_vulkan_module) {
        ERR("[ZLUDA] LoadLibraryA(\"winevulkan.dll\"|\"vulkan-1.dll\") failed: %lu\n",
            (unsigned long)GetLastError());
        return -1;
    }
    g_vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)GetProcAddress(g_vulkan_module, "vkGetInstanceProcAddr");
    if (!g_vkGetInstanceProcAddr) {
        ERR("[ZLUDA] GetProcAddress(\"vkGetInstanceProcAddr\") returned NULL\n");
        return -1;
    }
    /* Defer vkGetDeviceProcAddr resolution to load_vulkan_fns — we need the
     * real VkInstance for vkGetInstanceProcAddr(inst, "vkGetDeviceProcAddr").
     * Per Vulkan spec, NULL instance only resolves global commands. */
    return 0;
}

/* Resolution strategy: winevulkan.dll exports every Vulkan function as a
 * regular DLL symbol with proper dispatch wrapping baked in. GetProcAddress
 * works directly. Fall back to vkGetDeviceProcAddr / vkGetInstanceProcAddr
 * only if a function is missing from the static export table (rare for
 * winevulkan.dll). */
#define RESOLVE_DEV(name) \
    do { \
        FIXME("[ZLUDA-vk] RESOLVE_DEV(\"%s\") begin\n", #name); \
        FARPROC __p = GetProcAddress(g_vulkan_module, #name); \
        FIXME("[ZLUDA-vk] RESOLVE_DEV(\"%s\") GetProcAddress=%p\n", #name, __p); \
        g_##name = (PFN_##name)__p; \
        if (!g_##name && g_vkGetDeviceProcAddr) { \
            FIXME("[ZLUDA-vk] RESOLVE_DEV(\"%s\") fallback dpa=%p\n", #name, g_vkGetDeviceProcAddr); \
            g_##name = (PFN_##name)g_vkGetDeviceProcAddr(dev, #name); \
        } \
        FIXME("[ZLUDA-vk] RESOLVE_DEV(\"%s\") final=%p\n", #name, g_##name); \
        if (!g_##name) { ERR("[ZLUDA] resolve(\"%s\") returned NULL\n", #name); return -1; } \
    } while (0)
#define RESOLVE_INST(name) \
    do { \
        FARPROC __p = GetProcAddress(g_vulkan_module, #name); \
        g_##name = (PFN_##name)__p; \
        if (!g_##name && g_vkGetInstanceProcAddr) \
            g_##name = (PFN_##name)g_vkGetInstanceProcAddr(inst, #name); \
        if (!g_##name) { ERR("[ZLUDA] resolve(\"%s\") returned NULL\n", #name); return -1; } \
    } while (0)

/* TEMP DEBUG: emit FIXME (visible without trace+) at each step inside
 * load_vulkan_fns so we can pinpoint where the iter4 SIGSEGV originated.
 * Remove once the crash is understood. */
#define STEP2(msg, ...) FIXME("[ZLUDA-vk] " msg "\n", ##__VA_ARGS__)

static int load_vulkan_fns(VkInstance inst, VkPhysicalDevice phys, VkDevice dev)
{
    STEP2("enter inst=%p phys=%p dev=%p", (void*)(uintptr_t)inst, (void*)(uintptr_t)phys, (void*)(uintptr_t)dev);
    if (g_vk_device == dev && g_vk_phys_device == phys) { STEP2("cached"); return 0; }
    STEP2("loader");
    if (load_vulkan_loader() != 0) return -1;
    /* Bootstrap vkGetDeviceProcAddr properly: it MUST come from
     * vkGetInstanceProcAddr(inst, ...) so it has the loader's per-device
     * dispatch wrapping. Wine's KHR extension functions (vkGetMemoryWin32HandleKHR
     * etc.) are NOT in the static PE export table — they're only reachable via
     * the dispatched vkGetDeviceProcAddr. */
    g_vkGetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)
        g_vkGetInstanceProcAddr(inst, "vkGetDeviceProcAddr");
    if (!g_vkGetDeviceProcAddr) {
        ERR("[ZLUDA] vkGetInstanceProcAddr(inst, \"vkGetDeviceProcAddr\") returned NULL\n");
        return -1;
    }
    STEP2("loader OK module=%p ipa=%p dpa=%p", g_vulkan_module, g_vkGetInstanceProcAddr, g_vkGetDeviceProcAddr);

    RESOLVE_DEV(vkCreateBuffer); STEP2("vkCreateBuffer=%p", g_vkCreateBuffer);
    RESOLVE_DEV(vkDestroyBuffer); STEP2("vkDestroyBuffer=%p", g_vkDestroyBuffer);
    RESOLVE_DEV(vkGetBufferMemoryRequirements); STEP2("vkGetBufferMemoryRequirements=%p", g_vkGetBufferMemoryRequirements);
    RESOLVE_DEV(vkAllocateMemory); STEP2("vkAllocateMemory=%p", g_vkAllocateMemory);
    RESOLVE_DEV(vkFreeMemory); STEP2("vkFreeMemory=%p", g_vkFreeMemory);
    RESOLVE_DEV(vkBindBufferMemory); STEP2("vkBindBufferMemory=%p", g_vkBindBufferMemory);
    RESOLVE_DEV(vkGetMemoryWin32HandleKHR); STEP2("vkGetMemoryWin32HandleKHR=%p", g_vkGetMemoryWin32HandleKHR);
    RESOLVE_INST(vkGetPhysicalDeviceMemoryProperties); STEP2("vkGetPhysicalDeviceMemoryProperties=%p", g_vkGetPhysicalDeviceMemoryProperties);

    /* Wine's HANDLE→fd translator. ntdll.so is already mapped into the
     * process; just look it up via the dynamic linker. The dlsym path is
     * needed because wine_server_handle_to_fd is not exported by ntdll.dll
     * (the PE-side wrapper) — only by ntdll.so. */
    if (!g_wine_server_handle_to_fd) {
        if (!g_ntdll_so_handle) g_ntdll_so_handle = dlopen("ntdll.so", RTLD_NOW | RTLD_NOLOAD);
        if (!g_ntdll_so_handle) g_ntdll_so_handle = dlopen("ntdll.so", RTLD_NOW);
        if (g_ntdll_so_handle) {
            g_wine_server_handle_to_fd = (PFN_wine_server_handle_to_fd)
                dlsym(g_ntdll_so_handle, "wine_server_handle_to_fd");
        }
        if (!g_wine_server_handle_to_fd) {
            ERR("[ZLUDA] dlsym(ntdll.so, wine_server_handle_to_fd) NULL: %s\n", dlerror());
            return -1;
        }
        STEP2("wine_server_handle_to_fd=%p", g_wine_server_handle_to_fd);
    }

    g_vk_device      = dev;
    g_vk_phys_device = phys;
    memset(&g_vk_mem_props, 0, sizeof(g_vk_mem_props));
    STEP2("calling vkGetPhysicalDeviceMemoryProperties(%p, %p)", (void*)(uintptr_t)phys, &g_vk_mem_props);
    g_vkGetPhysicalDeviceMemoryProperties(phys, &g_vk_mem_props);
    STEP2("memProps OK typeCount=%u heapCount=%u", g_vk_mem_props.memoryTypeCount, g_vk_mem_props.memoryHeapCount);
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

/* Step diagnostic — emit FIXME (visible without trace+) at each major stage so
 * we can pinpoint where setup_external_memory crashes. Remove once stable. */
#define STEP(msg, ...) FIXME("[ZLUDA-step] " msg "\n", ##__VA_ARGS__)

int zluda_d3d11_setup_external_memory(
    ID3D11Resource         *pD3DResource,
    unsigned                Flags,
    ZludaD3D11ImportedDesc *pDesc)
{
    STEP("entry pD3DResource=%p flags=%u pDesc=%p", pD3DResource, Flags, pDesc);
    if (!pD3DResource || !pDesc) return ZLUDA_D3D11_ERR_NOT_SUPPORTED;
    memset(pDesc, 0, sizeof(*pDesc));
    pDesc->fd = -1;
    pDesc->d3d_resource = pD3DResource;
    pDesc->register_flags = Flags;

    /* Only Tex2D supported in this iteration. The P0 probe has already logged
     * the descriptor type at FIXME level if it was something else; this is
     * the silent rejection step. */
    ID3D11Texture2D *pTex2D = NULL;
    STEP("QI Tex2D");
    if (FAILED(ID3D11Resource_QueryInterface(pD3DResource, &IID_ID3D11Texture2D, (void**)&pTex2D))) {
        WARN("[ZLUDA] zluda_d3d11_setup_external_memory: not a Tex2D\n");
        return ZLUDA_D3D11_ERR_NOT_SUPPORTED;
    }
    STEP("Tex2D pTex2D=%p, GetDesc", pTex2D);
    D3D11_TEXTURE2D_DESC tex_desc;
    ID3D11Texture2D_GetDesc(pTex2D, &tex_desc);
    pDesc->width       = tex_desc.Width;
    pDesc->height      = tex_desc.Height;
    pDesc->dxgi_format = tex_desc.Format;
    pDesc->bind_flags  = tex_desc.BindFlags;
    STEP("Tex2D got %ux%u fmt=%u, Release", tex_desc.Width, tex_desc.Height, tex_desc.Format);
    ID3D11Texture2D_Release(pTex2D);
    STEP("Tex2D released");

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
    STEP("QI IDXGIVkInteropSurface");
    if (FAILED(ID3D11Resource_QueryInterface(pD3DResource,
            &IID_IDXGIVkInteropSurface, (void**)&pInteropSurf))) {
        ERR("[ZLUDA] QueryInterface(IDXGIVkInteropSurface) failed — DXVK missing or wrong version\n");
        return ZLUDA_D3D11_ERR_DXVK_INTEROP_MISSING;
    }
    STEP("InteropSurface pInteropSurf=%p, GetDevice", pInteropSurf);
    IDXGIVkInteropDevice *pInteropDev = NULL;
    if (FAILED(pInteropSurf->lpVtbl->GetDevice(pInteropSurf, &pInteropDev))) {
        ERR("[ZLUDA] IDXGIVkInteropSurface::GetDevice failed\n");
        pInteropSurf->lpVtbl->Release(pInteropSurf);
        return ZLUDA_D3D11_ERR_DXVK_INTEROP_MISSING;
    }
    STEP("InteropDevice pInteropDev=%p, GetVulkanHandles", pInteropDev);
    VkInstance       vk_instance     = 0;
    VkPhysicalDevice vk_phys         = 0;
    VkDevice         vk_dev          = 0;
    pInteropDev->lpVtbl->GetVulkanHandles(pInteropDev, &vk_instance, &vk_phys, &vk_dev);
    STEP("GetVulkanHandles inst=%p phys=%p dev=%p", (void*)(uintptr_t)vk_instance, (void*)(uintptr_t)vk_phys, (void*)(uintptr_t)vk_dev);
    pInteropDev->lpVtbl->Release(pInteropDev);
    pInteropSurf->lpVtbl->Release(pInteropSurf);

    if (!vk_instance || !vk_phys || !vk_dev) {
        ERR("[ZLUDA] GetVulkanHandles returned NULL handle (inst=%p phys=%p dev=%p)\n",
            (void*)(uintptr_t)vk_instance, (void*)(uintptr_t)vk_phys, (void*)(uintptr_t)vk_dev);
        return ZLUDA_D3D11_ERR_DXVK_INTEROP_MISSING;
    }

    STEP("load_vulkan_fns");
    if (load_vulkan_fns(vk_instance, vk_phys, vk_dev) != 0) {
        return ZLUDA_D3D11_ERR_VULKAN_LOAD;
    }
    STEP("load_vulkan_fns OK");

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

    /* Wine substitutes external_memory_fd → external_memory_win32 in the
     * PE-visible extension list, so we have to use OPAQUE_WIN32_BIT here even
     * though the underlying mechanism is dma_buf. The HANDLE we get from
     * vkGetMemoryWin32HandleKHR will be unwrapped to its real Linux fd via
     * wine_server_handle_to_fd below. */
    VkExternalMemoryBufferCreateInfo ext_buf_info = {
        .sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .pNext       = NULL,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
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
    STEP("vkCreateBuffer size=%llu", (unsigned long long)alloc_size);
    VkResult vr = g_vkCreateBuffer(vk_dev, &buf_info, NULL, &vk_buf);
    STEP("vkCreateBuffer vr=%d vk_buf=%p", vr, (void*)(uintptr_t)vk_buf);
    if (vr != VK_SUCCESS) {
        ERR("[ZLUDA] vkCreateBuffer failed: %d\n", vr);
        return ZLUDA_D3D11_ERR_VULKAN_CALL;
    }

    VkMemoryRequirements req = { 0 };
    STEP("vkGetBufferMemoryRequirements");
    g_vkGetBufferMemoryRequirements(vk_dev, vk_buf, &req);
    STEP("memReqs size=%llu align=%llu typeBits=0x%x", (unsigned long long)req.size, (unsigned long long)req.alignment, req.memoryTypeBits);

    int mem_type = find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mem_type < 0) {
        ERR("[ZLUDA] no DEVICE_LOCAL memory type found (typeBits=0x%x)\n", req.memoryTypeBits);
        g_vkDestroyBuffer(vk_dev, vk_buf, NULL);
        return ZLUDA_D3D11_ERR_NO_MEMORY_TYPE;
    }

    VkExportMemoryAllocateInfo export_info = {
        .sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        .pNext       = NULL,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
    };
    VkMemoryAllocateInfo alloc_info = {
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext           = &export_info,
        .allocationSize  = req.size,
        .memoryTypeIndex = (uint32_t)mem_type,
    };
    VkDeviceMemory vk_mem = 0;
    STEP("vkAllocateMemory typeIdx=%d", mem_type);
    vr = g_vkAllocateMemory(vk_dev, &alloc_info, NULL, &vk_mem);
    STEP("vkAllocateMemory vr=%d vk_mem=%p", vr, (void*)(uintptr_t)vk_mem);
    if (vr != VK_SUCCESS) {
        ERR("[ZLUDA] vkAllocateMemory failed: %d\n", vr);
        g_vkDestroyBuffer(vk_dev, vk_buf, NULL);
        return ZLUDA_D3D11_ERR_VULKAN_CALL;
    }

    STEP("vkBindBufferMemory");
    vr = g_vkBindBufferMemory(vk_dev, vk_buf, vk_mem, 0);
    if (vr != VK_SUCCESS) {
        ERR("[ZLUDA] vkBindBufferMemory failed: %d\n", vr);
        g_vkFreeMemory(vk_dev, vk_mem, NULL);
        g_vkDestroyBuffer(vk_dev, vk_buf, NULL);
        return ZLUDA_D3D11_ERR_VULKAN_CALL;
    }

    HANDLE win32_handle = NULL;
    VkMemoryGetWin32HandleInfoKHR handle_info = {
        .sType      = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,
        .pNext      = NULL,
        .memory     = vk_mem,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
    };
    STEP("vkGetMemoryWin32HandleKHR");
    vr = g_vkGetMemoryWin32HandleKHR(vk_dev, &handle_info, &win32_handle);
    STEP("vkGetMemoryWin32HandleKHR vr=%d handle=%p", vr, win32_handle);
    if (vr != VK_SUCCESS || !win32_handle) {
        ERR("[ZLUDA] vkGetMemoryWin32HandleKHR failed: vr=%d handle=%p\n", vr, win32_handle);
        g_vkFreeMemory(vk_dev, vk_mem, NULL);
        g_vkDestroyBuffer(vk_dev, vk_buf, NULL);
        return ZLUDA_D3D11_ERR_VULKAN_CALL;
    }

    /* Translate the NT HANDLE to the underlying Linux fd. Wine wrapped the fd
     * inside an NT HANDLE during vkGetMemoryWin32HandleKHR; we now unwrap.
     * After this call, BOTH the HANDLE and the fd refer to the same kernel
     * resource — closing the HANDLE would close the fd. We dup() the fd so
     * libcuda.so can own it independently and close the HANDLE here. */
    int dma_buf_fd = -1;
    unsigned int access = 0, options = 0;
    int rc = g_wine_server_handle_to_fd(win32_handle, 0, &dma_buf_fd, &options);
    STEP("wine_server_handle_to_fd rc=%d fd=%d access=0x%x options=0x%x", rc, dma_buf_fd, access, options);
    if (rc != 0 || dma_buf_fd < 0) {
        ERR("[ZLUDA] wine_server_handle_to_fd failed: rc=%d fd=%d\n", rc, dma_buf_fd);
        CloseHandle(win32_handle);
        g_vkFreeMemory(vk_dev, vk_mem, NULL);
        g_vkDestroyBuffer(vk_dev, vk_buf, NULL);
        return ZLUDA_D3D11_ERR_VULKAN_CALL;
    }
    int owned_fd = dup(dma_buf_fd);
    if (owned_fd < 0) {
        ERR("[ZLUDA] dup(fd=%d) failed: %s\n", dma_buf_fd, strerror(errno));
        CloseHandle(win32_handle);
        g_vkFreeMemory(vk_dev, vk_mem, NULL);
        g_vkDestroyBuffer(vk_dev, vk_buf, NULL);
        return ZLUDA_D3D11_ERR_VULKAN_CALL;
    }
    CloseHandle(win32_handle);  /* releases the kernel-side wrapped reference */
    dma_buf_fd = owned_fd;

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
