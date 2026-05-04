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
#include "winternl.h"
#include "winioctl.h"
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

DEFINE_GUID(IID_IDXGIVkInteropSurface,  0x5546cf8c, 0x77e7, 0x4341, 0xb0, 0x5d, 0x8d, 0x4d, 0x50, 0x00, 0xe7, 0x7d);
DEFINE_GUID(IID_IDXGIVkInteropDevice,   0xe2ef5fa5, 0xdc21, 0x4af7, 0x90, 0xc4, 0xf6, 0x7e, 0xf6, 0xa0, 0x93, 0x23);
DEFINE_GUID(IID_IDXGIVkInteropDevice1,  0xe2ef5fa5, 0xdc21, 0x4af7, 0x90, 0xc4, 0xf6, 0x7e, 0xf6, 0xa0, 0x93, 0x24);
DEFINE_GUID(IID_IDXGIVkInteropDevice2,  0xe2ef5fa5, 0xdc21, 0x4af7, 0x90, 0xc4, 0xf6, 0x7e, 0xf6, 0xa0, 0x93, 0x25);

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

/* IDXGIVkInteropDevice2 — chkuendig/dxvk@zluda-physx fork extension that
 * adds AllocateExternalBuffer / CopySurfaceToExternalBuffer /
 * FreeExternalBuffer methods. We declare ALL methods up to and including
 * the new ones so the vtable layout matches DXVK's class. Methods we
 * don't call still need to be in the right slot positions. */
struct VkImageSubresourceRange_stub { uint32_t members[6]; };
typedef struct VkCmdQueue_stub { uint64_t handle; } VkCmdQueue_stub;

typedef struct IDXGIVkInteropDeviceVtbl {
    /* IUnknown */
    HRESULT (STDMETHODCALLTYPE *QueryInterface)(IDXGIVkInteropDevice*, REFIID, void**);
    ULONG   (STDMETHODCALLTYPE *AddRef)(IDXGIVkInteropDevice*);
    ULONG   (STDMETHODCALLTYPE *Release)(IDXGIVkInteropDevice*);
    /* IDXGIVkInteropDevice (slots 3..8) */
    void    (STDMETHODCALLTYPE *GetVulkanHandles)(IDXGIVkInteropDevice*, VkInstance*, VkPhysicalDevice*, VkDevice*);
    void    (STDMETHODCALLTYPE *GetSubmissionQueue)(IDXGIVkInteropDevice*, uint64_t*, uint32_t*);
    void    (STDMETHODCALLTYPE *TransitionSurfaceLayout)(IDXGIVkInteropDevice*, void*, void*, uint32_t, uint32_t);
    void    (STDMETHODCALLTYPE *FlushRenderingCommands)(IDXGIVkInteropDevice*);
    void    (STDMETHODCALLTYPE *LockSubmissionQueue)(IDXGIVkInteropDevice*);
    void    (STDMETHODCALLTYPE *ReleaseSubmissionQueue)(IDXGIVkInteropDevice*);
    /* IDXGIVkInteropDevice1 (slots 9..10) */
    void    (STDMETHODCALLTYPE *GetSubmissionQueue1)(IDXGIVkInteropDevice*, uint64_t*, uint32_t*, uint32_t*);
    HRESULT (STDMETHODCALLTYPE *CreateTexture2DFromVkImage)(IDXGIVkInteropDevice*, void*, VkImage, void**);
    /* IDXGIVkInteropDevice2 — ZLUDA fork extension (slots 11..13) */
    HRESULT (STDMETHODCALLTYPE *AllocateExternalBuffer)(IDXGIVkInteropDevice*,
        uint64_t Size, HANDLE* pHandle, uint64_t* pBufferOut, uint64_t* pMemoryOut, uint64_t* pAllocSizeOut);
    HRESULT (STDMETHODCALLTYPE *CopySurfaceToExternalBuffer)(IDXGIVkInteropDevice*,
        IDXGIVkInteropSurface* pSrc, uint64_t Dst, uint64_t DstSize, unsigned int DstRowPitch);
    void    (STDMETHODCALLTYPE *FreeExternalBuffer)(IDXGIVkInteropDevice*,
        uint64_t Buffer, uint64_t Memory);
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
static PFN_vkGetMemoryFdKHR                    g_vkGetMemoryFdKHR __attribute__((unused)) = NULL;
static PFN_vkGetMemoryWin32HandleKHR           g_vkGetMemoryWin32HandleKHR __attribute__((unused)) = NULL;
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
/* Resolution: GetProcAddress first (covers the core fns winevulkan.dll
 * statically exports), fall back to vkGetInstanceProcAddr(inst, fn) which
 * works for ANY Vulkan command per spec — instance, physical-device, OR
 * device-level — at the cost of an extra dispatch hop. Wine's KHR
 * extension functions only resolve via this path. */
#define RESOLVE_DEV(name) RESOLVE_INST(name)
#define RESOLVE_INST(name) \
    do { \
        FIXME("[ZLUDA-vk] RESOLVE(\"%s\") begin\n", #name); \
        g_##name = (PFN_##name)GetProcAddress(g_vulkan_module, #name); \
        FIXME("[ZLUDA-vk] RESOLVE(\"%s\") GetProcAddress=%p\n", #name, g_##name); \
        if (!g_##name) \
            g_##name = (PFN_##name)g_vkGetInstanceProcAddr(inst, #name); \
        FIXME("[ZLUDA-vk] RESOLVE(\"%s\") final=%p\n", #name, g_##name); \
        if (!g_##name) { ERR("[ZLUDA] resolve(\"%s\") returned NULL\n", #name); return -1; } \
    } while (0)

/* TEMP DEBUG: emit FIXME (visible without trace+) at each step inside
 * load_vulkan_fns so we can pinpoint where the iter4 SIGSEGV originated.
 * Remove once the crash is understood. */
#define STEP2(msg, ...) FIXME("[ZLUDA-vk] " msg "\n", ##__VA_ARGS__)

__attribute__((unused))
static int load_vulkan_fns(VkInstance inst, VkPhysicalDevice phys, VkDevice dev)
{
    STEP2("enter inst=%p phys=%p dev=%p", (void*)(uintptr_t)inst, (void*)(uintptr_t)phys, (void*)(uintptr_t)dev);
    if (g_vk_device == dev && g_vk_phys_device == phys) { STEP2("cached"); return 0; }
    STEP2("loader");
    if (load_vulkan_loader() != 0) return -1;
    /* Wine's winevulkan.dll doesn't expose vkGetDeviceProcAddr through
     * vkGetInstanceProcAddr(inst, ...) — that path returns NULL. Per
     * Vulkan spec, vkGetInstanceProcAddr(inst, fn) IS valid for ANY
     * function (instance/physical-device/device-level), with the cost
     * of an extra dispatch hop. Use it directly for everything. */
    g_vkGetDeviceProcAddr = NULL;  /* not used in this iteration */
    STEP2("loader OK module=%p ipa=%p (skipping dpa bootstrap)",
          g_vulkan_module, g_vkGetInstanceProcAddr);

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
__attribute__((unused))
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

    /* Query DXVK's IDXGIVkInteropSurface, then its IDXGIVkInteropDevice2
     * (the chkuendig/dxvk@zluda-physx fork-only extension that adds
     * AllocateExternalBuffer). The whole Vulkan dance happens inside
     * DXVK, where the per-device function pointers are correctly resolved;
     * application PE-side code can't reach vkGetMemoryWin32HandleKHR
     * directly (Wine's loader filters it out, GetProcAddress returns NULL,
     * vkGetInstanceProcAddr also returns NULL). */
    IDXGIVkInteropSurface *pInteropSurf = NULL;
    STEP("QI IDXGIVkInteropSurface");
    if (FAILED(ID3D11Resource_QueryInterface(pD3DResource,
            &IID_IDXGIVkInteropSurface, (void**)&pInteropSurf))) {
        ERR("[ZLUDA] QueryInterface(IDXGIVkInteropSurface) failed — DXVK missing\n");
        return ZLUDA_D3D11_ERR_DXVK_INTEROP_MISSING;
    }
    IDXGIVkInteropDevice *pInteropDev = NULL;
    if (FAILED(pInteropSurf->lpVtbl->GetDevice(pInteropSurf, &pInteropDev))) {
        ERR("[ZLUDA] IDXGIVkInteropSurface::GetDevice failed\n");
        pInteropSurf->lpVtbl->Release(pInteropSurf);
        return ZLUDA_D3D11_ERR_DXVK_INTEROP_MISSING;
    }
    pInteropSurf->lpVtbl->Release(pInteropSurf);

    /* QI for the fork-extended interface. If this fails we're running with
     * upstream DXVK without our patch — degrade gracefully to NOT_SUPPORTED
     * (caller will fall back to pcuGraphicsD3D11RegisterResource which
     * returns NOT_SUPPORTED, and Batman has a CPU fallback). */
    IDXGIVkInteropDevice *pInteropDev2 = NULL;
    STEP("QI IDXGIVkInteropDevice2");
    HRESULT hr = pInteropDev->lpVtbl->QueryInterface(pInteropDev,
        &IID_IDXGIVkInteropDevice2, (void**)&pInteropDev2);
    pInteropDev->lpVtbl->Release(pInteropDev);
    if (FAILED(hr) || !pInteropDev2) {
        ERR("[ZLUDA] QueryInterface(IDXGIVkInteropDevice2) failed — chkuendig/dxvk fork required\n");
        return ZLUDA_D3D11_ERR_DXVK_INTEROP_MISSING;
    }

    /* Pixel size + size for canonical R32G32B32A32_FLOAT */
    const uint64_t pixel_size = 16;
    const uint64_t pitch      = (uint64_t)tex_desc.Width * pixel_size;
    const uint64_t alloc_size = pitch * tex_desc.Height;

    HANDLE         win32_handle = NULL;
    uint64_t       vk_buffer    = 0;
    uint64_t       vk_memory    = 0;
    uint64_t       actual_size  = 0;

    STEP("AllocateExternalBuffer size=%llu", (unsigned long long)alloc_size);
    hr = pInteropDev2->lpVtbl->AllocateExternalBuffer(pInteropDev2,
        alloc_size, &win32_handle, &vk_buffer, &vk_memory, &actual_size);
    STEP("AllocateExternalBuffer hr=0x%x handle=%p buf=0x%llx mem=0x%llx size=%llu",
         hr, win32_handle, (unsigned long long)vk_buffer,
         (unsigned long long)vk_memory, (unsigned long long)actual_size);
    if (FAILED(hr) || !win32_handle) {
        ERR("[ZLUDA] AllocateExternalBuffer failed: hr=0x%x\n", hr);
        pInteropDev2->lpVtbl->Release(pInteropDev2);
        return ZLUDA_D3D11_ERR_VULKAN_CALL;
    }

    /* Translate the SharedGpuResource HANDLE to the underlying Linux fd.
     *
     * Wine's vkGetMemoryWin32HandleKHR doesn't return a raw NT-section
     * HANDLE — it returns a HANDLE backed by Wine's `\\??\\SharedGpuResource`
     * device driver. To get the real fd we must ioctl the driver with
     * IOCTL_SHARED_GPU_RESOURCE_GET_UNIX_RESOURCE first, which gives us a
     * wine_server-managed HANDLE, and only THEN can wine_server_handle_to_fd
     * unwrap that to the Linux fd. See Wine's own
     * winevulkan/vulkan.c::get_shared_resource_fd() — same two-step. */
    if (!g_wine_server_handle_to_fd) {
        if (!g_ntdll_so_handle) g_ntdll_so_handle = dlopen("ntdll.so", RTLD_NOW | RTLD_NOLOAD);
        if (!g_ntdll_so_handle) g_ntdll_so_handle = dlopen("ntdll.so", RTLD_NOW);
        if (g_ntdll_so_handle) {
            g_wine_server_handle_to_fd = (PFN_wine_server_handle_to_fd)
                dlsym(g_ntdll_so_handle, "wine_server_handle_to_fd");
        }
        if (!g_wine_server_handle_to_fd) {
            ERR("[ZLUDA] dlsym(ntdll.so, wine_server_handle_to_fd) NULL: %s\n", dlerror());
            pInteropDev2->lpVtbl->FreeExternalBuffer(pInteropDev2, vk_buffer, vk_memory);
            CloseHandle(win32_handle);
            pInteropDev2->lpVtbl->Release(pInteropDev2);
            return ZLUDA_D3D11_ERR_VULKAN_LOAD;
        }
    }

    /* IOCTL definition from wine/dlls/sharedgpures.sys/shared_resource.c.
     * FILE_DEVICE_VIDEO=0x23, function=3, METHOD_BUFFERED=0, FILE_READ_DATA=0x1
     *   CTL_CODE = (DeviceType<<16) | (Access<<14) | (Function<<2) | Method
     *            = (0x23<<16) | (1<<14) | (3<<2) | 0
     *            = 0x230000 | 0x4000 | 0xC | 0
     *            = 0x23400C */
    #define IOCTL_SHARED_GPU_RESOURCE_GET_UNIX_RESOURCE  0x23400C
    int dma_buf_fd = -1;
    {
        IO_STATUS_BLOCK iosb;
        unsigned int unix_handle = 0;  /* obj_handle_t in Wine; 32-bit */
        STEP("NtDeviceIoControlFile GET_UNIX_RESOURCE");
        NTSTATUS st = NtDeviceIoControlFile(win32_handle, NULL, NULL, NULL, &iosb,
                                            IOCTL_SHARED_GPU_RESOURCE_GET_UNIX_RESOURCE,
                                            NULL, 0, &unix_handle, sizeof(unix_handle));
        STEP("ioctl status=%lx unix_handle=0x%x", (unsigned long)st, unix_handle);
        if (st != 0 /* STATUS_SUCCESS */) {
            ERR("[ZLUDA] IOCTL_SHARED_GPU_RESOURCE_GET_UNIX_RESOURCE failed: 0x%lx\n", (unsigned long)st);
            pInteropDev2->lpVtbl->FreeExternalBuffer(pInteropDev2, vk_buffer, vk_memory);
            CloseHandle(win32_handle);
            pInteropDev2->lpVtbl->Release(pInteropDev2);
            return ZLUDA_D3D11_ERR_VULKAN_CALL;
        }

        /* unix_handle is an obj_handle_t (32-bit) — promote to HANDLE for
         * wine_server_handle_to_fd. Wine's helper wine_server_ptr_handle
         * does just `(HANDLE)(ULONG_PTR)h`. */
        HANDLE unix_resource = (HANDLE)(uintptr_t)unix_handle;

        unsigned int options = 0;
        int          rc = g_wine_server_handle_to_fd(unix_resource, 1 /* FILE_READ_DATA */,
                                                     &dma_buf_fd, &options);
        STEP("wine_server_handle_to_fd rc=%d fd=%d", rc, dma_buf_fd);
        NtClose(unix_resource);
        if (rc != 0 || dma_buf_fd < 0) {
            ERR("[ZLUDA] wine_server_handle_to_fd failed: rc=%d fd=%d\n", rc, dma_buf_fd);
            pInteropDev2->lpVtbl->FreeExternalBuffer(pInteropDev2, vk_buffer, vk_memory);
            CloseHandle(win32_handle);
            pInteropDev2->lpVtbl->Release(pInteropDev2);
            return ZLUDA_D3D11_ERR_VULKAN_CALL;
        }
    }

    /* dup so libcuda.so owns its own fd. wine_server_handle_to_fd may not
     * always give us an exclusive fd; dup is cheap insurance. */
    int owned_fd = dup(dma_buf_fd);
    if (owned_fd < 0) {
        ERR("[ZLUDA] dup(fd=%d) failed: %s\n", dma_buf_fd, strerror(errno));
        pInteropDev2->lpVtbl->FreeExternalBuffer(pInteropDev2, vk_buffer, vk_memory);
        CloseHandle(win32_handle);
        pInteropDev2->lpVtbl->Release(pInteropDev2);
        return ZLUDA_D3D11_ERR_VULKAN_CALL;
    }
    CloseHandle(win32_handle);  /* releases the wrapper kernel ref */

    /* Free the VkBuffer + VkDeviceMemory now. The dma_buf has its own
     * refcount so it stays alive until libcuda.so closes the fd. */
    pInteropDev2->lpVtbl->FreeExternalBuffer(pInteropDev2, vk_buffer, vk_memory);
    pInteropDev2->lpVtbl->Release(pInteropDev2);

    pDesc->fd    = owned_fd;
    pDesc->size  = actual_size;
    pDesc->pitch = (uint32_t)pitch;

    FIXME("[ZLUDA] zluda_d3d11_setup_external_memory: %ux%u fmt=%u fd=%d size=%llu pitch=%u\n",
          pDesc->width, pDesc->height, pDesc->dxgi_format,
          pDesc->fd, (unsigned long long)pDesc->size, pDesc->pitch);
    return ZLUDA_D3D11_OK;
}
