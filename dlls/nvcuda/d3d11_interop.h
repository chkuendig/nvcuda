/*
 * D3D11 → ZLUDA texture interop helper.
 *
 * Defines a self-contained API used by wine_cuGraphicsD3D11RegisterResource
 * to do the Windows-COM (DXVK IDXGIVkInterop*) + Vulkan FFI dance needed to
 * convert an ID3D11Texture2D into a dma_buf fd that the Linux libcuda.so
 * (ZLUDA) can hipImportExternalMemory.
 *
 * All the heavy lifting (vkCreateBuffer, vkAllocateMemory with
 * ExportMemoryAllocateInfo, vkGetMemoryFdKHR, etc.) lives here so nvcuda.c
 * stays focused on its CUDA-relay role.
 *
 * Errors return < 0; zero means success and *pDesc->fd / handles are valid
 * and ownership of the fd has transferred to the caller.
 */

#ifndef __ZLUDA_NVCUDA_D3D11_INTEROP_H
#define __ZLUDA_NVCUDA_D3D11_INTEROP_H

#include <stdint.h>
#include "windef.h"
#include "d3d11.h"

#ifdef __cplusplus
extern "C" {
#endif

/* This struct's layout MUST match `ZludaD3D11ImportedDesc` in
 * zluda/zluda/src/impl/d3d11_interop.rs. */
typedef struct ZludaD3D11ImportedDesc {
    int      fd;             /* dma_buf fd; ownership transfers to libcuda.so */
    uint64_t size;           /* allocation size in bytes (Vulkan reqs.size) */
    uint32_t width;          /* texture width in texels */
    uint32_t height;         /* texture height in texels */
    uint32_t pitch;          /* row pitch in bytes */
    uint32_t dxgi_format;    /* DXGI_FORMAT enum (== 2 / R32G32B32A32_FLOAT) */
    uint32_t register_flags; /* CUgraphicsRegisterFlags (== 0) */
    uint32_t bind_flags;     /* D3D11_BIND_FLAG bitfield (diagnostics) */
    void    *d3d_resource;   /* original ID3D11Resource* (diagnostics) */
} ZludaD3D11ImportedDesc;

/* Result codes for zluda_d3d11_setup_external_memory. */
#define ZLUDA_D3D11_OK                       0
#define ZLUDA_D3D11_ERR_DXVK_INTEROP_MISSING -1   /* QueryInterface failed (no DXVK) */
#define ZLUDA_D3D11_ERR_NOT_SUPPORTED        -2   /* descriptor mismatch (logged) */
#define ZLUDA_D3D11_ERR_VULKAN_LOAD          -3   /* libvulkan.so.1 not found */
#define ZLUDA_D3D11_ERR_VULKAN_CALL          -4   /* a Vulkan API returned non-success */
#define ZLUDA_D3D11_ERR_NO_MEMORY_TYPE       -5   /* no exportable memory type matched */

/* Live Vulkan + DXVK objects that must outlive setup_external_memory so the
 * Map blit can run. Owned by the caller after setup_external_memory returns
 * ZLUDA_D3D11_OK; ownership transfers into the resource table on
 * zluda_d3d11_remember_resource. On failure callers must release them via
 * zluda_d3d11_release_alloc. */
typedef struct ZludaD3D11Alloc {
    void    *interop_device;   /* IDXGIVkInteropDevice2*, AddRef'd */
    uint64_t vk_buffer;
    uint64_t vk_memory;
} ZludaD3D11Alloc;

/* Inspect pD3DResource (a Tex2D), allocate a parallel exportable VkBuffer
 * + VkDeviceMemory via DXVK's IDXGIVkInteropDevice2::AllocateExternalBuffer,
 * extract the dma_buf fd via Wine's SharedGpuResource ioctl +
 * wine_server_handle_to_fd, and fill *pDesc.
 *
 * The VkBuffer + VkDeviceMemory are NOT freed here — they are kept alive so
 * later cuGraphicsMapResources calls can blit the current D3D11 image
 * contents into the same memory the imported HIP pointer reads from.
 * Caller must either pass *pAlloc to zluda_d3d11_remember_resource (transfers
 * ownership) or zluda_d3d11_release_alloc (frees) — never both, never neither.
 *
 * On success, the caller owns pDesc->fd and must either (a) close() it OR
 * (b) hand it to libcuda.so via zluda_d3d11_register_imported (which takes
 * ownership). Either way, do not call this function and then drop the fd.
 *
 * For now, supports ONLY Tex2D R32G32B32A32_FLOAT mips=1 — the canonical
 * descriptor observed from Batman: Arkham Knight (see
 * project_d3d11_p0_resource_descriptor.md). Returns
 * ZLUDA_D3D11_ERR_NOT_SUPPORTED otherwise after logging the unexpected
 * descriptor at FIXME level.
 */
int zluda_d3d11_setup_external_memory(
    ID3D11Resource         *pD3DResource,
    unsigned                Flags,           /* CUgraphicsRegisterFlags */
    ZludaD3D11ImportedDesc *pDesc,
    ZludaD3D11Alloc        *pAlloc);

/* Free pAlloc's VkBuffer/VkDeviceMemory and Release its interop_device. Use
 * when register_imported failed or any post-setup step errored. Safe to call
 * with zero-initialised pAlloc (no-op). */
void zluda_d3d11_release_alloc(ZludaD3D11Alloc *pAlloc);

/* Insert (cuda_resource → d3d_resource + alloc + pitch) into the PE-side
 * resource table. AddRefs d3d_resource for the table's own ownership;
 * pAlloc->interop_device's refcount transfers in (caller no longer owns it).
 * Subsequent Map calls will look up cuda_resource here. */
void zluda_d3d11_remember_resource(
    void                  *cuda_resource,
    ID3D11Resource        *d3d_resource,
    const ZludaD3D11Alloc *pAlloc,
    uint32_t               pitch,
    uint64_t               size);

/* Iterate `cuda_resources[0..count]`. For each entry that's in our table,
 * run IDXGIVkInteropDevice2::CopySurfaceToExternalBuffer to refresh the
 * imported memory with the current D3D11 image contents. Resources not in
 * the table (eg GL, or CUDA-only dummies that never reached our register
 * path) are silently skipped. */
void zluda_d3d11_blit_on_map(unsigned int count, void **cuda_resources);

/* Iterate `cuda_resources[0..count]`. For each entry that's in our table,
 * run IDXGIVkInteropDevice2::CopyExternalBufferToSurface to propagate
 * CUDA-written contents of the imported memory back into DXVK's image so
 * subsequent D3D11 sampling sees the new data. Resources not in the
 * table are silently skipped. Called from wine_cuGraphicsUnmapResources
 * BEFORE forwarding to libcuda.so. */
void zluda_d3d11_blit_on_unmap(unsigned int count, void **cuda_resources);

/* Remove cuda_resource from the table and release its alloc + d3d_resource.
 * Called from wine_cuGraphicsUnregisterResource. No-op for resources we
 * don't own (eg GL). */
void zluda_d3d11_forget_resource(void *cuda_resource);

#ifdef __cplusplus
}
#endif

#endif /* __ZLUDA_NVCUDA_D3D11_INTEROP_H */
