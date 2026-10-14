/* SPDX-License-Identifier: LGPL-2.1-or-later
 * Experimental real PE frontend. The existing 64-bit Winelib relay is unchanged.
 */
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "wire.h"

typedef LONG (WINAPI *query_memory_fn)(HANDLE, void *, int, void *, SIZE_T, SIZE_T *);
typedef LONG (WINAPI *unix_call_fn)(uint64_t, unsigned, void *);
static uint64_t unix_handle;
static unix_call_fn dispatcher;
static void storage_destroy(void *ctx);

BOOL WINAPI DllMain(HINSTANCE module, DWORD reason, void *reserved)
{
    (void)reserved;
    if (reason == DLL_PROCESS_ATTACH) {
        HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
        query_memory_fn query = (void *)GetProcAddress(ntdll, "NtQueryVirtualMemory");
        unix_call_fn *slot = (unix_call_fn *)GetProcAddress(ntdll, "__wine_unix_call_dispatcher");
        DisableThreadLibraryCalls(module);
        if (!query || !slot || query((HANDLE)(LONG_PTR)-1, module, 1000,
                                     &unix_handle, sizeof(unix_handle), NULL)) return FALSE;
        dispatcher = *slot;
        if (!dispatcher || !unix_handle) return FALSE;
    }
    return TRUE;
}

#define P(x) ((uint32_t)(uintptr_t)(x))
#define CALL(op, ...) do { struct cuda32_call p = {{__VA_ARGS__}}; \
    return dispatcher(unix_handle, op, &p); } while (0)

int WINAPI cuInit(unsigned flags) { CALL(C32_INIT, flags); }
int WINAPI cuDriverGetVersion(int *v) { CALL(C32_VERSION, P(v)); }
int WINAPI cuDeviceGetCount(int *v) { CALL(C32_DEVICE_COUNT, P(v)); }
int WINAPI cuDeviceGet(int *dev, int ordinal) { CALL(C32_DEVICE_GET, P(dev), ordinal); }
int WINAPI cuDeviceGetName(char *name, int length, int dev) { CALL(C32_DEVICE_NAME, P(name), length, dev); }
int WINAPI cuDeviceGetAttribute(int *value, int attr, int dev)
{
    if (attr == 0x20080402) {
        /* PhysXDevice correlates CUDA with NVAPI's physical adapter ID. Keep
         * this private namespace adaptation confined to the single-GPU
         * prototype, and obtain the actual ID from the loaded NVAPI DLL. */
        if (!value) return 1;
        int count = 0;
        int rc = cuDeviceGetCount(&count);
        if (rc) return rc;
        if (dev < 0 || dev >= count) return 101;
        if (count != 1 || dev != 0) return 801;
        HMODULE nvapi = GetModuleHandleA("nvapi.dll");
        if (!nvapi) return 801;
        void *(__cdecl *query)(uint32_t) = (void *)GetProcAddress(nvapi, "nvapi_QueryInterface");
        if (!query) return 801;
        struct topology { uint32_t version, count; struct { void *handle; uint32_t flags; } gpu[8]; } topo = {0};
        int (__cdecl *enumerate)(struct topology *) = query(0x5786cc6e);
        int (__cdecl *get_id)(void *, uint32_t *) = query(0x6533ea3e);
        if (!enumerate || !get_id) return 801;
        topo.version = sizeof(topo) | (1u << 16);
        if (enumerate(&topo) || topo.count != 1) return 801;
        uint32_t id = 0;
        if (get_id(topo.gpu[0].handle, &id)) return 801;
        memcpy(value, &id, 4);
        fprintf(stderr, "cuda32: PhysX adapter dev=%d nvapi_id=%08x\n", dev, id);
        return 0;
    }
    CALL(C32_DEVICE_ATTR, P(value), attr, dev);
}
int WINAPI cuDeviceGetProperties(void *properties, int dev) { CALL(C32_DEVICE_PROPERTIES, P(properties), dev); }
int WINAPI cuDeviceComputeCapability(int *major, int *minor, int dev)
{ CALL(C32_COMPUTE_CAPABILITY, P(major), P(minor), dev); }
int WINAPI cuDeviceTotalMem(unsigned *bytes, int dev) { CALL(C32_TOTAL_MEM, P(bytes), dev); }
int WINAPI cuDeviceTotalMem_v2(SIZE_T *bytes, int dev) { CALL(C32_TOTAL_MEM, P(bytes), dev); }
int WINAPI cuCtxPushCurrent(void *ctx) { CALL(C32_CTX_PUSH, P(ctx)); }
int WINAPI cuCtxPushCurrent_v2(void *ctx) { return cuCtxPushCurrent(ctx); }
int WINAPI cuCtxPopCurrent(void **ctx) { CALL(C32_CTX_POP, P(ctx)); }
int WINAPI cuCtxPopCurrent_v2(void **ctx) { return cuCtxPopCurrent(ctx); }
int WINAPI cuCtxCreate_v2(void **ctx, unsigned flags, int dev) { CALL(C32_CTX_CREATE, P(ctx), flags, dev); }
int WINAPI cuCtxCreate(void **ctx, unsigned flags, int dev) { return cuCtxCreate_v2(ctx, flags, dev); }
int WINAPI cuCtxDestroy_v2(void *ctx)
{
    struct cuda32_call p = {{P(ctx)}};
    int rc = dispatcher(unix_handle, C32_CTX_DESTROY, &p);
    if (!rc) storage_destroy(ctx);
    return rc;
}
int WINAPI cuCtxDestroy(void *ctx) { return cuCtxDestroy_v2(ctx); }
int WINAPI cuCtxAttach(void **ctx, unsigned flags) { CALL(C32_CTX_ATTACH, P(ctx), flags); }
int WINAPI cuCtxDetach(void *ctx)
{
    uint32_t destroyed = 0;
    struct cuda32_call p = {{P(ctx), P(&destroyed)}};
    int rc = dispatcher(unix_handle, C32_CTX_DETACH, &p);
    if (!rc && destroyed) storage_destroy(ctx);
    return rc;
}
int WINAPI cuCtxSynchronize(void) { CALL(C32_CTX_SYNC, 0); }
int WINAPI cuCtxGetCurrent(void **ctx) { CALL(C32_CTX_CURRENT, P(ctx)); }
int WINAPI cuCtxSetCurrent(void *ctx) { CALL(C32_CTX_SET, P(ctx)); }
int WINAPI cuCtxGetDevice(int *dev) { CALL(C32_CTX_DEVICE, P(dev)); }
int WINAPI cuCtxGetApiVersion(void *ctx, unsigned *version) { CALL(C32_CTX_API_VERSION, P(ctx), P(version)); }
int WINAPI cuMemAlloc_v2(uint32_t *ptr, SIZE_T bytes) { CALL(C32_ALLOC, P(ptr), bytes, 4); }
int WINAPI cuMemAlloc(uint32_t *ptr, unsigned bytes) { CALL(C32_ALLOC, P(ptr), bytes, 4); }
int WINAPI cuMemFree_v2(uint32_t ptr) { CALL(C32_FREE, (uint32_t)ptr); }
int WINAPI cuMemFree(uint32_t ptr) { CALL(C32_FREE, ptr); }
static void *host_allocations[128];
static SRWLOCK host_alloc_lock = SRWLOCK_INIT;
int WINAPI cuMemHostAlloc(void **out, SIZE_T bytes, unsigned flags)
{
    if (!out || !bytes) return 1;
    if (flags & ~1u) return 801;
    int rc = 2;
    AcquireSRWLockExclusive(&host_alloc_lock);
    for (unsigned i = 0; i < 128; ++i) if (!host_allocations[i]) {
        void *memory = VirtualAlloc(NULL, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (!memory) break;
        struct cuda32_call p = {{P(memory), bytes, flags}};
        rc = dispatcher(unix_handle, C32_HOST_REGISTER, &p);
        if (rc) VirtualFree(memory, 0, MEM_RELEASE);
        else { host_allocations[i] = memory; *out = memory; }
        break;
    }
    ReleaseSRWLockExclusive(&host_alloc_lock);
    return rc;
}
int WINAPI cuMemAllocHost(void **out, unsigned bytes) { return cuMemHostAlloc(out, bytes, 0); }
int WINAPI cuMemAllocHost_v2(void **out, SIZE_T bytes) { return cuMemHostAlloc(out, bytes, 0); }
int WINAPI cuMemFreeHost(void *memory)
{
    int rc = 1;
    AcquireSRWLockExclusive(&host_alloc_lock);
    for (unsigned i = 0; i < 128; ++i) if (memory && host_allocations[i] == memory) {
        struct cuda32_call p = {{P(memory)}};
        rc = dispatcher(unix_handle, C32_HOST_UNREGISTER, &p);
        if (!rc) { VirtualFree(memory, 0, MEM_RELEASE); host_allocations[i] = NULL; }
        break;
    }
    ReleaseSRWLockExclusive(&host_alloc_lock);
    return rc;
}
int WINAPI cuMemcpyHtoD_v2(uint32_t dst, const void *src, SIZE_T bytes)
{ CALL(C32_HTOD, (uint32_t)dst, P(src), bytes); }
int WINAPI cuMemcpyHtoD(uint32_t dst, const void *src, unsigned bytes) { CALL(C32_HTOD, dst, P(src), bytes); }
int WINAPI cuMemcpyDtoH_v2(void *dst, uint32_t src, SIZE_T bytes)
{ CALL(C32_DTOH, P(dst), (uint32_t)src, bytes); }
int WINAPI cuMemcpyDtoH(void *dst, uint32_t src, unsigned bytes) { CALL(C32_DTOH, P(dst), src, bytes); }
int WINAPI cuMemcpyHtoDAsync_v2(uint32_t dst, const void *src, SIZE_T bytes, void *stream)
{ CALL(C32_HTOD_ASYNC, (uint32_t)dst, P(src), bytes, P(stream)); }
int WINAPI cuMemcpyHtoDAsync(uint32_t dst, const void *src, unsigned bytes, void *stream)
{ CALL(C32_HTOD_ASYNC, dst, P(src), bytes, P(stream)); }
int WINAPI cuMemcpyDtoHAsync_v2(void *dst, uint32_t src, SIZE_T bytes, void *stream)
{ CALL(C32_DTOH_ASYNC, P(dst), (uint32_t)src, bytes, P(stream)); }
int WINAPI cuMemcpyDtoHAsync(void *dst, uint32_t src, unsigned bytes, void *stream)
{ CALL(C32_DTOH_ASYNC, P(dst), src, bytes, P(stream)); }
int WINAPI cuMemcpyDtoD(uint32_t dst, uint32_t src, unsigned bytes)
{ CALL(C32_DTOD, dst, src, bytes); }
int WINAPI cuMemcpyDtoD_v2(uint32_t dst, uint32_t src, SIZE_T bytes)
{ CALL(C32_DTOD, dst, src, bytes); }
int WINAPI cuMemcpyDtoDAsync(uint32_t dst, uint32_t src, unsigned bytes, void *stream)
{ CALL(C32_DTOD_ASYNC, dst, src, bytes, P(stream)); }
int WINAPI cuMemcpyDtoDAsync_v2(uint32_t dst, uint32_t src, SIZE_T bytes, void *stream)
{ CALL(C32_DTOD_ASYNC, dst, src, bytes, P(stream)); }
int WINAPI cuMemGetAddressRange(uint32_t *base, unsigned *bytes, uint32_t device)
{ CALL(C32_ADDRESS_RANGE, P(base), P(bytes), device); }
int WINAPI cuMemGetAddressRange_v2(uint32_t *base, SIZE_T *bytes, uint32_t device)
{ CALL(C32_ADDRESS_RANGE, P(base), P(bytes), device); }
int WINAPI cuEventCreate(void **event, unsigned flags) { CALL(C32_EVENT_CREATE, P(event), flags); }
int WINAPI cuEventDestroy_v2(void *event) { CALL(C32_EVENT_DESTROY, P(event)); }
int WINAPI cuEventDestroy(void *event) { return cuEventDestroy_v2(event); }
int WINAPI cuEventRecord(void *event, void *stream) { CALL(C32_EVENT_RECORD, P(event), P(stream)); }
int WINAPI cuEventSynchronize(void *event) { CALL(C32_EVENT_SYNC, P(event)); }
int WINAPI cuEventQuery(void *event) { CALL(C32_EVENT_QUERY, P(event)); }
int WINAPI cuEventElapsedTime(float *ms, void *start, void *end) { CALL(C32_EVENT_ELAPSED, P(ms), P(start), P(end)); }
int WINAPI cuStreamCreate(void **stream, unsigned flags) { CALL(C32_STREAM_CREATE, P(stream), flags); }
int WINAPI cuStreamDestroy_v2(void *stream) { CALL(C32_STREAM_DESTROY, P(stream)); }
int WINAPI cuStreamDestroy(void *stream) { return cuStreamDestroy_v2(stream); }
int WINAPI cuStreamSynchronize(void *stream) { CALL(C32_STREAM_SYNC, P(stream)); }
int WINAPI cuStreamQuery(void *stream) { CALL(C32_STREAM_QUERY, P(stream)); }
int WINAPI cuStreamWaitEvent(void *stream, void *event, unsigned flags)
{ CALL(C32_STREAM_WAIT, P(event), P(stream), flags); }
int WINAPI cuMemsetD32_v2(uint32_t dst, unsigned value, SIZE_T count)
{ CALL(C32_MEMSET, (uint32_t)dst, value, count); }
int WINAPI cuMemsetD32(uint32_t dst, unsigned value, unsigned count)
{ CALL(C32_MEMSET, dst, value, count); }
int WINAPI cuMemsetD8_v2(uint32_t dst, unsigned char value, SIZE_T count)
{ CALL(C32_MEMSET8, (uint32_t)dst, value, count); }
int WINAPI cuMemsetD8(uint32_t dst, unsigned char value, unsigned count)
{ CALL(C32_MEMSET8, dst, value, count); }
int WINAPI cuMemsetD16_v2(uint32_t dst, unsigned short value, SIZE_T count)
{ CALL(C32_MEMSET16, (uint32_t)dst, value, count); }
int WINAPI cuMemsetD16(uint32_t dst, unsigned short value, unsigned count)
{ CALL(C32_MEMSET16, dst, value, count); }
int WINAPI cuMemsetD8Async(uint32_t dst, unsigned char value, SIZE_T count, void *stream)
{ CALL(C32_MEMSET_ASYNC, (uint32_t)dst, value, count, P(stream), 1); }
int WINAPI cuMemsetD16Async(uint32_t dst, unsigned short value, SIZE_T count, void *stream)
{ CALL(C32_MEMSET_ASYNC, (uint32_t)dst, value, count, P(stream), 2); }
int WINAPI cuMemsetD32Async(uint32_t dst, unsigned value, SIZE_T count, void *stream)
{ CALL(C32_MEMSET_ASYNC, (uint32_t)dst, value, count, P(stream), 4); }
int WINAPI cuModuleLoadData(void **module, const void *image) { CALL(C32_MODULE_LOAD, P(module), P(image)); }
int WINAPI cuModuleUnload(void *module) { CALL(C32_MODULE_UNLOAD, P(module)); }
int WINAPI cuModuleGetGlobal(uint32_t *device, unsigned *bytes, void *module, const char *name)
{ CALL(C32_MODULE_GLOBAL, P(device), P(bytes), P(module), P(name), 4); }
int WINAPI cuModuleGetGlobal_v2(uint32_t *device, SIZE_T *bytes, void *module, const char *name)
{ CALL(C32_MODULE_GLOBAL, P(device), P(bytes), P(module), P(name), 4); }
int WINAPI cuModuleGetTexRef(void **tex, void *module, const char *name)
{ CALL(C32_TEX_GET, P(tex), P(module), P(name)); }
int WINAPI cuTexRefSetFormat(void *tex, int format, int channels)
{ CALL(C32_TEX_FORMAT, P(tex), format, channels); }
int WINAPI cuTexRefSetFlags(void *tex, unsigned flags) { CALL(C32_TEX_FLAGS, P(tex), flags); }
int WINAPI cuTexRefSetFilterMode(void *tex, int mode) { CALL(C32_TEX_FILTER, P(tex), mode); }
int WINAPI cuTexRefSetAddressMode(void *tex, int dim, int mode) { CALL(C32_TEX_ADDRESS_MODE, P(tex), dim, mode); }
static uint32_t float_bits(float value) { uint32_t bits; memcpy(&bits, &value, 4); return bits; }
int WINAPI cuTexRefSetMipmapFilterMode(void *tex, int mode) { CALL(C32_TEX_MIPMAP, P(tex), 0, mode); }
int WINAPI cuTexRefSetMipmapLevelBias(void *tex, float bias) { CALL(C32_TEX_MIPMAP, P(tex), 1, float_bits(bias)); }
int WINAPI cuTexRefSetMipmapLevelClamp(void *tex, float lo, float hi)
{ CALL(C32_TEX_MIPMAP, P(tex), 2, float_bits(lo), float_bits(hi)); }
int WINAPI cuTexRefSetMaxAnisotropy(void *tex, unsigned value) { CALL(C32_TEX_MIPMAP, P(tex), 3, value); }
int WINAPI cuTexRefSetAddress(SIZE_T *offset, void *tex, uint32_t device, unsigned bytes)
{ CALL(C32_TEX_ADDRESS, P(tex), device, bytes, P(offset)); }
int WINAPI cuTexRefSetAddress_v2(SIZE_T *offset, void *tex, uint32_t device, SIZE_T bytes)
{ CALL(C32_TEX_ADDRESS, P(tex), (uint32_t)device, bytes, P(offset)); }
int WINAPI cuModuleGetFunction(void **fn, void *module, const char *name)
{ CALL(C32_FUNCTION, P(fn), P(module), P(name)); }
int WINAPI cuLaunchKernel(void *fn, unsigned gx, unsigned gy, unsigned gz,
                         unsigned bx, unsigned by, unsigned bz, unsigned shared,
                         void *stream, void **params, void **extra)
{ CALL(C32_LAUNCH, P(fn), gx, gy, gz, bx, by, bz, shared, P(stream), P(params), P(extra)); }

/* PE32 function tables: each entry is a Windows stdcall thunk, never an ELF
 * function address. Slot numbers and signatures follow dark_api/src/lib.rs. */
static int WINAPI rt_load(void **m, const void *image) { return cuModuleLoadData(m, image); }
static int WINAPI rt_context(void **ctx, int dev) { CALL(C32_DARK_RUNTIME, 2, P(ctx), dev); }
static int WINAPI rt_flags(int dev, unsigned flags) { CALL(C32_DARK_RUNTIME, 4, dev, flags); }
static int WINAPI rt_device(int dev) { CALL(C32_DARK_RUNTIME, 5, dev); }
static int WINAPI rt_load_ext(void **m, const void *image, void *a, void *b, unsigned c)
{ if (a || b || c) return 801; return cuModuleLoadData(m, image); }
static void WINAPI rt_done(uint32_t value)
{ struct cuda32_call p = {{7, value}}; dispatcher(unix_handle, C32_DARK_RUNTIME, &p); }
static int WINAPI rt_load_header(const void *image, void **m, void *a, void *b, unsigned c)
{ if (a || b || c) return 801; return cuModuleLoadData(m, image); }
static int WINAPI rt_compilers(void) { CALL(C32_DARK_RUNTIME, 12); }
static const void *runtime_table[13] = {
    (const void *)(uintptr_t)(13 * sizeof(void *)), rt_load, rt_context, NULL,
    rt_flags, rt_device, rt_load_ext, rt_done, rt_load_header, NULL, NULL, NULL, rt_compilers
};
static const void *ctx_create_table[2] = {(const void *)(uintptr_t)(2 * sizeof(void *)), cuCtxCreate_v2};
/* Same per-process writable u32 buffers as upstream zluda32's DarkApi32;
 * PE storage keeps their addresses representable without truncating ELF data. */
static uint32_t hook_buffer1[1024], hook_buffer2[14];
static void WINAPI hooks_buffer1(void **buffer, SIZE_T *count)
{ if (buffer) *buffer = hook_buffer1; if (count) *count = 1024; }
static void WINAPI hooks_buffer2(void **buffer, SIZE_T *count)
{ if (buffer) *buffer = hook_buffer2; if (count) *count = 14; }
static const void *hooks_table[7] = {
    (const void *)(uintptr_t)(7 * sizeof(void *)), NULL, hooks_buffer1, NULL, NULL, NULL, hooks_buffer2
};
static const void *tls_table[4] = {(const void *)(uintptr_t)(4 * sizeof(void *)), NULL, NULL, NULL};

typedef void (WINAPI *storage_dtor)(void *, void *, void *);
struct storage_entry { void *ctx, *key, *value; storage_dtor dtor; int used; };
static struct storage_entry storage[128];
static SRWLOCK storage_lock = SRWLOCK_INIT;
static int storage_context(void **ctx)
{
    int rc;
    unsigned version;
    if (!*ctx && (rc = cuCtxGetCurrent(ctx))) return rc;
    return cuCtxGetApiVersion(*ctx, &version);
}
static int WINAPI storage_put(void *ctx, void *key, void *value, storage_dtor dtor)
{
    int rc = storage_context(&ctx); if (rc) return rc;
    int slot = -1;
    AcquireSRWLockExclusive(&storage_lock);
    for (unsigned i = 0; i < 128; ++i) {
        if (storage[i].used && storage[i].ctx == ctx && storage[i].key == key) { slot = i; break; }
        if (!storage[i].used && slot < 0) slot = i;
    }
    if (slot >= 0) storage[slot] = (struct storage_entry){ctx, key, value, dtor, 1};
    ReleaseSRWLockExclusive(&storage_lock);
    return slot < 0 ? 2 : 0;
}
static int WINAPI storage_delete(void *ctx, void *key)
{
    int rc = storage_context(&ctx); if (rc) return rc;
    rc = 1;
    AcquireSRWLockExclusive(&storage_lock);
    for (unsigned i = 0; i < 128; ++i)
        if (storage[i].used && storage[i].ctx == ctx && storage[i].key == key) {
            memset(&storage[i], 0, sizeof(storage[i])); rc = 0; break;
        }
    ReleaseSRWLockExclusive(&storage_lock);
    return rc;
}
static int WINAPI storage_get(void **value, void *ctx, void *key)
{
    if (!value) return 1;
    int rc = storage_context(&ctx); if (rc) return rc;
    rc = 1;
    AcquireSRWLockShared(&storage_lock);
    for (unsigned i = 0; i < 128; ++i)
        if (storage[i].used && storage[i].ctx == ctx && storage[i].key == key) {
            *value = storage[i].value; rc = 0; break;
        }
    ReleaseSRWLockShared(&storage_lock);
    return rc;
}
static void storage_destroy(void *ctx)
{
    struct storage_entry callbacks[128];
    unsigned n = 0;
    AcquireSRWLockExclusive(&storage_lock);
    for (unsigned i = 0; i < 128; ++i)
        if (storage[i].used && storage[i].ctx == ctx) {
            callbacks[n++] = storage[i]; memset(&storage[i], 0, sizeof(storage[i]));
        }
    ReleaseSRWLockExclusive(&storage_lock);
    for (unsigned i = 0; i < n; ++i)
        if (callbacks[i].dtor) callbacks[i].dtor(ctx, callbacks[i].key, callbacks[i].value);
}
static const void *storage_table[4] = {storage_put, storage_delete, storage_get, NULL};

/* CUDA's private heap records contain two pointer-sized words. Keep them in
 * PE-owned memory and validate frees; passing the ELF64 record to PE32 would
 * expose both the wrong layout and an unrepresentable address. */
struct heap_record { uint32_t x, y; };
static struct heap_record *heap_records[128];
static SRWLOCK heap_lock = SRWLOCK_INIT;
static int WINAPI heap_record_alloc(const void **out, SIZE_T x, SIZE_T y)
{
    if (!out) return 1;
    int rc = 2;
    AcquireSRWLockExclusive(&heap_lock);
    for (unsigned i = 0; i < 128; ++i) if (!heap_records[i]) {
        struct heap_record *record = HeapAlloc(GetProcessHeap(), 0, sizeof(*record));
        if (record) { *record = (struct heap_record){x, y}; heap_records[i] = record; *out = record; rc = 0; }
        break;
    }
    ReleaseSRWLockExclusive(&heap_lock);
    return rc;
}
static int WINAPI heap_record_free(const void *record, SIZE_T *y)
{
    int rc = 1;
    AcquireSRWLockExclusive(&heap_lock);
    for (unsigned i = 0; i < 128; ++i) if (record && heap_records[i] == record) {
        if (y) *y = heap_records[i]->y;
        HeapFree(GetProcessHeap(), 0, heap_records[i]); heap_records[i] = NULL; rc = 0; break;
    }
    ReleaseSRWLockExclusive(&heap_lock);
    return rc;
}
static const void *heap_table[3] = {(const void *)(uintptr_t)(3 * sizeof(void *)), heap_record_alloc, heap_record_free};

int WINAPI cuGetExportTable(const void **table, const unsigned char *guid)
{
    static const unsigned char runtime_guid[16] = {
        0x6b,0xd5,0xfb,0x6c,0x5b,0xf4,0xe7,0x4a,0x89,0x87,0xd9,0x39,0x12,0xfd,0x9d,0xf9
    };
    static const unsigned char hooks_guid[16] = {
        0xa0,0x94,0x79,0x8c,0x2e,0x74,0x2e,0x74,0x93,0xf2,0x08,0x00,0x20,0x0c,0x0a,0x66
    };
    static const unsigned char tls_guid[16] = {
        0x42,0xd8,0x5a,0x81,0x23,0xf6,0xcb,0x47,0x82,0x98,0xf6,0xe7,0x8a,0x3a,0xec,0xdc
    };
    static const unsigned char storage_guid[16] = {
        0xc6,0x93,0x33,0x6e,0x11,0x21,0xdf,0x11,0xa8,0xc3,0x68,0xf3,0x55,0xd8,0x95,0x93
    };
    static const unsigned char heap_guid[16] = {
        0x19,0x5b,0xcb,0xf4,0xd6,0x7d,0x02,0x4a,0xac,0xc5,0x1d,0x29,0xce,0xa6,0x31,0xae
    };
    static const unsigned char create_guid[16] = {
        0x0c,0xa5,0x0b,0x8c,0x10,0x04,0x92,0x9a,0x89,0xa7,0xd0,0xdf,0x10,0xe7,0x72,0x86
    };
    if (!table || !guid) return 1;
    *table = NULL;
    if (!memcmp(guid, runtime_guid, 16)) { *table = runtime_table; return 0; }
    if (!memcmp(guid, hooks_guid, 16)) { *table = hooks_table; return 0; }
    if (!memcmp(guid, tls_guid, 16)) { *table = tls_table; return 0; }
    if (!memcmp(guid, storage_guid, 16)) { *table = storage_table; return 0; }
    if (!memcmp(guid, heap_guid, 16)) { *table = heap_table; return 0; }
    if (!memcmp(guid, create_guid, 16)) { *table = ctx_create_table; return 0; }
    fprintf(stderr, "cuda32: export table ");
    for (unsigned i = 0; i < 16; ++i) fprintf(stderr, "%02x", guid[i]);
    fprintf(stderr, " unsupported\n");
    return 801;
}
