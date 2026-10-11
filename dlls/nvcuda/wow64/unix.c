/* SPDX-License-Identifier: LGPL-2.1-or-later
 * Bounded compute prototype, not a complete CUDA implementation. Checked
 * contexts/streams, bounded module globals and linear textures. Reuses ZLUDA's
 * upstream 32-bit compiler and appends its required hidden pool-base argument.
 */
#define _GNU_SOURCE /* dladdr */
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wire.h"

#define INVALID 1
#define OOM 2
#define INVALID_CONTEXT 201
#define INVALID_HANDLE 400
#define UNSUPPORTED 801
#define POOL_BYTES (512u * 1024 * 1024)
#define MAX_HANDLES 2048
#define MAX_ALLOCS 256
#define MAX_ARGS 32
#define MAX_GLOBALS 32
#define MAX_CONTEXTS 16
typedef int (*entry_fn)(void *);
struct arg_info { uint32_t size, align; };
struct dark_table {
    int (*globals)(char **, const void **, uint32_t *, uint32_t *, uint32_t *, void *);
    int (*args)(struct arg_info *, uint32_t *, void *);
};
static struct dark_table *dark;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static void *library;
static uint32_t generation;
/* ZLUDA_WOW64_TRACE=1 logs every call; =2 also logs launch arguments. */
static int trace_level(void)
{
    static int level = -1;
    if (level < 0) { const char *t = getenv("ZLUDA_WOW64_TRACE"); level = t ? atoi(t) : 0; if (level < 0) level = 0; }
    return level;
}
struct global { char *name; uint32_t offset, bytes; };
struct handle {
    void *ptr; char *name; uint32_t id, owner, ctx_id, nargs, nglobals; int type;
    struct arg_info args[MAX_ARGS]; struct global globals[MAX_GLOBALS];
};
static struct handle handles[MAX_HANDLES];
struct allocation { uint32_t offset, bytes, reserved, owner; };
struct context_state {
    void *native;
    uint64_t pool_base;
    uint32_t id, refs;
    struct allocation ranges[MAX_ALLOCS];
};
static struct context_state contexts[MAX_CONTEXTS], *active;
/* The dispatcher mutex protects this selected view. It is resolved from
 * the backend's thread-current context for every resource operation. */
#define context (active->native)
#define pool (active->pool_base)
#define context_refs (active->refs)
#define allocations (active->ranges)

#define API(name, ...) static int (*p_##name)(__VA_ARGS__)
API(cuInit, unsigned);
API(cuDriverGetVersion, int *);
API(cuDeviceGetCount, int *);
API(cuDeviceGet, int *, int);
API(cuDeviceGetName, char *, int, int);
API(cuDeviceGetAttribute, int *, int, int);
API(cuDeviceGetProperties, void *, int);
API(cuDeviceComputeCapability, int *, int *, int);
API(cuDeviceTotalMem_v2, size_t *, int);
API(cuCtxCreate_v2, void **, unsigned, int);
API(cuCtxDestroy_v2, void *);
API(cuCtxGetCurrent, void **);
API(cuCtxSetCurrent, void *);
API(cuCtxGetDevice, int *);
API(cuCtxGetApiVersion, void *, unsigned *);
API(cuCtxPushCurrent_v2, void *);
API(cuCtxPopCurrent_v2, void **);
API(cuCtxSynchronize, void);
API(cuMemAlloc_v2, uint64_t *, size_t);
API(cuMemFree_v2, uint64_t);
API(cuMemcpyHtoD_v2, uint64_t, const void *, size_t);
API(cuMemcpyDtoH_v2, void *, uint64_t, size_t);
API(cuMemcpyHtoDAsync_v2, uint64_t, const void *, size_t, void *);
API(cuMemcpyDtoHAsync_v2, void *, uint64_t, size_t, void *);
API(cuMemcpyDtoD_v2, uint64_t, uint64_t, size_t);
API(cuMemcpyDtoDAsync_v2, uint64_t, uint64_t, size_t, void *);
API(cuEventCreate, void **, unsigned);
API(cuEventDestroy_v2, void *);
API(cuEventRecord, void *, void *);
API(cuEventSynchronize, void *);
API(cuEventQuery, void *);
API(cuEventElapsedTime, float *, void *, void *);
API(cuStreamCreate, void **, unsigned);
API(cuStreamDestroy_v2, void *);
API(cuStreamSynchronize, void *);
API(cuStreamQuery, void *);
API(cuStreamWaitEvent, void *, void *, unsigned);
API(cuMemsetD32_v2, uint64_t, unsigned, size_t);
API(cuMemsetD8_v2, uint64_t, unsigned char, size_t);
API(cuMemsetD16_v2, uint64_t, unsigned short, size_t);
API(cuMemsetD8Async, uint64_t, unsigned char, size_t, void *);
API(cuMemsetD16Async, uint64_t, unsigned short, size_t, void *);
API(cuMemsetD32Async, uint64_t, unsigned, size_t, void *);
API(cuModuleLoadData, void **, const void *);
API(cuModuleUnload, void *);
API(cuModuleGetFunction, void **, void *, const char *);
API(cuModuleGetTexRef, void **, void *, const char *);
API(cuTexRefSetFormat, void *, int, int);
API(cuTexRefSetFlags, void *, unsigned);
API(cuTexRefSetFilterMode, void *, int);
API(cuTexRefSetAddressMode, void *, int, int);
API(cuTexRefSetMipmapFilterMode, void *, int);
API(cuTexRefSetMipmapLevelBias, void *, float);
API(cuTexRefSetMipmapLevelClamp, void *, float, float);
API(cuTexRefSetMaxAnisotropy, void *, unsigned);
API(cuTexRefSetAddress_v2, size_t *, void *, uint64_t, size_t);
API(cuLaunchKernel, void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
    unsigned, void *, void **, void **);
API(cuGetExportTable, const void **, const void *);

static void *ptr(uint32_t p) { return (void *)(uintptr_t)p; }
/* ZLUDA translates PTX, not NVIDIA ELF machine code. Its release loader can
 * currently return an empty module for a SASS-only fatbin. Reject that case
 * with CUDA's actual no-binary result before it can become a metadata error.
 * Unknown or malformed containers remain the backend's responsibility. */
static int fatbin_has_ptx(const void *data)
{
    const unsigned char *b = data;
    uint32_t magic;
    uint16_t version, header;
    uint64_t bytes;
    memcpy(&magic, b, 4); memcpy(&version, b + 4, 2);
    memcpy(&header, b + 6, 2); memcpy(&bytes, b + 8, 8);
    if (magic != 0xba55ed50 || version != 1 || header < 16 ||
        bytes > 64u * 1024 * 1024) return -1;
    b += header;
    int ptx = 0;
    while (bytes) {
        uint16_t kind, entry_version;
        uint32_t entry_header;
        uint64_t payload;
        if (bytes < 16) return -1;
        memcpy(&kind, b, 2); memcpy(&entry_version, b + 2, 2);
        memcpy(&entry_header, b + 4, 4); memcpy(&payload, b + 8, 8);
        if ((entry_version != 0x100 && entry_version != 0x101) ||
            (kind != 1 && kind != 2) || entry_header < 16 ||
            entry_header > bytes || payload > bytes - entry_header) return -1;
        if (kind == 1) ptx = 1;
        uint64_t step = entry_header + payload;
        b += step; bytes -= step;
    }
    return ptx;
}
static int load_backend(void)
{
    if (library) return 0;
    /* ZLUDA's libcuda.so sits next to this library in the compat tool
     * (files/lib/wine/x86_64-unix); ZLUDA_WOW64_LIBRARY overrides it. */
    char own[4096];
    const char *path = getenv("ZLUDA_WOW64_LIBRARY");
    if (!path || !path[0]) {
        Dl_info info;
        const char *slash;
        if (!dladdr((void *)load_backend, &info) || !info.dli_fname ||
            !(slash = strrchr(info.dli_fname, '/')) ||
            snprintf(own, sizeof(own), "%.*s/libcuda.so", (int)(slash - info.dli_fname),
                     info.dli_fname) >= (int)sizeof(own)) return UNSUPPORTED;
        path = own;
    }
    if (path[0] != '/') return UNSUPPORTED;
    void *lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!lib) { const char *err = dlerror(); fprintf(stderr, "nvcuda wow64: dlopen %s failed: %s\n", path, err ? err : "(null)"); return UNSUPPORTED; }
    fprintf(stderr, "nvcuda wow64: loaded %s\n", path);
#define LOAD(name) do { *(void **)(&p_##name) = dlsym(lib, #name); \
    if (!p_##name) { dlclose(lib); return UNSUPPORTED; } } while (0)
    LOAD(cuInit); LOAD(cuDriverGetVersion); LOAD(cuDeviceGetCount); LOAD(cuDeviceGet);
    LOAD(cuDeviceGetName); LOAD(cuDeviceGetAttribute); LOAD(cuCtxCreate_v2);
    LOAD(cuDeviceGetProperties); LOAD(cuDeviceComputeCapability); LOAD(cuDeviceTotalMem_v2);
    LOAD(cuCtxPushCurrent_v2); LOAD(cuCtxPopCurrent_v2);
    LOAD(cuCtxDestroy_v2); LOAD(cuCtxGetCurrent); LOAD(cuCtxSetCurrent); LOAD(cuCtxSynchronize);
    LOAD(cuCtxGetDevice); LOAD(cuCtxGetApiVersion);
    LOAD(cuMemAlloc_v2); LOAD(cuMemFree_v2); LOAD(cuMemcpyHtoD_v2); LOAD(cuMemcpyDtoH_v2);
    LOAD(cuMemsetD32_v2); LOAD(cuModuleLoadData); LOAD(cuModuleUnload);
    LOAD(cuModuleGetFunction); LOAD(cuLaunchKernel); LOAD(cuGetExportTable);
    LOAD(cuModuleGetTexRef); LOAD(cuTexRefSetFormat); LOAD(cuTexRefSetFlags); LOAD(cuTexRefSetAddress_v2);
    LOAD(cuTexRefSetFilterMode); LOAD(cuTexRefSetAddressMode);
    LOAD(cuTexRefSetMipmapFilterMode); LOAD(cuTexRefSetMipmapLevelBias);
    LOAD(cuTexRefSetMipmapLevelClamp); LOAD(cuTexRefSetMaxAnisotropy);
    LOAD(cuEventCreate); LOAD(cuEventDestroy_v2); LOAD(cuEventRecord); LOAD(cuEventSynchronize);
    LOAD(cuEventQuery); LOAD(cuEventElapsedTime); LOAD(cuStreamCreate); LOAD(cuStreamDestroy_v2);
    LOAD(cuStreamSynchronize); LOAD(cuStreamQuery); LOAD(cuStreamWaitEvent);
    LOAD(cuMemcpyHtoDAsync_v2); LOAD(cuMemcpyDtoHAsync_v2);
    LOAD(cuMemcpyDtoD_v2); LOAD(cuMemcpyDtoDAsync_v2);
    LOAD(cuMemsetD8_v2); LOAD(cuMemsetD16_v2);
    LOAD(cuMemsetD8Async); LOAD(cuMemsetD16Async); LOAD(cuMemsetD32Async);
#undef LOAD
    library = lib;
    return 0;
}
static struct handle *get_handle(uint32_t id, int type)
{
    for (unsigned i = 0; i < MAX_HANDLES; ++i)
        if (handles[i].id == id && handles[i].type == type &&
            (type == 1 || type == 4 || (active && handles[i].ctx_id == active->id)))
            return &handles[i];
    return NULL;
}
static struct handle *new_handle(void *value, int type)
{
    for (unsigned i = 0; i < MAX_HANDLES; ++i) if (!handles[i].type) {
        if (generation == 0x00ffffff) return NULL;
        struct handle *h = &handles[i];
        memset(h, 0, sizeof(*h));
        h->ptr = value; h->type = type; h->id = 0x40000000u | (++generation << 4);
        h->ctx_id = (type == 1 || type == 4) ? h->id : active->id;
        return h;
    }
    return NULL;
}
static int current_context(void)
{
    void *actual = NULL;
    active = NULL;
    int rc = p_cuCtxGetCurrent(&actual);
    if (rc) return rc;
    for (unsigned i = 0; i < MAX_CONTEXTS; ++i)
        if (actual && contexts[i].native == actual) { active = &contexts[i]; return 0; }
    return INVALID_CONTEXT;
}
static struct context_state *context_slot(void *native)
{
    for (unsigned i = 0; i < MAX_CONTEXTS; ++i)
        if (contexts[i].native == native) return &contexts[i];
    for (unsigned i = 0; i < MAX_CONTEXTS; ++i)
        if (!contexts[i].native) return &contexts[i];
    return NULL;
}
static int destroy_context(struct handle *h)
{
    struct context_state *state = context_slot(h->ptr);
    if (!state || state->native != h->ptr) return INVALID_CONTEXT;
    void *previous = NULL, *popped = NULL;
    int rc = p_cuCtxGetCurrent(&previous);
    if (rc) return rc;
    if (previous != h->ptr) {
        rc = p_cuCtxPushCurrent_v2(h->ptr);
        if (rc) return rc;
    }
    active = state;
    if (pool) {
        rc = p_cuMemFree_v2(pool);
        if (!rc) pool = 0;
    }
    if (previous != h->ptr) {
        int restore = p_cuCtxPopCurrent_v2(&popped);
        if (!rc) rc = restore;
    }
    if (rc) return rc;
    rc = p_cuCtxDestroy_v2(h->ptr);
    if (rc) return rc;
    for (unsigned i = 0; i < MAX_HANDLES; ++i)
        if (handles[i].type && handles[i].ctx_id == state->id) {
            if (handles[i].type == 2)
                for (unsigned j = 0; j < handles[i].nglobals; ++j)
                    free(handles[i].globals[j].name);
            free(handles[i].name);
            memset(&handles[i], 0, sizeof(handles[i]));
        }
    memset(state, 0, sizeof(*state));
    active = NULL;
    return 0;
}
static int translate(uint32_t offset, uint32_t bytes, uint64_t *out)
{
    if (!offset || !bytes) return INVALID;
    for (unsigned i = 0; i < MAX_ALLOCS; ++i) {
        struct allocation *a = &allocations[i];
        if (a->bytes && offset >= a->offset && offset - a->offset < a->bytes &&
            bytes <= a->bytes - (offset - a->offset)) {
            *out = pool + offset; return 0;
        }
    }
    return INVALID;
}
static int alloc_range(uint32_t requested, uint32_t owner, uint32_t *result)
{
    if (!requested) return INVALID;
    if (requested > POOL_BYTES - 256) return OOM;
    unsigned slot;
    for (slot = 0; slot < MAX_ALLOCS && allocations[slot].bytes; ++slot) {}
    if (slot == MAX_ALLOCS) return OOM;
    uint32_t bytes = (requested + 255u) & ~255u, offset = 256;
    for (;;) {
        uint32_t next = offset;
        if (offset > POOL_BYTES - bytes) return OOM;
        for (unsigned i = 0; i < MAX_ALLOCS; ++i) {
            struct allocation *a = &allocations[i];
            if (a->bytes && offset < a->offset + a->reserved && offset + bytes > a->offset)
                if (next < a->offset + a->reserved) next = a->offset + a->reserved;
        }
        if (next == offset) break;
        offset = next;
    }
    if (!pool) {
        int rc = p_cuMemAlloc_v2(&pool, POOL_BYTES); if (rc) return rc;
        if (trace_level()) fprintf(stderr, "cuda32: pool ctx=%u base=%016llx bytes=%08x\n", active->id, (unsigned long long)pool, POOL_BYTES);
    }
    allocations[slot] = (struct allocation){offset, requested, bytes, owner};
    if (trace_level()) fprintf(stderr, "cuda32: alloc ctx=%u offset=%08x bytes=%08x reserved=%08x owner=%u\n", active->id, offset, requested, bytes, owner);
    *result = offset;
    return 0;
}
static int alloc32(struct cuda32_call *p)
{
    if (!p->a[0] || p->a[2] != 4) return INVALID;
    uint32_t offset;
    int rc = alloc_range(p->a[1], 0, &offset); if (rc) return rc;
    memcpy(ptr(p->a[0]), &offset, 4);
    return 0;
}
static void release_globals(struct handle *h)
{
    for (unsigned i = 0; i < h->nglobals; ++i) free(h->globals[i].name);
    for (unsigned i = 0; i < MAX_ALLOCS; ++i)
        if (allocations[i].owner == h->id) memset(&allocations[i], 0, sizeof(allocations[i]));
    h->nglobals = 0;
}
static int init_globals(struct handle *h, uint32_t count)
{
    if (count > MAX_GLOBALS) return UNSUPPORTED;
    char *names[MAX_GLOBALS]; const void *initializers[MAX_GLOBALS];
    uint32_t sizes[MAX_GLOBALS], alignments[MAX_GLOBALS], returned = count;
    if (!count) return 0;
    int rc = dark->globals(names, initializers, sizes, alignments, &returned, h->ptr);
    if (rc) return rc;
    if (returned != count) return UNSUPPORTED;
    for (unsigned i = 0; i < count; ++i) {
        if (!names[i] || !sizes[i] || !initializers[i] || !alignments[i] ||
            alignments[i] > 256 || (alignments[i] & (alignments[i] - 1))) return UNSUPPORTED;
        char *name = strdup(names[i]); if (!name) return OOM;
        uint32_t offset;
        rc = alloc_range(sizes[i], h->id, &offset);
        if (rc) { free(name); return rc; }
        h->globals[h->nglobals++] = (struct global){name, offset, sizes[i]};
        rc = p_cuMemcpyHtoD_v2(pool + offset, initializers[i], sizes[i]);
        if (rc) return rc;
    }
    return 0;
}
static int dispatch(unsigned op, struct cuda32_call *p)
{
    int rc = load_backend();
    if (rc) return rc;
    switch (op) {
    case C32_INIT: return p_cuInit(p->a[0]);
    case C32_VERSION: return p->a[0] ? p_cuDriverGetVersion(ptr(p->a[0])) : INVALID;
    case C32_DEVICE_COUNT: return p->a[0] ? p_cuDeviceGetCount(ptr(p->a[0])) : INVALID;
    case C32_DEVICE_GET: return p->a[0] ? p_cuDeviceGet(ptr(p->a[0]), p->a[1]) : INVALID;
    case C32_DEVICE_NAME: return p->a[0] ? p_cuDeviceGetName(ptr(p->a[0]), p->a[1], p->a[2]) : INVALID;
    case C32_DEVICE_ATTR: return p->a[0] ? p_cuDeviceGetAttribute(ptr(p->a[0]), p->a[1], p->a[2]) : INVALID;
    /* CUdevprop contains only 32-bit ints; its layout is identical on both sides. */
    case C32_DEVICE_PROPERTIES: return p->a[0] ? p_cuDeviceGetProperties(ptr(p->a[0]), p->a[1]) : INVALID;
    case C32_COMPUTE_CAPABILITY:
        return p->a[0] && p->a[1] ? p_cuDeviceComputeCapability(ptr(p->a[0]), ptr(p->a[1]), p->a[2]) : INVALID;
    case C32_TOTAL_MEM: {
        if (!p->a[0]) return INVALID;
        size_t total = 0;
        rc = p_cuDeviceTotalMem_v2(&total, p->a[1]); if (rc) return rc;
        /* A PE32 size_t is four bytes even for the v2 entry point. Report
         * the bounded prototype pool's usable capacity, not unlimited VRAM. */
        uint32_t available = total < POOL_BYTES - 256 ? (uint32_t)total : POOL_BYTES - 256;
        memcpy(ptr(p->a[0]), &available, 4); return 0;
    }
    case C32_CTX_PUSH: {
        struct handle *h = get_handle(p->a[0], 1);
        if (!h) h = get_handle(p->a[0], 4);
        if (!h) return INVALID_HANDLE;
        return p_cuCtxPushCurrent_v2(h->ptr);
    }
    case C32_CTX_POP: {
        if (!p->a[0]) return INVALID;
        void *value = NULL;
        rc = p_cuCtxPopCurrent_v2(&value); if (rc) return rc;
        for (unsigned i = 0; i < MAX_HANDLES; ++i)
            if ((handles[i].type == 1 || handles[i].type == 4) && handles[i].ptr == value) {
                memcpy(ptr(p->a[0]), &handles[i].id, 4); return 0;
            }
        return INVALID_HANDLE;
    }
    case C32_CTX_CURRENT: {
        if (!p->a[0]) return INVALID;
        uint32_t id = 0;
        void *value = NULL;
        rc = p_cuCtxGetCurrent(&value); if (rc) return rc;
        for (unsigned i = 0; i < MAX_HANDLES; ++i)
            if ((handles[i].type == 1 || handles[i].type == 4) && handles[i].ptr == value)
                id = handles[i].id;
        memcpy(ptr(p->a[0]), &id, 4); return 0;
    }
    case C32_CTX_SET: {
        if (!p->a[0]) return p_cuCtxSetCurrent(NULL);
        struct handle *h = get_handle(p->a[0], 1);
        if (!h) h = get_handle(p->a[0], 4);
        return h ? p_cuCtxSetCurrent(h->ptr) : INVALID_CONTEXT;
    }
    case C32_CTX_API_VERSION: {
        struct handle *h = get_handle(p->a[0], 1);
        if (!h) h = get_handle(p->a[0], 4);
        if (!h) return INVALID_CONTEXT;
        return p->a[1] ? p_cuCtxGetApiVersion(h->ptr, ptr(p->a[1])) : INVALID;
    }
    case C32_DARK_RUNTIME: {
        static const unsigned char guid[16] = {
            0x6b,0xd5,0xfb,0x6c,0x5b,0xf4,0xe7,0x4a,0x89,0x87,0xd9,0x39,0x12,0xfd,0x9d,0xf9
        };
        const void **table = NULL;
        rc = p_cuGetExportTable((const void **)&table, guid); if (rc) return rc;
        unsigned slot = p->a[0];
        if (!table || (uintptr_t)table[0] < 13 * sizeof(void *) || slot >= 13 || !table[slot]) return UNSUPPORTED;
        switch (slot) {
        case 2: {
            if (!p->a[1]) return INVALID;
            void *native = NULL;
            int (*fn)(void **, int) = (void *)table[slot];
            rc = fn(&native, p->a[2]); if (rc) return rc;
            struct handle *h = NULL;
            for (unsigned i = 0; i < MAX_HANDLES; ++i)
                if ((handles[i].type == 1 || handles[i].type == 4) && handles[i].ptr == native) h = &handles[i];
            if (!h) h = new_handle(native, 4); /* borrowed primary context */
            if (!h) return OOM;
            struct context_state *state = context_slot(native);
            if (!state) return OOM;
            if (!state->native) { state->native = native; state->id = h->id; }
            memcpy(ptr(p->a[1]), &h->id, 4); return 0;
        }
        case 4: {
            int (*fn)(int, unsigned) = (void *)table[slot]; return fn(p->a[1], p->a[2]);
        }
        case 5: {
            int (*fn)(int) = (void *)table[slot]; return fn(p->a[1]);
        }
        case 7: {
            void (*fn)(uintptr_t) = (void *)table[slot]; fn(p->a[1]); return 0;
        }
        case 12: {
            int (*fn)(void) = (void *)table[slot]; return fn();
        }
        default: return UNSUPPORTED;
        }
    }
    case C32_CTX_CREATE: {
        if (!p->a[0]) return INVALID;
        struct context_state *state = context_slot(NULL);
        if (!state) return OOM;
        void *value = NULL;
        rc = p_cuCtxCreate_v2(&value, p->a[1], p->a[2]); if (rc) return rc;
        struct handle *h = new_handle(value, 1);
        if (!h) { p_cuCtxDestroy_v2(value); return OOM; }
        state->native = value; state->id = h->id; state->refs = 1;
        active = state;
        memcpy(ptr(p->a[0]), &h->id, 4); return 0;
    }
    case C32_CTX_DESTROY: {
        struct handle *h = get_handle(p->a[0], 1);
        return h ? destroy_context(h) : INVALID_HANDLE;
    }
    }
    rc = current_context(); if (rc) return rc;
    switch (op) {
    case C32_CTX_DEVICE: return p->a[0] ? p_cuCtxGetDevice(ptr(p->a[0])) : INVALID;
    case C32_CTX_ATTACH: {
        if (!p->a[0] || p->a[1] || context_refs == UINT32_MAX) return INVALID;
        for (unsigned i = 0; i < MAX_HANDLES; ++i)
            if (handles[i].type == 1 && handles[i].ptr == context) {
                ++context_refs;
                memcpy(ptr(p->a[0]), &handles[i].id, 4); return 0;
            }
        return INVALID_CONTEXT;
    }
    case C32_CTX_DETACH: {
        if (p->a[1]) { uint32_t zero = 0; memcpy(ptr(p->a[1]), &zero, 4); }
        struct handle *h = get_handle(p->a[0], 1);
        if (!h || h->ptr != context || !context_refs) return INVALID_CONTEXT;
        if (context_refs > 1) { --context_refs; return 0; }
        /* The bridge owns this native context. Last detach follows the
         * same checked destruction path as explicit destroy, not a no-op. */
        rc = destroy_context(h);
        if (!rc && p->a[1]) { uint32_t one = 1; memcpy(ptr(p->a[1]), &one, 4); }
        return rc;
    }
    case C32_CTX_SYNC: return p_cuCtxSynchronize();
    case C32_ALLOC: return alloc32(p);
    case C32_FREE:
        for (unsigned i = 0; i < MAX_ALLOCS; ++i) if (allocations[i].bytes && allocations[i].offset == p->a[0]) {
            if (allocations[i].owner) return INVALID;
            rc = p_cuCtxSynchronize();
            if (trace_level()) fprintf(stderr, "cuda32: free ctx=%u offset=%08x bytes=%08x rc=%d\n", active->id, allocations[i].offset, allocations[i].bytes, rc);
            if (!rc) memset(&allocations[i], 0, sizeof(allocations[i]));
            return rc;
        }
        return INVALID;
    case C32_MEMSET8: case C32_MEMSET16: case C32_MEMSET_ASYNC: {
        unsigned width = op == C32_MEMSET8 ? 1 : op == C32_MEMSET16 ? 2 : p->a[4];
        if (width != 1 && width != 2 && width != 4) return INVALID;
        if (p->a[2] > UINT32_MAX / width || (p->a[0] & (width - 1))) return INVALID;
        struct handle *stream = op == C32_MEMSET_ASYNC && p->a[3] ? get_handle(p->a[3], 6) : NULL;
        if (op == C32_MEMSET_ASYNC && p->a[3] && !stream) return INVALID_HANDLE;
        if (!p->a[2]) return 0;
        uint64_t address;
        rc = translate(p->a[0], p->a[2] * width, &address); if (rc) return rc;
        if (op == C32_MEMSET8) return p_cuMemsetD8_v2(address, p->a[1], p->a[2]);
        if (op == C32_MEMSET16) return p_cuMemsetD16_v2(address, p->a[1], p->a[2]);
        void *s = stream ? stream->ptr : NULL;
        if (width == 1) return p_cuMemsetD8Async(address, p->a[1], p->a[2], s);
        if (width == 2) return p_cuMemsetD16Async(address, p->a[1], p->a[2], s);
        return p_cuMemsetD32Async(address, p->a[1], p->a[2], s);
    }
    case C32_HTOD: case C32_DTOH: case C32_MEMSET: {
        uint64_t address;
        uint32_t offset = op == C32_DTOH ? p->a[1] : p->a[0], bytes = p->a[2];
        if (op == C32_MEMSET) { if (bytes > UINT32_MAX / 4 || (offset & 3)) return INVALID; bytes *= 4; }
        if (!bytes) return 0;
        rc = translate(offset, bytes, &address); if (rc) return rc;
        if (op == C32_MEMSET) return p_cuMemsetD32_v2(address, p->a[1], p->a[2]);
        if (!(op == C32_HTOD ? p->a[1] : p->a[0])) return INVALID;
        return op == C32_HTOD ? p_cuMemcpyHtoD_v2(address, ptr(p->a[1]), bytes)
                              : p_cuMemcpyDtoH_v2(ptr(p->a[0]), address, bytes);
    }
    case C32_MODULE_LOAD: {
        if (!p->a[0] || !p->a[1]) return INVALID;
        void *module = NULL;
        const void *image = ptr(p->a[1]);
        uint32_t magic;
        memcpy(&magic, image, 4);
        struct { uint32_t magic, version; const void *data, *extra; } wrapper;
        if (magic == 0x466243b1) {
            const uint32_t *w = image;
            if (w[1] != 1) return UNSUPPORTED; /* v2 contains another pointer array */
            wrapper.magic = magic; wrapper.version = w[1];
            wrapper.data = ptr(w[2]); wrapper.extra = ptr(w[3]);
            if (!wrapper.data) return INVALID;
            const char *capture = getenv("ZLUDA_WOW64_FATBIN_CAPTURE");
            if (capture && capture[0] == '/') {
                uint32_t header_magic;
                uint16_t header_size;
                uint64_t file_bytes;
                memcpy(&header_magic, wrapper.data, 4);
                memcpy(&header_size, (const char *)wrapper.data + 6, 2);
                memcpy(&file_bytes, (const char *)wrapper.data + 8, 8);
                if (header_magic == 0xba55ed50 && header_size >= 16 &&
                    file_bytes <= 64u * 1024 * 1024) {
                    static unsigned capture_id;
                    char filename[1024];
                    int length = snprintf(filename, sizeof(filename), "%s/module-%u.fatbin", capture, capture_id++);
                    if (length > 0 && (size_t)length < sizeof(filename)) {
                        FILE *file = fopen(filename, "wb");
                        if (file) {
                            fwrite(wrapper.data, 1, header_size + (size_t)file_bytes, file);
                            fclose(file);
                        }
                    }
                }
            }
            if (fatbin_has_ptx(wrapper.data) == 0) {
                fprintf(stderr, "cuda32: fatbin contains no PTX: CUDA_ERROR_NO_BINARY_FOR_GPU\n");
                return 209;
            }
            image = &wrapper;
        }
        rc = p_cuModuleLoadData(&module, image);
        fprintf(stderr, "cuda32: module image=%08x backend_load=%d\n", magic, rc);
        if (rc) return rc;
        static const unsigned char guid[16] = {0xEE,0x37,0x6A,0x1B,0xB8,0x58,0x42,0xEB,0xA5,0x2D,0x9A,0x10,0x6D,0x50,0xD9,0x14};
        rc = p_cuGetExportTable((const void **)&dark, guid);
        if (!rc && (!dark || !dark->globals || !dark->args)) rc = UNSUPPORTED;
        uint32_t count = 0;
        if (!rc) rc = dark->globals(NULL, NULL, NULL, NULL, &count, module);
        fprintf(stderr, "cuda32: module metadata32=%d globals=%u\n", rc, count);
        if (rc) { p_cuModuleUnload(module); return rc; }
        struct handle *h = new_handle(module, 2);
        if (!h) { p_cuModuleUnload(module); return OOM; }
        rc = init_globals(h, count);
        if (rc) { release_globals(h); memset(h, 0, sizeof(*h)); p_cuModuleUnload(module); return rc; }
        memcpy(ptr(p->a[0]), &h->id, 4); return 0;
    }
    case C32_MODULE_UNLOAD: {
        struct handle *h = get_handle(p->a[0], 2); if (!h) return INVALID_HANDLE;
        rc = p_cuCtxSynchronize(); if (rc) return rc;
        rc = p_cuModuleUnload(h->ptr);
        if (!rc) {
            for (unsigned i = 0; i < MAX_HANDLES; ++i)
                if ((handles[i].type == 3 || handles[i].type == 5) && handles[i].owner == h->id) {
                    free(handles[i].name);
                    memset(&handles[i], 0, sizeof(handles[i]));
                }
            release_globals(h); memset(h, 0, sizeof(*h));
        }
        return rc;
    }
    case C32_MODULE_GLOBAL: {
        struct handle *m = get_handle(p->a[2], 2); if (!m) return INVALID_HANDLE;
        if (!p->a[3] || (!p->a[0] && !p->a[1]) || p->a[4] != 4) return INVALID;
        for (unsigned i = 0; i < m->nglobals; ++i)
            if (!strcmp(m->globals[i].name, ptr(p->a[3]))) {
                if (p->a[0]) memcpy(ptr(p->a[0]), &m->globals[i].offset, 4);
                if (p->a[1]) memcpy(ptr(p->a[1]), &m->globals[i].bytes, 4);
                return 0;
            }
        return 500;
    }
    case C32_TEX_GET: {
        if (!p->a[0] || !p->a[2]) return INVALID;
        struct handle *m = get_handle(p->a[1], 2); if (!m) return INVALID_HANDLE;
        for (unsigned i = 0; i < MAX_HANDLES; ++i)
            if (handles[i].type == 5 && handles[i].owner == m->id &&
                !strcmp(handles[i].name, ptr(p->a[2]))) {
                memcpy(ptr(p->a[0]), &handles[i].id, 4); return 0;
            }
        char *name = strdup(ptr(p->a[2])); if (!name) return OOM;
        void *tex = NULL;
        rc = p_cuModuleGetTexRef(&tex, m->ptr, name); if (rc) { free(name); return rc; }
        struct handle *h = new_handle(tex, 5);
        if (!h) { free(name); return OOM; }
        h->owner = m->id; h->name = name; memcpy(ptr(p->a[0]), &h->id, 4); return 0;
    }
    case C32_TEX_MIPMAP: {
        struct handle *h = get_handle(p->a[0], 5); if (!h) return INVALID_HANDLE;
        float lo, hi; memcpy(&lo, &p->a[2], 4); memcpy(&hi, &p->a[3], 4);
        switch (p->a[1]) {
        case 0: return p->a[2] <= 1 ? p_cuTexRefSetMipmapFilterMode(h->ptr, p->a[2]) : INVALID;
        case 1: return p_cuTexRefSetMipmapLevelBias(h->ptr, lo);
        case 2: return p_cuTexRefSetMipmapLevelClamp(h->ptr, lo, hi);
        case 3: return p_cuTexRefSetMaxAnisotropy(h->ptr, p->a[2]);
        default: return INVALID;
        }
    }
    case C32_TEX_FORMAT: case C32_TEX_FLAGS: case C32_TEX_ADDRESS: case C32_TEX_FILTER: case C32_TEX_ADDRESS_MODE: {
        struct handle *h = get_handle(p->a[0], 5); if (!h) return INVALID_HANDLE;
        if (op == C32_TEX_FILTER) return p->a[1] <= 1 ? p_cuTexRefSetFilterMode(h->ptr, p->a[1]) : INVALID;
        if (op == C32_TEX_ADDRESS_MODE) return p->a[1] <= 2 && p->a[2] <= 3 ? p_cuTexRefSetAddressMode(h->ptr, p->a[1], p->a[2]) : INVALID;
        if (op == C32_TEX_FORMAT) return p_cuTexRefSetFormat(h->ptr, p->a[1], p->a[2]);
        if (op == C32_TEX_FLAGS) return p_cuTexRefSetFlags(h->ptr, p->a[1]);
        uint64_t address = 0;
        if (p->a[1]) { rc = translate(p->a[1], p->a[2] ? p->a[2] : 1, &address); if (rc) return rc; }
        size_t offset = 0;
        rc = p_cuTexRefSetAddress_v2(&offset, h->ptr, address, p->a[2]);
        if (trace_level()) fprintf(stderr, "cuda32: texbind ref=%08x offset=%08x bytes=%08x address=%016llx rc=%d\n", p->a[0], p->a[1], p->a[2], (unsigned long long)address, rc);
        if (rc) return rc;
        if (offset > UINT32_MAX) return UNSUPPORTED;
        if (p->a[3]) { uint32_t narrow = offset; memcpy(ptr(p->a[3]), &narrow, 4); }
        return 0;
    }
    case C32_HOST_REGISTER: case C32_HOST_UNREGISTER: {
        /* Resolve HIP through the already loaded ZLUDA dependency closure,
         * never a second runtime. Wine allocates representable PE32 memory;
         * HIP pins it for real asynchronous transfers. Mapped device pointers
         * and write-combined allocations are deliberately unsupported. */
        static int (*register_host)(void *, size_t, unsigned);
        static int (*unregister_host)(void *);
        if (!register_host) *(void **)(&register_host) = dlsym(library, "hipHostRegister");
        if (!unregister_host) *(void **)(&unregister_host) = dlsym(library, "hipHostUnregister");
        if (!register_host || !unregister_host) return UNSUPPORTED;
        if (!p->a[0]) return INVALID;
        if (op == C32_HOST_REGISTER) {
            if (!p->a[1]) return INVALID;
            if (p->a[2] & ~1u) return UNSUPPORTED;
            rc = register_host(ptr(p->a[0]), p->a[1], p->a[2]);
            if (trace_level()) fprintf(stderr, "cuda32: hostreg ptr=%08x bytes=%08x flags=%08x rc=%d\n", p->a[0], p->a[1], p->a[2], rc);
        } else {
            void *active = NULL;
            rc = p_cuCtxGetCurrent(&active); if (rc) return rc;
            if (active) { rc = p_cuCtxSynchronize(); if (rc) return rc; }
            rc = unregister_host(ptr(p->a[0]));
        }
        switch (rc) { /* common HIP/CUDA values; unknown runtime errors stay errors */
        case 0: case 1: case 2: case 100: case 101: case 201: case 712: case 713: case 801: return rc;
        default: fprintf(stderr, "cuda32: HIP host registration error=%d\n", rc); return 999;
        }
    }
    case C32_EVENT_CREATE: case C32_STREAM_CREATE: {
        if (!p->a[0]) return INVALID;
        void *native = NULL;
        rc = op == C32_EVENT_CREATE ? p_cuEventCreate(&native, p->a[1]) : p_cuStreamCreate(&native, p->a[1]);
        if (rc) return rc;
        struct handle *h = new_handle(native, op == C32_EVENT_CREATE ? 7 : 6);
        if (!h) { if (op == C32_EVENT_CREATE) p_cuEventDestroy_v2(native); else p_cuStreamDestroy_v2(native); return OOM; }
        memcpy(ptr(p->a[0]), &h->id, 4); return 0;
    }
    case C32_EVENT_DESTROY: case C32_EVENT_SYNC: case C32_EVENT_QUERY: {
        struct handle *h = get_handle(p->a[0], 7); if (!h) return INVALID_HANDLE;
        if (op == C32_EVENT_SYNC) return p_cuEventSynchronize(h->ptr);
        if (op == C32_EVENT_QUERY) return p_cuEventQuery(h->ptr);
        rc = p_cuEventDestroy_v2(h->ptr); if (!rc) memset(h, 0, sizeof(*h)); return rc;
    }
    case C32_STREAM_DESTROY: case C32_STREAM_SYNC: case C32_STREAM_QUERY: {
        struct handle *h = p->a[0] ? get_handle(p->a[0], 6) : NULL;
        if (p->a[0] && !h) return INVALID_HANDLE;
        if (op == C32_STREAM_SYNC) return p_cuStreamSynchronize(h ? h->ptr : NULL);
        if (op == C32_STREAM_QUERY) return p_cuStreamQuery(h ? h->ptr : NULL);
        if (!h) return INVALID_HANDLE;
        rc = p_cuStreamDestroy_v2(h->ptr); if (!rc) memset(h, 0, sizeof(*h)); return rc;
    }
    case C32_EVENT_RECORD: case C32_STREAM_WAIT: {
        struct handle *event = get_handle(p->a[0], 7);
        /* CUDA explicitly allows waiting on an event from another context. */
        if (!event && op == C32_STREAM_WAIT)
            for (unsigned i = 0; i < MAX_HANDLES; ++i)
                if (handles[i].id == p->a[0] && handles[i].type == 7) event = &handles[i];
        struct handle *stream = p->a[1] ? get_handle(p->a[1], 6) : NULL;
        if (!event || (p->a[1] && !stream)) return INVALID_HANDLE;
        return op == C32_EVENT_RECORD ? p_cuEventRecord(event->ptr, stream ? stream->ptr : NULL) :
            p_cuStreamWaitEvent(stream ? stream->ptr : NULL, event->ptr, p->a[2]);
    }
    case C32_EVENT_ELAPSED: {
        struct handle *start = get_handle(p->a[1], 7), *end = get_handle(p->a[2], 7);
        if (!start || !end) return INVALID_HANDLE;
        return p->a[0] ? p_cuEventElapsedTime(ptr(p->a[0]), start->ptr, end->ptr) : INVALID;
    }
    case C32_ADDRESS_RANGE: {
        if (!p->a[0] && !p->a[1]) return INVALID;
        for (unsigned i = 0; i < MAX_ALLOCS; ++i) {
            struct allocation *a = &allocations[i];
            if (a->bytes && p->a[2] >= a->offset && p->a[2] - a->offset < a->bytes) {
                if (p->a[0]) memcpy(ptr(p->a[0]), &a->offset, 4);
                if (p->a[1]) memcpy(ptr(p->a[1]), &a->bytes, 4);
                return 0;
            }
        }
        return INVALID;
    }
    case C32_DTOD: case C32_DTOD_ASYNC: {
        struct handle *stream = op == C32_DTOD_ASYNC && p->a[3] ? get_handle(p->a[3], 6) : NULL;
        if (op == C32_DTOD_ASYNC && p->a[3] && !stream) return INVALID_HANDLE;
        if (!p->a[2]) return 0;
        uint64_t dst, src;
        rc = translate(p->a[0], p->a[2], &dst); if (rc) return rc;
        rc = translate(p->a[1], p->a[2], &src); if (rc) return rc;
        return op == C32_DTOD ? p_cuMemcpyDtoD_v2(dst, src, p->a[2]) :
            p_cuMemcpyDtoDAsync_v2(dst, src, p->a[2], stream ? stream->ptr : NULL);
    }
    case C32_HTOD_ASYNC: case C32_DTOH_ASYNC: {
        struct handle *stream = p->a[3] ? get_handle(p->a[3], 6) : NULL;
        if (p->a[3] && !stream) return INVALID_HANDLE;
        if (!p->a[2]) return 0;
        uint32_t offset = op == C32_HTOD_ASYNC ? p->a[0] : p->a[1];
        uint32_t host = op == C32_HTOD_ASYNC ? p->a[1] : p->a[0];
        if (!host) return INVALID;
        uint64_t address;
        rc = translate(offset, p->a[2], &address); if (rc) return rc;
        return op == C32_HTOD_ASYNC ? p_cuMemcpyHtoDAsync_v2(address, ptr(host), p->a[2], stream ? stream->ptr : NULL) :
            p_cuMemcpyDtoHAsync_v2(ptr(host), address, p->a[2], stream ? stream->ptr : NULL);
    }
    case C32_FUNCTION: {
        if (!p->a[0] || !p->a[2]) return INVALID;
        struct handle *m = get_handle(p->a[1], 2); if (!m) return INVALID_HANDLE;
        void *function = NULL;
        rc = p_cuModuleGetFunction(&function, m->ptr, ptr(p->a[2])); if (rc) return rc;
        uint32_t count = 0;
        rc = dark->args(NULL, &count, function); if (rc) return rc;
        if (count > MAX_ARGS) return UNSUPPORTED;
        struct handle *h = new_handle(function, 3); if (!h) return OOM;
        h->owner = m->id; h->nargs = count;
        rc = dark->args(h->args, &count, function);
        if (rc) { memset(h, 0, sizeof(*h)); return rc; }
        if (trace_level()) fprintf(stderr, "cuda32: function id=%08x module=%08x nargs=%u name=%s\n", h->id, m->id, count, (const char *)ptr(p->a[2]));
        memcpy(ptr(p->a[0]), &h->id, 4); return 0;
    }
    case C32_LAUNCH: {
        struct handle *h = get_handle(p->a[0], 3); if (!h) return INVALID_HANDLE;
        if (p->a[9] && p->a[10]) return INVALID;
        struct handle *stream = p->a[8] ? get_handle(p->a[8], 6) : NULL;
        if (p->a[8] && !stream) return INVALID_HANDLE;
        if (!pool || (h->nargs && !p->a[9] && !p->a[10])) return INVALID;
        uint32_t *source = ptr(p->a[9]);
        struct handle *m = get_handle(h->owner, 2); if (!m) return INVALID_HANDLE;
        void *params[MAX_ARGS + 1 + MAX_GLOBALS];
        if (p->a[10]) {
            uint32_t *extra = ptr(p->a[10]), buffer = 0, bytes = 0;
            int have_buffer = 0, have_size = 0, ended = 0;
            for (unsigned i = 0; i < 16; i += 2) {
                if (!extra[i]) { ended = 1; break; }
                if (extra[i] == 1 && !have_buffer) { buffer = extra[i+1]; have_buffer = 1; }
                else if (extra[i] == 2 && !have_size && extra[i+1]) {
                    memcpy(&bytes, ptr(extra[i+1]), 4); have_size = 1;
                } else return INVALID;
            }
            if (!ended || !have_buffer || !have_size || (!buffer && h->nargs) || bytes > UINT32_MAX - buffer) return INVALID;
            uint32_t offset = 0;
            for (unsigned i = 0; i < h->nargs; ++i) {
                uint32_t align = h->args[i].align, size = h->args[i].size;
                if (!align || (align & (align - 1)) || offset > UINT32_MAX - (align - 1)) return INVALID;
                offset = (offset + align - 1) & ~(align - 1);
                if (offset > bytes || size > bytes - offset) return INVALID;
                params[i] = ptr(buffer + offset); offset += size;
            }
        } else {
            for (unsigned i = 0; i < h->nargs; ++i) { if (!source[i]) return INVALID; params[i] = ptr(source[i]); }
        }
        params[h->nargs] = &pool;
        for (unsigned i = 0; i < m->nglobals; ++i) params[h->nargs + 1 + i] = &m->globals[i].offset;
        if (trace_level() >= 2) {
            fprintf(stderr, "cuda32: launch fn=%08x grid=%u,%u,%u block=%u,%u,%u shared=%u stream=%08x args=", p->a[0], p->a[1], p->a[2], p->a[3], p->a[4], p->a[5], p->a[6], p->a[7], p->a[8]);
            for (unsigned i = 0; i < h->nargs; ++i) {
                const unsigned char *b = params[i];
                unsigned n = h->args[i].size > 16 ? 16 : h->args[i].size;
                /* little-endian value, most significant byte first */
                for (unsigned k = n; k-- > 0;) fprintf(stderr, "%02x", b[k]);
                fputc(i + 1 < h->nargs ? ',' : '\n', stderr);
            }
            if (!h->nargs) fputc('\n', stderr);
        }
        return p_cuLaunchKernel(h->ptr, p->a[1], p->a[2], p->a[3], p->a[4], p->a[5], p->a[6],
                               p->a[7], stream ? stream->ptr : NULL, params, NULL);
    }
    default: return UNSUPPORTED;
    }
}

static int call(unsigned op, void *args)
{
    struct cuda32_call snapshot = *(struct cuda32_call *)args;
    pthread_mutex_lock(&lock);
    int rc = dispatch(op, args);
    pthread_mutex_unlock(&lock);
    if (trace_level() >= 1)
        fprintf(stderr, "cuda32: op=%u args=%08x,%08x,%08x result=%d\n",
                op, snapshot.a[0], snapshot.a[1], snapshot.a[2], rc);
    return rc;
}
#define ENTRY(name, op) static int name(void *p) { return call(op, p); }
ENTRY(init,C32_INIT) ENTRY(version,C32_VERSION) ENTRY(count,C32_DEVICE_COUNT)
ENTRY(device,C32_DEVICE_GET) ENTRY(name,C32_DEVICE_NAME) ENTRY(attr,C32_DEVICE_ATTR)
ENTRY(create,C32_CTX_CREATE) ENTRY(destroy,C32_CTX_DESTROY) ENTRY(sync_ctx,C32_CTX_SYNC)
ENTRY(alloc,C32_ALLOC) ENTRY(release,C32_FREE) ENTRY(htod,C32_HTOD) ENTRY(dtoh,C32_DTOH)
ENTRY(memset32,C32_MEMSET) ENTRY(load,C32_MODULE_LOAD) ENTRY(unload,C32_MODULE_UNLOAD)
ENTRY(function,C32_FUNCTION) ENTRY(launch,C32_LAUNCH)
ENTRY(properties,C32_DEVICE_PROPERTIES) ENTRY(capability,C32_COMPUTE_CAPABILITY)
ENTRY(total_mem,C32_TOTAL_MEM) ENTRY(push_ctx,C32_CTX_PUSH) ENTRY(pop_ctx,C32_CTX_POP)
ENTRY(attach_ctx,C32_CTX_ATTACH) ENTRY(detach_ctx,C32_CTX_DETACH)
ENTRY(current_ctx,C32_CTX_CURRENT) ENTRY(device_ctx,C32_CTX_DEVICE) ENTRY(api_ctx,C32_CTX_API_VERSION) ENTRY(runtime_dark,C32_DARK_RUNTIME)
ENTRY(module_global,C32_MODULE_GLOBAL)
ENTRY(tex_get,C32_TEX_GET) ENTRY(tex_format,C32_TEX_FORMAT) ENTRY(tex_flags,C32_TEX_FLAGS) ENTRY(tex_address,C32_TEX_ADDRESS)
ENTRY(host_register,C32_HOST_REGISTER) ENTRY(host_unregister,C32_HOST_UNREGISTER)
ENTRY(event_create,C32_EVENT_CREATE) ENTRY(event_destroy,C32_EVENT_DESTROY) ENTRY(event_record,C32_EVENT_RECORD)
ENTRY(event_sync,C32_EVENT_SYNC) ENTRY(event_query,C32_EVENT_QUERY) ENTRY(event_elapsed,C32_EVENT_ELAPSED)
ENTRY(stream_create,C32_STREAM_CREATE) ENTRY(stream_destroy,C32_STREAM_DESTROY) ENTRY(stream_sync,C32_STREAM_SYNC)
ENTRY(stream_query,C32_STREAM_QUERY) ENTRY(stream_wait,C32_STREAM_WAIT) ENTRY(htod_async,C32_HTOD_ASYNC) ENTRY(dtoh_async,C32_DTOH_ASYNC)
ENTRY(set_ctx,C32_CTX_SET)
ENTRY(memset8,C32_MEMSET8) ENTRY(memset16,C32_MEMSET16) ENTRY(memset_async,C32_MEMSET_ASYNC)
ENTRY(dtod,C32_DTOD) ENTRY(dtod_async,C32_DTOD_ASYNC) ENTRY(address_range,C32_ADDRESS_RANGE)
ENTRY(tex_filter,C32_TEX_FILTER) ENTRY(tex_mode,C32_TEX_ADDRESS_MODE)
ENTRY(tex_mipmap,C32_TEX_MIPMAP)
#define ENTRIES init,version,count,device,name,attr,create,destroy,sync_ctx,alloc,release,htod,dtoh,memset32,load,unload,function,launch,properties,capability,total_mem,push_ctx,pop_ctx,attach_ctx,detach_ctx,current_ctx,device_ctx,api_ctx,runtime_dark,module_global,tex_get,tex_format,tex_flags,tex_address,host_register,host_unregister,event_create,event_destroy,event_record,event_sync,event_query,event_elapsed,stream_create,stream_destroy,stream_sync,stream_query,stream_wait,htod_async,dtoh_async,set_ctx,memset8,memset16,memset_async,dtod,dtod_async,address_range,tex_filter,tex_mode,tex_mipmap
__attribute__((visibility("default"))) const entry_fn __wine_unix_call_funcs[] = { ENTRIES };
__attribute__((visibility("default"))) const entry_fn __wine_unix_call_wow64_funcs[] = { ENTRIES };
_Static_assert(sizeof(__wine_unix_call_funcs)/sizeof(entry_fn) == C32_COUNT, "opcode table mismatch");
