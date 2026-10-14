/* SPDX-License-Identifier: LGPL-2.1-or-later
 * Read-only check of the legacy NVAPI topology consumed by PhysXDevice 2.8.4.
 * Layout and IDs match NVIDIA's NVAPI header included by dxvk-nvapi.
 */
#include <windows.h>
#include <stdint.h>
#include <stdio.h>

struct topology {
    uint32_t version, count;
    struct { void *handle; uint32_t flags; } gpu[8];
};
struct gpu { void *handle; uint32_t flags; };
struct topology_v2 { uint32_t version, count; struct gpu *gpus; };

int main(void)
{
    _Static_assert(sizeof(struct topology) == 72, "requires PE32 topology V1");
    HMODULE dll = LoadLibraryA("nvapi.dll");
    if (!dll) { printf("FAIL nvapi.dll load: %lu\n", GetLastError()); return 1; }
    void *(__cdecl *query)(uint32_t) = (void *)GetProcAddress(dll, "nvapi_QueryInterface");
    if (!query) return 1;
    int (__cdecl *init)(void) = query(0x0150e828);
    int (__cdecl *enumerate)(struct topology *) = query(0x5786cc6e);
    if (!init || !enumerate) { puts("FAIL missing NVAPI entry point"); return 1; }
    int rc = init();
    if (rc) { printf("FAIL NVAPI init: %d\n", rc); return 1; }
    struct topology t = {0};
    t.version = sizeof(t) | (1u << 16);
    rc = enumerate(&t);
    printf("NVAPI CUDA topology result=%d count=%u\n", rc, t.count);
    if (rc || t.count > 8) return 1;
    for (unsigned i = 0; i < t.count; ++i)
        printf("GPU %u handle=%p flags=%08x\n", i, t.gpu[i].handle, t.gpu[i].flags);
    if (!t.count) { puts("FAIL no CUDA-capable GPUs advertised by NVAPI"); return 2; }
    puts("PASS legacy NVAPI CUDA-capable GPU enumeration");
    struct topology_v2 v2 = {sizeof(v2) | (2u << 16), 0xdeadbeef, (struct gpu *)1};
    int (__cdecl *enumerate_v2)(struct topology_v2 *) = (void *)enumerate;
    if (enumerate_v2(&v2) || v2.count != t.count || !v2.gpus || v2.gpus == (struct gpu *)1 ||
        v2.gpus[0].handle != t.gpu[0].handle || v2.gpus[0].flags != 0x21001b) return 1;
    struct gpu *first_output = v2.gpus;
    v2.count = 0; v2.gpus = NULL;
    if (enumerate_v2(&v2) || v2.count != t.count || v2.gpus != first_output ||
        v2.gpus[0].handle != t.gpu[0].handle || v2.gpus[0].flags != 0x21001b) return 1;
    v2.version = sizeof(v2) | (3u << 16);
    if (enumerate_v2(&v2) != -9) return 1;
    puts("PASS topology V2 poisoned outputs, stable returned storage and invalid version");
    int (__cdecl *physx_topology)(struct topology_v2 *) = query(0x36e39e6b);
    v2.version = 0xface1234; v2.count = 0xdeadbeef; v2.gpus = (struct gpu *)1;
    if (!physx_topology || physx_topology(&v2) || v2.version != 0xface1234 ||
        v2.count != t.count || v2.gpus != first_output ||
        v2.gpus[0].handle != t.gpu[0].handle || v2.gpus[0].flags != 0x21001b ||
        physx_topology(NULL) != -5 || query(0x33c7358c)) return 1;
    puts("PASS PhysX private topology V2, poisoned fields and null argument");
    HMODULE cuda = LoadLibraryA("nvcuda.dll");
    if (!cuda || t.count != 1) return 1;
    int (WINAPI *cuda_init)(unsigned) = (void *)GetProcAddress(cuda, "cuInit");
    int (WINAPI *attribute)(int *, int, int) = (void *)GetProcAddress(cuda, "cuDeviceGetAttribute");
    int (__cdecl *get_id)(void *, uint32_t *) = query(0x6533ea3e);
    if (!cuda_init || !attribute || !get_id || cuda_init(0)) return 1;
    uint32_t expected = 0;
    struct { int value; uint32_t guard; } actual = {-1, 0xabcdef01};
    if (get_id(t.gpu[0].handle, &expected) || attribute(&actual.value, 0x20080402, 0) ||
        (uint32_t)actual.value != expected || actual.guard != 0xabcdef01) return 1;
    if (attribute(&actual.value, 0x20080402, 1) != 101 ||
        attribute(NULL, 0x20080402, 0) != 1 || attribute(&actual.value, 0x12345678, 0) != 1) return 1;
    printf("PASS PhysX private adapter correlation: %08x; unrelated errors preserved\n", expected);
    return 0;
}
