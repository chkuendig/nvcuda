/* SPDX-License-Identifier: LGPL-2.1-or-later */
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define FN(name, ...) static int (WINAPI *name)(__VA_ARGS__)
FN(cuInit, unsigned);
FN(cuDriverGetVersion, int *);
FN(cuDeviceGetCount, int *);
FN(cuDeviceGet, int *, int);
FN(cuDeviceGetName, char *, int, int);
FN(cuDeviceGetAttribute, int *, int, int);
FN(cuDeviceGetProperties, void *, int);
FN(cuDeviceComputeCapability, int *, int *, int);
FN(cuDeviceTotalMem, unsigned *, int);
FN(cuDeviceTotalMem_v2, SIZE_T *, int);
FN(cuCtxPushCurrent, void *);
FN(cuCtxPopCurrent_v2, void **);
FN(cuCtxCreate_v2, void **, unsigned, int);
FN(cuCtxDestroy_v2, void *);
FN(cuCtxSynchronize, void);
FN(cuCtxAttach, void **, unsigned);
FN(cuCtxDetach, void *);
FN(cuCtxGetCurrent, void **);
FN(cuCtxSetCurrent, void *);
FN(cuCtxGetDevice, int *);
FN(cuCtxGetApiVersion, void *, unsigned *);
FN(cuGetExportTable, const void **, const unsigned char *);
FN(cuEventCreate, void **, unsigned);
FN(cuGraphCreate, void **, unsigned);
FN(cuEventDestroy_v2, void *);
FN(cuEventRecord, void *, void *);
FN(cuEventSynchronize, void *);
FN(cuEventQuery, void *);
FN(cuEventElapsedTime, float *, void *, void *);
FN(cuStreamCreate, void **, unsigned);
FN(cuStreamDestroy_v2, void *);
FN(cuStreamSynchronize, void *);
FN(cuStreamQuery, void *);
FN(cuStreamWaitEvent, void *, void *, unsigned);
FN(cuMemAlloc, uint32_t *, unsigned);
FN(cuMemAlloc_v2, uint32_t *, SIZE_T);
FN(cuMemFree, uint32_t);
FN(cuMemFree_v2, uint32_t);
FN(cuMemHostAlloc, void **, SIZE_T, unsigned);
FN(cuMemFreeHost, void *);
FN(cuMemsetD8, uint32_t, unsigned char, unsigned);
FN(cuMemsetD8_v2, uint32_t, unsigned char, SIZE_T);
FN(cuMemsetD16_v2, uint32_t, unsigned short, SIZE_T);
FN(cuMemsetD32_v2, uint32_t, unsigned, SIZE_T);
FN(cuMemsetD8Async, uint32_t, unsigned char, SIZE_T, void *);
FN(cuMemsetD16Async, uint32_t, unsigned short, SIZE_T, void *);
FN(cuMemsetD32Async, uint32_t, unsigned, SIZE_T, void *);
FN(cuMemcpyHtoD, uint32_t, const void *, unsigned);
FN(cuMemcpyDtoH_v2, void *, uint32_t, SIZE_T);
FN(cuMemcpyHtoDAsync_v2, uint32_t, const void *, SIZE_T, void *);
FN(cuMemcpyDtoHAsync_v2, void *, uint32_t, SIZE_T, void *);
FN(cuMemcpyDtoD_v2, uint32_t, uint32_t, SIZE_T);
FN(cuMemcpyDtoDAsync_v2, uint32_t, uint32_t, SIZE_T, void *);
FN(cuMemGetAddressRange_v2, uint32_t *, SIZE_T *, uint32_t);
FN(cuModuleLoadData, void **, const void *);
FN(cuModuleUnload, void *);
FN(cuModuleGetGlobal, uint32_t *, unsigned *, void *, const char *);
FN(cuModuleGetGlobal_v2, uint32_t *, SIZE_T *, void *, const char *);
FN(cuModuleGetTexRef, void **, void *, const char *);
FN(cuTexRefSetFormat, void *, int, int);
FN(cuTexRefSetFlags, void *, unsigned);
FN(cuTexRefSetFilterMode, void *, int);
FN(cuTexRefSetAddressMode, void *, int, int);
FN(cuTexRefSetMipmapFilterMode, void *, int);
FN(cuTexRefSetMipmapLevelBias, void *, float);
FN(cuTexRefSetMipmapLevelClamp, void *, float, float);
FN(cuTexRefSetMaxAnisotropy, void *, unsigned);
FN(cuTexRefSetAddress, SIZE_T *, void *, uint32_t, unsigned);
FN(cuTexRefSetAddress_v2, SIZE_T *, void *, uint32_t, SIZE_T);
FN(cuModuleGetFunction, void **, void *, const char *);
FN(cuLaunchKernel, void *, unsigned, unsigned, unsigned, unsigned, unsigned,
    unsigned, unsigned, void *, void **, void **);

#define CHECK(expr) do { int rc = (expr); if (rc) { \
    fprintf(stderr,"FAIL line %d %s returned %d\n",__LINE__,#expr,rc); return 1; } } while(0)
#define EXPECT(expr, code) do { int rc = (expr); if (rc != (code)) { \
    fprintf(stderr,"FAIL line %d expected %d got %d: %s\n",__LINE__,code,rc,#expr); return 1; } } while(0)

static const char ptx[] =
    ".version 6.5\n.target sm_30\n.address_size 32\n"
    ".visible .entry pointer_chase(.param .u32 node) {\n"
    " .reg .u32 n, target, value, tid, address;\n"
    " ld.param.u32 n, [node];\n"
    " ld.global.u32 target, [n];\n"
    " ld.global.u32 value, [n+4];\n"
    " mov.u32 tid, %tid.x;\n"
    " mad.lo.u32 address, tid, 4, target;\n"
    " xor.b32 value, value, tid;\n"
    " st.global.u32 [address], value;\n"
    " ret;\n}\n";

static DWORD WINAPI context_thread(void *ctx)
{
    void *popped = NULL;
    CHECK(cuCtxPushCurrent(ctx));
    CHECK(cuCtxSynchronize());
    CHECK(cuCtxPopCurrent_v2(&popped));
    if (popped != ctx) return 1;
    return 0;
}

static const char global_ptx[] =
    ".version 6.5\n.target sm_30\n.address_size 32\n"
    ".global .align 4 .b8 scalar[4] = {7,0,0,0};\n"
    ".const .align 4 .b8 factor[4] = {3,0,0,0};\n"
    ".visible .entry globals_test(.param .u32 output) {\n"
    " .reg .u32 out, a, b, s, f;\n"
    " ld.param.u32 out, [output];\n"
    " mov.u32 s, scalar;\n"
    " mov.u32 f, factor;\n"
    " ld.global.u32 a, [s];\n"
    " ld.const.u32 b, [f];\n"
    " add.u32 a, a, b;\n"
    " st.global.u32 [s], a;\n"
    " st.global.u32 [out], a;\n"
    " ret;\n}\n";

static int test_globals(void)
{
    void *m = NULL, *second = NULL, *fn = NULL;
    CHECK(cuModuleLoadData(&m, global_ptx));
    uint32_t legacy[2] = {0, 0xaabbccdd}, size[2] = {0, 0xaabbccdd};
    struct { uint32_t p, guard; } address = {0, 0xaabbccdd};
    uint32_t wide = 0;
    CHECK(cuModuleGetGlobal(&legacy[0], &size[0], m, "scalar"));
    CHECK(cuModuleGetGlobal_v2(&address.p, (SIZE_T *)&size[0], m, "scalar"));
    wide = address.p;
    if (address.guard != 0xaabbccdd || wide != legacy[0] || size[0] != 4 || size[1] != 0xaabbccdd || legacy[1] != 0xaabbccdd) return 1;
    uint32_t initial = 0, output = 0, result = 0;
    CHECK(cuMemcpyDtoH_v2(&initial, wide, 4));
    if (initial != 7) return 1;
    EXPECT(cuMemFree(legacy[0]), 1);
    uint32_t missing = 0x12345678;
    EXPECT(cuModuleGetGlobal_v2(&missing, NULL, m, "missing"), 500);
    if (missing != 0x12345678) return 1;
    CHECK(cuModuleGetFunction(&fn, m, "globals_test"));
    CHECK(cuMemAlloc(&output, 4));
    void *params[] = {&output};
    CHECK(cuLaunchKernel(fn,1,1,1,1,1,1,0,NULL,params,NULL));
    CHECK(cuCtxSynchronize());
    CHECK(cuMemcpyDtoH_v2(&result, output, 4));
    if (result != 10) return 1;
    CHECK(cuModuleLoadData(&second, global_ptx));
    uint32_t second_address = 0;
    CHECK(cuModuleGetGlobal_v2(&second_address, NULL, second, "scalar"));
    CHECK(cuMemcpyDtoH_v2(&initial, second_address, 4));
    if (initial != 7 || second_address == wide) return 1;
    CHECK(cuLaunchKernel(fn,1,1,1,1,1,1,0,NULL,params,NULL));
    CHECK(cuCtxSynchronize());
    CHECK(cuMemcpyDtoH_v2(&result, output, 4));
    if (result != 13) return 1;
    CHECK(cuModuleUnload(m));
    EXPECT(cuModuleGetGlobal_v2(&wide, NULL, m, "scalar"), 400);
    EXPECT(cuMemcpyDtoH_v2(&initial, legacy[0], 4), 1);
    CHECK(cuModuleUnload(second));
    CHECK(cuMemFree(output));
    puts("PASS module globals: initialized constant/global GPU access, hidden arguments, widths, module isolation and unload");
    return 0;
}

static const char texture_ptx[] =
    ".version 6.5\n.target sm_30\n.address_size 32\n"
    ".global .texref tex;\n"
    ".visible .entry texture_test(.param .u32 output, .param .s32 coord) {\n"
    " .reg .u32 out; .reg .s32 x,a,b,c,d;\n"
    " ld.param.u32 out, [output]; ld.param.s32 x, [coord];\n"
    " tex.1d.v4.s32.s32 {a,b,c,d}, [tex, {x}];\n"
    " st.global.v4.s32 [out], {a,b,c,d}; ret;\n}\n";
static int test_texture(void)
{
    void *m = NULL, *tex = NULL, *same = NULL, *fn = NULL;
    CHECK(cuModuleLoadData(&m, texture_ptx));
    CHECK(cuModuleGetTexRef(&tex, m, "tex"));
    CHECK(cuModuleGetTexRef(&same, m, "tex"));
    if (!tex || tex != same) return 1;
    CHECK(cuTexRefSetFormat(tex, 10, 1)); /* signed int32 */
    CHECK(cuTexRefSetFlags(tex, 1)); /* read as integer */
    CHECK(cuTexRefSetFilterMode(tex, 0)); /* point */
    CHECK(cuTexRefSetAddressMode(tex, 0, 1)); /* clamp */
    CHECK(cuTexRefSetMipmapFilterMode(tex, 0));
    CHECK(cuTexRefSetMipmapLevelBias(tex, 0.25f));
    CHECK(cuTexRefSetMipmapLevelClamp(tex, 0.0f, 4.0f));
    CHECK(cuTexRefSetMaxAnisotropy(tex, 1));
    EXPECT(cuTexRefSetFilterMode(tex, 2), 1);
    EXPECT(cuTexRefSetAddressMode(tex, 3, 0), 1);
    uint32_t source = 0, output = 0;
    CHECK(cuMemAlloc(&source, 16)); CHECK(cuMemAlloc(&output, 16));
    int data[4] = {12, 34, 56, 78}, result[4] = {0};
    CHECK(cuMemcpyHtoD(source, data, sizeof(data)));
    uint32_t offset[2] = {0xabcdef01, 0xabcdef01};
    EXPECT(cuTexRefSetAddress_v2((SIZE_T *)&offset[0], tex, source, 17), 1);
    /* The pinned backend stores these settings but rejects binding this
     * linear texture with the non-default configuration. Preserve its error. */
    EXPECT(cuTexRefSetAddress((SIZE_T *)&offset[0], tex, source, 16), 801);
    CHECK(cuTexRefSetMipmapLevelBias(tex, 0.0f));
    CHECK(cuTexRefSetMipmapLevelClamp(tex, 0.0f, 0.0f));
    CHECK(cuTexRefSetMaxAnisotropy(tex, 0));
    CHECK(cuTexRefSetAddress((SIZE_T *)&offset[0], tex, source, 16));
    if (offset[0] || offset[1] != 0xabcdef01) return 1;
    CHECK(cuTexRefSetAddress_v2((SIZE_T *)&offset[0], tex, source, 16));
    CHECK(cuModuleGetFunction(&fn, m, "texture_test"));
    int coordinate = 2; void *params[] = {&output, &coordinate};
    CHECK(cuLaunchKernel(fn,1,1,1,1,1,1,0,NULL,params,NULL));
    CHECK(cuCtxSynchronize()); CHECK(cuMemcpyDtoH_v2(result, output, 16));
    if (result[0] != 56) { fprintf(stderr, "FAIL texture read: %d expected 56\n", result[0]); return 1; }
    CHECK(cuModuleUnload(m));
    EXPECT(cuTexRefSetFlags(tex, 1), 400);
    CHECK(cuMemFree(source)); CHECK(cuMemFree(output));
    puts("PASS PE32 texref: stable token, pool translation, GPU read, offset width and stale handle");
    return 0;
}

static int test_async(void)
{
    static const char async_ptx[] =
        ".version 6.5\n.target sm_30\n.address_size 32\n"
        ".visible .entry increment(.param .u32 ptr) {\n"
        ".reg .u32 p,v; ld.param.u32 p,[ptr]; ld.global.u32 v,[p];\n"
        "add.u32 v,v,1; st.global.u32 [p],v; ret;\n}\n";
    void *m = NULL, *fn = NULL, *producer = NULL, *consumer = NULL;
    struct { void *p; uint32_t guard; } ready = {NULL, 0xabcdef01}, done = {NULL, 0xabcdef01};
    CHECK(cuModuleLoadData(&m, async_ptx)); CHECK(cuModuleGetFunction(&fn, m, "increment"));
    CHECK(cuStreamCreate(&producer, 1)); CHECK(cuStreamCreate(&consumer, 1));
    CHECK(cuEventCreate(&ready.p, 0)); CHECK(cuEventCreate(&done.p, 0));
    if (!ready.p || !done.p || ready.guard != 0xabcdef01 || done.guard != 0xabcdef01) return 1;
    uint32_t *host = NULL, device = 0;
    CHECK(cuMemHostAlloc((void **)&host, 65536, 0)); CHECK(cuMemAlloc(&device, 4));
    host[0] = 41; host[1] = 0;
    CHECK(cuMemcpyHtoDAsync_v2(device, host, 4, producer));
    void *params[] = {&device};
    CHECK(cuLaunchKernel(fn,1,1,1,1,1,1,0,producer,params,NULL));
    CHECK(cuEventRecord(ready.p, producer));
    CHECK(cuStreamWaitEvent(consumer, ready.p, 0));
    CHECK(cuMemcpyDtoHAsync_v2(host + 1, device, 4, consumer));
    CHECK(cuEventRecord(done.p, consumer)); CHECK(cuEventSynchronize(done.p));
    if (host[1] != 42) return 1;
    CHECK(cuEventQuery(ready.p)); CHECK(cuEventQuery(done.p));
    CHECK(cuStreamQuery(consumer)); CHECK(cuStreamSynchronize(producer));
    struct { float ms; uint32_t guard; } timing = {-1, 0xabcdef01};
    CHECK(cuEventElapsedTime(&timing.ms, ready.p, done.p));
    if (timing.ms < 0 || timing.guard != 0xabcdef01) return 1;
    EXPECT(cuMemcpyDtoHAsync_v2(host, device, 5, consumer), 1);
    CHECK(cuEventDestroy_v2(ready.p)); CHECK(cuEventDestroy_v2(done.p));
    EXPECT(cuEventQuery(ready.p), 400);
    CHECK(cuStreamDestroy_v2(producer)); CHECK(cuStreamDestroy_v2(consumer));
    EXPECT(cuStreamQuery(consumer), 400);
    CHECK(cuMemFree(device)); CHECK(cuMemFreeHost(host)); CHECK(cuModuleUnload(m));
    puts("PASS real two-stream GPU compute: pinned async transfers, event dependency, timing widths and stale handles");
    return 0;
}

static int test_fills(void)
{
    uint32_t device = 0, words[16]; void *stream = NULL;
    CHECK(cuMemAlloc(&device, sizeof(words))); CHECK(cuStreamCreate(&stream, 1));
    CHECK(cuMemsetD8(device, 0x5a, sizeof(words)));
    CHECK(cuMemcpyDtoH_v2(words, device, sizeof(words)));
    for (unsigned i = 0; i < 16; ++i) if (words[i] != 0x5a5a5a5a) return 1;
    CHECK(cuMemsetD16_v2(device, 0x1234, 32));
    CHECK(cuMemcpyDtoH_v2(words, device, sizeof(words)));
    for (unsigned i = 0; i < 16; ++i) if (words[i] != 0x12341234) return 1;
    CHECK(cuMemsetD32_v2(device, 0x98765432, 16));
    CHECK(cuMemcpyDtoH_v2(words, device, sizeof(words)));
    for (unsigned i = 0; i < 16; ++i) if (words[i] != 0x98765432) return 1;
    CHECK(cuMemsetD8Async(device, 0xa5, 64, stream));
    CHECK(cuMemsetD16Async(device + 16, 0x2345, 8, stream));
    CHECK(cuMemsetD32Async(device + 32, 0x12345678, 8, stream));
    CHECK(cuStreamSynchronize(stream));
    CHECK(cuMemcpyDtoH_v2(words, device, sizeof(words)));
    for (unsigned i = 0; i < 16; ++i) {
        unsigned expected = i < 4 ? 0xa5a5a5a5 : i < 8 ? 0x23452345 : 0x12345678;
        if (words[i] != expected) return 1;
    }
    EXPECT(cuMemsetD8_v2(device, 0, 65), 1);
    EXPECT(cuMemsetD16_v2(device + 1, 0, 1), 1);
    EXPECT(cuMemsetD32_v2(device, 0, 0x40000000), 1);
    EXPECT(cuMemsetD16Async(device, 0, 33, stream), 1);
    CHECK(cuStreamDestroy_v2(stream));
    EXPECT(cuMemsetD8Async(device, 0, 1, stream), 400);
    CHECK(cuMemFree(device));
    puts("PASS byte/halfword/word GPU fills: exact data, async stream, range, alignment, overflow and stale handle");
    return 0;
}

static int test_packed_launch(void)
{
    static const char packed_ptx[] =
        ".version 6.5\n.target sm_30\n.address_size 32\n"
        ".visible .entry packed(.param .u32 dst,.param .u64 wide,.param .u32 salt) {\n"
        ".reg .u32 p,v,s; .reg .u64 w; ld.param.u32 p,[dst]; ld.param.u64 w,[wide];\n"
        "cvt.u32.u64 v,w; ld.param.u32 s,[salt]; xor.b32 v,v,s; st.global.u32 [p],v; ret;\n}\n";
    void *m = NULL, *fn = NULL, *stream = NULL;
    uint32_t dst = 0, copy = 0, result = 0, salt = 0x1234;
    uint64_t wide = 0x1122334455667788ull;
    CHECK(cuModuleLoadData(&m, packed_ptx)); CHECK(cuModuleGetFunction(&fn, m, "packed"));
    CHECK(cuMemAlloc(&dst, 12)); CHECK(cuMemAlloc(&copy, 12)); CHECK(cuStreamCreate(&stream, 1));
    unsigned char buffer[20]; memset(buffer, 0xcd, sizeof(buffer));
    memcpy(buffer, &dst, 4); memcpy(buffer + 8, &wide, 8); memcpy(buffer + 16, &salt, 4);
    SIZE_T bytes = sizeof(buffer);
    void *extra[] = {(void *)1, buffer, (void *)2, &bytes, NULL};
    CHECK(cuLaunchKernel(fn,1,1,1,1,1,1,0,stream,NULL,extra));
    CHECK(cuMemcpyDtoDAsync_v2(copy + 4, dst, 4, stream)); CHECK(cuStreamSynchronize(stream));
    CHECK(cuMemcpyDtoH_v2(&result, copy + 4, 4));
    if (result != (0x55667788u ^ salt)) return 1;
    CHECK(cuMemcpyDtoD_v2(copy + 8, copy + 4, 4));
    result = 0; CHECK(cuMemcpyDtoH_v2(&result, copy + 8, 4));
    if (result != (0x55667788u ^ salt)) return 1;
    struct { uint32_t p, guard; } base = {0, 0xaabbccdd}, size = {0, 0xaabbccdd};
    CHECK(cuMemGetAddressRange_v2(&base.p, (SIZE_T *)&size.p, copy + 5));
    if (base.p != copy || size.p != 12 || base.guard != 0xaabbccdd || size.guard != 0xaabbccdd) return 1;
    EXPECT(cuMemGetAddressRange_v2(&base.p, (SIZE_T *)&size.p, copy + 12), 1);
    EXPECT(cuMemcpyDtoD_v2(copy, dst + 11, 2), 1);
    EXPECT(cuMemcpyDtoDAsync_v2(copy + 11, dst, 2, stream), 1);
    bytes = 19; EXPECT(cuLaunchKernel(fn,1,1,1,1,1,1,0,stream,NULL,extra), 1); bytes = 20;
    void *params[] = {&dst, &wide, &salt};
    EXPECT(cuLaunchKernel(fn,1,1,1,1,1,1,0,stream,params,extra), 1);
    extra[0] = (void *)3; EXPECT(cuLaunchKernel(fn,1,1,1,1,1,1,0,stream,NULL,extra), 1);
    CHECK(cuStreamDestroy_v2(stream)); CHECK(cuMemFree(dst)); CHECK(cuMemFree(copy)); CHECK(cuModuleUnload(m));
    puts("PASS packed launch: 64-bit scalar alignment, GPU result, device copies, allocation-range widths and rejected malformed buffers");
    return 0;
}

static unsigned destroyed_storage;
static void *storage_expected_ctx;
static void WINAPI storage_destructor(void *ctx, void *key, void *value)
{
    if (ctx != storage_expected_ctx || key != (void *)0x123456 || value != (void *)0x334455)
        destroyed_storage = 1000;
    ++destroyed_storage;
    /* Re-entry proves neither bridge nor storage locks surround callbacks. */
    int version = 0;
    if (cuDriverGetVersion(&version) || version < 1000) destroyed_storage = 1000;
}

static int test_multiple_contexts(int dev)
{
    void *ctx[2] = {0}, *module[2] = {0}, *function[2] = {0}, *stream[2] = {0};
    uint32_t node[2] = {0}, result[2] = {0}, output[64];
    for (unsigned i = 0; i < 2; ++i) {
        CHECK(cuCtxCreate_v2(&ctx[i], i ? 4 : 0, dev));
        CHECK(cuMemAlloc(&node[i], 8)); CHECK(cuMemAlloc(&result[i], sizeof(output)));
        uint32_t input[2] = {result[i], 0x12340000 + i};
        CHECK(cuMemcpyHtoD(node[i], input, sizeof(input)));
        CHECK(cuModuleLoadData(&module[i], ptx));
        CHECK(cuModuleGetFunction(&function[i], module[i], "pointer_chase"));
        CHECK(cuStreamCreate(&stream[i], 1));
        void *params[] = {&node[i]};
        CHECK(cuLaunchKernel(function[i], 1,1,1,64,1,1,0,stream[i],params,NULL));
        CHECK(cuStreamSynchronize(stream[i]));
    }
    if (ctx[0] == ctx[1] || node[0] != node[1] || result[0] != result[1]) return 1;
    for (unsigned i = 0; i < 2; ++i) {
        CHECK(cuCtxSetCurrent(ctx[i]));
        CHECK(cuMemcpyDtoH_v2(output, result[i], sizeof(output)));
        for (unsigned lane = 0; lane < 64; ++lane)
            if (output[lane] != ((0x12340000 + i) ^ lane)) return 1;
        EXPECT(cuStreamSynchronize(stream[1-i]), 400);
        void *wrong = NULL;
        EXPECT(cuModuleGetFunction(&wrong, module[1-i], "pointer_chase"), 400);
    }
    CHECK(cuCtxSetCurrent(ctx[0]));
    CHECK(cuCtxDestroy_v2(ctx[1])); /* non-current context, must preserve ctx[0] */
    void *current = NULL;
    CHECK(cuCtxGetCurrent(&current)); if (current != ctx[0]) return 1;
    CHECK(cuStreamSynchronize(stream[0]));
    CHECK(cuMemcpyDtoH_v2(output, result[0], sizeof(output)));
    for (unsigned lane = 0; lane < 64; ++lane)
        if (output[lane] != (0x12340000u ^ lane)) return 1;
    EXPECT(cuCtxSetCurrent(ctx[1]), 201);
    CHECK(cuCtxDestroy_v2(ctx[0]));
    puts("PASS two CUDA contexts: same PE32 offsets, isolated GPU results, scoped handles and non-current destruction");
    return 0;
}

int main(void)
{
    _Static_assert(sizeof(void *) == 4, "fixture must be a 32-bit PE");
    HMODULE cuda = LoadLibraryA("nvcuda.dll");
    if (!cuda) { fprintf(stderr,"FAIL LoadLibrary nvcuda error %lu\n",GetLastError()); return 1; }
#define LOAD(name) do { name = (void *)GetProcAddress(cuda,#name); \
    if (!name) { fprintf(stderr,"FAIL missing %s\n",#name); return 1; } } while(0)
    LOAD(cuInit); LOAD(cuDriverGetVersion); LOAD(cuDeviceGetCount); LOAD(cuDeviceGet);
    LOAD(cuDeviceGetName); LOAD(cuCtxCreate_v2); LOAD(cuCtxDestroy_v2); LOAD(cuCtxSynchronize);
    LOAD(cuDeviceGetAttribute); LOAD(cuDeviceGetProperties); LOAD(cuDeviceComputeCapability);
    LOAD(cuDeviceTotalMem); LOAD(cuDeviceTotalMem_v2); LOAD(cuCtxPushCurrent); LOAD(cuCtxPopCurrent_v2);
    LOAD(cuCtxAttach); LOAD(cuCtxDetach);
    LOAD(cuCtxGetCurrent); LOAD(cuCtxSetCurrent); LOAD(cuCtxGetDevice); LOAD(cuCtxGetApiVersion);
    LOAD(cuGetExportTable); LOAD(cuEventCreate); LOAD(cuGraphCreate);
    LOAD(cuEventDestroy_v2); LOAD(cuEventRecord); LOAD(cuEventSynchronize); LOAD(cuEventQuery); LOAD(cuEventElapsedTime);
    LOAD(cuStreamCreate); LOAD(cuStreamDestroy_v2); LOAD(cuStreamSynchronize); LOAD(cuStreamQuery); LOAD(cuStreamWaitEvent);
    LOAD(cuMemcpyHtoDAsync_v2); LOAD(cuMemcpyDtoHAsync_v2);
    LOAD(cuMemcpyDtoD_v2); LOAD(cuMemcpyDtoDAsync_v2); LOAD(cuMemGetAddressRange_v2);
    LOAD(cuMemAlloc); LOAD(cuMemAlloc_v2); LOAD(cuMemFree); LOAD(cuMemFree_v2);
    LOAD(cuMemHostAlloc); LOAD(cuMemFreeHost);
    LOAD(cuMemsetD8); LOAD(cuMemsetD8_v2); LOAD(cuMemsetD16_v2); LOAD(cuMemsetD32_v2);
    LOAD(cuMemsetD8Async); LOAD(cuMemsetD16Async); LOAD(cuMemsetD32Async);
    LOAD(cuMemcpyHtoD); LOAD(cuMemcpyDtoH_v2); LOAD(cuModuleLoadData);
    LOAD(cuModuleUnload); LOAD(cuModuleGetFunction); LOAD(cuLaunchKernel);
    LOAD(cuModuleGetGlobal); LOAD(cuModuleGetGlobal_v2);
    LOAD(cuModuleGetTexRef); LOAD(cuTexRefSetFormat); LOAD(cuTexRefSetFlags);
    LOAD(cuTexRefSetFilterMode); LOAD(cuTexRefSetAddressMode);
    LOAD(cuTexRefSetMipmapFilterMode); LOAD(cuTexRefSetMipmapLevelBias);
    LOAD(cuTexRefSetMipmapLevelClamp); LOAD(cuTexRefSetMaxAnisotropy);
    LOAD(cuTexRefSetAddress); LOAD(cuTexRefSetAddress_v2);
    CHECK(cuInit(0));
    int version = 0, count = 0, dev = -1;
    char name[128] = {0};
    CHECK(cuDriverGetVersion(&version)); CHECK(cuDeviceGetCount(&count));
    if (count < 1) { fprintf(stderr,"FAIL no CUDA devices\n"); return 1; }
    CHECK(cuDeviceGet(&dev,0)); CHECK(cuDeviceGetName(name,sizeof(name),dev));
    struct { int fields[14]; uint32_t guard; } properties = {{0}, 0xabcdef01};
    CHECK(cuDeviceGetProperties(&properties, dev));
    int major = 0, minor = 0, attr = 0;
    CHECK(cuDeviceComputeCapability(&major, &minor, dev));
    CHECK(cuDeviceGetAttribute(&attr, 75, dev));
    if (major != attr || major < 1 || minor < 0) return 1;
    CHECK(cuDeviceGetAttribute(&attr, 1, dev));
    if (properties.fields[0] != attr || properties.guard != 0xabcdef01) return 1;
    uint32_t memory[2] = {0, 0xabcdef01};
    CHECK(cuDeviceTotalMem(&memory[0], dev));
    uint32_t capacity = memory[0];
    if (!capacity || memory[1] != 0xabcdef01) return 1;
    memory[0] = 0;
    CHECK(cuDeviceTotalMem_v2((SIZE_T *)&memory[0], dev));
    if (memory[0] != capacity || memory[1] != 0xabcdef01) return 1;
    EXPECT(cuDeviceTotalMem(NULL, dev), 1);
    EXPECT(cuDeviceComputeCapability(NULL, &minor, dev), 1);
    printf("PASS discovery: legacy property layout, capability %d.%d, 32-bit size_t writes (%u bytes)\n", major, minor, capacity);
    printf("PASS 32-bit PE -> 64-bit Unix: CUDA %d device %s\n",version,name);
    void *ctx = NULL, *module = NULL, *function = NULL, *other = NULL;
    CHECK(cuCtxCreate_v2(&ctx,0,dev));
    CHECK(cuCtxGetCurrent(&other));
    if (other != ctx) return 1;
    CHECK(cuCtxSetCurrent(NULL)); CHECK(cuCtxGetCurrent(&other));
    if (other) return 1;
    EXPECT(cuCtxSynchronize(), 201);
    EXPECT(cuCtxSetCurrent((void *)0x1234), 201);
    CHECK(cuCtxSetCurrent(ctx)); CHECK(cuCtxGetCurrent(&other));
    if (other != ctx) return 1;
    uint32_t too_large = 0x12345678;
    EXPECT(cuMemAlloc(&too_large, capacity + 1), 2);
    if (too_large != 0x12345678) return 1;
    int ctx_dev = -1;
    CHECK(cuCtxGetDevice(&ctx_dev));
    if (ctx_dev != dev) return 1;
    const unsigned char runtime_guid[16] = {
        0x6b,0xd5,0xfb,0x6c,0x5b,0xf4,0xe7,0x4a,0x89,0x87,0xd9,0x39,0x12,0xfd,0x9d,0xf9
    };
    const unsigned char unknown_guid[16] = {0};
    const void **rt = NULL;
    EXPECT(cuGetExportTable(NULL, runtime_guid), 1);
    EXPECT(cuGetExportTable((const void **)&rt, NULL), 1);
    EXPECT(cuGetExportTable((const void **)&rt, unknown_guid), 801);
    if (rt) return 1;
    CHECK(cuGetExportTable((const void **)&rt, runtime_guid));
    if (!rt || (uintptr_t)rt[0] != 52) return 1;
    int (WINAPI *rt_context)(void **, int) = (void *)rt[2];
    struct { void *ctx; uint32_t guard; } primary = {NULL, 0xabcdef01};
    CHECK(rt_context(&primary.ctx, dev));
    if (!primary.ctx || primary.guard != 0xabcdef01) return 1;
    unsigned api_version = 0;
    CHECK(cuCtxGetApiVersion(primary.ctx, &api_version));
    if (api_version < 3000) return 1;
    CHECK(cuCtxGetCurrent(&other));
    if (other != ctx) return 1;
    void *event_canary = (void *)0x11223344;
    EXPECT(cuGraphCreate(&event_canary, 0), 801);
    if (event_canary != (void *)0x11223344) return 1;
    puts("PASS PE32 runtime table thunks, primary handle width and unsupported errors");
    const unsigned char heap_guid[16] = {
        0x19,0x5b,0xcb,0xf4,0xd6,0x7d,0x02,0x4a,0xac,0xc5,0x1d,0x29,0xce,0xa6,0x31,0xae
    };
    const void **heap = NULL;
    CHECK(cuGetExportTable((const void **)&heap, heap_guid));
    if ((uintptr_t)heap[0] != 12) return 1;
    int (WINAPI *heap_alloc)(const void **, SIZE_T, SIZE_T) = (void *)heap[1];
    int (WINAPI *heap_free)(const void *, SIZE_T *) = (void *)heap[2];
    struct { const void *p; uint32_t guard; } record = {NULL, 0xabcdef01};
    CHECK(heap_alloc(&record.p, 0x11223344, 0x55667788));
    if (!record.p || record.guard != 0xabcdef01 || ((const uint32_t *)record.p)[0] != 0x11223344 ||
        ((const uint32_t *)record.p)[1] != 0x55667788) return 1;
    uint32_t freed_y[2] = {0, 0xabcdef01};
    CHECK(heap_free(record.p, (SIZE_T *)&freed_y[0]));
    if (freed_y[0] != 0x55667788 || freed_y[1] != 0xabcdef01) return 1;
    EXPECT(heap_free(record.p, NULL), 1);
    EXPECT(heap_alloc(NULL, 1, 2), 1);
    puts("PASS PE32 heap record layout, output widths and rejected double free");
    const unsigned char hooks_guid[16] = {
        0xa0,0x94,0x79,0x8c,0x2e,0x74,0x2e,0x74,0x93,0xf2,0x08,0x00,0x20,0x0c,0x0a,0x66
    };
    const void **hooks = NULL;
    CHECK(cuGetExportTable((const void **)&hooks, hooks_guid));
    if ((uintptr_t)hooks[0] != 28) return 1;
    void (WINAPI *get_buffer)(void **, SIZE_T *) = (void *)hooks[2];
    struct { void *p; uint32_t guard; } buffer = {NULL, 0xabcdef01};
    struct { SIZE_T n; uint32_t guard; } length = {0, 0xabcdef01};
    get_buffer(&buffer.p, &length.n);
    if (!buffer.p || length.n != 1024 || buffer.guard != 0xabcdef01 || length.guard != 0xabcdef01) return 1;
    ((uint32_t *)buffer.p)[0] = 0x10203040;
    void *buffer_again = NULL;
    get_buffer(&buffer_again, &length.n);
    if (buffer_again != buffer.p || *(uint32_t *)buffer_again != 0x10203040) return 1;
    const unsigned char storage_guid[16] = {
        0xc6,0x93,0x33,0x6e,0x11,0x21,0xdf,0x11,0xa8,0xc3,0x68,0xf3,0x55,0xd8,0x95,0x93
    };
    const void **storage = NULL;
    CHECK(cuGetExportTable((const void **)&storage, storage_guid));
    int (WINAPI *put)(void *, void *, void *, void (WINAPI *)(void *, void *, void *)) = (void *)storage[0];
    int (WINAPI *remove_key)(void *, void *) = (void *)storage[1];
    int (WINAPI *get)(void **, void *, void *) = (void *)storage[2];
    void *stored = (void *)0x112233;
    EXPECT(get(&stored, ctx, (void *)0x123456), 1);
    if (stored != (void *)0x112233) return 1;
    EXPECT(put((void *)0x1234, NULL, NULL, NULL), 201);
    storage_expected_ctx = ctx;
    CHECK(put(ctx, (void *)0x123456, (void *)0x223344, storage_destructor));
    CHECK(put(ctx, (void *)0x123456, (void *)0x334455, storage_destructor));
    CHECK(get(&stored, NULL, (void *)0x123456));
    if (stored != (void *)0x334455 || destroyed_storage) return 1;
    CHECK(put(ctx, (void *)0x777, NULL, storage_destructor));
    CHECK(remove_key(ctx, (void *)0x777));
    EXPECT(cuCtxAttach(&other, 1), 1);
    EXPECT(cuCtxAttach(NULL, 0), 1);
    EXPECT(cuCtxDetach((void *)0x1234), 201);
    CHECK(cuCtxAttach(&other, 0));
    if (other != ctx) return 1;
    CHECK(cuCtxDetach(other));
    CHECK(cuCtxSynchronize());
    if (destroyed_storage) return 1;
    HANDLE thread = CreateThread(NULL, 0, context_thread, ctx, 0, NULL);
    DWORD thread_status = 1;
    if (!thread || WaitForSingleObject(thread, 30000) != WAIT_OBJECT_0 ||
        !GetExitCodeThread(thread, &thread_status) || thread_status) return 1;
    CloseHandle(thread);
    EXPECT(cuCtxPushCurrent((void *)0x1234), 400);
    puts("PASS context push/pop across a new Windows thread");
    CHECK(cuCtxCreate_v2(&other,4,dev));
    CHECK(cuCtxDestroy_v2(other));
    CHECK(cuCtxSetCurrent(ctx));
    int (WINAPI *rt_load)(void **, const void *) = (void *)rt[1];
    /* A valid SASS-only container must fail without changing the output.
     * Returning an empty module or metadata NOT_FOUND changes cudart's path. */
    struct {
        uint32_t magic; uint16_t version, header; uint64_t bytes;
        uint16_t kind, entry_version; uint32_t entry_header; uint64_t payload;
        uint32_t elf_magic;
    } sass = {0xba55ed50, 1, 16, 20, 2, 0x100, 16, 4, 0x464c457f};
    struct { uint32_t magic, version; const void *data, *extra; } wrapper =
        {0x466243b1, 1, &sass, NULL};
    void *no_binary = (void *)0x12345678;
    EXPECT(rt_load(&no_binary, &wrapper), 209);
    if (no_binary != (void *)0x12345678) return 1;
    puts("PASS SASS-only fatbin: no-binary error and unchanged module output");
    CHECK(rt_load(&module,ptx));
    CHECK(cuModuleGetFunction(&function,module,"pointer_chase"));
    for (unsigned round = 0; round < 4; ++round) {
        uint32_t node = 0, result = 0;
        struct { uint32_t p, guard; } allocation = {0, 0xaabbccdd};
        CHECK(cuMemAlloc(&node,8)); CHECK(cuMemAlloc_v2(&allocation.p,64*4));
        result = allocation.p;
        if (allocation.guard != 0xaabbccdd) return 1;
        if (!node || !result || result == node) return 1;
        uint32_t data[2] = {(uint32_t)result, 0x12345678u + round};
        uint32_t output[64] = {0};
        CHECK(cuMemcpyHtoD(node,data,sizeof(data)));
        void *params[] = {&node};
        CHECK(cuLaunchKernel(function,1,1,1,64,1,1,0,NULL,params,NULL));
        CHECK(cuCtxSynchronize());
        CHECK(cuMemcpyDtoH_v2(output,result,sizeof(output)));
        for (unsigned i=0;i<64;++i) if (output[i] != (data[1] ^ i)) {
            fprintf(stderr,"FAIL round %u lane %u got %08x expected %08x\n",round,i,output[i],data[1]^i);
            return 1;
        }
        EXPECT(cuMemcpyDtoH_v2(output,result,257),1);
        CHECK(cuMemFree(node)); CHECK(cuMemFree_v2(result));
        EXPECT(cuMemFree(node),1);
        printf("PASS round %u: 64 GPU pointer-chase results, legacy/v2 widths, allocation reuse\n",round);
    }
    CHECK(cuModuleUnload(module));
    void *params[] = {NULL};
    EXPECT(cuLaunchKernel(function,1,1,1,1,1,1,0,NULL,params,NULL),400);
    EXPECT(cuModuleUnload(module),400);
    CHECK(test_globals());
    CHECK(test_texture());
    struct { void *p; uint32_t guard; } pinned = {(void *)0x12345678, 0xabcdef01};
    EXPECT(cuMemHostAlloc(&pinned.p, 65536, 2), 801);
    if (pinned.p != (void *)0x12345678) return 1;
    EXPECT(cuMemHostAlloc(NULL, 65536, 0), 1);
    CHECK(cuMemHostAlloc(&pinned.p, 65536, 0));
    if (!pinned.p || pinned.guard != 0xabcdef01) return 1;
    *(uint32_t *)pinned.p = 0x12345678;
    uint32_t pinned_device = 0;
    CHECK(cuMemAlloc(&pinned_device, 4));
    CHECK(cuMemcpyHtoD(pinned_device, pinned.p, 4));
    *(uint32_t *)pinned.p = 0;
    CHECK(cuMemcpyDtoH_v2(pinned.p, pinned_device, 4));
    if (*(uint32_t *)pinned.p != 0x12345678) return 1;
    CHECK(cuMemFree(pinned_device)); CHECK(cuMemFreeHost(pinned.p));
    EXPECT(cuMemFreeHost(pinned.p), 1);
    puts("PASS real HIP-registered PE32 host memory, transfer data and lifetime checks");
    CHECK(test_async());
    CHECK(test_fills());
    CHECK(test_packed_launch());
    CHECK(cuCtxDestroy_v2(ctx));
    if (destroyed_storage != 1) return 1;
    CHECK(cuCtxCreate_v2(&ctx,0,dev));
    EXPECT(get(&stored, ctx, (void *)0x123456), 1);
    storage_expected_ctx = ctx;
    CHECK(put(ctx, (void *)0x123456, (void *)0x334455, storage_destructor));
    CHECK(cuCtxDetach(ctx));
    if (destroyed_storage != 2) return 1;
    EXPECT(cuCtxDetach(ctx), 201);
    EXPECT(cuCtxSynchronize(), 201);
    CHECK(cuCtxCreate_v2(&ctx,0,dev));
    CHECK(cuCtxDestroy_v2(ctx));
    puts("PASS legacy context reference count and final-detach invalidation");
    puts("PASS runtime buffers and context-storage replacement/delete/destruction/re-entry");
    CHECK(test_multiple_contexts(dev));
    FreeLibrary(cuda);
    puts("PASS WoW64 CUDA compute and negative cases");
    return 0;
}
