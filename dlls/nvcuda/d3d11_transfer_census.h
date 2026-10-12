/* Opt-in D2 transfer census (DRAFT revision 3, unbuilt, unreviewed).
 *
 * ZLUDA_D3D11_TRANSFER_CENSUS_DIR=<absolute dir, owned, 0700> is the only
 * opt-in; it is independent of the copy probe. At the exported wrapper
 * boundary (exports and cuGetProcAddress resolve to the same wrappers) it
 * records host-capable transfer calls, raw synchronisation observations,
 * allocation identity with generations and every graphics-interop call with
 * the relay branch taken; the relay's own staging calls are recorded
 * separately because they use pcu* directly.
 *
 * What it does NOT claim: completion of async work (records say "submitted"
 * or "returned"), context identity (never queried: ZLUDA's cuCtxGetCurrent
 * is not passive), zero-copy kernel writes, CPU-side copies into D3D, or any
 * D3D consumption. Size equality is never attribution.
 *
 * Protocol. Every admitted hook is counted from entry to the end of its
 * record fill in census_inflight; all census state, including admission
 * itself (inflight, pending, identity snapshot, setup-skip counts), is changed
 * only inside a census_mutating section, and no section spans a CUDA call. Publication closes admission and mutation with
 * atomic flag bits, waits only for in-progress mutation sections (bounded CPU
 * work, never a CUDA call), then reads frozen state. Calls still inside CUDA
 * at the cutoff are counted as lost and make the affected coverage
 * incomplete; their late returns are refused and change nothing.
 *
 * Disabled (default): one atomic load per hook. No added GPU work, waits,
 * payload copies, per-call file I/O or libcuda calls in any mode. */

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#ifndef CENSUS_TEST_POINT
#define CENSUS_TEST_POINT(id) do {} while(0)
#endif
enum { CENSUS_TP_SETUP, CENSUS_TP_MUTATE, CENSUS_TP_PUBLISH, CENSUS_TP_ADMIT };

#define CENSUS_COUNT_OF(a) (sizeof(a)/sizeof((a)[0]))
#ifndef CENSUS_WINDOW_NS
#define CENSUS_WINDOW_NS UINT64_C(60000000000)
#endif
#define CENSUS_BUCKETS 6
#define CENSUS_BUCKET_NS (CENSUS_WINDOW_NS/CENSUS_BUCKETS)
#ifndef CENSUS_CLOSE_GRACE_NS
#define CENSUS_CLOSE_GRACE_NS UINT64_C(2000000000) /* calls returning within this grace still count */
#endif
#define CENSUS_TRANSFER_LIMIT 16384
#define CENSUS_SYNC_LIMIT 32768
#define CENSUS_STAGING_LIMIT 8192
#define CENSUS_GFX_EVENT_LIMIT 32768
#define CENSUS_ALLOC_LIMIT 16384
#define CENSUS_GFX_RESOURCE_LIMIT 1024
#ifndef CENSUS_ARM_POLL_MS
#define CENSUS_ARM_POLL_MS 500
#endif
#define CENSUS_ARM_GIVE_UP_NS UINT64_C(7200000000000)
#ifndef CENSUS_DRAIN_WAIT_MS
#define CENSUS_DRAIN_WAIT_MS 1000
#endif
#define CENSUS_BATCH_SLOTS 6

/* class: 1 transfer record, 2 transfer record (v1 layout, operands not
 * decoded), 3 count only, 4 sync observation, 5 allocation identity, 6 graphics */
#define CENSUS_APIS(X) \
    X(cuMemcpyDtoH,1) X(cuMemcpyDtoH_v2,1) X(cuMemcpyDtoH_v2_ptds,1) \
    X(cuMemcpyDtoHAsync,1) X(cuMemcpyDtoHAsync_v2,1) X(cuMemcpyDtoHAsync_v2_ptsz,1) \
    X(cuMemcpyAtoH,1) X(cuMemcpyAtoH_v2,1) X(cuMemcpyAtoH_v2_ptds,1) \
    X(cuMemcpyAtoHAsync,1) X(cuMemcpyAtoHAsync_v2,1) X(cuMemcpyAtoHAsync_v2_ptsz,1) \
    X(cuMemcpy2D,2) X(cuMemcpy2D_v2,1) X(cuMemcpy2D_v2_ptds,1) \
    X(cuMemcpy2DUnaligned,2) X(cuMemcpy2DUnaligned_v2,1) X(cuMemcpy2DUnaligned_v2_ptds,1) \
    X(cuMemcpy2DAsync,2) X(cuMemcpy2DAsync_v2,1) X(cuMemcpy2DAsync_v2_ptsz,1) \
    X(cuMemcpy3D,2) X(cuMemcpy3D_v2,1) X(cuMemcpy3D_v2_ptds,1) \
    X(cuMemcpy3DAsync,2) X(cuMemcpy3DAsync_v2,1) X(cuMemcpy3DAsync_v2_ptsz,1) \
    X(cuMemcpy,1) X(cuMemcpy_ptds,1) X(cuMemcpyAsync,1) X(cuMemcpyAsync_ptsz,1) \
    X(cuMemcpyHtoD,3) X(cuMemcpyHtoD_v2,3) X(cuMemcpyHtoD_v2_ptds,3) \
    X(cuMemcpyHtoDAsync,3) X(cuMemcpyHtoDAsync_v2,3) X(cuMemcpyHtoDAsync_v2_ptsz,3) \
    X(cuMemcpyHtoA,3) X(cuMemcpyHtoA_v2,3) X(cuMemcpyHtoA_v2_ptds,3) \
    X(cuMemcpyHtoAAsync,3) X(cuMemcpyHtoAAsync_v2,3) X(cuMemcpyHtoAAsync_v2_ptsz,3) \
    X(cuMemcpyDtoD,3) X(cuMemcpyDtoD_v2,3) X(cuMemcpyDtoD_v2_ptds,3) \
    X(cuMemcpyDtoDAsync,3) X(cuMemcpyDtoDAsync_v2,3) X(cuMemcpyDtoDAsync_v2_ptsz,3) \
    X(cuMemcpyDtoA,3) X(cuMemcpyDtoA_v2,3) X(cuMemcpyDtoA_v2_ptds,3) \
    X(cuMemcpyAtoD,3) X(cuMemcpyAtoD_v2,3) X(cuMemcpyAtoD_v2_ptds,3) \
    X(cuMemcpyAtoA,3) X(cuMemcpyAtoA_v2,3) X(cuMemcpyAtoA_v2_ptds,3) \
    X(cuMemcpyPeer,3) X(cuMemcpyPeer_ptds,3) X(cuMemcpyPeerAsync,3) X(cuMemcpyPeerAsync_ptsz,3) \
    X(cuMemcpy3DPeer,3) X(cuMemcpy3DPeer_ptds,3) X(cuMemcpy3DPeerAsync,3) X(cuMemcpy3DPeerAsync_ptsz,3) \
    X(cuMemcpyBatchAsync,3) X(cuMemcpyBatchAsync_ptsz,3) X(cuMemcpyBatchAsync_v2,3) X(cuMemcpyBatchAsync_v2_ptsz,3) \
    X(cuMemcpy3DBatchAsync,3) X(cuMemcpy3DBatchAsync_ptsz,3) X(cuMemcpy3DBatchAsync_v2,3) X(cuMemcpy3DBatchAsync_v2_ptsz,3) \
    X(cuMemcpyWithAttributesAsync,3) X(cuMemcpyWithAttributesAsync_ptsz,3) \
    X(cuMemcpy3DWithAttributesAsync,3) X(cuMemcpy3DWithAttributesAsync_ptsz,3) \
    X(cuStreamSynchronize,4) X(cuStreamSynchronize_ptsz,4) X(cuStreamQuery,4) X(cuStreamQuery_ptsz,4) \
    X(cuCtxSynchronize,4) X(cuCtxSynchronize_v2,4) X(cuEventRecord,4) X(cuEventRecord_ptsz,4) \
    X(cuEventRecordWithFlags,4) X(cuEventRecordWithFlags_ptsz,4) X(cuEventSynchronize,4) X(cuEventQuery,4) \
    X(cuMemHostAlloc,5) X(cuMemAllocHost,5) X(cuMemAllocHost_v2,5) X(cuMemHostRegister,5) X(cuMemHostRegister_v2,5) \
    X(cuMemFreeHost,5) X(cuMemHostUnregister,5) X(cuMemHostGetDevicePointer,5) X(cuMemHostGetDevicePointer_v2,5) \
    X(cuMemAlloc,5) X(cuMemAlloc_v2,5) X(cuMemAllocPitch,5) X(cuMemAllocPitch_v2,5) X(cuMemAllocManaged,5) \
    X(cuMemAllocAsync,5) X(cuMemAllocAsync_ptsz,5) X(cuMemAllocFromPoolAsync,5) X(cuMemAllocFromPoolAsync_ptsz,5) \
    X(cuMemFree,5) X(cuMemFree_v2,5) X(cuMemFreeAsync,5) X(cuMemFreeAsync_ptsz,5) \
    X(cuArrayCreate,5) X(cuArrayCreate_v2,5) X(cuArray3DCreate,5) X(cuArray3DCreate_v2,5) X(cuArrayDestroy,5) \
    X(cuGraphicsD3D11RegisterResource,6) X(cuGraphicsMapResources,6) X(cuGraphicsMapResources_ptsz,6) \
    X(cuGraphicsUnmapResources,6) X(cuGraphicsUnmapResources_ptsz,6) X(cuGraphicsSubResourceGetMappedArray,6) \
    X(cuGraphicsResourceGetMappedPointer,6) X(cuGraphicsResourceGetMappedPointer_v2,6) \
    X(cuGraphicsResourceSetMapFlags,6) X(cuGraphicsResourceSetMapFlags_v2,6) \
    X(cuGraphicsResourceGetMappedMipmappedArray,6) X(cuGraphicsUnregisterResource,6)

enum {
#define CENSUS_ENUM(name,cls) CA_##name,
    CENSUS_APIS(CENSUS_ENUM)
#undef CENSUS_ENUM
    CA_COUNT
};
static const struct { const char *name; unsigned cls; } census_api_info[CA_COUNT]={
#define CENSUS_INFO(name,cls) {#name,cls},
    CENSUS_APIS(CENSUS_INFO)
#undef CENSUS_INFO
};

/* Record flags */
#define CF_ASYNC 0x1u              /* accepted submission only; completion is never inferred */
#define CF_PER_THREAD 0x2u         /* _ptds/_ptsz entry point (per-thread default stream semantics) */
#define CF_V1_UNDECODED 0x4u       /* v1 struct layout; operands deliberately not read */
#define CF_UNIFIED 0x8u            /* cuMemcpy/cuMemcpyAsync: types inferred only from the identity ledgers */
#define CF_ARGS_UNREAD 0x10u       /* call failed: struct/descriptor/batch arguments were not read */
#define CF_IDENTITY_AMBIGUOUS 0x20u /* identity could not be ordered against concurrent allocator/registry changes */
#define CF_LEDGER_INCOMPLETE 0x40u /* identity changes were lost (setup skip, full ledger or resource table) before this record */
#define CF_BATCH_TRUNCATED 0x80u   /* more relay resources than per-record slots; extra ones not counted */
#define CF_FORWARDED 0x100u        /* graphics call not handled by the CPU-copy relay path; handles not resolved */

/* Identity kinds (ledger lookups, never size matching) and modifiers. */
enum { CK_NONE, CK_HOST_ALLOC, CK_HOST_LEGACY, CK_HOST_REGISTERED, CK_HOST_DEVICEMAP_ALIAS,
       CK_DEVICE, CK_DEVICE_PITCH, CK_DEVICE_ASYNC, CK_MANAGED, CK_APP_ARRAY, CK_CPU_COPY_ARRAY };
#define CK_STALE 0x100u          /* only a retired generation matched */
#define CK_FREE_SUBMITTED 0x200u /* matched generation has a stream-ordered free submitted; retirement not observed */

/* Allocation states */
enum { CA_STATE_RETIRED, CA_STATE_LIVE, CA_STATE_FREE_SUBMITTED, CA_STATE_ORDER_AMBIGUOUS };
#define CA_FLAG_ORPHAN_FREE 0x1u /* tombstone: a free with no known live allocation */
#define CA_FLAG_REFERENCED 0x2u  /* a published record resolved to this generation; never recycled */

/* Relay branch codes reported by cpu_register/cpu_map/cpu_resource. */
enum { CB_FORWARDED, CB_RELAY_SUCCESS, CB_FORCED_REJECT, CB_NOT_MAPPED, CB_ALREADY_MAPPED,
       CB_CONTEXT, CB_INVALID_VALUE, CB_MIXED_BATCH, CB_SYNC_FAILED, CB_TRANSFER_FAILED,
       CB_UNREGISTER_FAILED, CB_REG_FLAGS, CB_REG_NOT_TEXTURE2D, CB_REG_IDENTITY, CB_REG_DESC,
       CB_REG_NO_MEMORY, CB_REG_SINGLE_THREADED, CB_REG_NOT_MT, CB_REG_STAGING, CB_REG_ARRAY,
       CB_REG_DUPLICATE, CB_OP_UNSUPPORTED };
/* Filled by the relay while it holds cpu_textures_lock; never read by the relay. */
typedef struct {
    unsigned branch, found, width, height, format, mips, array_size, samples;
    uint64_t gen[CENSUS_BATCH_SLOTS];
    uintptr_t array, context;
} CensusRelayInfo;

enum { CO_SUCCESS, CO_FORCED_REJECT, CO_NOT_MAPPED, CO_ALREADY_MAPPED, CO_CONTEXT, CO_INVALID_VALUE,
       CO_NOT_SUPPORTED, CO_OTHER, CO_COUNT };
enum { CG_REGISTER, CG_MAP, CG_UNMAP, CG_GET_ARRAY, CG_GET_POINTER, CG_SET_FLAGS, CG_GET_MIPMAPPED,
       CG_UNREGISTER, CG_COUNT };

typedef struct {
    uint32_t committed, seq, api, tid;
    int32_t result; uint32_t flags, src_type, dst_type;
    uint64_t t_entry, t_return, stream, args; /* args: opaque 2D/3D struct pointer value */
    uint64_t src_host, src_device, src_array, dst_host, dst_device, dst_array;
    uint64_t src_x, src_y, src_z, dst_x, dst_y, dst_z;
    uint64_t width_bytes, height, depth, src_pitch, src_height, dst_pitch, dst_height, byte_count;
    uint32_t src_kind, dst_kind;
    uint64_t src_gen, dst_gen, src_offset, dst_offset;
} CensusTransfer;
typedef struct {
    uint32_t committed, seq, api, tid;
    int32_t result; uint32_t flags;
    uint64_t t_entry, t_return, stream, event; /* raw handle values; no lifetime or context identity */
} CensusSync;
typedef struct {
    uint32_t committed, seq, kind, tid; /* kind: 0 relay stream sync, 1 D3D->CUDA (map), 2 CUDA->D3D (unmap) */
    int32_t result; uint32_t ptsz;
    uint64_t t_entry, t_return, resource_gen, stream, bytes, map_epoch;
} CensusStaging;
typedef struct {
    uint32_t committed, seq, api, tid;
    int32_t result; uint32_t flags, branch, count, found, reserved;
    uint64_t t_entry, t_return, stream, resource, resource_gen, value, arg_a, arg_b;
    uint32_t width, height, format, mips, array_size, samples;
    uint64_t batch_gen[CENSUS_BATCH_SLOTS];
} CensusGfxEvent;
typedef struct {
    uint64_t base, size, alias, gen, alloc_ns, free_ns, alloc_stream, free_stream;
    uint32_t kind, flags, state, api, free_api, width, height, depth, format;
    uint32_t cuda_flags; /* the application's allocation/registration flags (e.g. DEVICEMAP 0x2); flags are census-only */
} CensusAlloc;
typedef struct {
    uint64_t handle, texture, array, gen, register_ns, unregister_ns, context; /* context: relay's registration capture */
    uint32_t live, width, height, reserved;
    uint32_t outcomes[CG_COUNT][CO_COUNT];
} CensusGfxResource;
typedef struct {
    unsigned long long calls, errors, in_window, recorded, overflow;
} CensusCounter;

/* Hook roles. Gate fields count admitted, not-yet-filled hooks. */
enum { CENSUS_ROLE_COUNT, CENSUS_ROLE_WINDOWED, CENSUS_ROLE_LEDGER, CENSUS_ROLE_GRAPHICS, CENSUS_ROLES };
#define CENSUS_GATE_CLOSED (UINT64_C(1)<<63)
#define CENSUS_GATE_ALL UINT64_C(1)
#define CENSUS_GATE_WINDOW (UINT64_C(1)<<21)
#define CENSUS_GATE_GRAPHICS (UINT64_C(1)<<42)
#define CENSUS_GATE_FIELD(v,shift) ((unsigned)(((v)>>(shift))&((UINT64_C(1)<<21)-1)))
typedef struct {
    unsigned long long t0;  /* nonzero only when admitted */
    uint64_t delta, version;
    unsigned pending, role, bucket;
} CensusToken;

enum { CS_UNREAD, CS_OFF, CS_SETUP, CS_ON, CS_FAILED };
static unsigned census_state;
static unsigned long long census_setup_skipped[CENSUS_ROLES];
static pid_t census_owner;
static int census_dir=-1;
static void *census_block;
static uint64_t census_inflight, census_mutating;
static unsigned census_publish_state; /* 0 none, 1 publishing, 2 done */
static unsigned long long census_window_start, census_arm_ns, census_arm_mtime_rt, census_arm_detect_rt;
static pthread_mutex_t census_lock=PTHREAD_MUTEX_INITIALIZER; /* identity tables only; never across CUDA, D3D or I/O */
static uint64_t census_ledger_version; /* bumped by every identity change, under census_lock */
static unsigned census_ledger_pending; /* admitted allocation/free hooks whose ledger update is not done */
static CensusCounter census_counters[CA_COUNT], census_staging_counter;
static CensusTransfer *census_transfers;
static CensusSync *census_syncs;
static CensusStaging *census_stagings;
static CensusGfxEvent *census_gfx_events;
static CensusAlloc *census_allocs;
static CensusGfxResource *census_gfx;
static unsigned census_transfer_next, census_sync_next, census_staging_next, census_gfx_event_next;
static unsigned census_transfer_bucket[CENSUS_BUCKETS], census_sync_bucket[CENSUS_BUCKETS], census_staging_bucket[CENSUS_BUCKETS];
static unsigned long long census_transfer_overflow, census_sync_overflow, census_staging_overflow, census_gfx_event_overflow;
/* Below: census_lock */
static unsigned census_alloc_count, census_gfx_count;
static unsigned long long census_generation, census_alloc_recycled, census_alloc_untracked, census_free_unmatched;
static unsigned long long census_gfx_untracked, census_gfx_unknown[CG_COUNT][CO_COUNT];
static unsigned long long census_gfx_setup_untracked; /* relay registrations during setup: no generation */
static unsigned long long census_order_ambiguous_allocs, census_ambiguous_lookups;

/* ---- clock and output (census-owned; the probe is untouched) ---- */
static unsigned long long census_now(void)
{
    struct timespec ts;
    if(clock_gettime(CLOCK_MONOTONIC,&ts)) return 0;
    return (unsigned long long)ts.tv_sec*1000000000+ts.tv_nsec;
}
static int census_write_all(int fd, const void *data, size_t size)
{
    const char *p=data;
    while(size) { ssize_t n=write(fd,p,size); if(n<0 && errno==EINTR) continue; if(n<=0) return -1; p+=n; size-=n; }
    return 0;
}
/* Same no-replacement pending->link publication as the copy probe. */
static int census_publish_file(const char *name, const void *data, size_t size)
{
    char pending[128];
    int fd, ok;
    snprintf(pending,sizeof(pending),"%s.pending",name);
    fd=openat(census_dir,pending,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);
    if(fd<0) return -1;
    ok=!census_write_all(fd,data,size);
    if(close(fd)) ok=0;
    if(ok && linkat(census_dir,pending,census_dir,name,0)) ok=0;
    if(unlinkat(census_dir,pending,0)) ok=0;
    return ok?0:-1;
}

/* ---- gates ---- */
/* Every change to census state happens between these two. Refused after close. */
static BOOL census_mutate_begin(void)
{
    uint64_t old=__atomic_fetch_add(&census_mutating,1,__ATOMIC_SEQ_CST);
    if(old&CENSUS_GATE_CLOSED) { __atomic_fetch_sub(&census_mutating,1,__ATOMIC_SEQ_CST); return FALSE; }
    CENSUS_TEST_POINT(CENSUS_TP_MUTATE);
    return TRUE;
}
static void census_mutate_end(void)
{
    __atomic_fetch_sub(&census_mutating,1,__ATOMIC_SEQ_CST);
}
/* Exit of an admitted hook: refused (and left counted as lost) after close. */
static BOOL census_exit_begin(const CensusToken *tok)
{
    return tok->t0 && census_mutate_begin();
}
static void census_exit_end(const CensusToken *tok)
{
    /* inflight first: once mutation has drained, every filled hook has left it. */
    __atomic_fetch_sub(&census_inflight,tok->delta,__ATOMIC_SEQ_CST);
    census_mutate_end();
}

/* ---- setup and teardown ---- */
static void census_free_storage(void)
{
    free(census_block); census_block=NULL;
    census_transfers=NULL; census_syncs=NULL; census_stagings=NULL; census_gfx_events=NULL; census_allocs=NULL; census_gfx=NULL;
    if(census_dir>=0) { close(census_dir); census_dir=-1; }
}
static BOOL census_setup(const char *path)
{
    struct stat st;
    size_t total;
    char *block;
    int fd;
    if(!path || path[0]!='/') return FALSE;
    fd=open(path,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);
    if(fd<0) return FALSE;
    if(fstat(fd,&st) || !S_ISDIR(st.st_mode) || st.st_uid!=geteuid() || (st.st_mode&0777)!=0700) { close(fd); return FALSE; }
    census_dir=fd;
    total=sizeof(CensusTransfer)*CENSUS_TRANSFER_LIMIT+sizeof(CensusSync)*CENSUS_SYNC_LIMIT
         +sizeof(CensusStaging)*CENSUS_STAGING_LIMIT+sizeof(CensusGfxEvent)*CENSUS_GFX_EVENT_LIMIT
         +sizeof(CensusAlloc)*CENSUS_ALLOC_LIMIT+sizeof(CensusGfxResource)*CENSUS_GFX_RESOURCE_LIMIT;
    census_block=block=calloc(1,total); /* about 15 MiB, only when opted in */
    if(!block) { census_free_storage(); return FALSE; }
    census_transfers=(CensusTransfer*)block; block+=sizeof(CensusTransfer)*CENSUS_TRANSFER_LIMIT;
    census_syncs=(CensusSync*)block; block+=sizeof(CensusSync)*CENSUS_SYNC_LIMIT;
    census_stagings=(CensusStaging*)block; block+=sizeof(CensusStaging)*CENSUS_STAGING_LIMIT;
    census_gfx_events=(CensusGfxEvent*)block; block+=sizeof(CensusGfxEvent)*CENSUS_GFX_EVENT_LIMIT;
    census_allocs=(CensusAlloc*)block; block+=sizeof(CensusAlloc)*CENSUS_ALLOC_LIMIT;
    census_gfx=(CensusGfxResource*)block;
    census_owner=getpid();
    return TRUE;
}
static DWORD WINAPI census_thread(void *module);
/* Intentional pin: the census thread holds its own module reference until it
 * has published (or given up) and leaves through FreeLibraryAndExitThread, so
 * while opted in an application FreeLibrary cannot unload nvcuda before then. */
static BOOL census_start_thread(void)
{
    HMODULE module=NULL;
    HANDLE thread;
    if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,(LPCWSTR)(void*)census_thread,&module)) return FALSE;
    if(!(thread=CreateThread(NULL,0,census_thread,module,0,NULL))) { FreeLibrary(module); return FALSE; }
    CloseHandle(thread);
    return TRUE;
}
static unsigned census_init(void)
{
    const char *env=getenv("ZLUDA_D3D11_TRANSFER_CENSUS_DIR");
    unsigned expected=CS_UNREAD, want=env && env[0] ? CS_SETUP : CS_OFF;
    if(!__atomic_compare_exchange_n(&census_state,&expected,want,FALSE,__ATOMIC_SEQ_CST,__ATOMIC_SEQ_CST))
        return expected; /* another thread decided first */
    if(want==CS_OFF) return CS_OFF;
    CENSUS_TEST_POINT(CENSUS_TP_SETUP);
    if(!census_setup(env) || !census_start_thread()) {
        census_free_storage();
        __atomic_store_n(&census_state,CS_FAILED,__ATOMIC_SEQ_CST);
        ERR("[ZLUDA] transfer census opt-in could not start; no census evidence\n");
        return CS_FAILED;
    }
    __atomic_store_n(&census_state,CS_ON,__ATOMIC_SEQ_CST);
    FIXME("[ZLUDA] transfer census explicit opt-in enabled\n");
    return CS_ON;
}
/* FreeLibrary path only (never on process exit): the pin guarantees the
 * census thread has already finished; nothing else can call in. */
static void census_unload(void)
{
    if(__atomic_load_n(&census_state,__ATOMIC_SEQ_CST)!=CS_ON) return;
    __atomic_store_n(&census_state,CS_OFF,__ATOMIC_SEQ_CST);
    census_free_storage();
}

/* ---- admission ---- */
static void census_enter(CensusToken *tok, unsigned role)
{
    unsigned state=__atomic_load_n(&census_state,__ATOMIC_ACQUIRE);
    unsigned long long now, start;
    uint64_t delta, old;
    tok->t0=0;
    if(state==CS_OFF) return;
    if(state==CS_UNREAD) state=census_init();
    if(state==CS_SETUP) {
        if(census_mutate_begin()) { __atomic_fetch_add(&census_setup_skipped[role],1,__ATOMIC_SEQ_CST); census_mutate_end(); }
        return;
    }
    if(state!=CS_ON || census_owner!=getpid()) return;
    /* Admission is itself a mutation section: publication either waits for it
     * or it is refused. Entry time, gate counts, pending and the identity
     * snapshot are therefore taken consistently with the freeze. */
    if(!census_mutate_begin()) return;
    if(!(now=census_now())) now=1;
    start=__atomic_load_n(&census_window_start,__ATOMIC_ACQUIRE);
    delta=CENSUS_GATE_ALL;
    tok->bucket=CENSUS_BUCKETS;
    /* Window membership is decided once, here, and drives both records and lost counts. */
    if(role==CENSUS_ROLE_WINDOWED && start && now>=start && now-start<CENSUS_WINDOW_NS) {
        delta+=CENSUS_GATE_WINDOW; tok->bucket=(unsigned)((now-start)/CENSUS_BUCKET_NS);
    }
    if(role==CENSUS_ROLE_GRAPHICS) delta+=CENSUS_GATE_GRAPHICS;
    old=__atomic_fetch_add(&census_inflight,delta,__ATOMIC_SEQ_CST);
    if(old&CENSUS_GATE_CLOSED) { __atomic_fetch_sub(&census_inflight,delta,__ATOMIC_SEQ_CST); census_mutate_end(); return; }
    CENSUS_TEST_POINT(CENSUS_TP_ADMIT);
    if(role==CENSUS_ROLE_LEDGER) __atomic_fetch_add(&census_ledger_pending,1,__ATOMIC_SEQ_CST);
    /* Identity snapshot: version first, then pending (see census_identity_ambiguous_locked).
     * Any allocation/free overlapping the call either is pending here, is
     * pending at return, or bumped the version in between. */
    tok->version=__atomic_load_n(&census_ledger_version,__ATOMIC_SEQ_CST);
    tok->pending=__atomic_load_n(&census_ledger_pending,__ATOMIC_SEQ_CST);
    tok->delta=delta; tok->role=role; tok->t0=now;
    census_mutate_end();
}

/* ---- records (only inside a mutation section) ---- */
static void census_count(CensusCounter *c, CUresult result, BOOL in_window)
{
    __atomic_fetch_add(&c->calls,1,__ATOMIC_RELAXED);
    if(result) __atomic_fetch_add(&c->errors,1,__ATOMIC_RELAXED);
    if(in_window) __atomic_fetch_add(&c->in_window,1,__ATOMIC_RELAXED);
}
static void *census_claim(CensusCounter *c, unsigned bucket, void *base, size_t size, unsigned limit,
                          unsigned *next, unsigned *buckets, unsigned long long *overflow)
{
    unsigned slot;
    if(buckets) {
        if(bucket>=CENSUS_BUCKETS) return NULL;
        if(__atomic_fetch_add(&buckets[bucket],1,__ATOMIC_RELAXED)>=limit/CENSUS_BUCKETS) goto full;
    }
    slot=__atomic_fetch_add(next,1,__ATOMIC_RELAXED);
    if(slot>=limit) goto full;
    __atomic_fetch_add(&c->recorded,1,__ATOMIC_RELAXED);
    return (char*)base+(size_t)slot*size;
full:
    __atomic_fetch_add(overflow,1,__ATOMIC_RELAXED);
    __atomic_fetch_add(&c->overflow,1,__ATOMIC_RELAXED);
    return NULL;
}
static void census_commit(uint32_t *committed, uint32_t *seq, uint32_t *tid)
{
    static unsigned sequence;
    *seq=__atomic_add_fetch(&sequence,1,__ATOMIC_RELAXED);
    *tid=GetCurrentThreadId();
    __atomic_store_n(committed,1,__ATOMIC_RELEASE);
}

/* ---- allocation ledger (census_lock) ---- */
static CensusAlloc *census_alloc_slot_locked(void)
{
    CensusAlloc *oldest=NULL;
    unsigned i;
    BOOL tombstones_free=__atomic_load_n(&census_ledger_pending,__ATOMIC_SEQ_CST)<=1;
    if(census_alloc_count<CENSUS_ALLOC_LIMIT) return &census_allocs[census_alloc_count++];
    /* Only retired, never-referenced generations are recycled; tombstones only
     * while no other allocation call is in flight. */
    for(i=0;i<CENSUS_ALLOC_LIMIT;i++) {
        CensusAlloc *a=&census_allocs[i];
        if(a->state!=CA_STATE_RETIRED || (a->flags&CA_FLAG_REFERENCED)) continue;
        if((a->flags&CA_FLAG_ORPHAN_FREE) && !tombstones_free) continue;
        if(!oldest || a->free_ns<oldest->free_ns) oldest=a;
    }
    if(oldest) census_alloc_recycled++;
    return oldest;
}
static void census_ledger_changed_locked(BOOL was_pending)
{
    __atomic_add_fetch(&census_ledger_version,1,__ATOMIC_SEQ_CST);
    if(was_pending) __atomic_sub_fetch(&census_ledger_pending,1,__ATOMIC_SEQ_CST);
}
static BOOL census_host_kind(unsigned kind) { return kind>=CK_HOST_ALLOC && kind<=CK_HOST_REGISTERED; }

typedef struct { uint32_t kind; uint64_t gen, offset; BOOL ambiguous; } CensusIdentity;
/* side: 0 host pointer, 1 device pointer, 2 array handle, 3 unified address.
 * Pinned host allocations are device-addressable at their host address (UVA)
 * and at a cuMemHostGetDevicePointer alias; device-side matches report
 * CK_HOST_DEVICEMAP_ALIAS. More than one live match is ambiguous. */
static void census_identify_locked(uint64_t address, unsigned side, CensusIdentity *id)
{
    CensusAlloc *live=NULL, *stale=NULL;
    uint64_t live_offset=0, stale_offset=0;
    unsigned i, live_kind=0, stale_kind=0, live_matches=0;
    memset(id,0,sizeof(*id));
    if(!address) return;
    if(side==2) {
        for(i=0;i<census_gfx_count;i++) if(census_gfx[i].live && census_gfx[i].array==address) {
            id->kind=CK_CPU_COPY_ARRAY; id->gen=census_gfx[i].gen; live_matches++;
        }
        if(live_matches) { id->ambiguous=live_matches>1; return; }
    }
    for(i=0;i<census_alloc_count;i++) {
        CensusAlloc *a=&census_allocs[i];
        BOOL host=census_host_kind(a->kind);
        BOOL in_base=a->kind!=CK_APP_ARRAY && a->size && address>=a->base && address-a->base<a->size;
        BOOL in_alias=host && a->alias && address>=a->alias && address-a->alias<a->size;
        unsigned k=0; uint64_t o=0;
        if(a->flags&CA_FLAG_ORPHAN_FREE) continue;
        if(side==2) { if(a->kind==CK_APP_ARRAY && a->base==address) k=CK_APP_ARRAY; }
        else if(host) {
            if(side!=1 && in_base) { k=a->kind; o=address-a->base; }
            else if(side!=0 && in_base) { k=CK_HOST_DEVICEMAP_ALIAS; o=address-a->base; }
            else if(side!=0 && in_alias) { k=CK_HOST_DEVICEMAP_ALIAS; o=address-a->alias; }
        }
        else if(side!=0 && in_base) { k=a->kind; o=address-a->base; }
        if(!k) continue;
        if(a->state!=CA_STATE_RETIRED) {
            live_matches++;
            if(!live || a->gen>live->gen) { live=a; live_kind=k; live_offset=o; }
        }
        else if(!stale || a->gen>stale->gen) { stale=a; stale_kind=k; stale_offset=o; }
    }
    if(live) {
        id->kind=live_kind; id->gen=live->gen; id->offset=live_offset;
        if(live->state==CA_STATE_FREE_SUBMITTED) { id->kind|=CK_FREE_SUBMITTED; id->ambiguous=TRUE; }
        if(live->state==CA_STATE_ORDER_AMBIGUOUS || live_matches>1) id->ambiguous=TRUE;
        live->flags|=CA_FLAG_REFERENCED;
    }
    else if(stale) {
        id->kind=stale_kind|CK_STALE; id->gen=stale->gen; id->offset=stale_offset;
        stale->flags|=CA_FLAG_REFERENCED;
    }
    else if(census_alloc_recycled) id->ambiguous=TRUE; /* a recycled generation may have matched */
}

/* ---- graphics registry: called by the relay while it holds cpu_textures_lock ---- */
static uint64_t census_gfx_insert(uint64_t handle, uint64_t texture, uint64_t array, uint64_t context,
                                  unsigned width, unsigned height)
{
    CensusGfxResource *r;
    uint64_t gen=0;
    unsigned state=__atomic_load_n(&census_state,__ATOMIC_ACQUIRE);
    if(state==CS_SETUP) { /* registration the census cannot track: counted, never silent */
        if(census_mutate_begin()) { __atomic_fetch_add(&census_gfx_setup_untracked,1,__ATOMIC_SEQ_CST); census_mutate_end(); }
        return 0;
    }
    if(state!=CS_ON || census_owner!=getpid() || !census_mutate_begin()) return 0;
    pthread_mutex_lock(&census_lock);
    /* Never recycled: every published generation keeps its metadata. */
    if(census_gfx_count>=CENSUS_GFX_RESOURCE_LIMIT) census_gfx_untracked++;
    else {
        r=&census_gfx[census_gfx_count++];
        memset(r,0,sizeof(*r));
        r->handle=handle; r->texture=texture; r->array=array; r->context=context;
        r->gen=gen=++census_generation; r->register_ns=census_now(); r->live=1; r->width=width; r->height=height;
        census_ledger_changed_locked(FALSE);
    }
    pthread_mutex_unlock(&census_lock);
    census_mutate_end();
    return gen;
}
static CensusGfxResource *census_gfx_by_gen_locked(uint64_t gen)
{
    unsigned i;
    if(gen) for(i=0;i<census_gfx_count;i++) if(census_gfx[i].gen==gen) return &census_gfx[i];
    return NULL;
}
static void census_gfx_retire(uint64_t gen)
{
    CensusGfxResource *r;
    if(!gen || !census_mutate_begin()) return;
    pthread_mutex_lock(&census_lock);
    if((r=census_gfx_by_gen_locked(gen))) { r->live=0; r->unregister_ns=census_now(); census_ledger_changed_locked(FALSE); }
    pthread_mutex_unlock(&census_lock);
    census_mutate_end();
}
static unsigned census_outcome(CUresult result, unsigned branch)
{
    if(branch==CB_FORCED_REJECT) return CO_FORCED_REJECT;
    switch(result) {
    case CUDA_SUCCESS: return CO_SUCCESS;
    case 211: return CO_NOT_MAPPED;
    case 208: return CO_ALREADY_MAPPED;
    case CUDA_ERROR_INVALID_CONTEXT: return CO_CONTEXT;
    case CUDA_ERROR_INVALID_VALUE: return CO_INVALID_VALUE;
    case CUDA_ERROR_NOT_SUPPORTED: return CO_NOT_SUPPORTED;
    default: return CO_OTHER;
    }
}

/* ---- exits called from the wrappers (exactly one per census_enter) ---- */
static void census_exit_count(CensusToken *tok, unsigned api, CUresult result)
{
    if(!census_exit_begin(tok)) return;
    census_count(&census_counters[api],result,FALSE);
    census_exit_end(tok);
}
/* Identity at return is valid for the submission only if no ledger change
 * and no allocation/free hook was in flight between entry and return. */
static BOOL census_identity_ambiguous_locked(const CensusToken *tok)
{
    return tok->pending || __atomic_load_n(&census_ledger_pending,__ATOMIC_SEQ_CST)
        || __atomic_load_n(&census_ledger_version,__ATOMIC_SEQ_CST)!=tok->version;
}
static void census_exit_transfer(CensusToken *tok, unsigned api, CUresult result, unsigned flags,
                                 const CUDA_MEMCPY3D *copy3d, const CUDA_MEMCPY2D *copy2d,
                                 unsigned src_type, uint64_t src_host, uint64_t src_device, uint64_t src_array, uint64_t src_offset,
                                 unsigned dst_type, uint64_t dst_host, uint64_t dst_device, uint64_t dst_array, uint64_t dst_offset,
                                 uint64_t byte_count, CUstream stream)
{
    CensusTransfer *r;
    BOOL in_window;
    if(!census_exit_begin(tok)) return;
    in_window=tok->bucket<CENSUS_BUCKETS;
    census_count(&census_counters[api],result,in_window);
    if(in_window && (r=census_claim(&census_counters[api],tok->bucket,census_transfers,sizeof(*r),CENSUS_TRANSFER_LIMIT,
                                     &census_transfer_next,census_transfer_bucket,&census_transfer_overflow))) {
        CensusIdentity src, dst;
        r->api=api; r->result=result; r->flags=flags; r->t_entry=tok->t0; r->t_return=census_now();
        r->stream=(uintptr_t)stream; r->args=(uintptr_t)(copy3d?(const void*)copy3d:(const void*)copy2d);
        if(copy3d || copy2d || (flags&CF_V1_UNDECODED)) {
            /* Struct arguments are read only after the driver accepted them. */
            if(flags&CF_V1_UNDECODED) {}
            else if(result) r->flags|=CF_ARGS_UNREAD;
            else if(copy3d) {
                r->src_type=copy3d->srcMemoryType; r->dst_type=copy3d->dstMemoryType;
                r->src_host=(uintptr_t)copy3d->srcHost; r->src_device=copy3d->srcDevice; r->src_array=(uintptr_t)copy3d->srcArray;
                r->dst_host=(uintptr_t)copy3d->dstHost; r->dst_device=copy3d->dstDevice; r->dst_array=(uintptr_t)copy3d->dstArray;
                r->src_x=copy3d->srcXInBytes; r->src_y=copy3d->srcY; r->src_z=copy3d->srcZ;
                r->dst_x=copy3d->dstXInBytes; r->dst_y=copy3d->dstY; r->dst_z=copy3d->dstZ;
                r->width_bytes=copy3d->WidthInBytes; r->height=copy3d->Height; r->depth=copy3d->Depth;
                r->src_pitch=copy3d->srcPitch; r->src_height=copy3d->srcHeight; r->dst_pitch=copy3d->dstPitch; r->dst_height=copy3d->dstHeight;
            }
            else {
                r->src_type=copy2d->srcMemoryType; r->dst_type=copy2d->dstMemoryType;
                r->src_host=(uintptr_t)copy2d->srcHost; r->src_device=copy2d->srcDevice; r->src_array=(uintptr_t)copy2d->srcArray;
                r->dst_host=(uintptr_t)copy2d->dstHost; r->dst_device=copy2d->dstDevice; r->dst_array=(uintptr_t)copy2d->dstArray;
                r->src_x=copy2d->srcXInBytes; r->src_y=copy2d->srcY; r->dst_x=copy2d->dstXInBytes; r->dst_y=copy2d->dstY;
                r->width_bytes=copy2d->WidthInBytes; r->height=copy2d->Height; r->depth=1;
                r->src_pitch=copy2d->srcPitch; r->dst_pitch=copy2d->dstPitch;
            }
        }
        else { /* by-value arguments: no dereference, safe on any result */
            r->src_type=src_type; r->dst_type=dst_type;
            r->src_host=src_host; r->src_device=src_device; r->src_array=src_array; r->src_x=src_offset;
            r->dst_host=dst_host; r->dst_device=dst_device; r->dst_array=dst_array; r->dst_x=dst_offset;
            r->byte_count=byte_count;
        }
        pthread_mutex_lock(&census_lock);
        if(census_alloc_untracked || census_gfx_untracked || census_gfx_setup_untracked
           || census_setup_skipped[CENSUS_ROLE_LEDGER] || census_setup_skipped[CENSUS_ROLE_GRAPHICS]) r->flags|=CF_LEDGER_INCOMPLETE;
#define CENSUS_SIDE(type,host,device,array,id) \
        if((type)==1) census_identify_locked(host,0,&id); \
        else if((type)==2) census_identify_locked(device,1,&id); \
        else if((type)==3) census_identify_locked(array,2,&id); \
        else if((type)==4) census_identify_locked(device,3,&id); \
        else memset(&id,0,sizeof(id));
        CENSUS_SIDE(r->src_type,r->src_host,r->src_device,r->src_array,src)
        CENSUS_SIDE(r->dst_type,r->dst_host,r->dst_device,r->dst_array,dst)
#undef CENSUS_SIDE
        if(src.ambiguous || dst.ambiguous || census_identity_ambiguous_locked(tok)) {
            r->flags|=CF_IDENTITY_AMBIGUOUS; census_ambiguous_lookups++;
        }
        pthread_mutex_unlock(&census_lock);
        r->src_kind=src.kind; r->src_gen=src.gen; r->src_offset=src.offset;
        r->dst_kind=dst.kind; r->dst_gen=dst.gen; r->dst_offset=dst.offset;
        census_commit(&r->committed,&r->seq,&r->tid);
    }
    census_exit_end(tok);
}
/* Raw observation only: no stream/event lifetime, context or completion claim. */
static void census_exit_sync(CensusToken *tok, unsigned api, CUresult result, unsigned flags, CUstream stream, CUevent event)
{
    CensusSync *r;
    BOOL in_window;
    if(!census_exit_begin(tok)) return;
    in_window=tok->bucket<CENSUS_BUCKETS;
    census_count(&census_counters[api],result,in_window);
    /* Not-ready queries are counted, not recorded. */
    if(in_window && !((api==CA_cuStreamQuery || api==CA_cuStreamQuery_ptsz || api==CA_cuEventQuery) && result)
       && (r=census_claim(&census_counters[api],tok->bucket,census_syncs,sizeof(*r),CENSUS_SYNC_LIMIT,
                          &census_sync_next,census_sync_bucket,&census_sync_overflow))) {
        r->api=api; r->result=result; r->flags=flags; r->t_entry=tok->t0; r->t_return=census_now();
        r->stream=(uintptr_t)stream; r->event=(uintptr_t)event;
        census_commit(&r->committed,&r->seq,&r->tid);
    }
    census_exit_end(tok);
}
/* Successful allocation: base is valid only when result==0 (callers read
 * output pointers only then). A free observed during this allocation's flight
 * at the same base makes the new generation order-ambiguous, never live. */
static void census_exit_alloc(CensusToken *tok, unsigned api, CUresult result, unsigned kind, uint64_t base, uint64_t size,
                              unsigned flags, uint64_t stream, unsigned width, unsigned height, unsigned depth, unsigned format)
{
    CensusAlloc *a;
    unsigned i, state=CA_STATE_LIVE;
    if(!census_exit_begin(tok)) return;
    census_count(&census_counters[api],result,FALSE);
    pthread_mutex_lock(&census_lock);
    if(!result && base) {
        for(i=0;i<census_alloc_count;i++)
            if((census_allocs[i].flags&CA_FLAG_ORPHAN_FREE) && census_allocs[i].base==base && census_allocs[i].free_ns>=tok->t0) {
                state=CA_STATE_ORDER_AMBIGUOUS; census_order_ambiguous_allocs++; break;
            }
        if(!(a=census_alloc_slot_locked())) census_alloc_untracked++;
        else {
            memset(a,0,sizeof(*a));
            a->base=base; a->size=size; a->gen=++census_generation; a->alloc_ns=census_now(); a->alloc_stream=stream;
            a->kind=kind; a->cuda_flags=flags; a->state=state; a->api=api;
            a->width=width; a->height=height; a->depth=depth; a->format=format;
        }
        census_ledger_changed_locked(TRUE);
    }
    else __atomic_sub_fetch(&census_ledger_pending,1,__ATOMIC_SEQ_CST);
    pthread_mutex_unlock(&census_lock);
    census_exit_end(tok);
}
/* Successful free/unregister/destroy. Sync frees retire; stream-ordered frees
 * are only "submitted". With two live generations at one base (a reuse whose
 * allocation hook ran first) the older one is the one being freed. Unknown
 * bases leave a tombstone so a concurrent allocation cannot become live. */
static void census_exit_free(CensusToken *tok, unsigned api, CUresult result, unsigned first_kind, unsigned last_kind,
                             uint64_t base, BOOL async, uint64_t stream)
{
    CensusAlloc *a, *oldest=NULL;
    unsigned i;
    if(!census_exit_begin(tok)) return;
    census_count(&census_counters[api],result,FALSE);
    pthread_mutex_lock(&census_lock);
    if(!result && base) {
        for(i=0;i<census_alloc_count;i++) {
            a=&census_allocs[i];
            if((a->state==CA_STATE_LIVE || a->state==CA_STATE_ORDER_AMBIGUOUS) && !(a->flags&CA_FLAG_ORPHAN_FREE)
               && a->base==base && a->kind>=first_kind && a->kind<=last_kind && (!oldest || a->gen<oldest->gen)) oldest=a;
        }
        if(oldest) {
            oldest->state=async?CA_STATE_FREE_SUBMITTED:CA_STATE_RETIRED;
            oldest->free_ns=census_now(); oldest->free_api=api; oldest->free_stream=stream;
        }
        else {
            census_free_unmatched++;
            if(!(a=census_alloc_slot_locked())) census_alloc_untracked++;
            else {
                memset(a,0,sizeof(*a));
                a->base=base; a->gen=++census_generation; a->free_ns=census_now(); a->free_api=api; a->free_stream=stream;
                a->kind=first_kind; a->flags=CA_FLAG_ORPHAN_FREE; a->state=CA_STATE_RETIRED;
            }
        }
        census_ledger_changed_locked(TRUE);
    }
    else __atomic_sub_fetch(&census_ledger_pending,1,__ATOMIC_SEQ_CST);
    pthread_mutex_unlock(&census_lock);
    census_exit_end(tok);
}
static void census_exit_alias(CensusToken *tok, unsigned api, CUresult result, uint64_t host, uint64_t device)
{
    unsigned i;
    if(!census_exit_begin(tok)) return;
    census_count(&census_counters[api],result,FALSE);
    pthread_mutex_lock(&census_lock);
    if(!result && device) {
        for(i=0;i<census_alloc_count;i++) {
            CensusAlloc *a=&census_allocs[i];
            if(a->state==CA_STATE_LIVE && census_host_kind(a->kind) && !(a->flags&CA_FLAG_ORPHAN_FREE)
               && host>=a->base && host-a->base<a->size) { a->alias=device-(host-a->base); break; }
        }
        census_ledger_changed_locked(TRUE);
    }
    else __atomic_sub_fetch(&census_ledger_pending,1,__ATOMIC_SEQ_CST);
    pthread_mutex_unlock(&census_lock);
    census_exit_end(tok);
}
/* Graphics call. Generations come only from the relay (captured under its
 * registry lock); forwarded calls carry opaque handle values and no
 * generation. value/arg_* are by-value or read by the caller only on success. */
static void census_exit_gfx(CensusToken *tok, unsigned api, unsigned op, CUresult result, const CensusRelayInfo *info,
                            uint64_t resource, unsigned count, CUstream stream, uint64_t value, uint64_t arg_a, uint64_t arg_b)
{
    CensusGfxEvent *r;
    unsigned i, outcome, slots, flags=0;
    if(!census_exit_begin(tok)) return;
    census_count(&census_counters[api],result,FALSE);
    outcome=census_outcome(result,info->branch);
    slots=info->found<CENSUS_BATCH_SLOTS?info->found:CENSUS_BATCH_SLOTS;
    if(info->branch==CB_FORWARDED) flags|=CF_FORWARDED;
    if(info->found>CENSUS_BATCH_SLOTS) flags|=CF_BATCH_TRUNCATED;
    pthread_mutex_lock(&census_lock);
    if(!slots) census_gfx_unknown[op][outcome]++;
    for(i=0;i<slots;i++) {
        CensusGfxResource *g=census_gfx_by_gen_locked(info->gen[i]);
        if(g) g->outcomes[op][outcome]++; else census_gfx_unknown[op][outcome]++;
    }
    pthread_mutex_unlock(&census_lock);
    if((r=census_claim(&census_counters[api],0,census_gfx_events,sizeof(*r),CENSUS_GFX_EVENT_LIMIT,
                       &census_gfx_event_next,NULL,&census_gfx_event_overflow))) {
        r->api=api; r->result=result; r->flags=flags; r->branch=info->branch; r->count=count; r->found=info->found;
        r->t_entry=tok->t0; r->t_return=census_now(); r->stream=(uintptr_t)stream;
        r->resource=resource; r->resource_gen=info->gen[0]; r->value=value; r->arg_a=arg_a; r->arg_b=arg_b;
        r->width=info->width; r->height=info->height; r->format=info->format; r->mips=info->mips;
        r->array_size=info->array_size; r->samples=info->samples;
        memcpy(r->batch_gen,info->gen,sizeof(r->batch_gen));
        census_commit(&r->committed,&r->seq,&r->tid);
    }
    census_exit_end(tok);
}
/* Relay-internal staging (inside an admitted graphics call). */
static unsigned long long census_staging_begin(void)
{
    unsigned long long now;
    if(__atomic_load_n(&census_state,__ATOMIC_ACQUIRE)!=CS_ON || census_owner!=getpid()) return 0;
    return (now=census_now()) ? now : 1;
}
static void census_staging_end(unsigned kind, unsigned long long t0, CUresult result, BOOL ptsz, uint64_t gen,
                               CUstream stream, uint64_t bytes, uint64_t epoch)
{
    unsigned long long start;
    unsigned bucket=CENSUS_BUCKETS;
    CensusStaging *r;
    if(!t0 || !census_mutate_begin()) return;
    start=__atomic_load_n(&census_window_start,__ATOMIC_ACQUIRE);
    if(start && t0>=start && t0-start<CENSUS_WINDOW_NS) bucket=(unsigned)((t0-start)/CENSUS_BUCKET_NS);
    census_count(&census_staging_counter,result,bucket<CENSUS_BUCKETS);
    if(bucket<CENSUS_BUCKETS && (r=census_claim(&census_staging_counter,bucket,census_stagings,sizeof(*r),CENSUS_STAGING_LIMIT,
                                                 &census_staging_next,census_staging_bucket,&census_staging_overflow))) {
        r->kind=kind; r->result=result; r->ptsz=ptsz; r->t_entry=t0; r->t_return=census_now();
        r->resource_gen=gen; r->stream=(uintptr_t)stream; r->bytes=bytes; r->map_epoch=epoch;
        census_commit(&r->committed,&r->seq,&r->tid);
    }
    census_mutate_end();
}

/* ---- publication (frozen state only) ---- */
typedef struct { const char *name; size_t offset, size; } CensusField;
#define CENSUS_FIELD(type,field) {#field,offsetof(type,field),sizeof(((type*)0)->field)}
static const CensusField census_transfer_fields[]={
    CENSUS_FIELD(CensusTransfer,committed),CENSUS_FIELD(CensusTransfer,seq),CENSUS_FIELD(CensusTransfer,api),
    CENSUS_FIELD(CensusTransfer,tid),CENSUS_FIELD(CensusTransfer,result),CENSUS_FIELD(CensusTransfer,flags),
    CENSUS_FIELD(CensusTransfer,src_type),CENSUS_FIELD(CensusTransfer,dst_type),CENSUS_FIELD(CensusTransfer,t_entry),
    CENSUS_FIELD(CensusTransfer,t_return),CENSUS_FIELD(CensusTransfer,stream),CENSUS_FIELD(CensusTransfer,args),
    CENSUS_FIELD(CensusTransfer,src_host),CENSUS_FIELD(CensusTransfer,src_device),CENSUS_FIELD(CensusTransfer,src_array),
    CENSUS_FIELD(CensusTransfer,dst_host),CENSUS_FIELD(CensusTransfer,dst_device),CENSUS_FIELD(CensusTransfer,dst_array),
    CENSUS_FIELD(CensusTransfer,src_x),CENSUS_FIELD(CensusTransfer,src_y),CENSUS_FIELD(CensusTransfer,src_z),
    CENSUS_FIELD(CensusTransfer,dst_x),CENSUS_FIELD(CensusTransfer,dst_y),CENSUS_FIELD(CensusTransfer,dst_z),
    CENSUS_FIELD(CensusTransfer,width_bytes),CENSUS_FIELD(CensusTransfer,height),CENSUS_FIELD(CensusTransfer,depth),
    CENSUS_FIELD(CensusTransfer,src_pitch),CENSUS_FIELD(CensusTransfer,src_height),CENSUS_FIELD(CensusTransfer,dst_pitch),
    CENSUS_FIELD(CensusTransfer,dst_height),CENSUS_FIELD(CensusTransfer,byte_count),CENSUS_FIELD(CensusTransfer,src_kind),
    CENSUS_FIELD(CensusTransfer,dst_kind),CENSUS_FIELD(CensusTransfer,src_gen),CENSUS_FIELD(CensusTransfer,dst_gen),
    CENSUS_FIELD(CensusTransfer,src_offset),CENSUS_FIELD(CensusTransfer,dst_offset)};
static const CensusField census_sync_fields[]={
    CENSUS_FIELD(CensusSync,committed),CENSUS_FIELD(CensusSync,seq),CENSUS_FIELD(CensusSync,api),CENSUS_FIELD(CensusSync,tid),
    CENSUS_FIELD(CensusSync,result),CENSUS_FIELD(CensusSync,flags),CENSUS_FIELD(CensusSync,t_entry),
    CENSUS_FIELD(CensusSync,t_return),CENSUS_FIELD(CensusSync,stream),CENSUS_FIELD(CensusSync,event)};
static const CensusField census_staging_fields[]={
    CENSUS_FIELD(CensusStaging,committed),CENSUS_FIELD(CensusStaging,seq),CENSUS_FIELD(CensusStaging,kind),
    CENSUS_FIELD(CensusStaging,tid),CENSUS_FIELD(CensusStaging,result),CENSUS_FIELD(CensusStaging,ptsz),
    CENSUS_FIELD(CensusStaging,t_entry),CENSUS_FIELD(CensusStaging,t_return),CENSUS_FIELD(CensusStaging,resource_gen),
    CENSUS_FIELD(CensusStaging,stream),CENSUS_FIELD(CensusStaging,bytes),CENSUS_FIELD(CensusStaging,map_epoch)};
static const CensusField census_gfx_event_fields[]={
    CENSUS_FIELD(CensusGfxEvent,committed),CENSUS_FIELD(CensusGfxEvent,seq),CENSUS_FIELD(CensusGfxEvent,api),
    CENSUS_FIELD(CensusGfxEvent,tid),CENSUS_FIELD(CensusGfxEvent,result),CENSUS_FIELD(CensusGfxEvent,flags),
    CENSUS_FIELD(CensusGfxEvent,branch),CENSUS_FIELD(CensusGfxEvent,count),CENSUS_FIELD(CensusGfxEvent,found),
    CENSUS_FIELD(CensusGfxEvent,t_entry),CENSUS_FIELD(CensusGfxEvent,t_return),CENSUS_FIELD(CensusGfxEvent,stream),
    CENSUS_FIELD(CensusGfxEvent,resource),CENSUS_FIELD(CensusGfxEvent,resource_gen),CENSUS_FIELD(CensusGfxEvent,value),
    CENSUS_FIELD(CensusGfxEvent,arg_a),CENSUS_FIELD(CensusGfxEvent,arg_b),CENSUS_FIELD(CensusGfxEvent,width),
    CENSUS_FIELD(CensusGfxEvent,height),CENSUS_FIELD(CensusGfxEvent,format),CENSUS_FIELD(CensusGfxEvent,mips),
    CENSUS_FIELD(CensusGfxEvent,array_size),CENSUS_FIELD(CensusGfxEvent,samples),CENSUS_FIELD(CensusGfxEvent,batch_gen)};
static const CensusField census_alloc_fields[]={
    CENSUS_FIELD(CensusAlloc,base),CENSUS_FIELD(CensusAlloc,size),CENSUS_FIELD(CensusAlloc,alias),CENSUS_FIELD(CensusAlloc,gen),
    CENSUS_FIELD(CensusAlloc,alloc_ns),CENSUS_FIELD(CensusAlloc,free_ns),CENSUS_FIELD(CensusAlloc,alloc_stream),
    CENSUS_FIELD(CensusAlloc,free_stream),CENSUS_FIELD(CensusAlloc,kind),CENSUS_FIELD(CensusAlloc,flags),
    CENSUS_FIELD(CensusAlloc,state),CENSUS_FIELD(CensusAlloc,api),CENSUS_FIELD(CensusAlloc,free_api),
    CENSUS_FIELD(CensusAlloc,width),CENSUS_FIELD(CensusAlloc,height),CENSUS_FIELD(CensusAlloc,depth),
    CENSUS_FIELD(CensusAlloc,format),CENSUS_FIELD(CensusAlloc,cuda_flags)};
static const CensusField census_gfx_fields[]={
    CENSUS_FIELD(CensusGfxResource,handle),CENSUS_FIELD(CensusGfxResource,texture),CENSUS_FIELD(CensusGfxResource,array),
    CENSUS_FIELD(CensusGfxResource,gen),CENSUS_FIELD(CensusGfxResource,register_ns),CENSUS_FIELD(CensusGfxResource,unregister_ns),
    CENSUS_FIELD(CensusGfxResource,context),CENSUS_FIELD(CensusGfxResource,live),CENSUS_FIELD(CensusGfxResource,width),
    CENSUS_FIELD(CensusGfxResource,height),CENSUS_FIELD(CensusGfxResource,outcomes)};
#undef CENSUS_FIELD

typedef struct { char *data; size_t used, size; BOOL failed; } CensusText;
static void census_text(CensusText *b, const char *format, ...)
{
    va_list args;
    int n;
    if(b->failed) return;
    va_start(args,format); n=vsnprintf(b->data+b->used,b->size-b->used,format,args); va_end(args);
    if(n<0 || (size_t)n>=b->size-b->used) b->failed=TRUE; else b->used+=n;
}
static void census_layout(CensusText *b, const char *name, const CensusField *fields, size_t count, size_t record)
{
    size_t i;
    census_text(b,"\"%s\":{\"record_bytes\":%zu,\"fields\":[",name,record);
    for(i=0;i<count;i++) census_text(b,"%s[\"%s\",%zu,%zu]",i?",":"",fields[i].name,fields[i].offset,fields[i].size);
    census_text(b,"]}");
}
static unsigned census_table(const char *suffix, const void *base, size_t record, unsigned claimed, unsigned limit,
                             unsigned *committed, unsigned *io_errors)
{
    char name[96];
    unsigned i, n=claimed<limit?claimed:limit;
    *committed=0;
    for(i=0;i<n;i++) if(*(const uint32_t*)((const char*)base+(size_t)i*record)) ++*committed;
    snprintf(name,sizeof(name),"%u-census-%s.bin",(unsigned)getpid(),suffix);
    if(n && census_publish_file(name,base,(size_t)n*record)) ++*io_errors;
    return n;
}
static void census_bucket_list(CensusText *b, const char *name, const unsigned *buckets, unsigned quota, BOOL last)
{
    unsigned i;
    census_text(b,"\"%s\":{\"quota\":%u,\"attempts\":[",name,quota);
    for(i=0;i<CENSUS_BUCKETS;i++) census_text(b,"%s%u",i?",":"",buckets[i]);
    census_text(b,"]}%s",last?"":",");
}
/* Published when mutation sections did not drain: no table, counter or
 * ledger is read, because none of them is frozen. */
static void census_publish_minimal(const char *reason, unsigned long long mutating)
{
    char name[96], json[512];
    int n=snprintf(json,sizeof(json),"{\"schema\":2,\"kind\":\"zluda_d3d11_transfer_census\",\"pid\":%u,\"reason\":\"%s\","
                   "\"published\":\"minimal\",\"mutation_drain_timeout\":true,\"mutating_at_timeout\":%llu,"
                   "\"records_complete\":false,\"graphics_complete\":false,\"identity_tracking_complete\":false,"
                   "\"identity_unambiguous\":false,\"attribution_ready\":false,\"attribution_blockers\":[\"mutation_drain_timeout\"]}\n",
                   (unsigned)getpid(),reason,mutating);
    snprintf(name,sizeof(name),"%u-census-summary.json",(unsigned)getpid());
    if(n<0 || (size_t)n>=sizeof(json) || census_publish_file(name,json,n))
        ERR("[ZLUDA] transfer census summary failed; evidence incomplete\n");
}
static void census_publish(const char *reason)
{
    CensusText b={0};
    char name[96];
    unsigned expected=0, i, polls, io_errors=0, lost_all, lost_window, lost_graphics;
    unsigned transfers, syncs, stagings, events, c_transfers, c_syncs, c_stagings, c_events;
    uint64_t mutating, inflight;
    BOOL window_end, records_complete, graphics_complete, identity_tracking_complete, identity_unambiguous, attribution_ready;
    unsigned ledger_updates_lost;
    unsigned long long setup_untracked;
    if(!__atomic_compare_exchange_n(&census_publish_state,&expected,1,FALSE,__ATOMIC_SEQ_CST,__ATOMIC_SEQ_CST)) return;
    /* Close admission, then mutation; wait only for mutation sections. */
    __atomic_fetch_or(&census_inflight,CENSUS_GATE_CLOSED,__ATOMIC_SEQ_CST);
    __atomic_fetch_or(&census_mutating,CENSUS_GATE_CLOSED,__ATOMIC_SEQ_CST);
    CENSUS_TEST_POINT(CENSUS_TP_PUBLISH);
    for(polls=0;(mutating=__atomic_load_n(&census_mutating,__ATOMIC_SEQ_CST)&~CENSUS_GATE_CLOSED) && polls<CENSUS_DRAIN_WAIT_MS;polls++) Sleep(1);
    if(mutating) {
        census_publish_minimal(reason,mutating);
        __atomic_store_n(&census_publish_state,2,__ATOMIC_SEQ_CST);
        return;
    }
    /* Frozen from here on: admitted hooks that have not filled are lost. */
    inflight=__atomic_load_n(&census_inflight,__ATOMIC_SEQ_CST);
    lost_all=CENSUS_GATE_FIELD(inflight,0); lost_window=CENSUS_GATE_FIELD(inflight,21); lost_graphics=CENSUS_GATE_FIELD(inflight,42);
    transfers=census_table("transfers",census_transfers,sizeof(CensusTransfer),census_transfer_next,CENSUS_TRANSFER_LIMIT,&c_transfers,&io_errors);
    syncs=census_table("sync",census_syncs,sizeof(CensusSync),census_sync_next,CENSUS_SYNC_LIMIT,&c_syncs,&io_errors);
    stagings=census_table("staging",census_stagings,sizeof(CensusStaging),census_staging_next,CENSUS_STAGING_LIMIT,&c_stagings,&io_errors);
    events=census_table("graphics",census_gfx_events,sizeof(CensusGfxEvent),census_gfx_event_next,CENSUS_GFX_EVENT_LIMIT,&c_events,&io_errors);
    snprintf(name,sizeof(name),"%u-census-allocations.bin",(unsigned)getpid());
    if(census_alloc_count && census_publish_file(name,census_allocs,(size_t)census_alloc_count*sizeof(CensusAlloc))) io_errors++;
    snprintf(name,sizeof(name),"%u-census-resources.bin",(unsigned)getpid());
    if(census_gfx_count && census_publish_file(name,census_gfx,(size_t)census_gfx_count*sizeof(CensusGfxResource))) io_errors++;
    window_end=!strcmp(reason,"window_end");
    /* Admitted allocation/free hooks whose ledger update never happened. */
    ledger_updates_lost=census_ledger_pending;
    setup_untracked=census_setup_skipped[CENSUS_ROLE_LEDGER]+census_setup_skipped[CENSUS_ROLE_GRAPHICS]+census_gfx_setup_untracked;
    /* Coverage verdicts. identity_tracking_complete says every identity change
     * was tracked; it can coexist with per-record ambiguity, which
     * identity_unambiguous reports. Only attribution_ready combines them. */
    records_complete=window_end && !io_errors && !lost_window && !lost_graphics
        && !census_setup_skipped[CENSUS_ROLE_WINDOWED] && !census_setup_skipped[CENSUS_ROLE_GRAPHICS]
        && !census_transfer_overflow && !census_sync_overflow && !census_staging_overflow
        && c_transfers==transfers && c_syncs==syncs && c_stagings==stagings;
    graphics_complete=!io_errors && !lost_graphics && !census_setup_skipped[CENSUS_ROLE_GRAPHICS] && !census_gfx_setup_untracked
        && !census_gfx_event_overflow && !census_gfx_untracked && c_events==events;
    identity_tracking_complete=!io_errors && !setup_untracked && !census_alloc_untracked && !census_gfx_untracked && !ledger_updates_lost;
    identity_unambiguous=!census_ambiguous_lookups && !census_order_ambiguous_allocs;
    attribution_ready=records_complete && graphics_complete && identity_tracking_complete && identity_unambiguous;
    b.size=1u<<20; b.data=malloc(b.size); if(!b.data) b.failed=TRUE;
    census_text(&b,"{\"schema\":2,\"kind\":\"zluda_d3d11_transfer_census\",\"pid\":%u,\"reason\":\"%s\",\"published\":\"full\","
                "\"records_complete\":%s,\"graphics_complete\":%s,\"identity_tracking_complete\":%s,\"identity_unambiguous\":%s,"
                "\"attribution_ready\":%s,"
                "\"ambiguous_transfer_lookups\":%llu,\"lost_at_cutoff\":{\"all\":%u,\"window\":%u,\"graphics\":%u},"
                "\"setup_skipped\":[%llu,%llu,%llu,%llu],\"setup_skipped_roles\":[\"count\",\"windowed\",\"ledger\",\"graphics\"],"
                "\"arm_detected_ns\":%llu,\"window_start_ns\":%llu,\"window_ns\":%llu,\"bucket_ns\":%llu,\"close_grace_ns\":%llu,"
                "\"arm_mtime_realtime_ns\":%llu,\"arm_detect_realtime_ns\":%llu,\"summary_ns\":%llu,\"drain_polls\":%u,\"io_errors\":%u,",
                (unsigned)getpid(),reason,records_complete?"true":"false",graphics_complete?"true":"false",
                identity_tracking_complete?"true":"false",identity_unambiguous?"true":"false",attribution_ready?"true":"false",
                census_ambiguous_lookups,lost_all,lost_window,lost_graphics,
                census_setup_skipped[0],census_setup_skipped[1],census_setup_skipped[2],census_setup_skipped[3],
                census_arm_ns,census_window_start,(unsigned long long)CENSUS_WINDOW_NS,(unsigned long long)CENSUS_BUCKET_NS,
                (unsigned long long)CENSUS_CLOSE_GRACE_NS,census_arm_mtime_rt,census_arm_detect_rt,census_now(),polls,io_errors);
    census_text(&b,"\"attribution_blockers\":[");
    {   const char *sep="";
#define CENSUS_BLOCKER(cond,name) if(cond) { census_text(&b,"%s\"%s\"",sep,name); sep=","; }
        CENSUS_BLOCKER(!window_end,"not_window_end")
        CENSUS_BLOCKER(io_errors,"io_errors")
        CENSUS_BLOCKER(lost_window,"window_calls_lost_at_cutoff")
        CENSUS_BLOCKER(lost_graphics,"graphics_calls_lost_at_cutoff")
        CENSUS_BLOCKER(ledger_updates_lost,"ledger_updates_lost_at_cutoff")
        CENSUS_BLOCKER(census_setup_skipped[CENSUS_ROLE_WINDOWED],"windowed_setup_skips")
        CENSUS_BLOCKER(setup_untracked,"identity_setup_skips")
        CENSUS_BLOCKER(census_transfer_overflow || census_sync_overflow || census_staging_overflow,"record_overflow")
        CENSUS_BLOCKER(census_gfx_event_overflow,"graphics_event_overflow")
        CENSUS_BLOCKER(c_transfers!=transfers || c_syncs!=syncs || c_stagings!=stagings || c_events!=events,"uncommitted_slots")
        CENSUS_BLOCKER(census_alloc_untracked,"allocations_untracked")
        CENSUS_BLOCKER(census_gfx_untracked,"resources_untracked")
        CENSUS_BLOCKER(census_ambiguous_lookups,"ambiguous_transfer_identity")
        CENSUS_BLOCKER(census_order_ambiguous_allocs,"order_ambiguous_allocations")
#undef CENSUS_BLOCKER
    }
    census_text(&b,"],");
    census_text(&b,"\"tables\":{\"transfers\":[%u,%u,%llu],\"sync\":[%u,%u,%llu],\"staging\":[%u,%u,%llu],\"graphics\":[%u,%u,%llu],"
                "\"allocations\":%u,\"resources\":%u},\"table_columns\":[\"claimed\",\"committed\",\"overflow\"],",
                transfers,c_transfers,census_transfer_overflow,syncs,c_syncs,census_sync_overflow,
                stagings,c_stagings,census_staging_overflow,events,c_events,census_gfx_event_overflow,
                census_alloc_count,census_gfx_count);
    census_text(&b,"\"ledger\":{\"generations\":%llu,\"alloc_recycled\":%llu,\"alloc_untracked\":%llu,\"free_unmatched\":%llu,"
                "\"order_ambiguous_allocs\":%llu,\"resource_untracked\":%llu,\"resource_setup_untracked\":%llu,\"pending_at_cutoff\":%u},",
                census_generation,census_alloc_recycled,census_alloc_untracked,census_free_unmatched,
                census_order_ambiguous_allocs,census_gfx_untracked,census_gfx_setup_untracked,ledger_updates_lost);
    census_text(&b,"\"bucket_attempts\":{");
    census_bucket_list(&b,"transfers",census_transfer_bucket,CENSUS_TRANSFER_LIMIT/CENSUS_BUCKETS,FALSE);
    census_bucket_list(&b,"sync",census_sync_bucket,CENSUS_SYNC_LIMIT/CENSUS_BUCKETS,FALSE);
    census_bucket_list(&b,"staging",census_staging_bucket,CENSUS_STAGING_LIMIT/CENSUS_BUCKETS,TRUE);
    census_text(&b,"},\"unknown_generation_outcomes\":[");
    for(i=0;i<CG_COUNT;i++) {
        unsigned j;
        census_text(&b,"%s[",i?",":"");
        for(j=0;j<CO_COUNT;j++) census_text(&b,"%s%llu",j?",":"",census_gfx_unknown[i][j]);
        census_text(&b,"]");
    }
    census_text(&b,"],\"apis\":{");
    for(i=0;i<CA_COUNT;i++) {
        CensusCounter *c=&census_counters[i];
        census_text(&b,"%s\"%s\":[%u,%llu,%llu,%llu,%llu,%llu]",i?",":"",census_api_info[i].name,census_api_info[i].cls,
                    c->calls,c->errors,c->in_window,c->recorded,c->overflow);
    }
    census_text(&b,"},\"relay_staging\":[%llu,%llu,%llu,%llu,%llu]",census_staging_counter.calls,census_staging_counter.errors,
                census_staging_counter.in_window,census_staging_counter.recorded,census_staging_counter.overflow);
    census_text(&b,",\"api_columns\":[\"class\",\"calls\",\"errors\",\"in_window\",\"recorded\",\"overflow\"],\"layouts\":{");
    census_layout(&b,"transfers",census_transfer_fields,CENSUS_COUNT_OF(census_transfer_fields),sizeof(CensusTransfer)); census_text(&b,",");
    census_layout(&b,"sync",census_sync_fields,CENSUS_COUNT_OF(census_sync_fields),sizeof(CensusSync)); census_text(&b,",");
    census_layout(&b,"staging",census_staging_fields,CENSUS_COUNT_OF(census_staging_fields),sizeof(CensusStaging)); census_text(&b,",");
    census_layout(&b,"graphics",census_gfx_event_fields,CENSUS_COUNT_OF(census_gfx_event_fields),sizeof(CensusGfxEvent)); census_text(&b,",");
    census_layout(&b,"allocations",census_alloc_fields,CENSUS_COUNT_OF(census_alloc_fields),sizeof(CensusAlloc)); census_text(&b,",");
    census_layout(&b,"resources",census_gfx_fields,CENSUS_COUNT_OF(census_gfx_fields),sizeof(CensusGfxResource));
    census_text(&b,"}}\n");
    snprintf(name,sizeof(name),"%u-census-summary.json",(unsigned)getpid());
    /* Summary last: its presence marks the tables above as fully written. */
    if(b.failed || census_publish_file(name,b.data,b.used)) ERR("[ZLUDA] transfer census summary failed; evidence incomplete\n");
    free(b.data);
    __atomic_store_n(&census_publish_state,2,__ATOMIC_SEQ_CST);
}

/* Polls the census directory's arm file, opens the window, closes it after
 * the window plus a short grace and publishes once. Independent of any later
 * application call. Process exit before then publishes nothing. */
static DWORD WINAPI census_thread(void *module)
{
    unsigned long long started=census_now(), start, now, end;
    struct stat st;
    for(;;) {
        Sleep(CENSUS_ARM_POLL_MS);
        if(census_owner!=getpid()) FreeLibraryAndExitThread(module,0);
        now=census_now();
        if(!fstatat(census_dir,"arm",&st,AT_SYMLINK_NOFOLLOW) && S_ISREG(st.st_mode) && st.st_uid==geteuid() && (st.st_mode&0777)==0600) {
            struct timespec rt;
            census_arm_mtime_rt=(unsigned long long)st.st_mtim.tv_sec*1000000000+st.st_mtim.tv_nsec;
            if(!clock_gettime(CLOCK_REALTIME,&rt)) census_arm_detect_rt=(unsigned long long)rt.tv_sec*1000000000+rt.tv_nsec;
            break;
        }
        if(now-started>CENSUS_ARM_GIVE_UP_NS) { census_publish("never_armed"); FreeLibraryAndExitThread(module,0); }
    }
    if(!(start=census_now())) start=1;
    census_arm_ns=start;
    __atomic_store_n(&census_window_start,start,__ATOMIC_RELEASE);
    end=start+CENSUS_WINDOW_NS+CENSUS_CLOSE_GRACE_NS;
    while((now=census_now()) && now<end) Sleep((DWORD)((end-now)/1000000+1));
    census_publish("window_end");
    FreeLibraryAndExitThread(module,0);
    return 0;
}
