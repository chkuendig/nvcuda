/* Host-only tests for the D2 transfer census (DRAFT revision 3, written, NOT executed).
 *
 * Exercises the census header helpers directly; no Wine, CUDA, D3D or GPU.
 * LIMIT: these drive the hook functions the wrappers call, not the generated
 * nvcuda.c wrappers or the relay itself. Each test runs in a forked child
 * with a bounded timeout, so it starts from pristine static state; a crash
 * (e.g. a dereference of a protected page), a hang, a fork or a wait failure
 * all count as FAIL.
 *
 * Build (once a host-only gate is cleared):
 *   gcc -std=gnu11 -O1 -g -Wall -pthread \
 *       -DCENSUS_WINDOW_NS='UINT64_C(1000000000)' -DCENSUS_CLOSE_GRACE_NS='UINT64_C(100000000)' \
 *       -DCENSUS_DRAIN_WAIT_MS=200 -DCENSUS_ARM_POLL_MS=20 \
 *       -I../../include -I../../dlls/nvcuda census_tests.c -o census_tests
 */
#include "census_test_shim.h"
#include "d3d11_transfer_census.h"
#include <dirent.h>
#include <errno.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>

#define CHECK(c) do { if(!(c)) { fprintf(stderr,"%s:%d: CHECK failed: %s\n",__FILE__,__LINE__,#c); return 1; } } while(0)
#define CHECK_VOID(c) do { if(!(c)) { fprintf(stderr,"%s:%d: CHECK failed: %s\n",__FILE__,__LINE__,#c); setup_insert_failed=1; } } while(0)
#define MS(x) ((unsigned long long)(x)*1000000ull)
#define CHILD_TIMEOUT_S 20
static int setup_insert_failed;

static char test_dir[256];
static int setup_dir(void)
{
    snprintf(test_dir,sizeof(test_dir),"/tmp/census-test-XXXXXX");
    if(!mkdtemp(test_dir)) return -1; /* mkdtemp creates 0700 */
    return setenv("ZLUDA_D3D11_TRANSFER_CENSUS_DIR",test_dir,1);
}
static char *read_summary(void)
{
    static char buf[1<<20];
    char path[512];
    FILE *f;
    size_t n;
    snprintf(path,sizeof(path),"%s/%u-census-summary.json",test_dir,(unsigned)getpid());
    if(!(f=fopen(path,"r"))) return NULL;
    n=fread(buf,1,sizeof(buf)-1,f); fclose(f); buf[n]=0;
    return buf;
}
static int count_files(const char *suffix)
{
    DIR *d=opendir(test_dir);
    struct dirent *e;
    int n=0;
    size_t sl=strlen(suffix);
    if(!d) return -1;
    while((e=readdir(d))) { size_t l=strlen(e->d_name); if(l>=sl && !strcmp(e->d_name+l-sl,suffix)) n++; }
    closedir(d);
    return n;
}
/* First instrumented call: opt-in setup. */
static int start_census(void)
{
    CensusToken t;
    census_enter(&t,CENSUS_ROLE_COUNT);
    census_exit_count(&t,CA_cuMemcpyHtoD_v2,CUDA_SUCCESS);
    return __atomic_load_n(&census_state,__ATOMIC_SEQ_CST)==CS_ON ? 0 : -1;
}
static void open_window(void) { __atomic_store_n(&census_window_start,census_now(),__ATOMIC_RELEASE); }

/* Simulated application calls (the hook sequence the wrappers generate). */
static void sim_alloc(uint64_t base, uint64_t size)
{
    CensusToken t;
    census_enter(&t,CENSUS_ROLE_LEDGER);
    census_exit_alloc(&t,CA_cuMemAlloc_v2,CUDA_SUCCESS,CK_DEVICE,base,size,0,0,0,0,0,0);
}
static void sim_free(uint64_t base)
{
    CensusToken t;
    census_enter(&t,CENSUS_ROLE_LEDGER);
    census_exit_free(&t,CA_cuMemFree_v2,CUDA_SUCCESS,CK_DEVICE,CK_MANAGED,base,FALSE,0);
}
static CensusTransfer *sim_dtoh(uint64_t device, uint64_t host)
{
    CensusToken t;
    unsigned before=census_transfer_next;
    census_enter(&t,CENSUS_ROLE_WINDOWED);
    census_exit_transfer(&t,CA_cuMemcpyDtoH_v2,CUDA_SUCCESS,0,NULL,NULL,2,0,device,0,0,1,host,0,0,0,64,NULL);
    return census_transfer_next>before ? &census_transfers[census_transfer_next-1] : NULL;
}

/* ---- 1. blocked call at close ---- */
static pthread_mutex_t gate_lock=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t gate_cond=PTHREAD_COND_INITIALIZER;
static int gate_entered, gate_released;
static void *blocked_transfer(void *arg)
{
    CensusToken t;
    (void)arg;
    census_enter(&t,CENSUS_ROLE_WINDOWED);
    pthread_mutex_lock(&gate_lock);
    gate_entered=1; pthread_cond_broadcast(&gate_cond);
    while(!gate_released) pthread_cond_wait(&gate_cond,&gate_lock); /* "inside CUDA" */
    pthread_mutex_unlock(&gate_lock);
    census_exit_transfer(&t,CA_cuMemcpyDtoH_v2,CUDA_SUCCESS,0,NULL,NULL,2,0,0x1000,0,0,1,0x2000,0,0,0,64,NULL);
    return NULL;
}
static int test_blocked_call_close(void)
{
    pthread_t thread;
    unsigned long long t0, calls, next;
    char *summary;
    CHECK(!setup_dir()); CHECK(!start_census()); open_window();
    CHECK(sim_dtoh(0x10,0x20)); /* one completed in-window call */
    pthread_create(&thread,NULL,blocked_transfer,NULL);
    pthread_mutex_lock(&gate_lock); while(!gate_entered) pthread_cond_wait(&gate_cond,&gate_lock); pthread_mutex_unlock(&gate_lock);
    t0=census_now();
    census_publish("window_end");
    CHECK(census_now()-t0<MS(150)); /* never waited for the blocked call */
    CHECK((summary=read_summary()));
    CHECK(strstr(summary,"\"records_complete\":false"));
    CHECK(strstr(summary,"\"lost_at_cutoff\":{\"all\":1,\"window\":1,\"graphics\":0}"));
    CHECK(strstr(summary,"\"transfers\":[1,1,0]"));
    calls=census_counters[CA_cuMemcpyDtoH_v2].calls; next=census_transfer_next;
    pthread_mutex_lock(&gate_lock); gate_released=1; pthread_cond_broadcast(&gate_cond); pthread_mutex_unlock(&gate_lock);
    pthread_join(thread,NULL);
    CHECK(census_counters[CA_cuMemcpyDtoH_v2].calls==calls); /* late return changed nothing */
    CHECK(census_transfer_next==next);
    CHECK(count_files("-census-summary.json")==1);
    return 0;
}
/* Control: same sequence without a blocked call publishes complete records. */
static int test_clean_close_control(void)
{
    char *summary;
    CHECK(!setup_dir()); CHECK(!start_census()); open_window();
    CHECK(sim_dtoh(0x10,0x20));
    census_publish("window_end");
    CHECK((summary=read_summary()));
    CHECK(strstr(summary,"\"records_complete\":true"));
    CHECK(strstr(summary,"\"lost_at_cutoff\":{\"all\":0,\"window\":0,\"graphics\":0}"));
    return 0;
}
/* Positive control for the attribution gate. */
static int test_attribution_control(void)
{
    char *summary;
    CHECK(!setup_dir()); CHECK(!start_census()); open_window();
    sim_alloc(0x5000,256);
    CHECK(sim_dtoh(0x5000,0x20));
    census_publish("window_end");
    CHECK((summary=read_summary()));
    CHECK(strstr(summary,"\"attribution_ready\":true"));
    CHECK(strstr(summary,"\"attribution_blockers\":[]"));
    return 0;
}

/* ---- 2. setup race ---- */
static void slow_setup(int id) { if(id==CENSUS_TP_SETUP) Sleep(100); }
static pthread_barrier_t race_barrier;
static void *racer(void *arg)
{
    unsigned long long end;
    uintptr_t base=(uintptr_t)arg;
    pthread_barrier_wait(&race_barrier);
    end=census_now()+MS(200);
    while(census_now()<end) { sim_alloc(base,64); sim_free(base); }
    return NULL;
}
static int test_setup_race(void)
{
    pthread_t threads[8];
    CensusTransfer *r;
    char *summary;
    unsigned i;
    CHECK(!setup_dir());
    census_test_hook=slow_setup;
    pthread_barrier_init(&race_barrier,NULL,8);
    for(i=0;i<8;i++) pthread_create(&threads[i],NULL,racer,(void*)(uintptr_t)(0x100000*(i+1)));
    for(i=0;i<8;i++) pthread_join(threads[i],NULL);
    census_test_hook=NULL;
    CHECK(census_state==CS_ON);
    CHECK(census_setup_skipped[CENSUS_ROLE_LEDGER]>0); /* loss is counted, not silent */
    open_window();
    CHECK((r=sim_dtoh(0x100000,0x20)));
    CHECK(r->flags&CF_LEDGER_INCOMPLETE);
    census_publish("window_end");
    CHECK((summary=read_summary()));
    CHECK(strstr(summary,"\"identity_tracking_complete\":false"));
    CHECK(strstr(summary,"\"attribution_ready\":false"));
    CHECK(strstr(summary,"\"identity_setup_skips\""));
    return 0;
}

/* ---- 3. mutation (writer) drain timeout ---- */
static __thread int block_in_mutation;
static int mutation_released;
static void hold_mutation(int id)
{
    if(id!=CENSUS_TP_MUTATE || !block_in_mutation) return;
    pthread_mutex_lock(&gate_lock);
    gate_entered=1; pthread_cond_broadcast(&gate_cond);
    while(!mutation_released) pthread_cond_wait(&gate_cond,&gate_lock);
    pthread_mutex_unlock(&gate_lock);
}
static void *stuck_writer(void *arg)
{
    CensusToken t;
    (void)arg;
    census_enter(&t,CENSUS_ROLE_COUNT);
    block_in_mutation=1;
    census_exit_count(&t,CA_cuMemcpyHtoD_v2,CUDA_SUCCESS);
    return NULL;
}
static int test_writer_timeout(void)
{
    pthread_t thread;
    char *summary;
    CHECK(!setup_dir()); CHECK(!start_census()); open_window();
    CHECK(sim_dtoh(0x10,0x20));
    census_test_hook=hold_mutation;
    pthread_create(&thread,NULL,stuck_writer,NULL);
    pthread_mutex_lock(&gate_lock); while(!gate_entered) pthread_cond_wait(&gate_cond,&gate_lock); pthread_mutex_unlock(&gate_lock);
    census_publish("window_end");
    CHECK((summary=read_summary()));
    CHECK(strstr(summary,"\"published\":\"minimal\""));
    CHECK(strstr(summary,"\"mutation_drain_timeout\":true"));
    CHECK(count_files(".bin")==0); /* no live table was serialized */
    pthread_mutex_lock(&gate_lock); mutation_released=1; pthread_cond_broadcast(&gate_cond); pthread_mutex_unlock(&gate_lock);
    pthread_join(thread,NULL);
    CHECK(count_files(".bin")==0 && count_files("-census-summary.json")==1);
    return 0;
}

/* ---- 4. address reuse and register/unregister interleavings ---- */
static pthread_mutex_t mock_registry=PTHREAD_MUTEX_INITIALIZER; /* stands in for cpu_textures_lock */
static uint64_t mock_register(uint64_t handle, uint64_t array)
{
    uint64_t gen;
    pthread_mutex_lock(&mock_registry);
    gen=census_gfx_insert(handle,0x7000,array,0x8000,256,118);
    pthread_mutex_unlock(&mock_registry);
    return gen;
}
static void mock_unregister(uint64_t gen)
{
    pthread_mutex_lock(&mock_registry);
    census_gfx_retire(gen);
    pthread_mutex_unlock(&mock_registry);
}
static int test_identity_interleavings(void)
{
    CensusTransfer *r1, *r2, *r3, *r;
    CensusToken ta, tf, tt, tu;
    CensusRelayInfo info;
    CensusGfxResource *g1, *g2;
    uint64_t gen1, gen2, X=0x10000, Y=0x20000, Z=0x30000, W=0x40000, H=0xabc0;
    CHECK(!setup_dir()); CHECK(!start_census()); open_window();
    /* a) plain reuse: distinct generations, stale between free and realloc */
    sim_alloc(X,256); CHECK((r1=sim_dtoh(X,0x1))); sim_free(X);
    CHECK((r3=sim_dtoh(X+8,0x1))); sim_alloc(X,256); CHECK((r2=sim_dtoh(X,0x1)));
    CHECK(r1->src_kind==CK_DEVICE && r2->src_kind==CK_DEVICE && r1->src_gen!=r2->src_gen);
    CHECK(r3->src_kind==(CK_DEVICE|CK_STALE) && r3->src_gen==r1->src_gen && r3->src_offset==8);
    CHECK(!(r1->flags&CF_IDENTITY_AMBIGUOUS) && !(r2->flags&CF_IDENTITY_AMBIGUOUS));
    /* b) free hook runs before the allocation hook of the same address */
    census_enter(&ta,CENSUS_ROLE_LEDGER);
    sim_free(Y);
    census_exit_alloc(&ta,CA_cuMemAlloc_v2,CUDA_SUCCESS,CK_DEVICE,Y,64,0,0,0,0,0,0);
    CHECK(census_order_ambiguous_allocs==1);
    CHECK((r=sim_dtoh(Y,0x1)) && (r->flags&CF_IDENTITY_AMBIGUOUS));
    /* c) reuse whose allocation hook ran before the old free hook */
    sim_alloc(Z,64); gen1=census_generation;
    census_enter(&tf,CENSUS_ROLE_LEDGER);
    sim_alloc(Z,64); gen2=census_generation;
    CHECK((r=sim_dtoh(Z,0x1)) && (r->flags&CF_IDENTITY_AMBIGUOUS));
    census_exit_free(&tf,CA_cuMemFree_v2,CUDA_SUCCESS,CK_DEVICE,CK_MANAGED,Z,FALSE,0);
    CHECK((r=sim_dtoh(Z,0x1)) && r->src_gen==gen2 && !(r->flags&CF_IDENTITY_AMBIGUOUS));
    (void)gen1;
    /* d) unrelated ledger change during a transfer: ambiguous, not guessed */
    census_enter(&tt,CENSUS_ROLE_WINDOWED);
    sim_alloc(W,64);
    census_exit_transfer(&tt,CA_cuMemcpyDtoH_v2,CUDA_SUCCESS,0,NULL,NULL,2,0,X,0,0,1,0x1,0,0,0,64,NULL);
    CHECK(census_transfers[census_transfer_next-1].flags&CF_IDENTITY_AMBIGUOUS);
    /* e) graphics: generations from the (mock) registry lock, never by address */
    gen1=mock_register(H,0x9000);
    CHECK(gen1);
    {   CensusToken tm; memset(&info,0,sizeof(info)); info.branch=CB_FORCED_REJECT; info.found=1; info.gen[0]=gen1;
        census_enter(&tm,CENSUS_ROLE_GRAPHICS);
        census_exit_gfx(&tm,CA_cuGraphicsSubResourceGetMappedArray,CG_GET_ARRAY,CUDA_ERROR_NOT_SUPPORTED,&info,H,1,NULL,0,0,0); }
    /* f) unregister call returns after the same handle was re-registered */
    census_enter(&tu,CENSUS_ROLE_GRAPHICS);
    mock_unregister(gen1);
    gen2=mock_register(H,0x9000);            /* same handle and array address */
    CHECK(gen2 && gen2!=gen1);
    memset(&info,0,sizeof(info)); info.branch=CB_RELAY_SUCCESS; info.found=1; info.gen[0]=gen1;
    census_exit_gfx(&tu,CA_cuGraphicsUnregisterResource,CG_UNREGISTER,CUDA_SUCCESS,&info,H,1,NULL,0,0,0);
    pthread_mutex_lock(&census_lock);
    g1=census_gfx_by_gen_locked(gen1); g2=census_gfx_by_gen_locked(gen2);
    pthread_mutex_unlock(&census_lock);
    CHECK(g1 && g2 && !g1->live && g2->live);
    CHECK(g1->outcomes[CG_GET_ARRAY][CO_FORCED_REJECT]==1 && g1->outcomes[CG_UNREGISTER][CO_SUCCESS]==1);
    CHECK(g2->outcomes[CG_UNREGISTER][CO_SUCCESS]==0);
    /* CPU-copy array identity resolves to the live registration only */
    {   CensusToken tx; census_enter(&tx,CENSUS_ROLE_WINDOWED);
        census_exit_transfer(&tx,CA_cuMemcpyAtoH_v2,CUDA_SUCCESS,0,NULL,NULL,3,0,0,0x9000,0,1,0x1,0,0,0,64,NULL);
        r=&census_transfers[census_transfer_next-1];
        CHECK(r->src_kind==CK_CPU_COPY_ARRAY && r->src_gen==gen2); }
    return 0;
}

/* ---- 5. failed calls: opaque arguments are never dereferenced ---- */
static int test_failed_opaque_arguments(void)
{
    void *bad=mmap(NULL,4096,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    CensusToken t;
    CensusRelayInfo info;
    CensusTransfer *r;
    CensusGfxEvent *e;
    unsigned before;
    unsigned long long t0;
    CHECK(bad!=MAP_FAILED);
    CHECK(!setup_dir()); CHECK(!start_census()); open_window();
    census_enter(&t,CENSUS_ROLE_WINDOWED);
    census_exit_transfer(&t,CA_cuMemcpy2D_v2,CUDA_ERROR_INVALID_VALUE,0,NULL,(const CUDA_MEMCPY2D*)bad,0,0,0,0,0,0,0,0,0,0,0,NULL);
    r=&census_transfers[census_transfer_next-1];
    CHECK((r->flags&CF_ARGS_UNREAD) && r->args==(uintptr_t)bad && !r->src_type && !r->dst_type);
    census_enter(&t,CENSUS_ROLE_WINDOWED);
    census_exit_transfer(&t,CA_cuMemcpy3DAsync_v2,CUDA_ERROR_INVALID_VALUE,CF_ASYNC,(const CUDA_MEMCPY3D*)bad,NULL,0,0,0,0,0,0,0,0,0,0,0,NULL);
    CHECK(census_transfers[census_transfer_next-1].flags&CF_ARGS_UNREAD);
    /* failed allocation: no ledger entry, pending released */
    before=census_alloc_count;
    census_enter(&t,CENSUS_ROLE_LEDGER);
    census_exit_alloc(&t,CA_cuArrayCreate_v2,CUDA_ERROR_INVALID_VALUE,CK_APP_ARRAY,0,0,0,0,0,0,0,0);
    CHECK(census_alloc_count==before && census_ledger_pending==0);
    /* forwarded batch with a huge count and an unreadable array: bounded, no read */
    memset(&info,0,sizeof(info)); info.branch=CB_FORWARDED;
    t0=census_now();
    census_enter(&t,CENSUS_ROLE_GRAPHICS);
    census_exit_gfx(&t,CA_cuGraphicsMapResources,CG_MAP,CUDA_ERROR_INVALID_VALUE,&info,(uintptr_t)bad,0x7fffffff,NULL,0,0,0);
    CHECK(census_now()-t0<MS(50));
    e=&census_gfx_events[census_gfx_event_next-1];
    CHECK((e->flags&CF_FORWARDED) && e->found==0 && e->count==0x7fffffff && e->resource==(uintptr_t)bad);
    /* relay batch longer than the slots: truncated, only six outcomes counted */
    memset(&info,0,sizeof(info)); info.branch=CB_RELAY_SUCCESS; info.found=10;
    census_enter(&t,CENSUS_ROLE_GRAPHICS);
    census_exit_gfx(&t,CA_cuGraphicsMapResources,CG_MAP,CUDA_SUCCESS,&info,(uintptr_t)bad,10,NULL,0,0,0);
    CHECK(census_gfx_events[census_gfx_event_next-1].flags&CF_BATCH_TRUNCATED);
    CHECK(census_gfx_unknown[CG_MAP][CO_SUCCESS]==CENSUS_BATCH_SLOTS); /* gens were 0 */
    return 0;
}

/* ---- 6. per-thread default streams are not joined ---- */
static void *ptsz_sync(void *arg)
{
    CensusToken t;
    (void)arg;
    census_enter(&t,CENSUS_ROLE_WINDOWED);
    census_exit_sync(&t,CA_cuStreamSynchronize_ptsz,CUDA_SUCCESS,CF_PER_THREAD,NULL,NULL);
    return NULL;
}
static int test_ptds_ambiguity(void)
{
    pthread_t a, b;
    CensusToken t;
    char *summary, *layout;
    CHECK(!setup_dir()); CHECK(!start_census()); open_window();
    pthread_create(&a,NULL,ptsz_sync,NULL); pthread_join(a,NULL);
    pthread_create(&b,NULL,ptsz_sync,NULL); pthread_join(b,NULL);
    census_enter(&t,CENSUS_ROLE_WINDOWED);
    census_exit_sync(&t,CA_cuStreamSynchronize,CUDA_SUCCESS,0,NULL,NULL);
    census_enter(&t,CENSUS_ROLE_WINDOWED);
    census_exit_sync(&t,CA_cuStreamQuery,600 /* not ready */,0,NULL,NULL);
    CHECK(census_sync_next==3); /* not-ready query counted, not recorded */
    CHECK(census_syncs[0].tid!=census_syncs[1].tid);
    CHECK((census_syncs[0].flags&CF_PER_THREAD) && (census_syncs[1].flags&CF_PER_THREAD) && !(census_syncs[2].flags&CF_PER_THREAD));
    CHECK(census_counters[CA_cuStreamQuery].errors==1);
    census_publish("window_end");
    CHECK((summary=read_summary()));
    /* The sync layout carries raw handles only: no context, completion or join field. */
    CHECK((layout=strstr(summary,"\"sync\":{\"record_bytes\"")));
    {   char *end=strstr(layout,"\"staging\":{\"record_bytes\"");
        CHECK(end);
        *end=0;
        CHECK(!strstr(layout,"context") && !strstr(layout,"complet") && !strstr(layout,"generation")); }
    CHECK(!strstr(summary,"\"completed"));
    return 0;
}

/* ---- 7. early teardown and unload ---- */
static int test_early_process_exit(void)
{
    pid_t child;
    int status;
    CHECK(!setup_dir());
    shim_thread_mode=1;
    child=fork();
    CHECK(child>=0);
    if(!child) { if(start_census()) _exit(2); open_window(); _exit(0); } /* exits before window end */
    { pid_t r; while((r=waitpid(child,&status,0))<0 && errno==EINTR) {} CHECK(r==child); }
    CHECK(WIFEXITED(status) && !WEXITSTATUS(status));
    CHECK(count_files("-census-summary.json")==0 && count_files(".bin")==0 && count_files(".pending")==0);
    return 0;
}
static int test_unload_after_publication(void)
{
    char path[512];
    unsigned long long end;
    int fd;
    CHECK(!setup_dir());
    shim_thread_mode=1;
    CHECK(!start_census());
    CHECK(shim_module_refs==1); /* pinned by the census thread */
    snprintf(path,sizeof(path),"%s/arm",test_dir);
    CHECK((fd=open(path,O_WRONLY|O_CREAT|O_EXCL,0600))>=0); close(fd);
    end=census_now()+MS(5000);
    while((__atomic_load_n(&census_publish_state,__ATOMIC_SEQ_CST)!=2 || __atomic_load_n(&shim_module_refs,__ATOMIC_SEQ_CST))
          && census_now()<end) Sleep(10);
    CHECK(census_publish_state==2 && shim_module_refs==0); /* published, then the pin was dropped */
    CHECK(strstr(read_summary(),"\"reason\":\"window_end\""));
    census_unload();
    CHECK(census_block==NULL && census_dir==-1 && census_state==CS_OFF);
    return 0;
}
static int test_setup_failure_cleanup(void)
{
    CensusToken t;
    CHECK(!setup_dir());
    shim_module_handle_ok=0;
    census_enter(&t,CENSUS_ROLE_COUNT);
    CHECK(!t.t0 && census_state==CS_FAILED && census_block==NULL && census_dir==-1);
    return 0;
}

/* ---- 8. admission vs close, deterministic (no scheduling stress) ---- */
static __thread int pause_in_admission;
static int admit_entered, admit_released;
static void hold_admission(int id)
{
    if(id!=CENSUS_TP_ADMIT || !pause_in_admission) return;
    pthread_mutex_lock(&gate_lock);
    admit_entered=1; pthread_cond_broadcast(&gate_cond);
    while(!admit_released) pthread_cond_wait(&gate_cond,&gate_lock);
    pthread_mutex_unlock(&gate_lock);
}
static void *ledger_entrant(void *arg)
{
    CensusToken t;
    (void)arg;
    pause_in_admission=1;
    census_enter(&t,CENSUS_ROLE_LEDGER); /* pauses between inflight admission and the pending update */
    /* "inside CUDA" forever: never exits */
    return NULL;
}
static void *publisher(void *arg) { (void)arg; census_publish("window_end"); return NULL; }
static void release_admission(void)
{
    pthread_mutex_lock(&gate_lock); admit_released=1; pthread_cond_broadcast(&gate_cond); pthread_mutex_unlock(&gate_lock);
}
static int wait_admission(void)
{
    pthread_mutex_lock(&gate_lock); while(!admit_entered) pthread_cond_wait(&gate_cond,&gate_lock); pthread_mutex_unlock(&gate_lock);
    return 0;
}
/* Released within the drain wait: publication waits for admission, then
 * freezes a pending count that can no longer change. */
static int test_admission_close_waits(void)
{
    pthread_t e, p;
    char *summary;
    CHECK(!setup_dir()); CHECK(!start_census()); open_window();
    census_test_hook=hold_admission;
    pthread_create(&e,NULL,ledger_entrant,NULL);
    wait_admission();
    pthread_create(&p,NULL,publisher,NULL);
    Sleep(50);
    CHECK(__atomic_load_n(&census_publish_state,__ATOMIC_SEQ_CST)==1); /* still draining */
    CHECK(count_files("-census-summary.json")==0);
    release_admission();
    pthread_join(p,NULL); pthread_join(e,NULL);
    CHECK((summary=read_summary()));
    CHECK(strstr(summary,"\"published\":\"full\""));
    CHECK(strstr(summary,"\"pending_at_cutoff\":1"));
    CHECK(strstr(summary,"\"identity_tracking_complete\":false"));
    CHECK(strstr(summary,"\"ledger_updates_lost_at_cutoff\""));
    CHECK(census_ledger_pending==1); /* the value that was published */
    return 0;
}
/* Held past the drain wait: only the minimal summary, no ledger read; the
 * entrant's later pending update cannot reach any published file. */
static int test_admission_close_timeout(void)
{
    pthread_t e;
    char *summary;
    CHECK(!setup_dir()); CHECK(!start_census()); open_window();
    census_test_hook=hold_admission;
    pthread_create(&e,NULL,ledger_entrant,NULL);
    wait_admission();
    census_publish("window_end"); /* returns after CENSUS_DRAIN_WAIT_MS */
    CHECK((summary=read_summary()));
    CHECK(strstr(summary,"\"published\":\"minimal\""));
    CHECK(count_files(".bin")==0);
    release_admission();
    pthread_join(e,NULL);
    CHECK(count_files(".bin")==0 && count_files("-census-summary.json")==1);
    return 0;
}

/* ---- 9. summary verdicts never authorize attribution on known gaps ---- */
static int test_summary_blocked_ledger(void)
{
    CensusToken t;
    char *summary;
    CHECK(!setup_dir()); CHECK(!start_census()); open_window();
    CHECK(sim_dtoh(0x10,0x20));
    census_enter(&t,CENSUS_ROLE_LEDGER); /* blocked allocation: never returns */
    census_publish("window_end");
    CHECK((summary=read_summary()));
    CHECK(strstr(summary,"\"lost_at_cutoff\":{\"all\":1,\"window\":0,\"graphics\":0}"));
    CHECK(strstr(summary,"\"pending_at_cutoff\":1"));
    CHECK(strstr(summary,"\"records_complete\":true"));
    CHECK(strstr(summary,"\"identity_tracking_complete\":false"));
    CHECK(strstr(summary,"\"attribution_ready\":false"));
    return 0;
}
static int test_summary_ambiguous_lookup(void)
{
    CensusToken t;
    char *summary;
    CHECK(!setup_dir()); CHECK(!start_census()); open_window();
    sim_alloc(0x5000,256);
    census_enter(&t,CENSUS_ROLE_WINDOWED);
    sim_alloc(0x6000,256); /* overlapping identity change */
    census_exit_transfer(&t,CA_cuMemcpyDtoH_v2,CUDA_SUCCESS,0,NULL,NULL,2,0,0x5000,0,0,1,0x1,0,0,0,64,NULL);
    census_publish("window_end");
    CHECK((summary=read_summary()));
    CHECK(strstr(summary,"\"records_complete\":true"));
    CHECK(strstr(summary,"\"identity_tracking_complete\":true"));
    CHECK(strstr(summary,"\"identity_unambiguous\":false"));
    CHECK(strstr(summary,"\"attribution_ready\":false"));
    CHECK(strstr(summary,"\"ambiguous_transfer_identity\""));
    return 0;
}
static int test_summary_graphics_overflow(void)
{
    CensusToken t;
    CensusRelayInfo info;
    char *summary;
    CHECK(!setup_dir()); CHECK(!start_census()); open_window();
    CHECK(sim_dtoh(0x10,0x20));
    census_gfx_event_next=CENSUS_GFX_EVENT_LIMIT; /* white-box: table already full */
    memset(&info,0,sizeof(info)); info.branch=CB_FORWARDED;
    census_enter(&t,CENSUS_ROLE_GRAPHICS);
    census_exit_gfx(&t,CA_cuGraphicsMapResources,CG_MAP,CUDA_SUCCESS,&info,0,1,NULL,0,0,0);
    census_publish("window_end");
    CHECK((summary=read_summary()));
    CHECK(strstr(summary,"\"graphics_complete\":false"));
    CHECK(strstr(summary,"\"attribution_ready\":false"));
    CHECK(strstr(summary,"\"graphics_event_overflow\""));
    return 0;
}
static void register_during_setup(int id)
{
    if(id==CENSUS_TP_SETUP) CHECK_VOID(census_gfx_insert(0xabc0,0x7000,0x9000,0x8000,256,118)==0);
}
static int test_summary_setup_untracked_resource(void)
{
    char *summary;
    CHECK(!setup_dir());
    census_test_hook=register_during_setup;
    CHECK(!start_census());
    census_test_hook=NULL;
    CHECK(!setup_insert_failed);
    open_window();
    CHECK(sim_dtoh(0x10,0x20)->flags&CF_LEDGER_INCOMPLETE);
    census_publish("window_end");
    CHECK((summary=read_summary()));
    CHECK(strstr(summary,"\"resource_setup_untracked\":1"));
    CHECK(strstr(summary,"\"graphics_complete\":false"));
    CHECK(strstr(summary,"\"identity_tracking_complete\":false"));
    CHECK(strstr(summary,"\"attribution_ready\":false"));
    return 0;
}
/* Referenced generations survive recycling; when nothing is recyclable the
 * allocation is untracked and attribution is refused. */
static int test_summary_referenced_recycling(void)
{
    char *summary;
    unsigned i;
    CHECK(!setup_dir()); CHECK(!start_census()); open_window();
    for(i=0;i<CENSUS_ALLOC_LIMIT;i++) { /* white-box: full ledger of retired generations */
        CensusAlloc *a=&census_allocs[i];
        memset(a,0,sizeof(*a));
        a->base=0x100000000ull+(uint64_t)i*0x1000; a->size=0x1000; a->gen=i+1; a->kind=CK_DEVICE;
        a->state=CA_STATE_RETIRED; a->free_ns=i+1; a->flags=i?CA_FLAG_REFERENCED:0;
    }
    census_alloc_count=CENSUS_ALLOC_LIMIT; census_generation=CENSUS_ALLOC_LIMIT;
    sim_alloc(0x7000,64); /* recycles the one unreferenced generation */
    CHECK(census_alloc_recycled==1 && census_alloc_untracked==0);
    CHECK(census_allocs[1].gen==2 && census_allocs[1].base==0x100001000ull); /* referenced metadata kept */
    sim_alloc(0x8000,64); /* nothing recyclable: explicit loss */
    CHECK(census_alloc_untracked==1);
    census_publish("window_end");
    CHECK((summary=read_summary()));
    CHECK(strstr(summary,"\"identity_tracking_complete\":false"));
    CHECK(strstr(summary,"\"attribution_ready\":false"));
    CHECK(strstr(summary,"\"allocations_untracked\""));
    return 0;
}

static const struct { const char *name; int (*fn)(void); } tests[]={
    {"blocked_call_close",test_blocked_call_close},
    {"clean_close_control",test_clean_close_control},
    {"attribution_control",test_attribution_control},
    {"setup_race",test_setup_race},
    {"writer_timeout",test_writer_timeout},
    {"identity_interleavings",test_identity_interleavings},
    {"failed_opaque_arguments",test_failed_opaque_arguments},
    {"ptds_ambiguity",test_ptds_ambiguity},
    {"early_process_exit",test_early_process_exit},
    {"unload_after_publication",test_unload_after_publication},
    {"setup_failure_cleanup",test_setup_failure_cleanup},
    {"admission_close_waits",test_admission_close_waits},
    {"admission_close_timeout",test_admission_close_timeout},
    {"summary_blocked_ledger",test_summary_blocked_ledger},
    {"summary_ambiguous_lookup",test_summary_ambiguous_lookup},
    {"summary_graphics_overflow",test_summary_graphics_overflow},
    {"summary_setup_untracked_resource",test_summary_setup_untracked_resource},
    {"summary_referenced_recycling",test_summary_referenced_recycling},
};
/* Runs one test in a child. Fork or wait failure, a signal, a non-zero exit
 * or a timeout (child alarm, parent kill as backstop) is a failure. */
static int run_child(int (*fn)(void), int *status)
{
    unsigned long long deadline;
    pid_t child=fork(), r;
    if(child<0) return -1;
    if(!child) { alarm(CHILD_TIMEOUT_S); _exit(fn()); }
    deadline=census_now()+MS((CHILD_TIMEOUT_S+5)*1000);
    for(;;) {
        r=waitpid(child,status,WNOHANG);
        if(r==child) return 0;
        if(r<0 && errno!=EINTR) return -1;
        if(census_now()>deadline) {
            kill(child,SIGKILL);
            while((r=waitpid(child,status,0))<0 && errno==EINTR) {}
            return -2;
        }
        Sleep(10);
    }
}
int main(void)
{
    unsigned i, failed=0;
    for(i=0;i<CENSUS_COUNT_OF(tests);i++) {
        int status=-1, r=run_child(tests[i].fn,&status);
        if(r || !WIFEXITED(status) || WEXITSTATUS(status)) {
            failed++;
            printf("FAIL %s (%s, status %d)\n",tests[i].name,r==-1?"fork/wait error":r==-2?"timeout":"child failed",status);
        }
        else printf("ok   %s\n",tests[i].name);
    }
    printf("%u/%u failed\n",failed,(unsigned)CENSUS_COUNT_OF(tests));
    return failed!=0;
}
