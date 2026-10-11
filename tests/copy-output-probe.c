/* Host-only exercise of sampling and output; no CUDA, D3D or Wine runtime. */
#define _GNU_SOURCE
#include <pthread.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
typedef int BOOL;
typedef int CUresult;
typedef void *CUstream;
typedef struct {
    unsigned width,height;
    void *pixels,*texture,*array,*cuda_context;
    unsigned long long probe_id,map_epoch,probe_last_ns,probe_payload_ns;
    unsigned probe_payloads;
} ZludaCpuTexture;
#define ERR(...) fprintf(stderr,__VA_ARGS__)
static pthread_mutex_t cpu_textures_lock=PTHREAD_MUTEX_INITIALIZER;
#include "../dlls/nvcuda/d3d11_copy_probe.h"
static void *arm_worker(void *unused) { probe_poll(); return NULL; }
int main(int argc,char **argv)
{
    float pixels[16]={0,1,NAN,INFINITY, -0.0f,2,-INFINITY,4, 3,0,5,6, -2,-4,7,8};
    ZludaCpuTexture t={.width=2,.height=2,.pixels=pixels,.probe_id=1,.map_epoch=9};
    CopyProbeRecord *records=NULL;
    assert(argc==2);
    probe_poll();
    if(!strcmp(argv[1],"disabled")) { assert(probe_dir==-1); puts("PASS disabled"); return 0; }
    assert(probe_dir>=0);
    assert(probe_readback_enabled==!strcmp(argv[1],"readback"));
    assert(!probe_start_ns); /* Directory alone cannot arm. */
    int fd=openat(probe_dir,"arm",O_WRONLY|O_CREAT|O_EXCL,0600);assert(fd>=0);assert(!close(fd));
    probe_next_poll_ns=0;
    pthread_t threads[32];
    for(unsigned i=0;i<32;i++)assert(!pthread_create(&threads[i],NULL,arm_worker,NULL));
    for(unsigned i=0;i<32;i++)assert(!pthread_join(threads[i],NULL));
    assert(probe_start_ns && !probe_readback_start_ns); unsigned long long armed=probe_start_ns;
    if(!strcmp(argv[1],"readback")) {
        fd=openat(probe_dir,"readback-arm",O_WRONLY|O_CREAT|O_EXCL,0600);assert(fd>=0);assert(!close(fd));
        probe_next_poll_ns=0;probe_poll();
        assert(probe_readback_start_ns && probe_start_ns==armed);
    }
    probe_next_poll_ns=0;probe_poll();assert(probe_start_ns==armed);
    probe_observe(&t,(void*)2,1,0,&records);assert(records && records->payload);
    assert(records->channels[0].finite==4 && records->channels[0].nonzero==2);
    assert(records->channels[2].nan_count==1 && records->channels[2].inf_count==1);
    if(!strcmp(argv[1],"readback")) {
        unsigned char rows[2][48];memset(rows,0xa5,sizeof(rows));
        for(unsigned y=0;y<2;y++)memcpy(rows[y],(char*)pixels+y*32,32);
        records->readback_attempted=1;
        probe_compare_readback(records,rows,48,pixels);
        assert(!records->readback_result && !records->readback_mismatches && records->readback_hash==records->hash);
        CopyProbeRecord changed={.width=2,.height=2}; rows[1][7]^=1;
        probe_compare_readback(&changed,rows,48,pixels);assert(changed.readback_mismatches==1 && changed.readback_hash!=records->hash);
        assert(changed.readback_first_byte==39 && changed.readback_actual==(changed.readback_expected^1));
        CopyProbeRecord invalid={.width=2,.height=2};
        probe_compare_readback(&invalid,rows,31,pixels);assert(invalid.readback_result<0 && !invalid.readback_hash);
    }
    probe_observe(&t,(void*)2,1,0,&records);assert(probe_records==1); /* Throttle. */
    probe_flush(records);records=NULL;
    t.probe_last_ns=0;probe_observe(&t,NULL,0,999,&records);assert(records && !records->payload && !records->channels[0].finite);
    if(!strcmp(argv[1],"last-write-failure")) {
        char name[96];snprintf(name,sizeof(name),"%u-%06u.json",(unsigned)getpid(),records->sequence);
        fd=openat(probe_dir,name,O_WRONLY|O_CREAT|O_EXCL,0600);assert(fd>=0);assert(!close(fd));
    }
    probe_flush(records);records=NULL;
    if(!strcmp(argv[1],"last-write-failure")) {
        assert(probe_io_errors==1 && probe_written==1);
        __atomic_store_n(&probe_start_ns,probe_now()-PROBE_WINDOW_NS-1,__ATOMIC_RELEASE);
        probe_flush(NULL);assert(probe_terminal);puts("PASS final write failure retained");return 0;
    }
    probe_records=PROBE_RECORD_LIMIT;t.probe_last_ns=0;probe_observe(&t,NULL,0,0,&records);assert(!records && probe_dropped==1);
    assert(!probe_io_errors);
    __atomic_store_n(&probe_start_ns,probe_now()-PROBE_WINDOW_NS-1,__ATOMIC_RELEASE);
    probe_flush(NULL);assert(probe_terminal);
    close(probe_dir);puts("PASS pixels, result, throttle, cap and payload");return 0;
}
