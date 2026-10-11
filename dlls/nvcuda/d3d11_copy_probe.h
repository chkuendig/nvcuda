/* Opt-in, sampled inspection of the existing CUDA->host bounce buffer.
 * No additional CUDA/D3D calls. Stats/payload snapshots perturb host timing.
 * Registry lock protects capture state; output happens after that lock leaves. */
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <locale.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define PROBE_RECORD_LIMIT 1024
#define PROBE_PAYLOAD_LIMIT 16
#define PROBE_PAYLOAD_BYTES (4u*1024u*1024u)
#define PROBE_WINDOW_NS UINT64_C(30000000000)
#define PROBE_INTERVAL_NS UINT64_C(2000000000)

typedef struct {
    unsigned finite, nonzero, nan_count, inf_count;
    float minimum, maximum;
} CopyProbeChannel;
typedef struct CopyProbeRecord {
    struct CopyProbeRecord *next;
    unsigned sequence, width, height, ptsz;
    unsigned long long resource_id, epoch, ns, hash;
    uintptr_t resource, texture, array, context, stream;
    int result;
    size_t bytes;
    CopyProbeChannel channels[4];
    void *payload;
} CopyProbeRecord;
static pthread_once_t probe_once=PTHREAD_ONCE_INIT;
static pthread_mutex_t probe_control_lock=PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t probe_output_lock=PTHREAD_MUTEX_INITIALIZER;
static int probe_dir=-1;
static pid_t probe_owner;
static unsigned long long probe_start_ns, probe_resource_id;
static unsigned probe_records, probe_payloads, probe_inflight;
static unsigned probe_written, probe_terminal;
static locale_t probe_locale;
static unsigned probe_dropped, probe_io_errors;
static unsigned long long probe_next_poll_ns;

static unsigned long long probe_now(void)
{
    struct timespec ts;
    if(clock_gettime(CLOCK_MONOTONIC,&ts)) return 0;
    return (unsigned long long)ts.tv_sec*1000000000+ts.tv_nsec;
}
static void probe_setup(void)
{
    const char *path=getenv("ZLUDA_D3D11_COPY_PROBE_DIR");
    struct stat st;
    int fd;
    if(!path || path[0]!='/') return;
    fd=open(path,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);
    if(fd<0) return;
    if(fstat(fd,&st) || !S_ISDIR(st.st_mode) || st.st_uid!=geteuid() || (st.st_mode&0777)!=0700) { close(fd); return; }
    probe_locale=newlocale(LC_NUMERIC_MASK,"C",(locale_t)0);
    if(!probe_locale) { close(fd); return; }
    probe_owner=getpid();
    probe_dir=fd;
}
static void probe_poll(void)
{
    unsigned long long now;
    struct stat st;
    pthread_once(&probe_once,probe_setup);
    if(probe_dir<0 || probe_owner!=getpid() || __atomic_load_n(&probe_start_ns,__ATOMIC_ACQUIRE)) return;
    if(pthread_mutex_trylock(&probe_control_lock)) return;
    now=probe_now();
    if(!__atomic_load_n(&probe_start_ns,__ATOMIC_ACQUIRE) && now && now>=probe_next_poll_ns) {
        probe_next_poll_ns=now+UINT64_C(500000000);
        if(!fstatat(probe_dir,"arm",&st,AT_SYMLINK_NOFOLLOW) && S_ISREG(st.st_mode) && st.st_uid==geteuid() && (st.st_mode&0777)==0600)
            __atomic_store_n(&probe_start_ns,now,__ATOMIC_RELEASE);
    }
    pthread_mutex_unlock(&probe_control_lock);
}
static void probe_observe(ZludaCpuTexture *t, CUstream stream, BOOL ptsz, CUresult result, CopyProbeRecord **records)
{
    unsigned long long start=__atomic_load_n(&probe_start_ns,__ATOMIC_ACQUIRE), now;
    CopyProbeRecord *r;
    const unsigned char *raw=t->pixels;
    size_t i, bytes=(size_t)t->width*t->height*16;
    if(!start || probe_owner!=getpid()) return;
    now=probe_now();
    if(now<start || now-start>PROBE_WINDOW_NS || (t->probe_last_ns && now-t->probe_last_ns<PROBE_INTERVAL_NS)) return;
    t->probe_last_ns=now;
    if(__atomic_load_n(&probe_records,__ATOMIC_RELAXED)>=PROBE_RECORD_LIMIT) { __atomic_fetch_add(&probe_dropped,1,__ATOMIC_RELAXED); return; }
    r=calloc(1,sizeof(*r));
    if(!r) { __atomic_fetch_add(&probe_dropped,1,__ATOMIC_RELAXED); return; }
    r->sequence=__atomic_add_fetch(&probe_records,1,__ATOMIC_RELAXED); r->resource_id=t->probe_id; r->epoch=t->map_epoch; r->ns=now;
    r->width=t->width; r->height=t->height; r->ptsz=ptsz; r->result=result; r->bytes=bytes;
    r->resource=(uintptr_t)t; r->texture=(uintptr_t)t->texture; r->array=(uintptr_t)t->array;
    r->context=(uintptr_t)t->cuda_context; r->stream=(uintptr_t)stream;
    if(!result) {
        r->hash=UINT64_C(14695981039346656037);
        for(i=0;i<bytes;i++) { r->hash^=raw[i]; r->hash*=UINT64_C(1099511628211); }
        for(i=0;i<bytes/4;i++) {
            float f; CopyProbeChannel *c=&r->channels[i%4];
            memcpy(&f,raw+i*4,4);
            if(isnan(f)) c->nan_count++;
            else if(!isfinite(f)) c->inf_count++;
            else { if(!c->finite || f<c->minimum)c->minimum=f; if(!c->finite || f>c->maximum)c->maximum=f; c->finite++; c->nonzero+=(f!=0); }
        }
        /* Eight stable resource slots, up to two payloads each, separated by
         * ten seconds. Exact float4 bytes; no row padding in this host buffer. */
        if(t->probe_id%6==1 && t->probe_payloads<2 && (t->probe_payloads==0 || now-t->probe_payload_ns>=UINT64_C(10000000000)) && probe_payloads<PROBE_PAYLOAD_LIMIT && bytes<=PROBE_PAYLOAD_BYTES) {
            r->payload=malloc(bytes);
            if(r->payload) { memcpy(r->payload,t->pixels,bytes); t->probe_payloads++; t->probe_payload_ns=now; probe_payloads++; }
            else __atomic_fetch_add(&probe_dropped,1,__ATOMIC_RELAXED);
        }
    }
    __atomic_add_fetch(&probe_inflight,1,__ATOMIC_RELEASE);
    r->next=*records; *records=r;
}
static int probe_write_all(int fd,const void *data,size_t size)
{
    const char *p=data;
    while(size) { ssize_t n=write(fd,p,size); if(n<0 && errno==EINTR)continue; if(n<=0)return -1; p+=n; size-=n; }
    return 0;
}
/* Publish by a no-replacement link only after every byte has been written.
 * Pending files never qualify as completed evidence. */
static int probe_publish(const char *name,const void *data,size_t size)
{
    char pending[128];
    int fd, ok;
    snprintf(pending,sizeof(pending),"%s.pending",name);
    fd=openat(probe_dir,pending,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);
    if(fd<0)return -1;
    ok=!probe_write_all(fd,data,size);
    if(close(fd))ok=0;
    if(ok && linkat(probe_dir,pending,probe_dir,name,0))ok=0;
    if(unlinkat(probe_dir,pending,0))ok=0;
    return ok?0:-1;
}
static void probe_finish(void)
{
    unsigned long long start=__atomic_load_n(&probe_start_ns,__ATOMIC_ACQUIRE), now=probe_now();
    char name[96],json[512];int n;
    unsigned captured,payloads,dropped;
    if(!start || now<start || now-start<=PROBE_WINDOW_NS || probe_terminal)return;
    /* A capture can still be scanning under the registry lock. Wait for it
     * before deciding whether every detached record has finished writing. */
    pthread_mutex_lock(&cpu_textures_lock);
    if(__atomic_load_n(&probe_inflight,__ATOMIC_ACQUIRE)) { pthread_mutex_unlock(&cpu_textures_lock); return; }
    captured=probe_records;payloads=probe_payloads;dropped=__atomic_load_n(&probe_dropped,__ATOMIC_RELAXED);
    probe_terminal=1;
    pthread_mutex_unlock(&cpu_textures_lock);
    n=snprintf(json,sizeof(json),"{\"schema\":1,\"pid\":%u,\"complete\":true,\"window_start_ns\":%llu,\"window_end_ns\":%llu,\"summary_ns\":%llu,\"captured\":%u,\"written\":%u,\"payloads\":%u,\"dropped\":%u,\"io_errors\":%u}\n",(unsigned)getpid(),start,start+PROBE_WINDOW_NS,now,captured,probe_written,payloads,dropped,probe_io_errors);
    snprintf(name,sizeof(name),"%u-summary.json",(unsigned)getpid());
    if(probe_publish(name,json,n)) ERR("[ZLUDA] copy probe terminal summary failed; evidence incomplete\n");
}
static void probe_flush(CopyProbeRecord *records)
{
    CopyProbeRecord *r;
    locale_t previous;
    if(probe_dir<0 || probe_owner!=getpid())return;
    pthread_mutex_lock(&probe_output_lock);
    previous=uselocale(probe_locale);
    while((r=records)) {
        char name[96], payload[96]="", json[4096];
        int n, used=0, payload_ok=0;
        records=r->next;
        if(r->payload) {
            snprintf(payload,sizeof(payload),"%u-%06u.float4",(unsigned)getpid(),r->sequence);
            payload_ok=!probe_publish(payload,r->payload,r->bytes);
            if(!payload_ok)probe_io_errors++;
        }
        used=snprintf(json,sizeof(json),"{\"schema\":1,\"pid\":%u,\"sequence\":%u,\"monotonic_ns\":%llu,\"window_start_ns\":%llu,\"resource_id\":%llu,\"resource\":\"%" PRIxPTR "\",\"texture\":\"%" PRIxPTR "\",\"array\":\"%" PRIxPTR "\",\"context\":\"%" PRIxPTR "\",\"stream\":\"%" PRIxPTR "\",\"ptsz\":%u,\"map_epoch\":%llu,\"width\":%u,\"height\":%u,\"direction\":\"cuda_to_host_before_d3d_copyback\",\"cuda_result\":%d,\"bytes\":%zu,\"fnv1a64\":\"%016llx\",\"payload\":\"%s\",\"io_errors\":%u,\"dropped\":%u,\"channels\":[",(unsigned)getpid(),r->sequence,r->ns,__atomic_load_n(&probe_start_ns,__ATOMIC_ACQUIRE),r->resource_id,r->resource,r->texture,r->array,r->context,r->stream,r->ptsz,r->epoch,r->width,r->height,r->result,r->bytes,r->hash,payload_ok?payload:"",probe_io_errors,__atomic_load_n(&probe_dropped,__ATOMIC_RELAXED));
        for(n=0;n<4;n++) {
            CopyProbeChannel *c=&r->channels[n];char minimum[32]="null",maximum[32]="null";
            if(c->finite) { snprintf(minimum,sizeof(minimum),"%.9g",c->minimum); snprintf(maximum,sizeof(maximum),"%.9g",c->maximum); }
            used+=snprintf(json+used,sizeof(json)-used,"%s{\"finite\":%u,\"nonzero\":%u,\"nan\":%u,\"inf\":%u,\"min\":%s,\"max\":%s}",n?",":"",c->finite,c->nonzero,c->nan_count,c->inf_count,minimum,maximum);
        }
        used+=snprintf(json+used,sizeof(json)-used,"]}\n");
        snprintf(name,sizeof(name),"%u-%06u.json",(unsigned)getpid(),r->sequence);
        if(probe_publish(name,json,used))probe_io_errors++;else probe_written++;
        free(r->payload); free(r);
        __atomic_sub_fetch(&probe_inflight,1,__ATOMIC_RELEASE);
    }
    probe_finish();
    uselocale(previous);
    pthread_mutex_unlock(&probe_output_lock);
}
