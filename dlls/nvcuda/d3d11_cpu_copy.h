/* Opt-in correctness baseline: standard D3D11 staging copies and real CUDA
 * arrays. No Wine internals, imported graphics memory or fabricated arrays.
 * Included after the native CUDA function pointers in nvcuda.c. */
#include "d3d10.h"
#include <stdlib.h>
#include <string.h>

typedef struct ZludaCpuTexture {
    struct ZludaCpuTexture *next;
    ID3D11Texture2D *texture, *staging;
    ID3D11DeviceContext *context;
    ID3D10Multithread *multithread;
    IUnknown *identity;
    CUcontext cuda_context;
    CUarray array;
    void *pixels;
    UINT width, height;
    BOOL mapped;
    unsigned long long probe_id, map_epoch, probe_last_ns, probe_payload_ns;
    unsigned probe_payloads;
    BOOL marker; /* S2 diagnostic: copy-back overwritten with magenta (ZLUDA_D3D11_COPY_MARKER=1) */
    unsigned marker_witness; /* 0 none yet, 1 first copy logged but all zero, 2 non-zero copy characterised */
    struct ZludaCoverage *cov; /* S3-W diagnostic state, only for the ZLUDA_D3D11_WRITE_COVERAGE size */
    BOOL probe_active, probe_triple_due; /* last probe record had a non-zero channel; P-triple payload planned for this unmap */
} ZludaCpuTexture;
static ZludaCpuTexture *cpu_textures;
static pthread_mutex_t cpu_textures_lock = PTHREAD_MUTEX_INITIALIZER;
/* Remains set after the first publication so live handles still work if the
 * opt-in environment is later changed. Never-enabled mode avoids the lock. */
static LONG cpu_copy_active;
#include "d3d11_copy_probe.h"
#include "d3d11_write_coverage.h"

static int cpu_copy_enabled(void)
{
    const char *value = getenv("ZLUDA_D3D11_CPU_COPY");
    return value && !strcmp(value, "1");
}
/* Opt-in S2 diagnostic (ZLUDA_D3D11_COPY_MARKER=1, read once): on CUDA->D3D
 * copy-back, overwrite every texel of the third resource of each registration
 * triple with opaque magenta (1,0,1,1). The role comes from registration order:
 * the game registers its particle textures in consecutive triples, and in the
 * retained abe75a37 runs the third of each non-zero triple carries the
 * colour-like signature (RGB in [0,1], small A). A third registration is marked
 * only if the two registrations before it have the same size. No extra CUDA or
 * D3D calls and no waits; the probe still observes the unmodified CUDA output. */
static LONG marker_registrations, marker_copies, marker_first_copies, marker_first_suppressed, marker_unregisters;
#define MARKER_FIRST_COPY_LIMIT 64
static UINT marker_prev_w[2], marker_prev_h[2];
static int cpu_marker_enabled(void)
{
    static LONG state; /* 0 unread, 1 off, 2 on */
    LONG value=InterlockedCompareExchange(&state,0,0);
    if(!value) {
        const char *env=getenv("ZLUDA_D3D11_COPY_MARKER");
        value=env && !strcmp(env,"1") ? 2 : 1;
        if(!InterlockedCompareExchange(&state,value,0) && value==2)
            FIXME("[ZLUDA] CPU-copy explicit opt-in marks copy-back of the third resource of each registration triple with magenta\n");
        value=InterlockedCompareExchange(&state,0,0);
    }
    return value==2;
}
static BOOL marker_log_count(LONG count) { return count<=8 || !(count&(count-1)); }
/* Called with cpu_textures_lock held, after t->probe_id is assigned. */
static void marker_assign(ZludaCpuTexture *t)
{
    BOOL same=marker_prev_w[0]==t->width && marker_prev_h[0]==t->height && marker_prev_w[1]==t->width && marker_prev_h[1]==t->height;
    if(t->probe_id%3==0) {
        if(same) {
            LONG count=InterlockedIncrement(&marker_registrations);
            t->marker=TRUE;
            if(marker_log_count(count)) FIXME("[ZLUDA] CPU-copy marker resource count=%d ordinal=%llu %ux%u resource=%p\n",(int)count,t->probe_id,t->width,t->height,t);
        }
        else FIXME("[ZLUDA] CPU-copy marker skipped ordinal=%llu %ux%u: predecessors differ in size\n",t->probe_id,t->width,t->height);
    }
    marker_prev_w[1]=marker_prev_w[0]; marker_prev_h[1]=marker_prev_h[0];
    marker_prev_w[0]=t->width; marker_prev_h[0]=t->height;
}
/* Witness only: does the data about to be overwritten have the colour-like signature? */
static void marker_signature(const ZludaCpuTexture *t, size_t *nonzero, size_t *rgb_unit, size_t *alpha_below_one)
{
    const float *texel=t->pixels;
    size_t i, n=(size_t)t->width*t->height;
    *nonzero=*rgb_unit=*alpha_below_one=0;
    for(i=0;i<n;i++) {
        const float *v=texel+i*4;
        if(v[0]>=0.0f && v[0]<=1.0f && v[1]>=0.0f && v[1]<=1.0f && v[2]>=0.0f && v[2]<=1.0f) ++*rgb_unit;
        if(v[3]<1.0f) ++*alpha_below_one;
        if(v[0]!=0.0f || v[1]!=0.0f || v[2]!=0.0f || v[3]!=0.0f) ++*nonzero;
    }
}
/* Bounce buffer of a marked resource, after the CUDA copy and before the D3D write.
 * Every marked resource is characterised on its first copy-back (registration
 * order alone cannot exclude a shifted ordinal after a re-registration). */
static void marker_apply(ZludaCpuTexture *t)
{
    float *texel=t->pixels;
    size_t i, n=(size_t)t->width*t->height, nonzero, rgb_unit, alpha_below_one;
    LONG count=InterlockedIncrement(&marker_copies);
    /* First copy of each marked resource, and if that was all zero, its first
     * non-zero copy (all-zero triples can carry particles later). */
    if(t->marker_witness<2) {
        marker_signature(t,&nonzero,&rgb_unit,&alpha_below_one);
        if(!t->marker_witness || nonzero) {
            const char *kind=!t->marker_witness?"first-copy":"first-nonzero";
            LONG first=InterlockedIncrement(&marker_first_copies);
            if(first<=MARKER_FIRST_COPY_LIMIT)
                FIXME("[ZLUDA] CPU-copy marker %s ordinal=%llu %ux%u resource=%p texels=%zu nonzero=%zu rgb_in_unit=%zu alpha_below_one=%zu\n",
                      kind,t->probe_id,t->width,t->height,t,n,nonzero,rgb_unit,alpha_below_one);
            else {
                LONG suppressed=InterlockedIncrement(&marker_first_suppressed);
                if(marker_log_count(suppressed)) FIXME("[ZLUDA] CPU-copy marker first-copy witness suppressed count=%d (limit %d)\n",(int)suppressed,MARKER_FIRST_COPY_LIMIT);
            }
            t->marker_witness=nonzero?2:1;
        }
    }
    if(marker_log_count(count)) {
        marker_signature(t,&nonzero,&rgb_unit,&alpha_below_one);
        FIXME("[ZLUDA] CPU-copy marker copy count=%d ordinal=%llu %ux%u texels=%zu nonzero=%zu rgb_in_unit=%zu alpha_below_one=%zu\n",
              (int)count,t->probe_id,t->width,t->height,n,nonzero,rgb_unit,alpha_below_one);
    }
    for(i=0;i<n;i++) { texel[i*4]=1.0f; texel[i*4+1]=0.0f; texel[i*4+2]=1.0f; texel[i*4+3]=1.0f; }
}
static ZludaCpuTexture *cpu_find(CUgraphicsResource resource)
{
    ZludaCpuTexture *t;
    for (t=cpu_textures;t;t=t->next) if ((void*)t==resource) return t;
    return NULL;
}
static CUresult cpu_context_check(ZludaCpuTexture *t)
{
    CUcontext current;
    CUresult result=pcuCtxGetCurrent(&current);
    return result ? result : current==t->cuda_context ? CUDA_SUCCESS : CUDA_ERROR_INVALID_CONTEXT;
}
static void cpu_release(ZludaCpuTexture *t)
{
    cov_free(t->cov);
    HeapFree(GetProcessHeap(),0,t->pixels);
    if(t->staging) ID3D11Texture2D_Release(t->staging);
    if(t->texture) ID3D11Texture2D_Release(t->texture);
    if(t->identity) IUnknown_Release(t->identity);
    if(t->multithread) ID3D10Multithread_Release(t->multithread);
    if(t->context) ID3D11DeviceContext_Release(t->context);
    HeapFree(GetProcessHeap(),0,t);
}
static CUresult cpu_register(CUgraphicsResource *out, ID3D11Resource *resource, unsigned flags)
{
    struct { size_t width,height; int format; unsigned channels; } array_desc;
    D3D11_TEXTURE2D_DESC desc;
    ID3D11Device *device;
    CUresult result;
    ZludaCpuTexture *t;
    HRESULT hr;
    BOOL marker=cpu_marker_enabled(); /* read once; logs the opt-in at the first registration */
    BOOL coverage=cpu_coverage_enabled(); /* likewise */
    if(!out || !resource) return CUDA_ERROR_INVALID_VALUE;
    if(flags) return CUDA_ERROR_NOT_SUPPORTED;
    t=HeapAlloc(GetProcessHeap(),HEAP_ZERO_MEMORY,sizeof(*t));
    if(!t) return CUDA_ERROR_OUT_OF_MEMORY;
    hr=ID3D11Resource_QueryInterface(resource,&IID_ID3D11Texture2D,(void**)&t->texture);
    if(FAILED(hr)) { cpu_release(t); return CUDA_ERROR_NOT_SUPPORTED; }
    hr=ID3D11Texture2D_QueryInterface(t->texture,&IID_IUnknown,(void**)&t->identity);
    if(FAILED(hr)) { cpu_release(t); return CUDA_ERROR_UNKNOWN; }
    ID3D11Texture2D_GetDesc(t->texture,&desc);
    if(desc.Format!=DXGI_FORMAT_R32G32B32A32_FLOAT || desc.MipLevels!=1 || desc.ArraySize!=1 || desc.SampleDesc.Count!=1) {
        cpu_release(t); return CUDA_ERROR_NOT_SUPPORTED;
    }
    t->width=desc.Width; t->height=desc.Height;
    t->pixels=HeapAlloc(GetProcessHeap(),0,(size_t)t->width*t->height*16);
    if(!t->pixels) { cpu_release(t); return CUDA_ERROR_OUT_OF_MEMORY; }
    if(coverage && t->width==cov_width && t->height==cov_height && !(t->cov=cov_alloc(t))) { cpu_release(t); return CUDA_ERROR_OUT_OF_MEMORY; }
    result=pcuCtxGetCurrent(&t->cuda_context);
    if(result || !t->cuda_context) { cpu_release(t); return result ? result : CUDA_ERROR_INVALID_CONTEXT; }
    ID3D11Resource_GetDevice(resource,&device);
    if(ID3D11Device_GetCreationFlags(device)&D3D11_CREATE_DEVICE_SINGLETHREADED) {
        ERR("[ZLUDA] CPU-copy interop refuses a single-threaded D3D11 device\n");
        ID3D11Device_Release(device); cpu_release(t); return CUDA_ERROR_NOT_SUPPORTED;
    }
    ID3D11Device_GetImmediateContext(device,&t->context);
    hr=ID3D11Device_QueryInterface(device,&IID_ID3D10Multithread,(void**)&t->multithread);
    /* Changing context locking is a separate, explicit process-lifetime opt-in.
     * Do not restore it on resource release: other users may now rely on it. */
    if(SUCCEEDED(hr) && !ID3D10Multithread_GetMultithreadProtected(t->multithread)) {
        const char *enable=getenv("ZLUDA_D3D11_CPU_COPY_ENABLE_MT");
        if(enable && !strcmp(enable,"1")) {
            ID3D10Multithread_SetMultithreadProtected(t->multithread,TRUE);
            FIXME("[ZLUDA] CPU-copy explicit opt-in enabled D3D11 multithread protection\n");
        }
    }
    if(FAILED(hr) || !ID3D10Multithread_GetMultithreadProtected(t->multithread)) {
        ERR("[ZLUDA] CPU-copy interop requires a multithread-protected D3D11 context\n");
        ID3D11Device_Release(device); cpu_release(t); return CUDA_ERROR_NOT_SUPPORTED;
    }
    desc.Usage=D3D11_USAGE_STAGING; desc.BindFlags=desc.MiscFlags=0;
    desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ|D3D11_CPU_ACCESS_WRITE;
    hr=ID3D11Device_CreateTexture2D(device,&desc,NULL,&t->staging);
    ID3D11Device_Release(device);
    if(FAILED(hr)) { cpu_release(t); return CUDA_ERROR_UNKNOWN; }
    array_desc.width=t->width; array_desc.height=t->height; array_desc.format=0x20; array_desc.channels=4;
    result=pcuArrayCreate_v2(&t->array,(const CUDA_ARRAY_DESCRIPTOR*)&array_desc);
    if(result) { cpu_release(t); return result; }
    pthread_mutex_lock(&cpu_textures_lock);
    for(ZludaCpuTexture *existing=cpu_textures;existing;existing=existing->next) {
        if(existing->identity==t->identity) {
            pthread_mutex_unlock(&cpu_textures_lock);
            result=pcuArrayDestroy(t->array);
            cpu_release(t);
            return result ? result : CUDA_ERROR_INVALID_HANDLE;
        }
    }
    t->probe_id=++probe_resource_id;
    if(marker) marker_assign(t);
    if(t->cov) cov_publish(t);
    t->next=cpu_textures; cpu_textures=t;
    InterlockedExchange(&cpu_copy_active,1);
    pthread_mutex_unlock(&cpu_textures_lock);
    *out=(CUgraphicsResource)t;
    FIXME("[ZLUDA] CPU-copy registration OK %ux%u resource=%p array=%p\n",t->width,t->height,t,t->array);
    return CUDA_SUCCESS;
}
static CUresult cpu_transfer(ZludaCpuTexture *t, BOOL to_cuda, CUstream stream, BOOL ptsz, CopyProbeRecord **records)
{
    D3D11_MAPPED_SUBRESOURCE mapped;
    CUDA_MEMCPY2D copy={0};
    HRESULT hr;
    CUresult result;
    unsigned y;
    ID3D10Multithread_Enter(t->multithread);
    if(to_cuda) ID3D11DeviceContext_CopyResource(t->context,(ID3D11Resource*)t->staging,(ID3D11Resource*)t->texture);
    hr=ID3D11DeviceContext_Map(t->context,(ID3D11Resource*)t->staging,0,to_cuda?D3D11_MAP_READ:D3D11_MAP_WRITE,0,&mapped);
    if(FAILED(hr)) { ID3D10Multithread_Leave(t->multithread); return CUDA_ERROR_UNKNOWN; }
    copy.WidthInBytes=(size_t)t->width*16; copy.Height=t->height;
    copy.srcMemoryType=to_cuda?1:3; copy.dstMemoryType=to_cuda?3:1;
    if(to_cuda) {
        for(y=0;y<t->height;y++) memcpy((char*)t->pixels+y*copy.WidthInBytes,(char*)mapped.pData+y*mapped.RowPitch,copy.WidthInBytes);
        if(t->cov) cov_map_fill(t); /* shadow = D3D content; CUDA receives the sentinel */
        copy.srcHost=t->pixels; copy.srcPitch=copy.WidthInBytes; copy.dstArray=t->array;
    }
    else { copy.srcArray=t->array; copy.dstHost=t->pixels; copy.dstPitch=copy.WidthInBytes; }
    result=pcuMemcpy2D_v2(&copy);
    if(!to_cuda) probe_observe(t,stream,ptsz,result,records);
    if(!result && !to_cuda && t->cov) cov_copy_back(t); /* restore unwritten channels before any marker */
    if(!result && !to_cuda && t->marker) marker_apply(t);
    if(!result && !to_cuda) for(y=0;y<t->height;y++) memcpy((char*)mapped.pData+y*mapped.RowPitch,(char*)t->pixels+y*copy.WidthInBytes,copy.WidthInBytes);
    ID3D11DeviceContext_Unmap(t->context,(ID3D11Resource*)t->staging,0);
    if(!result && !to_cuda) ID3D11DeviceContext_CopyResource(t->context,(ID3D11Resource*)t->texture,(ID3D11Resource*)t->staging);
    ID3D10Multithread_Leave(t->multithread);
    return result;
}
/* Return -1 only when no copy resource is present (forward unchanged).
 * Reject mixed batches before changing any state. */
static int cpu_map(unsigned count, CUgraphicsResource *resources, CUstream stream, BOOL unmap, BOOL ptsz)
{
    unsigned i,j,found=0;
    BOOL planned=FALSE;
    CopyProbeRecord *records=NULL;
    CUresult result=CUDA_SUCCESS;
    if(!InterlockedCompareExchange(&cpu_copy_active,0,0) || !resources) return -1;
    probe_poll(); /* Owned arm-file check, outside both interop locks. */
    pthread_mutex_lock(&cpu_textures_lock);
    for(i=0;i<count;i++) if(cpu_find(resources[i])) found++;
    if(!found) { pthread_mutex_unlock(&cpu_textures_lock); return -1; }
    if(found!=count) { result=CUDA_ERROR_NOT_SUPPORTED; goto done; }
    for(i=0;i<count;i++) {
        ZludaCpuTexture *t=cpu_find(resources[i]);
        for(j=0;j<i;j++) if(resources[i]==resources[j]) { result=CUDA_ERROR_INVALID_VALUE; goto done; }
        result=cpu_context_check(t); if(result) goto done;
        if(t->mapped!=unmap) { result=unmap?211:208; goto done; }
    }
    result=ptsz?pcuStreamSynchronize_ptsz(stream):pcuStreamSynchronize(stream);
    if(result) goto done;
    if(unmap && probe_payload_mode==1) {
        /* P-triple: decide payloads for whole covered triples of this batch before any copy-back. */
        if(count<=PROBE_TRIPLE_BATCH) { for(i=0;i<count;i++) probe_triple_batch[i]=cpu_find(resources[i]); planned=probe_triple_plan(count); }
        else probe_triple_oversize++;
    }
    for(i=0;i<count;i++) { result=cpu_transfer(cpu_find(resources[i]),!unmap,stream,ptsz,&records); if(result) goto done; }
    for(i=0;i<count;i++) {
        ZludaCpuTexture *t=cpu_find(resources[i]);
        t->mapped=!unmap;
        if(!unmap) t->map_epoch++;
    }
    { static unsigned messages; if(messages++<4) FIXME("[ZLUDA] CPU-copy %s OK count=%u\n",unmap?"unmap":"map",count); }
done:
    if(planned) probe_triple_settle(count); /* members a failed batch never copied back */
    pthread_mutex_unlock(&cpu_textures_lock);
    probe_flush(records); /* No interop lock held during file output. */
    return result;
}
/* op: 0=array, 1=linear pointer, 2=map flags, 3=mipmapped array, 4=unregister */
static int cpu_resource(CUgraphicsResource resource, unsigned op, void *out, unsigned a, unsigned b)
{
    CUresult result;
    ZludaCpuTexture *t,**link;
    if(!InterlockedCompareExchange(&cpu_copy_active,0,0)) return -1;
    pthread_mutex_lock(&cpu_textures_lock);
    t=cpu_find(resource);
    if(!t) { pthread_mutex_unlock(&cpu_textures_lock); return -1; }
    result=cpu_context_check(t); if(result) goto done;
    switch(op) {
    case 0:
        if(!out || a || b) result=CUDA_ERROR_INVALID_VALUE;
        else if(!t->mapped) result=211;
        else *(CUarray*)out=t->array;
        break;
    case 1: result=t->mapped?213:211; break;
    case 2: result=t->mapped?208:a?CUDA_ERROR_NOT_SUPPORTED:CUDA_SUCCESS; break;
    case 3: result=CUDA_ERROR_NOT_SUPPORTED; break;
    case 4:
        if(t->mapped) { result=208; break; }
        result=pcuArrayDestroy(t->array); if(result) break;
        if(cpu_marker_enabled()) {
            /* Unregisters while marking: ordinal shifts after re-registration become visible. */
            LONG count=InterlockedIncrement(&marker_unregisters);
            if(marker_log_count(count)) FIXME("[ZLUDA] CPU-copy marker-mode unregister count=%d ordinal=%llu %ux%u resource=%p marked=%d\n",
                                              (int)count,t->probe_id,t->width,t->height,t,t->marker);
        }
        if(t->cov) cov_unpublish(t);
        for(link=&cpu_textures;*link!=t;link=&(*link)->next) {}
        *link=t->next; cpu_release(t); break;
    default: result=CUDA_ERROR_INVALID_VALUE;
    }
done:
    pthread_mutex_unlock(&cpu_textures_lock);
    return result;
}
