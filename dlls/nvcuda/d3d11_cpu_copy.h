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
    ID3D11Query *ready;
    IUnknown *identity;
    CUcontext cuda_context;
    CUarray array;
    void *pixels;
    UINT width, height;
    CUevent upload_done;
    BOOL mapped, pending, poisoned;
    unsigned map_flags; /* CUDA NONE=0, READ_ONLY=1, WRITE_DISCARD=2. */
} ZludaCpuTexture;
static ZludaCpuTexture *cpu_textures;
static pthread_mutex_t cpu_textures_lock = PTHREAD_MUTEX_INITIALIZER;
/* Remains set after the first publication so live handles still work if the
 * opt-in environment is later changed. Never-enabled mode avoids the lock. */
static LONG cpu_copy_active;

static int cpu_copy_enabled(void)
{
    const char *value = getenv("ZLUDA_D3D11_CPU_COPY");
    return value && !strcmp(value, "1");
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
    /* Only unpublished or drained objects reach here. */
    if(t->pixels) pcuMemFreeHost(t->pixels);
    if(t->upload_done) pcuEventDestroy_v2(t->upload_done);
    if(t->ready) ID3D11Query_Release(t->ready);
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
    D3D11_QUERY_DESC query_desc={D3D11_QUERY_EVENT,0};
    ID3D11Device *device;
    CUresult result;
    ZludaCpuTexture *t;
    HRESULT hr;
    if(!out || !resource) return CUDA_ERROR_INVALID_VALUE;
    /* Preserve the registration capability gate. Surface/gather allocation
     * flags need a different CUDA array descriptor and remain unsupported.
     * READ_ONLY/WRITE_DISCARD are map flags, never registration bits. */
    if(flags) return CUDA_ERROR_NOT_SUPPORTED;
    if(!pcuMemHostAlloc || !pcuMemFreeHost || !pcuMemcpy2DAsync_v2 ||
       !pcuMemcpy2DAsync_v2_ptsz || !pcuEventCreate || !pcuEventRecord ||
       !pcuEventRecord_ptsz || !pcuEventSynchronize || !pcuEventDestroy_v2)
        return CUDA_ERROR_NOT_SUPPORTED;
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
    result=pcuCtxGetCurrent(&t->cuda_context);
    if(result || !t->cuda_context) { cpu_release(t); return result ? result : CUDA_ERROR_INVALID_CONTEXT; }
    if(!t->width || !t->height || (size_t)t->width>SIZE_MAX/16/t->height) {
        cpu_release(t); return CUDA_ERROR_INVALID_VALUE;
    }
    result=pcuMemHostAlloc(&t->pixels,(size_t)t->width*t->height*16,0);
    if(result) { cpu_release(t); return result; }
    result=pcuEventCreate(&t->upload_done,2); /* disable timing */
    if(result) { cpu_release(t); return result; }
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
    if(SUCCEEDED(hr)) hr=ID3D11Device_CreateQuery(device,&query_desc,&t->ready);
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
    t->next=cpu_textures; cpu_textures=t;
    InterlockedExchange(&cpu_copy_active,1);
    pthread_mutex_unlock(&cpu_textures_lock);
    *out=(CUgraphicsResource)t;
    FIXME("[ZLUDA] CPU-copy registration OK %ux%u resource=%p array=%p\n",t->width,t->height,t,t->array);
    return CUDA_SUCCESS;
}
/* The registry mutex owns all private staging/bounce/event state. The game's
 * ID3D10Multithread lock serializes only immediate-context calls. Never wait,
 * yield, touch row memory or call CUDA under a relay-held D3D lock. If the
 * caller retains an outer recursive Enter, Leave cannot release its level:
 * correctness is preserved, but lock-wait savings require an unowned caller
 * entry. Never force-release a lock level owned by the application. */
static CUresult cpu_drain(ZludaCpuTexture *t)
{
    CUresult result;
    if(t->poisoned) return CUDA_ERROR_UNKNOWN;
    if(!t->pending) return CUDA_SUCCESS;
    result=pcuEventSynchronize(t->upload_done);
    if(!result) t->pending=FALSE;
    return result;
}
static void cpu_copy_resource(ZludaCpuTexture *t, BOOL to_cuda)
{
    ID3D10Multithread_Enter(t->multithread);
    ID3D11DeviceContext_CopyResource(t->context,
        (ID3D11Resource*)(to_cuda?t->staging:t->texture),
        (ID3D11Resource*)(to_cuda?t->texture:t->staging));
    ID3D10Multithread_Leave(t->multithread);
}
/* Poll back-off: a few immediate yields cover short GPU tails, then sleep so
 * waiting on a busy GPU or a staging copy does not spin a CPU core. */
static void cpu_poll_backoff(unsigned attempt)
{
    Sleep(attempt<4?0:1);
}
/* WRITE_DISCARD leaves the mapped contents undefined, so such a map needs
 * neither the D3D snapshot nor any wait for earlier D3D or CUDA work: CUDA
 * writes only the private array, and unmap publishes in D3D context order. */
static BOOL cpu_needs_input(ZludaCpuTexture *t)
{
    return t->map_flags!=2;
}
/* Index of the last input resource sharing t's immediate context, so each
 * context with input gets exactly one completion wait. */
static BOOL cpu_last_input_of_context(unsigned count, CUgraphicsResource *resources, unsigned i)
{
    ZludaCpuTexture *t=cpu_find(resources[i]);
    unsigned j;
    if(!cpu_needs_input(t)) return FALSE;
    for(j=i+1;j<count;j++) {
        ZludaCpuTexture *u=cpu_find(resources[j]);
        if(cpu_needs_input(u) && u->context==t->context) return FALSE;
    }
    return TRUE;
}
/* One EVENT after all copies per immediate context with input. Flush submits
 * the batch; DONOTFLUSH polls cannot strand unsubmitted work or repeatedly
 * flush the game. A batch spanning devices needs one wait per context. */
static CUresult cpu_wait_batch(unsigned count, CUgraphicsResource *resources)
{
    unsigned i,attempt;
    HRESULT hr;
    ULONGLONG start;
    for(i=0;i<count;i++) {
        ZludaCpuTexture *t=cpu_find(resources[i]);
        if(!cpu_last_input_of_context(count,resources,i)) continue;
        ID3D10Multithread_Enter(t->multithread);
        ID3D11DeviceContext_End(t->context,(ID3D11Asynchronous*)t->ready);
        ID3D10Multithread_Leave(t->multithread);
        ID3D10Multithread_Enter(t->multithread);
        ID3D11DeviceContext_Flush(t->context);
        ID3D10Multithread_Leave(t->multithread);
    }
    for(i=0;i<count;i++) {
        ZludaCpuTexture *t=cpu_find(resources[i]);
        if(!cpu_last_input_of_context(count,resources,i)) continue;
        start=GetTickCount64();
        attempt=0;
        do {
            ID3D10Multithread_Enter(t->multithread);
            hr=ID3D11DeviceContext_GetData(t->context,(ID3D11Asynchronous*)t->ready,NULL,0,D3D11_ASYNC_GETDATA_DONOTFLUSH);
            ID3D10Multithread_Leave(t->multithread);
            if(hr!=S_FALSE) break;
            if(GetTickCount64()-start>=30000) return CUDA_ERROR_UNKNOWN;
            cpu_poll_backoff(attempt++);
        } while(1);
        if(hr!=S_OK) return CUDA_ERROR_UNKNOWN;
    }
    return CUDA_SUCCESS;
}
static CUresult cpu_stage_map(ZludaCpuTexture *t, BOOL to_cuda, D3D11_MAPPED_SUBRESOURCE *mapped)
{
    HRESULT hr;
    ULONGLONG start=GetTickCount64();
    unsigned attempt=0;
    do {
        ID3D10Multithread_Enter(t->multithread);
        hr=ID3D11DeviceContext_Map(t->context,(ID3D11Resource*)t->staging,0,
            to_cuda?D3D11_MAP_READ:D3D11_MAP_WRITE,D3D11_MAP_FLAG_DO_NOT_WAIT,mapped);
        ID3D10Multithread_Leave(t->multithread);
        if(hr!=DXGI_ERROR_WAS_STILL_DRAWING) break;
        if(GetTickCount64()-start>=30000) return CUDA_ERROR_UNKNOWN;
        cpu_poll_backoff(attempt++);
    } while(1);
    return FAILED(hr)?CUDA_ERROR_UNKNOWN:CUDA_SUCCESS;
}
static void cpu_stage_unmap(ZludaCpuTexture *t)
{
    ID3D10Multithread_Enter(t->multithread);
    ID3D11DeviceContext_Unmap(t->context,(ID3D11Resource*)t->staging,0);
    ID3D10Multithread_Leave(t->multithread);
}
static CUresult cpu_upload(ZludaCpuTexture *t, CUstream stream, BOOL ptsz)
{
    D3D11_MAPPED_SUBRESOURCE mapped;
    CUDA_MEMCPY2D copy={0};
    CUresult result,drained;
    unsigned y;
    result=cpu_stage_map(t,TRUE,&mapped);
    if(result) return result;
    copy.WidthInBytes=(size_t)t->width*16; copy.Height=t->height;
    for(y=0;y<t->height;y++) memcpy((char*)t->pixels+y*copy.WidthInBytes,
        (char*)mapped.pData+y*mapped.RowPitch,copy.WidthInBytes);
    cpu_stage_unmap(t);
    copy.srcMemoryType=1; copy.srcHost=t->pixels; copy.srcPitch=copy.WidthInBytes;
    copy.dstMemoryType=3; copy.dstArray=t->array;
    result=ptsz?pcuMemcpy2DAsync_v2_ptsz(&copy,stream):pcuMemcpy2DAsync_v2(&copy,stream);
    if(!result) {
        result=ptsz?pcuEventRecord_ptsz(t->upload_done,stream):pcuEventRecord(t->upload_done,stream);
        if(!result) { t->pending=TRUE; return CUDA_SUCCESS; }
    }
    /* Even a failed enqueue/record must not let the pinned source be reused
     * while work could still reference it. Failed drain quarantines the handle. */
    drained=ptsz?pcuStreamSynchronize_ptsz(stream):pcuStreamSynchronize(stream);
    if(drained) t->poisoned=TRUE;
    return result;
}
static CUresult cpu_download(ZludaCpuTexture *t, CUstream stream, BOOL ptsz)
{
    CUDA_MEMCPY2D copy={0};
    copy.WidthInBytes=(size_t)t->width*16; copy.Height=t->height;
    copy.srcMemoryType=3; copy.srcArray=t->array;
    copy.dstMemoryType=1; copy.dstHost=t->pixels; copy.dstPitch=copy.WidthInBytes;
    return ptsz?pcuMemcpy2DAsync_v2_ptsz(&copy,stream):pcuMemcpy2DAsync_v2(&copy,stream);
}
static CUresult cpu_publish(ZludaCpuTexture *t)
{
    D3D11_MAPPED_SUBRESOURCE mapped;
    CUresult result=cpu_stage_map(t,FALSE,&mapped);
    size_t row=(size_t)t->width*16;
    unsigned y;
    if(result) return result;
    for(y=0;y<t->height;y++) memcpy((char*)mapped.pData+y*mapped.RowPitch,(char*)t->pixels+y*row,row);
    cpu_stage_unmap(t);
    cpu_copy_resource(t,FALSE);
    return CUDA_SUCCESS;
}
/* Return -1 only when no copy resource is present (forward unchanged).
 * Reject mixed batches before changing any state. */
static int cpu_map(unsigned count, CUgraphicsResource *resources, CUstream stream, BOOL unmap, BOOL ptsz)
{
    unsigned i,j,found=0;
    CUresult result=CUDA_SUCCESS;
    if(!InterlockedCompareExchange(&cpu_copy_active,0,0) || !resources) return -1;
    pthread_mutex_lock(&cpu_textures_lock);
    for(i=0;i<count;i++) if(cpu_find(resources[i])) found++;
    if(!found) { pthread_mutex_unlock(&cpu_textures_lock); return -1; }
    if(found!=count) { result=CUDA_ERROR_NOT_SUPPORTED; goto done; }
    for(i=0;i<count;i++) {
        ZludaCpuTexture *t=cpu_find(resources[i]);
        for(j=0;j<i;j++) if(resources[i]==resources[j]) { result=CUDA_ERROR_INVALID_VALUE; goto done; }
        result=cpu_context_check(t); if(result) goto done;
        result=cpu_drain(t); if(result) goto done;
        if(t->mapped!=unmap) { result=unmap?211:208; goto done; }
    }
    if(!unmap) {
        for(i=0;i<count;i++) if(cpu_needs_input(cpu_find(resources[i]))) break;
        if(i==count) goto mapped; /* all WRITE_DISCARD: no snapshot, no waits */
        result=ptsz?pcuStreamSynchronize_ptsz(stream):pcuStreamSynchronize(stream);
        if(result) goto done;
        for(i=0;i<count;i++) {
            ZludaCpuTexture *t=cpu_find(resources[i]);
            if(cpu_needs_input(t)) cpu_copy_resource(t,TRUE);
        }
        result=cpu_wait_batch(count,resources); if(result) goto done;
        for(i=0;i<count;i++) {
            ZludaCpuTexture *t=cpu_find(resources[i]);
            if(!cpu_needs_input(t)) continue; /* caller discards the entire old image */
            result=cpu_upload(t,stream,ptsz);
            if(result) {
                /* State stays unmapped after a partial batch failure. Drain
                 * earlier uploads before a retry, unregister or bounce reuse. */
                for(j=0;j<i;j++) cpu_drain(cpu_find(resources[j]));
                goto done;
            }
        }
    } else {
        CUresult queued=CUDA_SUCCESS,drained;
        for(i=0;i<count;i++) {
            ZludaCpuTexture *t=cpu_find(resources[i]);
            if(t->map_flags==1) continue; /* caller promises CUDA made no writes */
            queued=cpu_download(t,stream,ptsz);
            if(queued) break;
        }
        /* Exactly one CUDA batch drain before CPU reads any download. */
        drained=ptsz?pcuStreamSynchronize_ptsz(stream):pcuStreamSynchronize(stream);
        if(drained) for(i=0;i<count;i++) cpu_find(resources[i])->poisoned=TRUE;
        result=queued?queued:drained; if(result) goto done;
        for(i=0;i<count;i++) {
            ZludaCpuTexture *t=cpu_find(resources[i]);
            if(t->map_flags==1) continue;
            result=cpu_publish(t); if(result) goto done;
        }
    }
mapped:
    for(i=0;i<count;i++) cpu_find(resources[i])->mapped=!unmap;
    { static unsigned messages; if(messages++<4) FIXME("[ZLUDA] CPU-copy %s OK count=%u\n",unmap?"unmap":"map",count); }
done:
    pthread_mutex_unlock(&cpu_textures_lock);
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
    case 2:
        if(t->mapped) result=208;
        else if(a>2) result=CUDA_ERROR_INVALID_VALUE;
        else t->map_flags=a;
        break;
    case 3: result=CUDA_ERROR_NOT_SUPPORTED; break;
    case 4:
        if(t->mapped) { result=208; break; }
        result=cpu_drain(t); if(result) break;
        result=pcuArrayDestroy(t->array); if(result) break;
        for(link=&cpu_textures;*link!=t;link=&(*link)->next) {}
        *link=t->next; cpu_release(t); break;
    default: result=CUDA_ERROR_INVALID_VALUE;
    }
done:
    pthread_mutex_unlock(&cpu_textures_lock);
    return result;
}
