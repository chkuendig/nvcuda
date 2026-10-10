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
    CUcontext cuda_context;
    CUarray array;
    void *pixels;
    UINT width, height;
    BOOL mapped;
} ZludaCpuTexture;
static ZludaCpuTexture *cpu_textures;
static pthread_mutex_t cpu_textures_lock = PTHREAD_MUTEX_INITIALIZER;

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
    HeapFree(GetProcessHeap(),0,t->pixels);
    if(t->staging) ID3D11Texture2D_Release(t->staging);
    if(t->texture) ID3D11Texture2D_Release(t->texture);
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
    if(!out || !resource) return CUDA_ERROR_INVALID_VALUE;
    if(flags) return CUDA_ERROR_NOT_SUPPORTED;
    t=HeapAlloc(GetProcessHeap(),HEAP_ZERO_MEMORY,sizeof(*t));
    if(!t) return CUDA_ERROR_OUT_OF_MEMORY;
    hr=ID3D11Resource_QueryInterface(resource,&IID_ID3D11Texture2D,(void**)&t->texture);
    if(FAILED(hr)) { cpu_release(t); return CUDA_ERROR_NOT_SUPPORTED; }
    ID3D11Texture2D_GetDesc(t->texture,&desc);
    if(desc.Format!=DXGI_FORMAT_R32G32B32A32_FLOAT || desc.MipLevels!=1 || desc.ArraySize!=1 || desc.SampleDesc.Count!=1) {
        cpu_release(t); return CUDA_ERROR_NOT_SUPPORTED;
    }
    t->width=desc.Width; t->height=desc.Height;
    t->pixels=HeapAlloc(GetProcessHeap(),0,(size_t)t->width*t->height*16);
    if(!t->pixels) { cpu_release(t); return CUDA_ERROR_OUT_OF_MEMORY; }
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
    t->next=cpu_textures; cpu_textures=t;
    pthread_mutex_unlock(&cpu_textures_lock);
    *out=(CUgraphicsResource)t;
    FIXME("[ZLUDA] CPU-copy registration OK %ux%u resource=%p array=%p\n",t->width,t->height,t,t->array);
    return CUDA_SUCCESS;
}
static CUresult cpu_transfer(ZludaCpuTexture *t, BOOL to_cuda)
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
        copy.srcHost=t->pixels; copy.srcPitch=copy.WidthInBytes; copy.dstArray=t->array;
    }
    else { copy.srcArray=t->array; copy.dstHost=t->pixels; copy.dstPitch=copy.WidthInBytes; }
    result=pcuMemcpy2D_v2(&copy);
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
    CUresult result=CUDA_SUCCESS;
    if(!resources) return -1;
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
    for(i=0;i<count;i++) { result=cpu_transfer(cpu_find(resources[i]),!unmap); if(result) goto done; }
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
        for(link=&cpu_textures;*link!=t;link=&(*link)->next) {}
        *link=t->next; cpu_release(t); break;
    default: result=CUDA_ERROR_INVALID_VALUE;
    }
done:
    pthread_mutex_unlock(&cpu_textures_lock);
    return result;
}
