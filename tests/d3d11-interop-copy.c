/* Run under the target Proton variant. Verify a real DXVK -> CUDA -> DXVK
 * float4 texture round trip, including remap refresh and resource cleanup. */
#define COBJMACROS
#include <windows.h>
#include <d3d11.h>
#include <d3d10.h>
#include <stdio.h>
#include <string.h>
#include "cuda.h"
#define W 256
#define H 118
#define LOAD(name, type) type name = (type)GetProcAddress(cuda, #name); if (!name) { fprintf(stderr,"missing %s\n",#name); return 2; }
typedef CUresult (WINAPI *init_t)(unsigned);
typedef CUresult (WINAPI *ctx_t)(CUcontext *, unsigned, CUdevice);
typedef CUresult (WINAPI *destroy_t)(CUcontext);
typedef CUresult (WINAPI *reg_t)(CUgraphicsResource *, ID3D11Resource *, unsigned);
typedef CUresult (WINAPI *map_t)(unsigned, CUgraphicsResource *, CUstream);
typedef CUresult (WINAPI *array_t)(CUarray *, CUgraphicsResource, unsigned, unsigned);
typedef CUresult (WINAPI *copy_t)(const CUDA_MEMCPY2D *);
typedef CUresult (WINAPI *unreg_t)(CUgraphicsResource);
#define CU(call) do { CUresult r=(call); if(r) { fprintf(stderr,"FAIL %s = %d\n",#call,r); return 3; } } while(0)
#define HR(call) do { HRESULT r=(call); if(FAILED(r)) { fprintf(stderr,"FAIL %s = 0x%lx\n",#call,(unsigned long)r); return 4; } } while(0)
typedef CUresult (WINAPI *flags_t)(CUgraphicsResource,unsigned);
typedef CUresult (WINAPI *stream_create_t)(CUstream *,unsigned);
typedef CUresult (WINAPI *stream_t)(CUstream);
typedef struct RelayApi {
    reg_t reg; map_t map,unmap; array_t array; copy_t copy; unreg_t unreg;
    flags_t flags; stream_create_t create; stream_t sync,destroy;
} RelayApi;
static float input[W*H*4], output[W*H*4];
typedef struct LockWorker { ID3D10Multithread *mt; volatile LONG stop,progress; HANDLE started; } LockWorker;
static DWORD WINAPI lock_worker(void *arg)
{
    LockWorker *w=arg;
    SetEvent(w->started);
    while(!InterlockedCompareExchange(&w->stop,0,0)) {
        ID3D10Multithread_Enter(w->mt);
        InterlockedIncrement(&w->progress);
        ID3D10Multithread_Leave(w->mt);
        Sleep(0);
    }
    return 0;
}
static int relay_cases(RelayApi *a, ID3D11Device *dev, ID3D11DeviceContext *dc,
                       ID3D10Multithread *mt, ID3D11Texture2D *staging)
{
    D3D11_TEXTURE2D_DESC td={0};
    ID3D11Texture2D *textures[2];
    CUgraphicsResource resources[2];
    CUstream stream;
    LockWorker worker={0};
    HANDLE thread;
    td.Width=W; td.Height=H; td.MipLevels=td.ArraySize=td.SampleDesc.Count=1;
    td.Format=DXGI_FORMAT_R32G32B32A32_FLOAT; td.Usage=D3D11_USAGE_DEFAULT;
    td.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_RENDER_TARGET;
    CU(a->create(&stream,1)); /* nonblocking stream: no default-stream ordering crutch */
    for(unsigned n=0;n<2;n++) {
        HR(ID3D11Device_CreateTexture2D(dev,&td,NULL,&textures[n]));
        for(unsigned flag=1;flag<=8;flag*=2) {
            CUgraphicsResource bad=NULL;
            if(a->reg(&bad,(ID3D11Resource*)textures[n],flag)!=801) return 11;
        }
        CU(a->reg(&resources[n],(ID3D11Resource*)textures[n],0));
        if(a->flags(resources[n],3)!=1 || a->flags(resources[n],4)!=1 ||
           a->flags(resources[n],~0u)!=1) return 11;
    }
    worker.mt=mt; worker.started=CreateEventA(NULL,TRUE,FALSE,NULL);
    thread=CreateThread(NULL,0,lock_worker,&worker,0,NULL);
    if(!worker.started || !thread || WaitForSingleObject(worker.started,5000)!=WAIT_OBJECT_0) return 12;
    InterlockedExchange(&worker.progress,0);
    /* Same-content remaps exercise B's verified skip; intervening D3D writes
     * must invalidate it, even though these resources are RTV-bound. The
     * companion relay-contract.py forces NOT_READY and verifies each poll
     * drops the lock and that all copies precede the single context wait. */
    for(unsigned pass=0;pass<6;pass++) {
        for(unsigned i=0;i<W*H*4;i++) input[i]=(float)(i%311+pass*1000);
        if(pass%2==0) for(unsigned n=0;n<2;n++)
            ID3D11DeviceContext_UpdateSubresource(dc,(ID3D11Resource*)textures[n],0,NULL,input,W*16,0);
        else for(unsigned i=0;i<W*H*4;i++) input[i]=(float)(i%311+(pass-1)*1000);
        CU(a->map(2,resources,stream)); CU(a->sync(stream));
        if(a->flags(resources[0],1)!=208) return 11;
        for(unsigned n=0;n<2;n++) {
            CUarray array; CUDA_MEMCPY2D copy={0};
            CU(a->array(&array,resources[n],0,0));
            copy.srcMemoryType=3; copy.srcArray=array; copy.dstMemoryType=1;
            copy.dstHost=output; copy.dstPitch=copy.WidthInBytes=W*16; copy.Height=H;
            CU(a->copy(&copy));
            if(memcmp(input,output,sizeof(input))) return 13;
        }
        CU(a->unmap(2,resources,stream));
    }
    /* READ_ONLY refreshes D3D input and must never publish CUDA output.
     * WRITE_DISCARD requires no defined initial array contents; overwrite all. */
    for(unsigned mode=1;mode<=2;mode++) {
        for(unsigned n=0;n<2;n++) CU(a->flags(resources[n],mode));
        for(unsigned i=0;i<W*H*4;i++) input[i]=(float)(i%127+7000+mode);
        for(unsigned n=0;n<2;n++) ID3D11DeviceContext_UpdateSubresource(dc,(ID3D11Resource*)textures[n],0,NULL,input,W*16,0);
        CU(a->map(2,resources,stream)); CU(a->sync(stream));
        for(unsigned n=0;n<2;n++) {
            CUarray array; CUDA_MEMCPY2D copy={0};
            CU(a->array(&array,resources[n],0,0));
            copy.WidthInBytes=W*16; copy.Height=H;
            if(mode==1) {
                copy.srcMemoryType=3; copy.srcArray=array; copy.dstMemoryType=1;
                copy.dstHost=output; copy.dstPitch=W*16;
                CU(a->copy(&copy)); if(memcmp(input,output,sizeof(input))) return 14;
            } else {
                copy.srcMemoryType=1; copy.srcHost=input; copy.srcPitch=W*16;
                copy.dstMemoryType=3; copy.dstArray=array; CU(a->copy(&copy));
            }
        }
        CU(a->unmap(2,resources,stream));
        for(unsigned n=0;n<2;n++) {
            D3D11_MAPPED_SUBRESOURCE mapped;
            ID3D11DeviceContext_CopyResource(dc,(ID3D11Resource*)staging,(ID3D11Resource*)textures[n]);
            HR(ID3D11DeviceContext_Map(dc,(ID3D11Resource*)staging,0,D3D11_MAP_READ,0,&mapped));
            for(unsigned y=0;y<H;y++) if(memcmp(input+y*W*4,(char*)mapped.pData+y*mapped.RowPitch,W*16)) return 14;
            ID3D11DeviceContext_Unmap(dc,(ID3D11Resource*)staging,0);
        }
        printf("PASS batch map flag=%u nondefault stream\n",mode);
    }
    InterlockedExchange(&worker.stop,1);
    if(WaitForSingleObject(thread,5000)!=WAIT_OBJECT_0 || !worker.progress) return 12;
    printf("PASS concurrent D3D lock progress=%ld (forced polling checked by relay-contract)\n",worker.progress);
    CloseHandle(thread); CloseHandle(worker.started);
    for(unsigned n=0;n<2;n++) {
        CU(a->flags(resources[n],0)); CU(a->unreg(resources[n]));
        ID3D11Texture2D_Release(textures[n]);
    }
    CU(a->destroy(stream));
    puts("PASS batched same-content remap and D3D-write fallback");
    return 0;
}
int main(void) {
    if (!freopen("interop-result.txt","w",stdout) || !freopen("interop-result.txt","a",stderr)) return 7;
    setvbuf(stdout,NULL,_IONBF,0); setvbuf(stderr,NULL,_IONBF,0);
    HMODULE cuda=LoadLibraryA("nvcuda.dll"); if(!cuda) { fprintf(stderr,"LoadLibrary nvcuda failed %lu\n",GetLastError()); return 2; }
    LOAD(cuInit,init_t); LOAD(cuCtxCreate_v2,ctx_t); LOAD(cuCtxDestroy_v2,destroy_t);
    LOAD(cuGraphicsD3D11RegisterResource,reg_t); LOAD(cuGraphicsMapResources,map_t);
    LOAD(cuGraphicsUnmapResources,map_t); LOAD(cuGraphicsSubResourceGetMappedArray,array_t);
    LOAD(cuMemcpy2D_v2,copy_t); LOAD(cuGraphicsUnregisterResource,unreg_t);
    LOAD(cuGraphicsResourceSetMapFlags,flags_t); LOAD(cuStreamCreate,stream_create_t);
    LOAD(cuStreamSynchronize,stream_t); LOAD(cuStreamDestroy_v2,stream_t);
    CU(cuInit(0)); CUcontext ctx; CU(cuCtxCreate_v2(&ctx,0,0));
    ID3D11Device *dev; ID3D11DeviceContext *dc; D3D_FEATURE_LEVEL level;
    HR(D3D11CreateDevice(NULL,D3D_DRIVER_TYPE_HARDWARE,NULL,0,NULL,0,D3D11_SDK_VERSION,&dev,&level,&dc));
    ID3D10Multithread *mt; HR(ID3D11Device_QueryInterface(dev,&IID_ID3D10Multithread,(void**)&mt));
    char enable_mt[8];
    if(!GetEnvironmentVariableA("ZLUDA_D3D11_CPU_COPY_ENABLE_MT",enable_mt,sizeof(enable_mt)))
        ID3D10Multithread_SetMultithreadProtected(mt,TRUE);
    printf("D3D creation flags=0x%x initial protected=%d\n",ID3D11Device_GetCreationFlags(dev),ID3D10Multithread_GetMultithreadProtected(mt));
    D3D11_TEXTURE2D_DESC td={0}; td.Width=W; td.Height=H; td.MipLevels=td.ArraySize=td.SampleDesc.Count=1;
    td.Format=DXGI_FORMAT_R32G32B32A32_FLOAT; td.Usage=D3D11_USAGE_DEFAULT; td.BindFlags=D3D11_BIND_SHADER_RESOURCE;
    ID3D11Texture2D *texture,*staging;
    HR(ID3D11Device_CreateTexture2D(dev,&td,NULL,&texture));
    td.Usage=D3D11_USAGE_STAGING; td.BindFlags=0; td.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    HR(ID3D11Device_CreateTexture2D(dev,&td,NULL,&staging));
    for(unsigned trial=0;trial<3;trial++) {
        CUgraphicsResource res; CU(cuGraphicsD3D11RegisterResource(&res,(ID3D11Resource*)texture,0));
        CUgraphicsResource duplicate=NULL;
        if(cuGraphicsD3D11RegisterResource(&duplicate,(ID3D11Resource*)texture,0)!=400) { fprintf(stderr,"FAIL duplicate registration validation\n"); return 8; }
        printf("PASS duplicate registration rejected trial=%u\n",trial);
        if(!ID3D10Multithread_GetMultithreadProtected(mt)) { fprintf(stderr,"FAIL unprotected immediate context\n"); return 10; }
        CUarray unmapped_array;
        if(cuGraphicsSubResourceGetMappedArray(&unmapped_array,res,0,0)!=211) { fprintf(stderr,"FAIL unmapped array validation\n"); return 8; }
        CUgraphicsResource duplicates[2]={res,res};
        if(cuGraphicsMapResources(2,duplicates,NULL)!=1) { fprintf(stderr,"FAIL duplicate map validation\n"); return 8; }
        for(unsigned pass=0;pass<2;pass++) {
            for(unsigned i=0;i<W*H*4;i++) input[i]=(float)((i%997)+1000*pass+3000*trial);
            ID3D11DeviceContext_UpdateSubresource(dc,(ID3D11Resource*)texture,0,NULL,input,W*16,W*H*16);
            ID3D11DeviceContext_CopyResource(dc,(ID3D11Resource*)staging,(ID3D11Resource*)texture);
            D3D11_MAPPED_SUBRESOURCE direct;
            HR(ID3D11DeviceContext_Map(dc,(ID3D11Resource*)staging,0,D3D11_MAP_READ,0,&direct));
            for(unsigned y=0;y<H;y++) if(memcmp(input+y*W*4,(char*)direct.pData+y*direct.RowPitch,W*16)) {
                fprintf(stderr,"FAIL direct D3D readback row=%u pitch=%u\n",y,direct.RowPitch); return 9;
            }
            ID3D11DeviceContext_Unmap(dc,(ID3D11Resource*)staging,0);
            printf("PASS direct D3D readback trial=%u pass=%u\n",trial,pass);
            CU(cuGraphicsMapResources(1,&res,NULL)); CUarray array;
            CU(cuGraphicsSubResourceGetMappedArray(&array,res,0,0));
            if(cuGraphicsMapResources(1,&res,NULL)!=208 || cuGraphicsUnregisterResource(res)!=208) { fprintf(stderr,"FAIL mapped lifetime validation\n"); return 8; }
            CUDA_MEMCPY2D copy={0}; copy.srcMemoryType=3; copy.srcArray=array;
            copy.dstMemoryType=1; copy.dstHost=output; copy.dstPitch=W*16; copy.WidthInBytes=W*16; copy.Height=H;
            CU(cuMemcpy2D_v2(&copy));
            if(memcmp(input,output,sizeof(input))) {
                for(unsigned i=0;i<W*H*4;i++) if(input[i]!=output[i]) {
                    fprintf(stderr,"FAIL D3D->CUDA trial=%u pass=%u index=%u expected=%g actual=%g first=%g,%g,%g,%g\n",trial,pass,i,input[i],output[i],output[0],output[1],output[2],output[3]); break;
                }
                /* An explicit host upload isolates array-copy behavior from D3D. */
                CUDA_MEMCPY2D upload={0}; upload.srcMemoryType=1; upload.srcHost=input; upload.srcPitch=W*16;
                upload.dstMemoryType=3; upload.dstArray=array; upload.WidthInBytes=W*16; upload.Height=H;
                CU(cuMemcpy2D_v2(&upload)); memset(output,0,sizeof(output)); CU(cuMemcpy2D_v2(&copy));
                fprintf(stderr,"DIAG explicit CUDA upload/download equal=%d first=%g,%g,%g,%g\n",!memcmp(input,output,sizeof(input)),output[0],output[1],output[2],output[3]);
                return 5;
            }
            for(unsigned i=0;i<W*H*4;i++) input[i]=-input[i]-1;
            memset(&copy,0,sizeof(copy)); copy.srcMemoryType=1; copy.srcHost=input; copy.srcPitch=W*16;
            copy.dstMemoryType=3; copy.dstArray=array; copy.WidthInBytes=W*16; copy.Height=H;
            CU(cuMemcpy2D_v2(&copy)); CU(cuGraphicsUnmapResources(1,&res,NULL));
            ID3D11DeviceContext_CopyResource(dc,(ID3D11Resource*)staging,(ID3D11Resource*)texture);
            D3D11_MAPPED_SUBRESOURCE mapped; HR(ID3D11DeviceContext_Map(dc,(ID3D11Resource*)staging,0,D3D11_MAP_READ,0,&mapped));
            for(unsigned y=0;y<H;y++) if(memcmp(input+y*W*4,(char*)mapped.pData+y*mapped.RowPitch,W*16)) {
                fprintf(stderr,"FAIL CUDA->D3D trial=%u pass=%u row=%u\n",trial,pass,y); return 6;
            }
            ID3D11DeviceContext_Unmap(dc,(ID3D11Resource*)staging,0);
            printf("PASS float4 roundtrip trial=%u pass=%u pixels=%u\n",trial,pass,W*H);
        }
        CU(cuGraphicsUnregisterResource(res));
    }
    RelayApi api={cuGraphicsD3D11RegisterResource,cuGraphicsMapResources,cuGraphicsUnmapResources,
        cuGraphicsSubResourceGetMappedArray,cuMemcpy2D_v2,cuGraphicsUnregisterResource,
        cuGraphicsResourceSetMapFlags,cuStreamCreate,cuStreamSynchronize,cuStreamDestroy_v2};
    int relay_result=relay_cases(&api,dev,dc,mt,staging); if(relay_result) return relay_result;
    ID3D11Texture2D_Release(staging); ID3D11Texture2D_Release(texture);
    ID3D10Multithread_Release(mt); ID3D11DeviceContext_Release(dc); ID3D11Device_Release(dev); CU(cuCtxDestroy_v2(ctx));
    puts("PASS all interop checks"); return 0;
}
