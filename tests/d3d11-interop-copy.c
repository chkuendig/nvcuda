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
typedef CUresult (WINAPI *surf_t)(CUsurfObject *, const void *);
typedef CUresult (WINAPI *tex_t)(CUtexObject *, const void *, const void *, const void *);
typedef CUresult (WINAPI *objdestroy_t)(unsigned long long);
#define CU(call) do { CUresult r=(call); if(r) { fprintf(stderr,"FAIL %s = %d\n",#call,r); return 3; } } while(0)
#define HR(call) do { HRESULT r=(call); if(FAILED(r)) { fprintf(stderr,"FAIL %s = 0x%lx\n",#call,(unsigned long)r); return 4; } } while(0)
static float input[W*H*4], output[W*H*4];
/* S3-W coverage mode (ZLUDA_D3D11_WRITE_COVERAGE=256x118): the mapped array
 * holds the sentinel; partial uploads must come back with every channel the
 * upload did not write restored to the pre-map D3D content. */
#define SENTINEL 0x7FC0C0DEu
#define OTHER_NAN 0x7FC00001u
static float premap[W*H*4], expect[W*H*4];
static void put_bits(float *f, unsigned bits) { memcpy(f,&bits,4); }
/* Expected D3D content after unmap for one host->array upload of rows [y0,y0+rows) and texels [0,texels). */
static void expect_upload(unsigned y0, unsigned rows, unsigned texels)
{
    for(unsigned y=y0;y<y0+rows;y++) for(unsigned x=0;x<texels;x++) for(unsigned k=0;k<4;k++) {
        unsigned i=(y*W+x)*4+k, bits; memcpy(&bits,&input[i],4);
        memcpy(&expect[i],bits==SENTINEL?&premap[i]:&input[i],4); /* bit copies: NaN payloads must survive */
    }
}
static CUresult upload_rows(copy_t copy2d, CUarray array, unsigned y0, unsigned rows, unsigned texels)
{
    CUDA_MEMCPY2D c={0}; c.srcMemoryType=1; c.srcHost=input+y0*W*4; c.srcPitch=W*16;
    c.dstMemoryType=3; c.dstArray=array; c.dstY=y0; c.WidthInBytes=texels*16; c.Height=rows;
    expect_upload(y0,rows,texels);
    return copy2d(&c);
}
int main(void) {
    if (!freopen("interop-result.txt","w",stdout) || !freopen("interop-result.txt","a",stderr)) return 7;
    setvbuf(stdout,NULL,_IONBF,0); setvbuf(stderr,NULL,_IONBF,0);
    HMODULE cuda=LoadLibraryA("nvcuda.dll"); if(!cuda) { fprintf(stderr,"LoadLibrary nvcuda failed %lu\n",GetLastError()); return 2; }
    LOAD(cuInit,init_t); LOAD(cuCtxCreate_v2,ctx_t); LOAD(cuCtxDestroy_v2,destroy_t);
    LOAD(cuGraphicsD3D11RegisterResource,reg_t); LOAD(cuGraphicsMapResources,map_t);
    LOAD(cuGraphicsUnmapResources,map_t); LOAD(cuGraphicsSubResourceGetMappedArray,array_t);
    LOAD(cuMemcpy2D_v2,copy_t); LOAD(cuGraphicsUnregisterResource,unreg_t);
    CU(cuInit(0)); CUcontext ctx; CU(cuCtxCreate_v2(&ctx,0,0));
    ID3D11Device *dev; ID3D11DeviceContext *dc; D3D_FEATURE_LEVEL level;
    HR(D3D11CreateDevice(NULL,D3D_DRIVER_TYPE_HARDWARE,NULL,0,NULL,0,D3D11_SDK_VERSION,&dev,&level,&dc));
    ID3D10Multithread *mt; HR(ID3D11Device_QueryInterface(dev,&IID_ID3D10Multithread,(void**)&mt));
    char enable_mt[8], marker_env[8], coverage_env[16];
    /* S2 marker mode: the relay marks the third of three same-size registrations
     * (here trial 2); its CUDA->D3D copy-back must then read back as magenta. */
    int marker_mode=GetEnvironmentVariableA("ZLUDA_D3D11_COPY_MARKER",marker_env,sizeof(marker_env))==1 && marker_env[0]=='1';
    printf("marker mode=%d\n",marker_mode);
    int coverage_mode=GetEnvironmentVariableA("ZLUDA_D3D11_WRITE_COVERAGE",coverage_env,sizeof(coverage_env))==7 && !strcmp(coverage_env,"256x118");
    printf("coverage mode=%d\n",coverage_mode);
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
            if(coverage_mode) {
                /* Map uploaded the sentinel, not the D3D content. */
                for(unsigned i=0;i<W*H*4;i++) { unsigned bits; memcpy(&bits,&output[i],4); if(bits!=SENTINEL) {
                    fprintf(stderr,"FAIL coverage sentinel trial=%u pass=%u index=%u bits=0x%08x\n",trial,pass,i,bits); return 11; } }
                printf("PASS coverage sentinel trial=%u pass=%u texels=%u\n",trial,pass,W*H);
                memcpy(premap,input,sizeof(input)); memcpy(expect,premap,sizeof(premap));
                for(unsigned i=0;i<W*H*4;i++) input[i]=-input[i]-1;
                if(!pass) CU(upload_rows(cuMemcpy2D_v2,array,0,50,W)); /* rows 0..49 */
                else {
                    /* Rows 0..59 full, with three texels holding another NaN in G (nan_other=3) and
                     * two whose B is the sentinel bits (partial_texels=2, B restored); rows 60..61 first 100 texels. */
                    for(unsigned k=0;k<3;k++) put_bits(&input[(7*W+11+k)*4+1],OTHER_NAN);
                    for(unsigned k=0;k<2;k++) put_bits(&input[(20*W+40+k)*4+2],SENTINEL);
                    CU(upload_rows(cuMemcpy2D_v2,array,0,60,W));
                    CU(upload_rows(cuMemcpy2D_v2,array,60,2,100));
                }
                CU(cuGraphicsUnmapResources(1,&res,NULL));
                ID3D11DeviceContext_CopyResource(dc,(ID3D11Resource*)staging,(ID3D11Resource*)texture);
                D3D11_MAPPED_SUBRESOURCE back; HR(ID3D11DeviceContext_Map(dc,(ID3D11Resource*)staging,0,D3D11_MAP_READ,0,&back));
                for(unsigned y=0;y<H;y++) if(memcmp(expect+y*W*4,(char*)back.pData+y*back.RowPitch,W*16)) {
                    fprintf(stderr,"FAIL coverage restore trial=%u pass=%u row=%u\n",trial,pass,y); return 12;
                }
                ID3D11DeviceContext_Unmap(dc,(ID3D11Resource*)staging,0);
                printf("PASS coverage copy-back trial=%u pass=%u rows=%u\n",trial,pass,pass?62:50);
                continue;
            }
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
            if(marker_mode && trial==2) {
                static const float magenta[4]={1.0f,0.0f,1.0f,1.0f};
                for(unsigned y=0;y<H;y++) for(unsigned x=0;x<W;x++) if(memcmp((char*)mapped.pData+y*mapped.RowPitch+x*16,magenta,16)) {
                    fprintf(stderr,"FAIL marker CUDA->D3D trial=%u pass=%u row=%u col=%u\n",trial,pass,y,x); return 6;
                }
                ID3D11DeviceContext_Unmap(dc,(ID3D11Resource*)staging,0);
                printf("PASS marker copy-back trial=%u pass=%u pixels=%u\n",trial,pass,W*H);
                continue;
            }
            for(unsigned y=0;y<H;y++) if(memcmp(input+y*W*4,(char*)mapped.pData+y*mapped.RowPitch,W*16)) {
                fprintf(stderr,"FAIL CUDA->D3D trial=%u pass=%u row=%u\n",trial,pass,y); return 6;
            }
            ID3D11DeviceContext_Unmap(dc,(ID3D11Resource*)staging,0);
            printf("PASS float4 roundtrip trial=%u pass=%u pixels=%u\n",trial,pass,W*H);
        }
        if(coverage_mode && trial==2) {
            /* Nothing written while mapped: the whole texture must come back unchanged.
             * Texture/surface objects on the mapped array exercise the bind witness; any result is accepted. */
            surf_t cuSurfObjectCreate=(surf_t)GetProcAddress(cuda,"cuSurfObjectCreate"); tex_t cuTexObjectCreate=(tex_t)GetProcAddress(cuda,"cuTexObjectCreate");
            objdestroy_t cuSurfObjectDestroy=(objdestroy_t)GetProcAddress(cuda,"cuSurfObjectDestroy"), cuTexObjectDestroy=(objdestroy_t)GetProcAddress(cuda,"cuTexObjectDestroy");
            struct { int resType; int pad; CUarray hArray; unsigned char rest[136]; } desc={0};
            unsigned char texdesc[128]={0};
            CUsurfObject surface=0; CUtexObject tex=0; CUarray array; CUresult rs, rt;
            if(!cuSurfObjectCreate || !cuTexObjectCreate || !cuSurfObjectDestroy || !cuTexObjectDestroy) { fprintf(stderr,"FAIL missing object exports\n"); return 2; }
            CU(cuGraphicsMapResources(1,&res,NULL)); CU(cuGraphicsSubResourceGetMappedArray(&array,res,0,0));
            desc.resType=0; desc.hArray=array;
            rs=cuSurfObjectCreate(&surface,&desc); if(!rs) cuSurfObjectDestroy(surface);
            rt=cuTexObjectCreate(&tex,&desc,texdesc,NULL); if(!rt) cuTexObjectDestroy(tex);
            printf("INFO coverage objects surf_result=%d tex_result=%d\n",(int)rs,(int)rt);
            CU(cuGraphicsUnmapResources(1,&res,NULL));
            ID3D11DeviceContext_CopyResource(dc,(ID3D11Resource*)staging,(ID3D11Resource*)texture);
            D3D11_MAPPED_SUBRESOURCE back; HR(ID3D11DeviceContext_Map(dc,(ID3D11Resource*)staging,0,D3D11_MAP_READ,0,&back));
            for(unsigned y=0;y<H;y++) if(memcmp(expect+y*W*4,(char*)back.pData+y*back.RowPitch,W*16)) {
                fprintf(stderr,"FAIL coverage unwritten restore row=%u\n",y); return 12;
            }
            ID3D11DeviceContext_Unmap(dc,(ID3D11Resource*)staging,0);
            puts("PASS coverage unwritten map restored");
        }
        if(trial==2 && GetEnvironmentVariableA("ZLUDA_COPY_PROBE_FIXTURE",enable_mt,sizeof(enable_mt))) {
            Sleep(31000); /* Complete the explicitly armed diagnostic window. */
            CU(cuGraphicsMapResources(1,&res,NULL));
            CU(cuGraphicsUnmapResources(1,&res,NULL));
            puts("PASS probe window completed");
        }
        CU(cuGraphicsUnregisterResource(res));
    }
    ID3D11Texture2D_Release(staging); ID3D11Texture2D_Release(texture);
    ID3D10Multithread_Release(mt); ID3D11DeviceContext_Release(dc); ID3D11Device_Release(dev); CU(cuCtxDestroy_v2(ctx));
    puts("PASS all interop checks"); return 0;
}
