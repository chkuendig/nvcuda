/* Host-only shim for d3d11_transfer_census.h tests (DRAFT, not yet executed).
 * Provides the handful of Win32/Wine names the census header uses, backed by
 * pthreads, plus the CENSUS_TEST_POINT hook. Include before the header. */
#ifndef CENSUS_TEST_SHIM_H
#define CENSUS_TEST_SHIM_H
#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include "cuda.h"

typedef int BOOL;
#define TRUE 1
#define FALSE 0
typedef unsigned int DWORD;
typedef void *HANDLE;
typedef void *HMODULE;
typedef const unsigned short *LPCWSTR;
#define WINAPI
#define GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS 4
#define FIXME(...) fprintf(stderr,"fixme:census-test: " __VA_ARGS__)
#define ERR(...) fprintf(stderr,"err:census-test: " __VA_ARGS__)

/* 0: CreateThread reports success without starting the census thread (tests
 * drive census_publish directly); 1: a real pthread runs census_thread. */
static int shim_thread_mode;
static int shim_module_handle_ok=1;
static int shim_module_refs;
static void (*census_test_hook)(int id);
#define CENSUS_TEST_POINT(id) do { if(census_test_hook) census_test_hook(id); } while(0)

static BOOL GetModuleHandleExW(DWORD flags, LPCWSTR address, HMODULE *module)
{
    (void)flags; (void)address;
    if(!shim_module_handle_ok) return FALSE;
    __atomic_add_fetch(&shim_module_refs,1,__ATOMIC_SEQ_CST);
    *module=(HMODULE)&shim_module_refs;
    return TRUE;
}
static BOOL FreeLibrary(HMODULE module)
{
    (void)module;
    __atomic_sub_fetch(&shim_module_refs,1,__ATOMIC_SEQ_CST);
    return TRUE;
}
static void FreeLibraryAndExitThread(HMODULE module, DWORD code)
{
    (void)code;
    FreeLibrary(module);
    pthread_exit(NULL);
}
typedef DWORD (WINAPI *SHIM_THREAD_START)(void *);
struct shim_start { SHIM_THREAD_START fn; void *arg; };
static void *shim_trampoline(void *p)
{
    struct shim_start start=*(struct shim_start*)p;
    free(p);
    start.fn(start.arg);
    return NULL;
}
static HANDLE CreateThread(void *security, size_t stack, SHIM_THREAD_START fn, void *arg, DWORD flags, DWORD *id)
{
    pthread_t thread;
    struct shim_start *start;
    (void)security; (void)stack; (void)flags; (void)id;
    if(!shim_thread_mode) return (HANDLE)1;
    if(!(start=malloc(sizeof(*start)))) return NULL;
    start->fn=fn; start->arg=arg;
    if(pthread_create(&thread,NULL,shim_trampoline,start)) { free(start); return NULL; }
    pthread_detach(thread);
    return (HANDLE)2;
}
static BOOL CloseHandle(HANDLE handle) { (void)handle; return TRUE; }
static void Sleep(DWORD ms)
{
    struct timespec ts={ms/1000,(long)(ms%1000)*1000000L};
    while(nanosleep(&ts,&ts)) {}
}
static DWORD GetCurrentThreadId(void) { return (DWORD)syscall(SYS_gettid); }
#endif
