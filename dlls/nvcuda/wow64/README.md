# WoW64 compute prototype

This is an opt-in experiment based on nvcuda `e903f13c`. It adds a real PE32
frontend and a 64-bit Unix library alongside the existing relay. It does not
change the normal build, Steam tool selection, or game prefixes.

## Call path

The PE frontend uses Wine's Unix-call dispatcher to enter the 64-bit ELF
library in the same process. There is no RPC server. Arguments cross this
boundary as fixed-width words. CUDA context, module and function handles are
tokens, not truncated native pointers.

The backend loads ZLUDA's `libcuda.so` from its own directory (the compat
tool installs both in `files/lib/wine/x86_64-unix`); an absolute path in
`ZLUDA_WOW64_LIBRARY` overrides that. Neither enables Wine WoW64. The launcher must also
enable Wine's new WoW64 mode, with the matching PE and Unix nvcuda modules
installed in a disposable runtime. GE11 supports `PROTON_USE_WOW64=1`, which
sets `WINEARCH=wow64`; the direct-Wine test sets the latter.

Device pointers exposed to the application are offsets in a per-context
512 MiB pool, allocated lazily.
ZLUDA's existing 32-bit PTX pass translates device accesses using the pool
base, which the relay appends after the explicit kernel arguments. This also
handles pointers stored inside GPU memory; merely widening API arguments
would not.

`CUdeviceptr` is pointer-sized in NVIDIA's API: **four bytes in PE32,
including v2 calls**, eight in the native backend. The PE exports and their
stdcall cleanup sizes follow PE32, not Wine's shared 64-bit `double` entries.
The unsupported-export generator applies a pinned list of by-value
CUdeviceptr corrections from ZLUDA `2173d35d`'s CUDA declarations; genuine
64-bit objects retain eight bytes. Output canaries check four-byte allocation
and global addresses. Earlier fixtures incorrectly agreed with eight-byte
prototype v2 signatures and did not validate that part of the CUDA ABI.

The metadata interface comes from the pinned ZLUDA source's two-entry
`ZLUDA32_INTERNAL` export table. It has no leading size member. Kernel argument
metadata describes the application's explicit parameters only.

## Build and run

On the Deck, `bash tests/build-wow64.sh` builds the two libraries and the
PE32 compute fixture using the existing devroot MinGW compiler and native C
compiler. `NVCUDA_DEVROOT` and `NVCUDA_WOW64_OUT` override their locations.

The hub's `runbooks/check-zluda-wow64.sh` prepares a hard-linked disposable
GE11 runtime, unlinking each replacement before writing it. It uses the
standard loader/preloader and an isolated prefix with `WINEARCH=wow64`.
Manual loader redirects from the first experiment are no longer needed.
Run it through `run-quiet.sh` for muted audio and actual panel sleep.

## Limits

This is not yet a complete PhysX implementation. It supports up to 16
contexts, checked streams/events, at most 256 allocations per context,
2,048 total handles and 32
explicit arguments. Bounded byte-array module globals, linear texture
references and HIP-registered PE32 host allocations are implemented. Arrays,
mapped/write-combined host allocations and the legacy parameter/launch API
are absent.
It serializes its own calls with a mutex and synchronizes before pool-offset
reuse. Those are prototype constraints, not adopted production defaults.
Legacy device properties, compute capability, 32-bit total-memory outputs and
context push/pop are now covered by the fixture, including a new Windows
thread. Total memory advertises at most the bounded pool's usable capacity.
`ZLUDA_WOW64_TRACE=1` enables opcode/result logging for startup diagnostics.

Asylum exposed the former one-context limit after its private NVAPI topology
query was implemented. A fresh-prefix fixture now exercises two contexts
with identical PE32 offsets and different GPU results, context-scoped
module/stream handles, and destruction of a non-current context while
preserving the current context's data. Existing compute tests also pass.
The limit of 16 contexts and borrowed-primary lifetime handling remain
prototype constraints; this is not a full CUDA context implementation.

It validates allocation ranges and handle identities, but does not contain
Wine exception handling for arbitrary invalid application pointers. The
fixture covers valid host pointers and explicit negative API cases, not the
entire CUDA contract.


`nvapi-compute-probe.exe` separately checks the legacy topology query used by
PhysXDevice 2.8.4. It is an enumeration check, not a PhysX compute pass;
probe logs and DLL hashes are kept in the hub evidence.

## Upstream design credit

The device-offset pool and appended hidden argument follow vosen/ZLUDA's
`zluda64_server/src/main.rs`, `ptx/src/pass/convert_32bit_to_64bit.rs` and
`dark_api/src/lib.rs` at the audited Stage3 pin. See `ZLUDA-MIT.txt` for its
MIT permission notice. The new relay files use LGPL-2.1-or-later, matching
nvcuda.
