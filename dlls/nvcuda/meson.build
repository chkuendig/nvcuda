nvcuda_src = [
  'internal.c',
  'nvcuda.c',
  'function_mappings.c',
  'encryption.c',
  'd3d11_interop.c',
]

thread_dep = dependency('threads')


shared_library(
  'nvcuda.dll',
  nvcuda_src,
  name_prefix: '',
  dependencies: [thread_dep, lib_dl],
  include_directories: include_path,
  link_depends: ['nvcuda.spec'],
  link_args: [meson.current_source_dir() / 'nvcuda.spec'],
  install: true,
  install_dir: install_dir_unix,
)

# ROCm stack fix — LD_PRELOAD shim for pthread_getattr_np (ROCm/clr#61).
# Wine's main thread has an unusually large reported stack that confuses
# ROCm's HSA runtime thread pool init. This shim clamps to 8MB.
shared_library(
  'wine_stack_fix',
  'wine_stack_fix.c',
  name_prefix: '',
  dependencies: [thread_dep, lib_dl],
  install: true,
  install_dir: install_dir_unix,
)
