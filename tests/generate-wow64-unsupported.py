#!/usr/bin/env python3
"""Expose unimplemented driver entries as explicit NOT_SUPPORTED errors.

Uses the Wine spec plus pinned CUDA pointer-width corrections for PE32. This provides loader
compatibility only: there is no success emulation and no argument dereference.
Implemented entries in wow64/exports.def always take precedence.
"""
import pathlib
import json
import re
import sys

repo, out = map(pathlib.Path, sys.argv[1:])
base = (repo / 'dlls/nvcuda/wow64/exports.def').read_text()
implemented = set(re.findall(r'^\s+(cu\w+)@\d+\s*$', base, re.M))
deviceptr_args = json.loads((repo / 'tests/wow64-deviceptr-args.json').read_text())['functions']
types = {'ptr': ('uint32_t', 4), 'str': ('uint32_t', 4),
         'wstr': ('uint32_t', 4), 'long': ('uint32_t', 4),
         'int64': ('uint64_t', 8), 'float': ('float', 4),
         'double': ('double', 8)}
c = ['/* Generated from nvcuda.spec: unsupported entries never return success. */',
     '#include <windows.h>', '#include <stdint.h>', '#include <stdio.h>']
exports = [base.rstrip()]
seen = set()
for line in (repo / 'dlls/nvcuda/nvcuda.spec').read_text().splitlines():
    match = re.fullmatch(r'@ stdcall (cu\w+)\(([^)]*)\) \w+', line)
    if not match:
        if line.startswith('@ stdcall cu'):
            raise ValueError('Unrecognized CUDA spec: ' + line)
        continue
    name, args = match.groups()
    if name in implemented or name in seen:
        continue
    seen.add(name)
    signature, unused, size = [], [], 0
    for i, arg in enumerate(args.split()):
        # Wine's shared spec describes CUdeviceptr as an 8-byte "double".
        # NVIDIA defines it as pointer-sized, including the v2 CUDA APIs.
        # Only the pinned by-value CUdeviceptr arguments are narrowed; actual
        # 64-bit texture/surface objects and scalar values retain their width.
        if i in deviceptr_args.get(name, []):
            arg = 'long'
        kind, width = types[arg]
        signature.append(f'{kind} a{i}')
        unused.append(f'(void)a{i};')
        size += width
    c.append(f'int WINAPI {name}({", ".join(signature) or "void"}) {{ '
             + ' '.join(unused)
             + f' fprintf(stderr, "cuda32: unsupported {name}\\n"); return 801; }}')
    exports.append(f' {name}@{size}')
(out / 'unsupported.c').write_text('\n'.join(c) + '\n')
(out / 'exports.def').write_text('\n'.join(exports) + '\n')
print(f'Generated {len(seen)} explicit unsupported exports; {len(implemented)} real exports')
