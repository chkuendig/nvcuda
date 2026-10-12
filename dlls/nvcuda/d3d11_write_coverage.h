/* Opt-in S3-W diagnostic (ZLUDA_D3D11_WRITE_COVERAGE=WxH, read once).
 * For registrations of exactly that size, the map-time upload carries a
 * sentinel (quiet NaN 0x7FC0C0DE in every channel) instead of the D3D content,
 * which is kept in a shadow. At copy-back, every channel still holding the
 * sentinel was not written through CUDA during that map and is restored from
 * the shadow, so D3D receives what a normal run would. Row-granular witness
 * lines report what was written, per registration ordinal.
 * T, the array transfer witness (same opt-in): forwarded CUDA calls whose
 * source or destination is a covered array, and texture/surface bindings of
 * it, are logged change-only. No extra CUDA or D3D calls and no waits; a
 * matching call takes the registry lock, nothing else does.
 * Included from d3d11_cpu_copy.h after the probe header. */
#define COV_SENTINEL 0x7FC0C0DEu
#define COV_MAX_W 16384
#define COV_MAX_H 512          /* keeps two row bitmaps inside one debug line */
#define COV_LINE_LIMIT 256     /* change lines per resource, coverage and T each */
#define COV_BEAT_LIMIT 1024    /* heartbeat lines per resource, coverage and T each */
#define COV_BEAT_NS UINT64_C(1000000000)
#define COV_TABLE 64
#define COV_PARTIAL_ROWS 8

#define COV_FUNCTIONS(X) \
    X(cuMemcpy2D) X(cuMemcpy2DAsync) X(cuMemcpy2DAsync_v2) X(cuMemcpy2DAsync_v2_ptsz) \
    X(cuMemcpy2DUnaligned) X(cuMemcpy2DUnaligned_v2) X(cuMemcpy2DUnaligned_v2_ptds) \
    X(cuMemcpy2D_v2) X(cuMemcpy2D_v2_ptds) \
    X(cuMemcpy3D) X(cuMemcpy3DAsync) X(cuMemcpy3DAsync_v2) X(cuMemcpy3DAsync_v2_ptsz) \
    X(cuMemcpy3DPeer) X(cuMemcpy3DPeer_ptds) X(cuMemcpy3DPeerAsync) X(cuMemcpy3DPeerAsync_ptsz) \
    X(cuMemcpy3D_v2) X(cuMemcpy3D_v2_ptds) \
    X(cuMemcpyAtoA) X(cuMemcpyAtoA_v2) X(cuMemcpyAtoA_v2_ptds) \
    X(cuMemcpyAtoD) X(cuMemcpyAtoD_v2) X(cuMemcpyAtoD_v2_ptds) \
    X(cuMemcpyAtoH) X(cuMemcpyAtoH_v2) X(cuMemcpyAtoH_v2_ptds) \
    X(cuMemcpyAtoHAsync) X(cuMemcpyAtoHAsync_v2) X(cuMemcpyAtoHAsync_v2_ptsz) \
    X(cuMemcpyDtoA) X(cuMemcpyDtoA_v2) X(cuMemcpyDtoA_v2_ptds) \
    X(cuMemcpyHtoA) X(cuMemcpyHtoA_v2) X(cuMemcpyHtoA_v2_ptds) \
    X(cuMemcpyHtoAAsync) X(cuMemcpyHtoAAsync_v2) X(cuMemcpyHtoAAsync_v2_ptsz) \
    X(cuSurfObjectCreate) X(cuTexObjectCreate) X(cuSurfRefSetArray) X(cuTexRefSetArray)
#define COV_ENUM(name) COV_##name,
enum { COV_FUNCTIONS(COV_ENUM) COV_FUNCTION_COUNT };
#undef COV_ENUM
#define COV_NAME(name) #name,
static const char *const cov_function_names[] = { COV_FUNCTIONS(COV_NAME) };
#undef COV_NAME

enum { COV_SIDE_SRC, COV_SIDE_DST, COV_SIDE_BIND, COV_SIDES };
static const char *const cov_side_names[] = { "src", "dst", "bind" };

/* One forwarded call, normalised. Linear (1D) entry points: x = byte offset,
 * width = ByteCount, height = 1, depth = 0. 2D: depth = 0. Binds: no dims. */
typedef struct {
    int src_type, dst_type, res_type, result;
    unsigned flags, linear; /* no padding: keys are compared with memcmp */
    size_t src_x, src_y, src_z, dst_x, dst_y, dst_z, width, height, depth;
} CovTransferKey;
typedef struct {
    unsigned function;
    CUarray src_array, dst_array, bind_array;
    CUstream stream;
    CovTransferKey key;
} CovTransfer;

typedef struct ZludaCoverage {
    uint32_t *shadow;                 /* D3D content read at map, float4 bits */
    unsigned char *rows, *last_rows;  /* bit set = row has a written texel; row 0 = MSB of byte 0 */
    unsigned char *t_rows, *last_t_rows; /* rows covered by successful transfers into the array this map */
    size_t written, nan_other, partial_texels;
    unsigned lines, suppressed, beats, role_witness, table_slot;
    unsigned long long copies, beat_ns;
    BOOL logged;
    /* T: per map (reset at the sentinel upload), totals, and the last logged key per function and side */
    unsigned t_dst, t_src, t_bind;
    unsigned long long t_total, t_since_beat, t_unmapped, t_beat_ns;
    unsigned t_lines, t_suppressed, t_beats;
    BOOL t_seen[COV_FUNCTION_COUNT][COV_SIDES];
    CovTransferKey t_last[COV_FUNCTION_COUNT][COV_SIDES];
} ZludaCoverage;

static UINT cov_width, cov_height;
static LONG cov_state; /* 0 unread, 1 off, 2 on */
/* Covered arrays, read without a lock by the wrappers. Written only under
 * cpu_textures_lock; a hit is re-checked against the registry under that lock. */
static CUarray cov_arrays[COV_TABLE];
static LONG cov_table_live; /* set once the first covered array is published */
static LONG cov_table_full;

static const char *cov_role(unsigned long long ordinal)
{
    static const char *const roles[3] = { "colour", "position", "scale" };
    return roles[ordinal % 3];
}
static BOOL cov_log_count(unsigned count) { return count <= 8 || !(count & (count - 1)); }

/* Strict "WxH": decimal, no sign, no leading zero, W <= 16384, H <= 512. */
static BOOL cov_parse(const char *s, UINT *w, UINT *h)
{
    unsigned v[2] = { 0, 0 }, i, digits;
    for (i = 0; i < 2; i++)
    {
        if (*s < '1' || *s > '9') return FALSE;
        for (digits = 0; *s >= '0' && *s <= '9'; s++)
        {
            if (++digits > 5) return FALSE;
            v[i] = v[i] * 10 + (*s - '0');
        }
        if (*s++ != (i ? '\0' : 'x')) return FALSE;
    }
    if (v[0] > COV_MAX_W || v[1] > COV_MAX_H) return FALSE;
    *w = v[0]; *h = v[1];
    return TRUE;
}
static BOOL cpu_coverage_enabled(void)
{
    LONG value = InterlockedCompareExchange(&cov_state, 0, 0);
    if (!value)
    {
        const char *env = getenv("ZLUDA_D3D11_WRITE_COVERAGE");
        UINT w = 0, h = 0;
        BOOL valid = env && cov_parse(env, &w, &h);
        if (valid) { cov_width = w; cov_height = h; } /* racing first readers store the same values */
        value = valid ? 2 : 1;
        if (!InterlockedCompareExchange(&cov_state, value, 0) && env)
        {
            if (valid) FIXME("[ZLUDA] CPU-copy explicit opt-in write coverage for %ux%u: sentinel upload at map, unwritten channels restored at copy-back, array transfer witness\n", w, h);
            else FIXME("[ZLUDA] CPU-copy write coverage value rejected (expected WxH, W 1..16384, H 1..512); coverage disabled\n");
        }
        value = InterlockedCompareExchange(&cov_state, 0, 0);
    }
    return value == 2;
}

static void cov_free(ZludaCoverage *c)
{
    if (!c) return;
    HeapFree(GetProcessHeap(), 0, c->shadow);
    HeapFree(GetProcessHeap(), 0, c->rows);
    HeapFree(GetProcessHeap(), 0, c);
}
/* Registration of a covered size; t->width/height are set. */
static ZludaCoverage *cov_alloc(const ZludaCpuTexture *t)
{
    size_t row_bytes = (t->height + 7) / 8;
    ZludaCoverage *c = HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, sizeof(*c));
    if (!c) return NULL;
    c->shadow = HeapAlloc(GetProcessHeap(), 0, (size_t)t->width * t->height * 16);
    c->rows = HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, row_bytes * 4);
    if (!c->shadow || !c->rows) { cov_free(c); return NULL; }
    c->last_rows = c->rows + row_bytes;
    c->t_rows = c->rows + row_bytes * 2;
    c->last_t_rows = c->rows + row_bytes * 3;
    c->table_slot = COV_TABLE;
    return c;
}
/* cpu_textures_lock held, after t->probe_id is assigned. */
static void cov_publish(ZludaCpuTexture *t)
{
    unsigned i;
    for (i = 0; i < COV_TABLE; i++)
        if (!__atomic_load_n(&cov_arrays[i], __ATOMIC_RELAXED))
        {
            __atomic_store_n(&cov_arrays[i], t->array, __ATOMIC_RELEASE);
            t->cov->table_slot = i;
            InterlockedExchange(&cov_table_live, 1);
            break;
        }
    if (i == COV_TABLE && !InterlockedExchange(&cov_table_full, 1))
        FIXME("[ZLUDA] CPU-copy array transfer table full (%d); later covered arrays have coverage lines but no array-transfer lines\n", COV_TABLE);
    FIXME("[ZLUDA] CPU-copy coverage resource ordinal=%llu role=%s %ux%u resource=%p array=%p transfer_witness=%d\n",
          t->probe_id, cov_role(t->probe_id), t->width, t->height, t, t->array, i < COV_TABLE);
}
/* cpu_textures_lock held, after the CUDA array is destroyed. */
static void cov_unpublish(ZludaCpuTexture *t)
{
    ZludaCoverage *c = t->cov;
    if (c->table_slot < COV_TABLE) __atomic_store_n(&cov_arrays[c->table_slot], NULL, __ATOMIC_RELEASE);
    FIXME("[ZLUDA] CPU-copy coverage unregister ordinal=%llu role=%s %ux%u resource=%p copies=%llu transfers=%llu\n",
          t->probe_id, cov_role(t->probe_id), t->width, t->height, t, c->copies, c->t_total);
}

/* Map (D3D->CUDA), after the D3D read into the bounce buffer and before the upload. */
static void cov_map_fill(ZludaCpuTexture *t)
{
    ZludaCoverage *c = t->cov;
    uint32_t *texel = t->pixels;
    size_t i, n = (size_t)t->width * t->height * 4;
    memcpy(c->shadow, t->pixels, n * 4);
    for (i = 0; i < n; i++) texel[i] = COV_SENTINEL;
    c->t_dst = c->t_src = c->t_bind = 0;
    memset(c->t_rows, 0, (t->height + 7) / 8);
}

static void cov_hex(char *out, const unsigned char *bits, unsigned rows)
{
    static const char digits[] = "0123456789abcdef";
    unsigned i, nibbles = (rows + 3) / 4;
    for (i = 0; i < nibbles; i++) out[i] = digits[(bits[i / 2] >> (i % 2 ? 0 : 4)) & 15];
    out[nibbles] = 0;
}
static BOOL cov_is_nan(uint32_t bits) { return (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu); }
static float cov_float(uint32_t bits) { float f; memcpy(&f, &bits, 4); return f; }

/* Unmap (CUDA->D3D), after the CUDA copy and probe_observe, before the marker
 * and the D3D write. Restores unwritten channels from the shadow. */
static void cov_copy_back(ZludaCpuTexture *t)
{
    ZludaCoverage *c = t->cov;
    uint32_t *texel = t->pixels;
    const uint32_t *shadow = c->shadow;
    size_t row_bytes = (t->height + 7) / 8, written = 0, nan_other = 0, partial_texels = 0;
    size_t clean = 0, finite = 0, nonzero = 0, a_zero = 0, a_one = 0, a_in_01 = 0, rgb_nonneg = 0, rgb_in_unit = 0;
    unsigned x, y, k, full = 0, partial_rows = 0, empty = 0, listed = 0, w = t->width, h = t->height;
    int first_unwritten = -1, last_unwritten = -1, used = 0;
    char partial[COV_PARTIAL_ROWS * 12 + 4] = "", bitmap[COV_MAX_H / 4 + 1], t_bitmap[COV_MAX_H / 4 + 1];
    unsigned long long now = probe_now();
    BOOL sign = c->role_witness < 2, changed;
    c->copies++;
    memset(c->rows, 0, row_bytes);
    for (y = 0; y < h; y++)
    {
        unsigned row_written = 0;
        for (x = 0; x < w; x++)
        {
            uint32_t *v = texel + ((size_t)y * w + x) * 4;
            const uint32_t *s = shadow + ((size_t)y * w + x) * 4;
            unsigned sentinels = 0, other_nan = 0;
            for (k = 0; k < 4; k++)
            {
                if (v[k] == COV_SENTINEL) { sentinels++; v[k] = s[k]; }
                else if (cov_is_nan(v[k])) other_nan = 1;
            }
            if (sentinels == 4) continue;
            row_written++;
            if (sentinels) partial_texels++;
            if (other_nan) nan_other++;
            if (sign && !sentinels)
            {
                /* Role signature over texels written in all four channels this map. */
                float r = cov_float(v[0]), g = cov_float(v[1]), b = cov_float(v[2]), a = cov_float(v[3]);
                clean++;
                if (!isfinite(r) || !isfinite(g) || !isfinite(b) || !isfinite(a)) continue;
                finite++;
                if (r != 0.0f || g != 0.0f || b != 0.0f || a != 0.0f) nonzero++;
                if (a == 0.0f) a_zero++;
                if (a == 1.0f) a_one++;
                if (a >= 0.0f && a <= 1.0f) a_in_01++;
                if (r >= 0.0f && g >= 0.0f && b >= 0.0f) rgb_nonneg++;
                if (r >= 0.0f && r <= 1.0f && g >= 0.0f && g <= 1.0f && b >= 0.0f && b <= 1.0f) rgb_in_unit++;
            }
        }
        written += row_written;
        if (row_written) c->rows[y / 8] |= 0x80 >> (y % 8);
        if (row_written == w) { full++; continue; }
        if (row_written)
        {
            partial_rows++;
            if (listed++ < COV_PARTIAL_ROWS) used += snprintf(partial + used, sizeof(partial) - used, "%s%u:%u", used ? "," : "", y, row_written);
        }
        else empty++;
        if (first_unwritten < 0) first_unwritten = y;
        last_unwritten = y;
    }
    if (listed > COV_PARTIAL_ROWS) memcpy(partial + used, ",+", 3); /* at most 8 entries of 10 chars precede it */
    if (!used) strcpy(partial, "-");
    changed = !c->logged || written != c->written || nan_other != c->nan_other || partial_texels != c->partial_texels
              || memcmp(c->rows, c->last_rows, row_bytes) || memcmp(c->t_rows, c->last_t_rows, row_bytes);
    cov_hex(bitmap, c->rows, h);
    cov_hex(t_bitmap, c->t_rows, h);
#define COV_FIELDS "ordinal=%llu role=%s %ux%u epoch=%llu written=%zu texels=%zu rows_full=%u rows_partial=%u rows_empty=%u first_unwritten_row=%d last_unwritten_row=%d partial_texels=%zu nan_other=%zu t_dst=%u t_src=%u t_bind=%u partial_rows=%s rows=%s t_rows=%s\n"
#define COV_ARGS t->probe_id, cov_role(t->probe_id), w, h, t->map_epoch, written, (size_t)w * h, full, partial_rows, empty, first_unwritten, last_unwritten, \
                 partial_texels, nan_other, c->t_dst, c->t_src, c->t_bind, partial, bitmap, t_bitmap
    if (changed)
    {
        if (c->lines < COV_LINE_LIMIT) { c->lines++; FIXME("[ZLUDA] CPU-copy coverage " COV_FIELDS, COV_ARGS); }
        else if (cov_log_count(++c->suppressed))
            FIXME("[ZLUDA] CPU-copy coverage lines suppressed ordinal=%llu count=%u (limit %d)\n", t->probe_id, c->suppressed, COV_LINE_LIMIT);
        c->logged = TRUE;
        c->written = written; c->nan_other = nan_other; c->partial_texels = partial_texels;
        memcpy(c->last_rows, c->rows, row_bytes);
        memcpy(c->last_t_rows, c->t_rows, row_bytes);
    }
    if (now && now - c->beat_ns >= COV_BEAT_NS && c->beats < COV_BEAT_LIMIT)
    {
        c->beat_ns = now; c->beats++;
        FIXME("[ZLUDA] CPU-copy coverage heartbeat copies=%llu " COV_FIELDS, c->copies, COV_ARGS);
    }
#undef COV_FIELDS
#undef COV_ARGS
    /* Once at the first copy-back with a written texel, and again at the first
     * one with a non-zero written texel if the first was all zero. */
    if (sign && clean && (!c->role_witness || nonzero))
    {
        FIXME("[ZLUDA] CPU-copy role-signature %s ordinal=%llu role=%s %ux%u epoch=%llu clean=%zu finite=%zu nonzero=%zu a_zero=%zu a_one=%zu a_in_01=%zu rgb_nonneg=%zu rgb_in_unit=%zu\n",
              c->role_witness ? "first-nonzero" : "first-written", t->probe_id, cov_role(t->probe_id), w, h, t->map_epoch,
              clean, finite, nonzero, a_zero, a_one, a_in_01, rgb_nonneg, rgb_in_unit);
        c->role_witness = nonzero ? 2 : 1;
    }
}

/* T. Lock-free pre-filter for the wrappers: one acquire load while nothing is covered. */
static BOOL cov_lookup(CUarray array)
{
    unsigned i;
    if (!array) return FALSE;
    for (i = 0; i < COV_TABLE; i++) if (__atomic_load_n(&cov_arrays[i], __ATOMIC_ACQUIRE) == array) return TRUE;
    return FALSE;
}
#define COV_ACTIVE() __atomic_load_n(&cov_table_live, __ATOMIC_ACQUIRE)

static void cov_mark_rows(ZludaCpuTexture *t, const CovTransfer *x)
{
    size_t row_bytes = (size_t)t->width * 16, first, last, y;
    const CovTransferKey *k = &x->key;
    if (!k->width || !k->height || k->dst_z) return; /* 3D: slice 0 is the only slice of a 2D array */
    if (k->linear)
    {
        /* Linear entry point: byte offset range over packed rows. */
        first = k->dst_x / row_bytes; last = (k->dst_x + k->width - 1) / row_bytes;
    }
    else { first = k->dst_y; last = k->dst_y + k->height - 1; }
    for (y = first; y <= last && y < t->height; y++) t->cov->t_rows[y / 8] |= 0x80 >> (y % 8);
}
/* cpu_textures_lock held. */
static void cov_note(ZludaCpuTexture *t, const CovTransfer *x, unsigned side)
{
    ZludaCoverage *c = t->cov;
    unsigned long long now = probe_now();
    c->t_total++; c->t_since_beat++;
    if (!t->mapped) c->t_unmapped++;
    else if (side == COV_SIDE_DST) { c->t_dst++; if (!x->key.result) cov_mark_rows(t, x); }
    else if (side == COV_SIDE_SRC) c->t_src++;
    else c->t_bind++;
    if (!c->t_seen[x->function][side] || memcmp(&c->t_last[x->function][side], &x->key, sizeof(x->key)))
    {
        c->t_seen[x->function][side] = TRUE;
        c->t_last[x->function][side] = x->key;
        if (c->t_lines < COV_LINE_LIMIT)
        {
            const CovTransferKey *k = &x->key;
            c->t_lines++;
            if (side == COV_SIDE_BIND)
                FIXME("[ZLUDA] CPU-copy array-bind fn=%s ordinal=%llu role=%s %ux%u epoch=%llu mapped=%d result=%d res_type=%d flags=%u count=%llu\n",
                      cov_function_names[x->function], t->probe_id, cov_role(t->probe_id), t->width, t->height, t->map_epoch, t->mapped,
                      k->result, k->res_type, k->flags, c->t_total);
            else
                FIXME("[ZLUDA] CPU-copy array-transfer fn=%s side=%s ordinal=%llu role=%s %ux%u epoch=%llu mapped=%d result=%d src_type=%d dst_type=%d src_x=%zu src_y=%zu src_z=%zu dst_x=%zu dst_y=%zu dst_z=%zu width_bytes=%zu height=%zu depth=%zu stream=%p count=%llu\n",
                      cov_function_names[x->function], cov_side_names[side], t->probe_id, cov_role(t->probe_id), t->width, t->height, t->map_epoch,
                      t->mapped, k->result, k->src_type, k->dst_type, k->src_x, k->src_y, k->src_z, k->dst_x, k->dst_y, k->dst_z,
                      k->width, k->height, k->depth, x->stream, c->t_total);
        }
        else if (cov_log_count(++c->t_suppressed))
            FIXME("[ZLUDA] CPU-copy array-transfer lines suppressed ordinal=%llu count=%u (limit %d)\n", t->probe_id, c->t_suppressed, COV_LINE_LIMIT);
    }
    if (now && now - c->t_beat_ns >= COV_BEAT_NS && c->t_beats < COV_BEAT_LIMIT)
    {
        FIXME("[ZLUDA] CPU-copy array-transfer heartbeat ordinal=%llu role=%s epoch=%llu calls=%llu total=%llu unmapped=%llu map_dst=%u map_src=%u map_bind=%u last=%s\n",
              t->probe_id, cov_role(t->probe_id), t->map_epoch, c->t_since_beat, c->t_total, c->t_unmapped, c->t_dst, c->t_src, c->t_bind,
              cov_function_names[x->function]);
        c->t_beat_ns = now; c->t_beats++; c->t_since_beat = 0;
    }
}
/* After the forwarded call returned. Only called when COV_ACTIVE(). */
static void cov_transfer(const CovTransfer *x)
{
    CUarray arrays[COV_SIDES];
    unsigned side;
    ZludaCpuTexture *t;
    arrays[COV_SIDE_SRC] = x->key.src_type == 3 ? x->src_array : NULL;
    arrays[COV_SIDE_DST] = x->key.dst_type == 3 ? x->dst_array : NULL;
    arrays[COV_SIDE_BIND] = x->bind_array;
    if (!cov_lookup(arrays[0]) && !cov_lookup(arrays[1]) && !cov_lookup(arrays[2])) return;
    pthread_mutex_lock(&cpu_textures_lock);
    for (side = 0; side < COV_SIDES; side++)
        if (arrays[side])
            for (t = cpu_textures; t; t = t->next)
                if (t->cov && t->array == arrays[side]) { cov_note(t, x, side); break; }
    pthread_mutex_unlock(&cpu_textures_lock);
}

/* Entry-point decoders. The relay's v1 and peer descriptors are opaque or
 * share the v2 type, so the legacy layouts are spelled out here. */
typedef struct {
    unsigned int srcXInBytes, srcY;
    CUmemorytype srcMemoryType; const void *srcHost; CUdeviceptr_v1 srcDevice; CUarray srcArray; unsigned int srcPitch;
    unsigned int dstXInBytes, dstY;
    CUmemorytype dstMemoryType; void *dstHost; CUdeviceptr_v1 dstDevice; CUarray dstArray; unsigned int dstPitch;
    unsigned int WidthInBytes, Height;
} CovMemcpy2DV1;
typedef struct {
    unsigned int srcXInBytes, srcY, srcZ, srcLOD;
    CUmemorytype srcMemoryType; const void *srcHost; CUdeviceptr_v1 srcDevice; CUarray srcArray; void *reserved0;
    unsigned int srcPitch, srcHeight;
    unsigned int dstXInBytes, dstY, dstZ, dstLOD;
    CUmemorytype dstMemoryType; void *dstHost; CUdeviceptr_v1 dstDevice; CUarray dstArray; void *reserved1;
    unsigned int dstPitch, dstHeight;
    unsigned int WidthInBytes, Height, Depth;
} CovMemcpy3DV1;
/* CUDA_MEMCPY3D_PEER has the v2 CUDA_MEMCPY3D layout with srcContext/dstContext in place of reserved0/1. */
typedef struct { int resType; CUarray hArray; } CovResourceDescHead; /* CU_RESOURCE_TYPE_ARRAY == 0; union at offset 8 */

static void cov_copy2d(unsigned function, const CUDA_MEMCPY2D *p, CUstream stream, CUresult result)
{
    CovTransfer x = { function };
    if (!p) return;
    x.stream = stream; x.key.result = result;
    x.key.src_type = p->srcMemoryType; x.key.dst_type = p->dstMemoryType;
    x.src_array = p->srcArray; x.dst_array = p->dstArray;
    x.key.src_x = p->srcXInBytes; x.key.src_y = p->srcY; x.key.dst_x = p->dstXInBytes; x.key.dst_y = p->dstY;
    x.key.width = p->WidthInBytes; x.key.height = p->Height;
    cov_transfer(&x);
}
static void cov_copy2d_v1(unsigned function, const void *copy, CUstream stream, CUresult result)
{
    const CovMemcpy2DV1 *p = copy;
    CovTransfer x = { function };
    if (!p) return;
    x.stream = stream; x.key.result = result;
    x.key.src_type = p->srcMemoryType; x.key.dst_type = p->dstMemoryType;
    x.src_array = p->srcArray; x.dst_array = p->dstArray;
    x.key.src_x = p->srcXInBytes; x.key.src_y = p->srcY; x.key.dst_x = p->dstXInBytes; x.key.dst_y = p->dstY;
    x.key.width = p->WidthInBytes; x.key.height = p->Height;
    cov_transfer(&x);
}
static void cov_copy3d(unsigned function, const void *copy, CUstream stream, CUresult result)
{
    const CUDA_MEMCPY3D *p = copy; /* also CUDA_MEMCPY3D_PEER */
    CovTransfer x = { function };
    if (!p) return;
    x.stream = stream; x.key.result = result;
    x.key.src_type = p->srcMemoryType; x.key.dst_type = p->dstMemoryType;
    x.src_array = p->srcArray; x.dst_array = p->dstArray;
    x.key.src_x = p->srcXInBytes; x.key.src_y = p->srcY; x.key.src_z = p->srcZ;
    x.key.dst_x = p->dstXInBytes; x.key.dst_y = p->dstY; x.key.dst_z = p->dstZ;
    x.key.width = p->WidthInBytes; x.key.height = p->Height; x.key.depth = p->Depth;
    cov_transfer(&x);
}
static void cov_copy3d_v1(unsigned function, const void *copy, CUstream stream, CUresult result)
{
    const CovMemcpy3DV1 *p = copy;
    CovTransfer x = { function };
    if (!p) return;
    x.stream = stream; x.key.result = result;
    x.key.src_type = p->srcMemoryType; x.key.dst_type = p->dstMemoryType;
    x.src_array = p->srcArray; x.dst_array = p->dstArray;
    x.key.src_x = p->srcXInBytes; x.key.src_y = p->srcY; x.key.src_z = p->srcZ;
    x.key.dst_x = p->dstXInBytes; x.key.dst_y = p->dstY; x.key.dst_z = p->dstZ;
    x.key.width = p->WidthInBytes; x.key.height = p->Height; x.key.depth = p->Depth;
    cov_transfer(&x);
}
/* Linear entry points; memory types: 1 host, 2 device, 3 array. */
static void cov_linear(unsigned function, int dst_type, CUarray dst, size_t dst_offset, int src_type, CUarray src, size_t src_offset,
                       size_t bytes, CUstream stream, CUresult result)
{
    CovTransfer x = { function };
    x.stream = stream; x.key.result = result;
    x.key.src_type = src_type; x.key.dst_type = dst_type;
    x.src_array = src; x.dst_array = dst;
    x.key.src_x = src_offset; x.key.dst_x = dst_offset;
    x.key.width = bytes; x.key.height = 1; x.key.linear = 1;
    cov_transfer(&x);
}
static void cov_bind(unsigned function, CUarray array, int res_type, unsigned flags, CUresult result)
{
    CovTransfer x = { function };
    x.bind_array = array; x.key.res_type = res_type; x.key.flags = flags; x.key.result = result;
    cov_transfer(&x);
}
static void cov_bind_desc(unsigned function, const void *desc, CUresult result)
{
    const CovResourceDescHead *d = desc;
    if (d && d->resType == 0) cov_bind(function, d->hArray, d->resType, 0, result);
}
