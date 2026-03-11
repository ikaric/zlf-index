# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: profile=False

"""Zipf Latent Fusion Index – Cython-accelerated core (v6: 64-byte records).

64-byte records (one cache line) using unsigned int for targets/link/length.
Pre-extracted C pointer array eliminates Python list access from the hot loop.
Within-automaton prefetching hides memory latency.
"""

from libc.stdlib cimport malloc, free, calloc, realloc
from libc.string cimport memcpy, memset
from cython.parallel cimport prange
from libc.math cimport log

cdef extern from *:
    void __builtin_prefetch(const void *addr, int rw, int locality) nogil
    int __builtin_ctzll(unsigned long long x) nogil

cdef extern from "<stdlib.h>":
    int posix_memalign(void **memptr, size_t alignment, size_t size) nogil

DEF MAX_INLINE = 8
DEF RECORD_SIZE = 64

# Record layout (64 bytes — full cache line, one state per line):
#   [0:4]   unsigned int length
#   [4:8]   unsigned int link
#   [8]     unsigned char  total_count
#   [9]     unsigned char  inline_count
#   [12:16] unsigned int overflow_start
#   [16:24] unsigned char  chars[8]
#   [24:56] unsigned int targets[8]

DEF OFF_LENGTH = 0
DEF OFF_LINK = 4
DEF OFF_COUNT = 8
DEF OFF_ICOUNT = 9
DEF OFF_OVERFLOW = 12
DEF OFF_CHARS = 16
DEF OFF_TARGETS = 24


# ============================================================
# Build phase – full 256-entry table (temporary)
# ============================================================

cdef struct BuildSA:
    int *length
    int *link
    int *trans
    int size
    int capacity


cdef void bsa_init(BuildSA *sa, int cap) noexcept nogil:
    sa.capacity = cap
    sa.size = 1
    sa.length = <int *>malloc(cap * sizeof(int))
    sa.link = <int *>malloc(cap * sizeof(int))
    sa.trans = <int *>calloc(cap * 256, sizeof(int))
    sa.length[0] = 0
    sa.link[0] = -1


cdef void bsa_free(BuildSA *sa) noexcept nogil:
    if sa.length != NULL:
        free(sa.length)
    if sa.link != NULL:
        free(sa.link)
    if sa.trans != NULL:
        free(sa.trans)
    sa.length = NULL
    sa.link = NULL
    sa.trans = NULL


cdef void bsa_build(BuildSA *sa, const unsigned char *data, int data_len) noexcept nogil:
    cdef int last = 0, cur, p, q, clone
    cdef unsigned char c
    cdef int i

    for i in range(data_len):
        c = data[i]
        cur = sa.size
        sa.size += 1
        sa.length[cur] = sa.length[last] + 1
        sa.link[cur] = 0
        p = last

        while p != -1 and sa.trans[p * 256 + c] == 0:
            sa.trans[p * 256 + c] = cur
            p = sa.link[p]

        if p == -1:
            sa.link[cur] = 0
        else:
            q = sa.trans[p * 256 + c]
            if sa.length[p] + 1 == sa.length[q]:
                sa.link[cur] = q
            else:
                clone = sa.size
                sa.size += 1
                sa.length[clone] = sa.length[p] + 1
                sa.link[clone] = sa.link[q]
                memcpy(&sa.trans[clone * 256], &sa.trans[q * 256], 256 * sizeof(int))
                while p != -1 and sa.trans[p * 256 + c] == q:
                    sa.trans[p * 256 + c] = clone
                    p = sa.link[p]
                sa.link[q] = clone
                sa.link[cur] = clone
        last = cur


# ============================================================
# Compact phase – 64-byte records with unsigned int
# ============================================================

cdef struct CompactSA:
    int num_states
    char *records               # num_states * 64 bytes
    unsigned char *ovf_chars    # overflow chars
    unsigned int *ovf_targets   # overflow targets (unsigned int)
    int ovf_size


cdef void csa_from_build(CompactSA *csa, BuildSA *bsa) noexcept nogil:
    cdef int ns = bsa.size
    cdef int i, c, count, ic, target
    cdef int ovf_total, ovf_idx
    cdef char *rec

    csa.num_states = ns
    cdef void *mem = NULL
    if posix_memalign(&mem, 64, ns * RECORD_SIZE) != 0:
        csa.records = <char *>calloc(ns, RECORD_SIZE)
    else:
        csa.records = <char *>mem
        memset(csa.records, 0, ns * RECORD_SIZE)

    # Pass 1: count overflow
    ovf_total = 0
    for i in range(ns):
        count = 0
        for c in range(256):
            if bsa.trans[i * 256 + c] != 0:
                count += 1
        if count > MAX_INLINE:
            ovf_total += count - MAX_INLINE

    if ovf_total > 0:
        csa.ovf_chars = <unsigned char *>malloc(ovf_total)
        csa.ovf_targets = <unsigned int *>malloc(ovf_total * sizeof(unsigned int))
    else:
        csa.ovf_chars = NULL
        csa.ovf_targets = NULL
    csa.ovf_size = ovf_total

    # Pass 2: fill records
    ovf_idx = 0
    for i in range(ns):
        rec = csa.records + i * RECORD_SIZE

        (<unsigned int *>(rec + OFF_LENGTH))[0] = <unsigned int>bsa.length[i]
        # link can be -1 for root; stored wrapped around as 0xFFFFFFFF (never read in query)
        (<unsigned int *>(rec + OFF_LINK))[0] = <unsigned int>(bsa.link[i])

        count = 0
        ic = 0

        for c in range(256):
            target = bsa.trans[i * 256 + c]
            if target != 0:
                if ic < MAX_INLINE:
                    (rec + OFF_CHARS)[ic] = <char>c
                    (<unsigned int *>(rec + OFF_TARGETS))[ic] = <unsigned int>target
                    ic += 1
                else:
                    csa.ovf_chars[ovf_idx] = <unsigned char>c
                    csa.ovf_targets[ovf_idx] = <unsigned int>target
                    ovf_idx += 1
                count += 1

        (rec + OFF_COUNT)[0] = <char>count
        (rec + OFF_ICOUNT)[0] = <char>ic
        if count > MAX_INLINE:
            (<unsigned int *>(rec + OFF_OVERFLOW))[0] = <unsigned int>(ovf_idx - (count - MAX_INLINE))
        else:
            (<unsigned int *>(rec + OFF_OVERFLOW))[0] = 0


cdef void csa_free(CompactSA *csa) noexcept nogil:
    if csa.records != NULL:
        free(csa.records)
    if csa.ovf_chars != NULL:
        free(csa.ovf_chars)
    if csa.ovf_targets != NULL:
        free(csa.ovf_targets)
    csa.records = NULL
    csa.ovf_chars = NULL
    csa.ovf_targets = NULL


cdef inline int csa_get_trans(CompactSA *csa, char *rec, unsigned char c) noexcept nogil:
    cdef int ic = <int>(<unsigned char>(rec[OFF_ICOUNT]))
    cdef int total = <int>(<unsigned char>(rec[OFF_COUNT]))
    cdef int j, remaining
    cdef unsigned int ovf_start
    cdef unsigned char *chars = <unsigned char *>(rec + OFF_CHARS)
    cdef unsigned int *targets = <unsigned int *>(rec + OFF_TARGETS)
    cdef unsigned long long V, mask, diff, has_zero

    # SWAR bit-banging search on 8-byte chars array
    V = (<unsigned long long *>chars)[0]
    mask = c
    mask |= mask << 8
    mask |= mask << 16
    mask |= mask << 32
    diff = V ^ mask
    has_zero = (diff - 0x0101010101010101ULL) & ~diff & 0x8080808080808080ULL

    if has_zero:
        j = __builtin_ctzll(has_zero) >> 3
        if j < ic:
            return <int>targets[j]

    if total > ic:
        ovf_start = (<unsigned int *>(rec + OFF_OVERFLOW))[0]
        remaining = total - ic
        for j in range(remaining):
            if csa.ovf_chars[ovf_start + j] == c:
                return <int>csa.ovf_targets[ovf_start + j]

    return 0


# ============================================================
# Query phase – with prefetching
# ============================================================

cdef double csa_query(CompactSA *csa, const unsigned char *data, int data_len) noexcept nogil:
    cdef long long delta_score = 0
    cdef int v = 0, length = 0
    cdef unsigned char c
    cdef int i, found
    cdef char *rec

    if data_len == 0:
        return 0.0

    for i in range(data_len):
        c = data[i]

        while True:
            rec = csa.records + v * RECORD_SIZE
            found = csa_get_trans(csa, rec, c)
            if found != 0 or v == 0:
                break
            v = <int>(<unsigned int *>(rec + OFF_LINK))[0]
            rec = csa.records + v * RECORD_SIZE
            length = <int>(<unsigned int *>(rec + OFF_LENGTH))[0]

        if found != 0:
            v = found
            length += 1
            # Prefetch the next state's record
            __builtin_prefetch(csa.records + v * RECORD_SIZE, 0, 1)
        else:
            v = 0
            length = 0

        delta_score += (<long long>length * (<long long>length + 1)) >> 1

    cdef long long n_len = data_len
    cdef long long t_delta = (n_len * (n_len + 1) * (n_len + 2)) // 6
    if t_delta == 0:
        return 0.0
    return <double>delta_score / <double>t_delta


# ============================================================
# Python wrapper
# ============================================================

cdef class SuffixAutomatonWrapper:
    cdef CompactSA _csa
    cdef bint _built

    def __cinit__(self):
        self._built = False

    def __dealloc__(self):
        if self._built:
            csa_free(&self._csa)

    def dumps(self) -> bytes:
        """Serialize the Suffix Automaton to raw bytes."""
        if not self._built:
            raise ValueError("SuffixAutomaton is not built.")
            
        # We need to serialize:
        # 1. num_states (4 bytes)
        # 2. ovf_size (4 bytes)
        # 3. records (num_states * RECORD_SIZE bytes)
        # 4. ovf_chars (ovf_size bytes)
        # 5. ovf_targets (ovf_size * 4 bytes)
        
        cdef int total_size = 8 + (self._csa.num_states * RECORD_SIZE) + self._csa.ovf_size + (self._csa.ovf_size * sizeof(unsigned int))
        cdef bytes result = b'\0' * total_size
        cdef char *buf = result
        cdef int offset = 0
        
        # Write num_states
        (<int *>(buf + offset))[0] = self._csa.num_states
        offset += sizeof(int)
        
        # Write ovf_size
        (<int *>(buf + offset))[0] = self._csa.ovf_size
        offset += sizeof(int)
        
        # Write records
        if self._csa.num_states > 0:
            memcpy(buf + offset, self._csa.records, self._csa.num_states * RECORD_SIZE)
            offset += self._csa.num_states * RECORD_SIZE
            
        # Write ovf_chars
        if self._csa.ovf_size > 0:
            memcpy(buf + offset, self._csa.ovf_chars, self._csa.ovf_size)
            offset += self._csa.ovf_size
            
        # Write ovf_targets
        if self._csa.ovf_size > 0:
            memcpy(buf + offset, self._csa.ovf_targets, self._csa.ovf_size * sizeof(unsigned int))
            
        return result
        
    @classmethod
    def loads(cls, bytes data):
        """Deserialize a Suffix Automaton from raw bytes."""
        if len(data) < 8:
            raise ValueError("Invalid serialized data: too short.")
            
        cdef int expected_num_states = (<int *>(<const char *>data))[0]
        cdef int expected_ovf_size = (<int *>(<const char *>data + sizeof(int)))[0]
        cdef int expected_size = 8 + (expected_num_states * RECORD_SIZE) + expected_ovf_size + (expected_ovf_size * sizeof(unsigned int))
        
        if len(data) < expected_size:
            raise ValueError(f"Invalid serialized data: expected {expected_size} bytes, got {len(data)}")
            
        cdef SuffixAutomatonWrapper wrapper = cls()
        cdef const char *buf = data
        cdef int offset = 0
        
        # Read num_states
        wrapper._csa.num_states = (<int *>(buf + offset))[0]
        offset += sizeof(int)
        
        # Read ovf_size
        wrapper._csa.ovf_size = (<int *>(buf + offset))[0]
        offset += sizeof(int)
        
        # Allocate and read records
        if wrapper._csa.num_states > 0:
            wrapper._csa.records = <char *>malloc(wrapper._csa.num_states * RECORD_SIZE)
            memcpy(wrapper._csa.records, buf + offset, wrapper._csa.num_states * RECORD_SIZE)
            offset += wrapper._csa.num_states * RECORD_SIZE
        else:
            wrapper._csa.records = NULL
            
        # Allocate and read ovf_chars
        if wrapper._csa.ovf_size > 0:
            wrapper._csa.ovf_chars = <unsigned char *>malloc(wrapper._csa.ovf_size)
            memcpy(wrapper._csa.ovf_chars, buf + offset, wrapper._csa.ovf_size)
            offset += wrapper._csa.ovf_size
        else:
            wrapper._csa.ovf_chars = NULL
            
        # Allocate and read ovf_targets
        if wrapper._csa.ovf_size > 0:
            wrapper._csa.ovf_targets = <unsigned int *>malloc(wrapper._csa.ovf_size * sizeof(unsigned int))
            memcpy(wrapper._csa.ovf_targets, buf + offset, wrapper._csa.ovf_size * sizeof(unsigned int))
        else:
            wrapper._csa.ovf_targets = NULL
            
        wrapper._built = True
        return wrapper

    cdef void _build(self, bytes data):
        cdef const unsigned char *buf = <const unsigned char *>data
        cdef int length = len(data)
        cdef int cap = 2 * length + 2
        cdef BuildSA bsa
        bsa_init(&bsa, cap)
        bsa_build(&bsa, buf, length)
        csa_from_build(&self._csa, &bsa)
        bsa_free(&bsa)
        self._built = True

    cdef double _query(self, const unsigned char *buf, int length):
        return csa_query(&self._csa, buf, length)


# ============================================================
# BM25 Inverted Index — ATIRE variant (matches rank_bm25.BM25Okapi)
# ============================================================

cdef struct PostingList:
    int *doc_ids
    int *tfs
    int size
    int capacity

cdef void posting_list_init(PostingList *pl) noexcept nogil:
    pl.size = 0
    pl.capacity = 2
    pl.doc_ids = <int *>malloc(pl.capacity * sizeof(int))
    pl.tfs = <int *>malloc(pl.capacity * sizeof(int))

cdef void posting_list_append(PostingList *pl, int doc_id, int tf) noexcept nogil:
    if pl.size >= pl.capacity:
        pl.capacity *= 2
        pl.doc_ids = <int *>realloc(pl.doc_ids, pl.capacity * sizeof(int))
        pl.tfs = <int *>realloc(pl.tfs, pl.capacity * sizeof(int))
    pl.doc_ids[pl.size] = doc_id
    pl.tfs[pl.size] = tf
    pl.size += 1

cdef void posting_list_free(PostingList *pl) noexcept nogil:
    if pl.doc_ids != NULL:
        free(pl.doc_ids)
    if pl.tfs != NULL:
        free(pl.tfs)
    pl.doc_ids = NULL
    pl.tfs = NULL
    pl.size = 0
    pl.capacity = 0


cdef class BM25Index:
    cdef double k1
    cdef double b
    cdef double epsilon
    cdef int num_docs
    cdef double avg_doc_len

    cdef int *doc_lengths
    cdef int doc_lengths_cap

    cdef int *doc_freqs
    cdef int doc_freqs_cap
    cdef int max_term_id

    cdef PostingList *postings
    cdef int postings_cap

    cdef double *idf_cache
    cdef int idf_cache_cap
    cdef bint idf_dirty

    def __cinit__(self, double k1=1.5, double b=0.75, double epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.num_docs = 0
        self.avg_doc_len = 0.0

        self.doc_lengths_cap = 16
        self.doc_lengths = <int *>malloc(self.doc_lengths_cap * sizeof(int))

        self.doc_freqs_cap = 1024
        self.doc_freqs = <int *>calloc(self.doc_freqs_cap, sizeof(int))
        self.max_term_id = -1

        self.postings_cap = 1024
        self.postings = <PostingList *>calloc(self.postings_cap, sizeof(PostingList))

        self.idf_cache = NULL
        self.idf_cache_cap = 0
        self.idf_dirty = True

    def __dealloc__(self):
        cdef int i
        if self.doc_lengths != NULL:
            free(self.doc_lengths)
        if self.doc_freqs != NULL:
            free(self.doc_freqs)
        if self.postings != NULL:
            for i in range(self.postings_cap):
                if self.postings[i].capacity > 0:
                    posting_list_free(&self.postings[i])
            free(self.postings)
        if self.idf_cache != NULL:
            free(self.idf_cache)

    cdef void _ensure_term_capacity(self, int max_id) noexcept nogil:
        cdef int old_cap = self.postings_cap
        if max_id >= self.postings_cap:
            while self.postings_cap <= max_id:
                self.postings_cap *= 2
            self.postings = <PostingList *>realloc(self.postings, self.postings_cap * sizeof(PostingList))
            self.doc_freqs = <int *>realloc(self.doc_freqs, self.postings_cap * sizeof(int))
            memset(self.postings + old_cap, 0, (self.postings_cap - old_cap) * sizeof(PostingList))
            memset(self.doc_freqs + old_cap, 0, (self.postings_cap - old_cap) * sizeof(int))

    cpdef void add_doc(self, int[:] token_ids):
        cdef int doc_id = self.num_docs
        self.num_docs += 1

        cdef int doc_len = token_ids.shape[0]
        if self.num_docs > self.doc_lengths_cap:
            self.doc_lengths_cap *= 2
            self.doc_lengths = <int *>realloc(self.doc_lengths, self.doc_lengths_cap * sizeof(int))
        self.doc_lengths[doc_id] = doc_len

        self.avg_doc_len = self.avg_doc_len + (doc_len - self.avg_doc_len) / self.num_docs

        if doc_len == 0:
            return

        cdef int max_id = -1
        cdef int i, term_id
        for i in range(doc_len):
            term_id = token_ids[i]
            if term_id > max_id:
                max_id = term_id

        if max_id > self.max_term_id:
            self._ensure_term_capacity(max_id)
            self.max_term_id = max_id

        # To extract term frequencies within this document efficiently:
        # We can allocate a temporary array bounded by max_id, 
        # or use a python dict if max_id is very large.
        # Python dict is very fast for small docs vs calloc large array.
        cdef dict freqs = {}
        for i in range(doc_len):
            term_id = token_ids[i]
            if term_id in freqs:
                freqs[term_id] += 1
            else:
                freqs[term_id] = 1

        for term_id_obj, tf_obj in freqs.items():
            term_id = <int>term_id_obj
            tf = <int>tf_obj

            if self.postings[term_id].capacity == 0:
                posting_list_init(&self.postings[term_id])

            posting_list_append(&self.postings[term_id], doc_id, tf)
            self.doc_freqs[term_id] += 1

        self.idf_dirty = True

    cdef void _rebuild_idf(self):
        """Recompute IDF cache using the ATIRE BM25 variant (matches rank_bm25)."""
        cdef int num_terms = self.max_term_id + 1
        cdef int i
        cdef double idf_sum = 0.0, val, avg_idf, eps
        cdef int df
        cdef int n = self.num_docs
        cdef int idf_count = 0

        if num_terms > self.idf_cache_cap:
            if self.idf_cache != NULL:
                free(self.idf_cache)
            self.idf_cache = <double *>malloc(num_terms * sizeof(double))
            self.idf_cache_cap = num_terms

        memset(self.idf_cache, 0, num_terms * sizeof(double))

        for i in range(num_terms):
            df = self.doc_freqs[i]
            if df == 0:
                continue
            val = log(<double>(n - df) + 0.5) - log(<double>df + 0.5)
            self.idf_cache[i] = val
            idf_sum += val
            idf_count += 1

        avg_idf = idf_sum / idf_count if idf_count > 0 else 0.0
        eps = self.epsilon * avg_idf

        for i in range(num_terms):
            if self.doc_freqs[i] > 0 and self.idf_cache[i] < 0:
                self.idf_cache[i] = eps

        self.idf_dirty = False

    cpdef list score_all(self, int[:] query_token_ids):
        cdef int q_len = query_token_ids.shape[0]
        cdef int doc_count = self.num_docs
        cdef list result = [0.0] * doc_count

        if doc_count == 0 or q_len == 0:
            return result

        if self.idf_dirty:
            self._rebuild_idf()

        cdef double *scores = <double *>calloc(doc_count, sizeof(double))
        
        cdef dict q_freqs = {}
        cdef int i, term_id
        for i in range(q_len):
            term_id = query_token_ids[i]
            if term_id in q_freqs:
                q_freqs[term_id] += 1
            else:
                q_freqs[term_id] = 1

        cdef double idf, norm, term_score
        cdef int doc_id, tf, j
        cdef PostingList *pl
        cdef int q_tf

        for term_id_obj, q_tf_obj in q_freqs.items():
            term_id = <int>term_id_obj
            q_tf = <int>q_tf_obj

            if term_id > self.max_term_id or self.doc_freqs[term_id] == 0:
                continue

            idf = self.idf_cache[term_id]

            pl = &self.postings[term_id]
            for j in range(pl.size):
                doc_id = pl.doc_ids[j]
                tf = pl.tfs[j]
                norm = 1.0 - self.b + self.b * (<double>self.doc_lengths[doc_id] / self.avg_doc_len)
                term_score = idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
                scores[doc_id] += term_score * q_tf

        for doc_id in range(doc_count):
            result[doc_id] = scores[doc_id]

        free(scores)
        return result

    def dumps(self) -> bytes:
        import struct
        
        cdef bytes header = struct.pack(
            "dddiid",
            self.k1, self.b, self.epsilon,
            self.num_docs, self.max_term_id, self.avg_doc_len,
        )
        
        cdef bytearray buf = bytearray(header)
        cdef int i, j
        
        # Serialize doc_lengths
        if self.num_docs > 0:
            bytes_len = self.num_docs * sizeof(int)
            buf.extend((<char*>self.doc_lengths)[:bytes_len])
            
        # Serialize doc_freqs and postings up to max_term_id
        if self.max_term_id >= 0:
            count = self.max_term_id + 1
            bytes_count = count * sizeof(int)
            buf.extend((<char*>self.doc_freqs)[:bytes_count])
            
            for i in range(count):
                if self.postings[i].size > 0:
                    buf.extend(struct.pack("i", self.postings[i].size))
                    
                    bytes_size = self.postings[i].size * sizeof(int)
                    buf.extend((<char*>self.postings[i].doc_ids)[:bytes_size])
                    buf.extend((<char*>self.postings[i].tfs)[:bytes_size])
                else:
                    buf.extend(struct.pack("i", 0))
                    
        return bytes(buf)

    @classmethod
    def loads(cls, data: bytes):
        import struct
        cdef BM25Index obj = cls()
        
        cdef int header_size = 40  # 8+8+8+4+4+8
        if len(data) < header_size:
            raise ValueError("Invalid serialized BM25Index data: too short for header")
            
        (obj.k1, obj.b, obj.epsilon, obj.num_docs, obj.max_term_id, obj.avg_doc_len) = struct.unpack("dddiid", data[:header_size])
        
        cdef int offset = header_size
        cdef int count
        cdef int i, j, size
        
        if obj.num_docs > 0:
            if obj.num_docs > obj.doc_lengths_cap:
                obj.doc_lengths_cap = obj.num_docs * 2
                obj.doc_lengths = <int *>realloc(obj.doc_lengths, obj.doc_lengths_cap * sizeof(int))
                
            dl_format = f"{obj.num_docs}i"
            dl_size = struct.calcsize(dl_format)
            dl_data = struct.unpack(dl_format, data[offset:offset+dl_size])
            offset += dl_size
            
            for i in range(obj.num_docs):
                obj.doc_lengths[i] = dl_data[i]
                
        if obj.max_term_id >= 0:
            obj._ensure_term_capacity(obj.max_term_id)
            count = obj.max_term_id + 1
            
            df_format = f"{count}i"
            df_size = struct.calcsize(df_format)
            df_data = struct.unpack(df_format, data[offset:offset+df_size])
            offset += df_size
            
            for i in range(count):
                obj.doc_freqs[i] = df_data[i]
                
            for i in range(count):
                size = struct.unpack("i", data[offset:offset+4])[0]
                offset += 4
                
                if size > 0:
                    posting_list_init(&obj.postings[i])
                    while obj.postings[i].capacity <= size:
                        obj.postings[i].capacity *= 2
                        
                    obj.postings[i].doc_ids = <int *>realloc(obj.postings[i].doc_ids, obj.postings[i].capacity * sizeof(int))
                    obj.postings[i].tfs = <int *>realloc(obj.postings[i].tfs, obj.postings[i].capacity * sizeof(int))
                    obj.postings[i].size = size
                    
                    ids_format = f"{size}i"
                    ids_size = struct.calcsize(ids_format)
                    ids_data = struct.unpack(ids_format, data[offset:offset+ids_size])
                    offset += ids_size
                    for j in range(size):
                        obj.postings[i].doc_ids[j] = ids_data[j]
                        
                    tfs_format = f"{size}i"
                    tfs_size = struct.calcsize(tfs_format)
                    tfs_data = struct.unpack(tfs_format, data[offset:offset+tfs_size])
                    offset += tfs_size
                    for j in range(size):
                        obj.postings[i].tfs[j] = tfs_data[j]

        obj.idf_dirty = True
        return obj


# ============================================================
# Public API
# ============================================================

def build_suffix_automaton(str m not None):
    """Build a Suffix Automaton for string m."""
    cdef bytes m_bytes = m.encode('utf-8')
    cdef SuffixAutomatonWrapper wrapper = SuffixAutomatonWrapper()
    wrapper._build(m_bytes)
    return wrapper


def zlfi(str n not None, m) -> float:
    """Compute the ZLF Index between query n and reference m."""
    if len(n) == 0:
        return 0.0
    cdef bytes n_bytes = n.encode('utf-8')
    cdef const unsigned char *buf = <const unsigned char *>n_bytes
    cdef int length = len(n_bytes)
    cdef SuffixAutomatonWrapper wrapper
    if isinstance(m, SuffixAutomatonWrapper):
        wrapper = <SuffixAutomatonWrapper>m
        return wrapper._query(buf, length)
    elif isinstance(m, str):
        wrapper = SuffixAutomatonWrapper()
        wrapper._build((<str>m).encode('utf-8'))
        return wrapper._query(buf, length)
    else:
        raise TypeError(f"m must be a str or SuffixAutomatonWrapper, got {type(m).__name__}")


def batch_scores(str query not None, list automatons not None) -> list:
    """Score a query against all automatons in one call using multithreading."""
    cdef int n = len(automatons)
    cdef bytes q_bytes = query.encode('utf-8')
    cdef const unsigned char *q_buf = <const unsigned char *>q_bytes
    cdef int q_len = len(q_bytes)
    cdef SuffixAutomatonWrapper wrapper
    cdef int i

    if q_len == 0 or n == 0:
        return [0.0] * n

    cdef double *scores = <double *>malloc(n * sizeof(double))
    cdef CompactSA **csa_ptrs = <CompactSA **>malloc(n * sizeof(CompactSA *))
    for i in range(n):
        wrapper = <SuffixAutomatonWrapper>automatons[i]
        csa_ptrs[i] = &wrapper._csa

    with nogil:
        for i in prange(n, schedule='static'):
            scores[i] = csa_query(csa_ptrs[i], q_buf, q_len)

    cdef list result = [0.0] * n
    for i in range(n):
        result[i] = scores[i]

    free(csa_ptrs)
    free(scores)
    return result


cdef struct TopKItem:
    double score
    int index

cdef inline void min_heapify(TopKItem *heap, int i, int n) noexcept nogil:
    cdef int smallest, left, right
    cdef TopKItem temp
    while True:
        smallest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and heap[left].score < heap[smallest].score:
            smallest = left
        if right < n and heap[right].score < heap[smallest].score:
            smallest = right

        if smallest != i:
            temp = heap[i]
            heap[i] = heap[smallest]
            heap[smallest] = temp
            i = smallest
        else:
            break


def batch_top_k(str query not None, list automatons not None, int k=10) -> list:
    """Score query against all automatons and return top-k indices (descending).

    Computes all scores in parallel using OpenMP, then extracts top-k using an O(n log k) C min-heap.
    """
    cdef int n = len(automatons)
    cdef bytes q_bytes = query.encode('utf-8')
    cdef const unsigned char *q_buf = <const unsigned char *>q_bytes
    cdef int q_len = len(q_bytes)
    cdef SuffixAutomatonWrapper wrapper
    cdef int i

    if k > n:
        k = n
    if q_len == 0 or n == 0 or k <= 0:
        return []

    cdef double *all_scores = <double *>malloc(n * sizeof(double))
    cdef CompactSA **csa_ptrs = <CompactSA **>malloc(n * sizeof(CompactSA *))
    for i in range(n):
        wrapper = <SuffixAutomatonWrapper>automatons[i]
        csa_ptrs[i] = &wrapper._csa

    # Parallel scoring phase
    with nogil:
        for i in prange(n, schedule='static'):
            all_scores[i] = csa_query(csa_ptrs[i], q_buf, q_len)

    # Extract sorted result using a min-heap
    cdef TopKItem *heap = <TopKItem *>malloc(k * sizeof(TopKItem))
    
    # Initialize heap with first k elements
    for i in range(k):
        heap[i].score = all_scores[i]
        heap[i].index = i
        
    # Build heap O(k)
    for i in range((k - 2) // 2, -1, -1):
        min_heapify(heap, i, k)
        
    # Process remaining elements O((n-k) log k)
    for i in range(k, n):
        if all_scores[i] > heap[0].score:
            heap[0].score = all_scores[i]
            heap[0].index = i
            min_heapify(heap, 0, k)
            
    # Sort the top k descending O(k log k)
    cdef TopKItem temp
    for i in range(k - 1, 0, -1):
        temp = heap[0]
        heap[0] = heap[i]
        heap[i] = temp
        min_heapify(heap, 0, i)

    cdef list result = [0] * k
    for i in range(k):
        result[i] = heap[i].index

    free(csa_ptrs)
    free(all_scores)
    free(heap)
    return result
