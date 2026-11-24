"""ctypes binding to the pinned llama.cpp build b12549 (commit 8af68d486), Windows x86_64.

Struct layouts mirror include/llama.h and ggml/include/ggml.h at that exact tag. The layout
is validated at import-of-use time by `Llama.check_layout()`, which reads documented default
values back out of `llama_*_default_params()`; a mismatch raises instead of running with a
silently shifted struct.

Purpose: read intermediate hidden states (residual stream, FFN keys, gate activations) out of
the same runtime and the same content-addressed GGUF that serves the adapters, through the
scheduler eval callback (the mechanism llama.cpp's own imatrix / cvector tools use).
"""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

import numpy as np

c_bool, c_int32, c_uint32, c_float, c_void_p, c_char_p, c_size_t, c_int64, c_int8 = (
    C.c_bool, C.c_int32, C.c_uint32, C.c_float, C.c_void_p, C.c_char_p, C.c_size_t, C.c_int64, C.c_int8)

EVAL_CB = C.CFUNCTYPE(c_bool, c_void_p, c_bool, c_void_p)
LOG_CB = C.CFUNCTYPE(None, C.c_int, c_char_p, c_void_p)

GGML_MAX_DIMS = 4
GGML_TYPE_F32 = 0


class ModelParams(C.Structure):
    _fields_ = [
        ("devices", c_void_p),
        ("tensor_buft_overrides", c_void_p),
        ("n_gpu_layers", c_int32),
        ("split_mode", C.c_int),
        ("load_mode", C.c_int),
        ("lazy_mode", C.c_int),
        ("main_gpu", c_int32),
        ("tensor_split", c_void_p),
        ("progress_callback", c_void_p),
        ("progress_callback_user_data", c_void_p),
        ("kv_overrides", c_void_p),
        ("vocab_only", c_bool),
        ("check_tensors", c_bool),
        ("use_extra_bufts", c_bool),
        ("no_host", c_bool),
        ("no_alloc", c_bool),
        ("load_mtp", c_bool),
    ]


class ContextParams(C.Structure):
    _fields_ = [
        ("n_ctx", c_uint32),
        ("n_batch", c_uint32),
        ("n_ubatch", c_uint32),
        ("n_seq_max", c_uint32),
        ("n_rs_seq", c_uint32),
        ("n_outputs_max", c_uint32),
        ("n_outputs_max_per_seq", c_uint32),
        ("n_threads", c_int32),
        ("n_threads_batch", c_int32),
        ("ctx_type", C.c_int),
        ("rope_scaling_type", C.c_int),
        ("pooling_type", C.c_int),
        ("attention_type", C.c_int),
        ("flash_attn_type", C.c_int),
        ("rope_freq_base", c_float),
        ("rope_freq_scale", c_float),
        ("yarn_ext_factor", c_float),
        ("yarn_attn_factor", c_float),
        ("yarn_beta_fast", c_float),
        ("yarn_beta_slow", c_float),
        ("yarn_orig_ctx", c_uint32),
        ("cb_eval", EVAL_CB),
        ("cb_eval_user_data", c_void_p),
        ("type_k", C.c_int),
        ("type_v", C.c_int),
        ("moe_cache_size", c_size_t),
        ("moe_cache_cpu_share", c_float),
        ("abort_callback", c_void_p),
        ("abort_callback_data", c_void_p),
        ("embeddings", c_bool),
        ("offload_kqv", c_bool),
        ("no_perf", c_bool),
        ("op_offload", c_bool),
        ("swa_full", c_bool),
        ("kv_unified", c_bool),
        ("samplers", c_void_p),
        ("n_samplers", c_size_t),
        ("ctx_other", c_void_p),
    ]


class Batch(C.Structure):
    _fields_ = [
        ("n_tokens", c_int32),
        ("token", C.POINTER(c_int32)),
        ("embd", C.POINTER(c_float)),
        ("pos", C.POINTER(c_int32)),
        ("n_seq_id", C.POINTER(c_int32)),
        ("seq_id", C.POINTER(C.POINTER(c_int32))),
        ("logits", C.POINTER(c_int8)),
    ]


class TensorHead(C.Structure):
    """Leading fields of struct ggml_tensor (type, buffer, ne, nb). Only these are read."""
    _fields_ = [
        ("type", C.c_int),
        ("buffer", c_void_p),
        ("ne", c_int64 * GGML_MAX_DIMS),
        ("nb", c_size_t * GGML_MAX_DIMS),
    ]


class Llama:
    def __init__(self, bin_dir: str | os.PathLike, quiet: bool = True):
        self.bin_dir = Path(bin_dir)
        os.add_dll_directory(str(self.bin_dir))
        # ggml.dll loads backends with a plain native LoadLibrary, which ignores add_dll_directory.
        # Without the next two steps ggml-cuda.dll cannot resolve cuBLAS and llama.cpp silently
        # falls back to CPU (observed 2026-11-19: 0 MiB VRAM in use during a "GPU" harvest).
        os.environ["PATH"] = str(self.bin_dir) + os.pathsep + os.environ.get("PATH", "")
        self._preloaded = []
        for dep in ("cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll"):
            if (self.bin_dir / dep).exists():
                self._preloaded.append(C.CDLL(str(self.bin_dir / dep)))
        self.ggml_base = C.CDLL(str(self.bin_dir / "ggml-base.dll"))
        self.ggml = C.CDLL(str(self.bin_dir / "ggml.dll"))
        self.lib = C.CDLL(str(self.bin_dir / "llama.dll"))
        L, G, GB = self.lib, self.ggml, self.ggml_base

        self._log_lines: list[str] = []
        self._quiet = quiet

        def _log(level, text, _ud):
            try:
                s = text.decode("utf-8", "replace") if text else ""
                self._log_lines.append(s)
                if not self._quiet or level >= 4:
                    print(s, end="")
            except Exception:
                pass

        self._log_cb = LOG_CB(_log)
        L.llama_log_set.argtypes = [LOG_CB, c_void_p]
        L.llama_log_set.restype = None
        L.llama_log_set(self._log_cb, None)

        G.ggml_backend_load_all_from_path.argtypes = [c_char_p]
        G.ggml_backend_load_all_from_path.restype = None
        L.llama_backend_init.argtypes = []
        L.llama_backend_init.restype = None
        L.llama_model_default_params.argtypes = []
        L.llama_model_default_params.restype = ModelParams
        L.llama_context_default_params.argtypes = []
        L.llama_context_default_params.restype = ContextParams
        L.llama_model_load_from_file.argtypes = [c_char_p, ModelParams]
        L.llama_model_load_from_file.restype = c_void_p
        L.llama_init_from_model.argtypes = [c_void_p, ContextParams]
        L.llama_init_from_model.restype = c_void_p
        L.llama_model_free.argtypes = [c_void_p]
        L.llama_model_free.restype = None
        L.llama_free.argtypes = [c_void_p]
        L.llama_free.restype = None
        L.llama_model_get_vocab.argtypes = [c_void_p]
        L.llama_model_get_vocab.restype = c_void_p
        L.llama_model_n_embd.argtypes = [c_void_p]
        L.llama_model_n_embd.restype = c_int32
        L.llama_model_n_layer.argtypes = [c_void_p]
        L.llama_model_n_layer.restype = c_int32
        L.llama_vocab_n_tokens.argtypes = [c_void_p]
        L.llama_vocab_n_tokens.restype = c_int32
        L.llama_n_ctx.argtypes = [c_void_p]
        L.llama_n_ctx.restype = c_uint32
        L.llama_tokenize.argtypes = [c_void_p, c_char_p, c_int32, C.POINTER(c_int32), c_int32, c_bool, c_bool]
        L.llama_tokenize.restype = c_int32
        L.llama_token_to_piece.argtypes = [c_void_p, c_int32, c_char_p, c_int32, c_int32, c_bool]
        L.llama_token_to_piece.restype = c_int32
        L.llama_batch_init.argtypes = [c_int32, c_int32, c_int32]
        L.llama_batch_init.restype = Batch
        L.llama_batch_free.argtypes = [Batch]
        L.llama_batch_free.restype = None
        L.llama_decode.argtypes = [c_void_p, Batch]
        L.llama_decode.restype = c_int32
        L.llama_get_logits_ith.argtypes = [c_void_p, c_int32]
        L.llama_get_logits_ith.restype = C.POINTER(c_float)
        L.llama_get_memory.argtypes = [c_void_p]
        L.llama_get_memory.restype = c_void_p
        L.llama_memory_clear.argtypes = [c_void_p, c_bool]
        L.llama_memory_clear.restype = None
        L.llama_adapter_lora_init.argtypes = [c_void_p, c_char_p]
        L.llama_adapter_lora_init.restype = c_void_p
        L.llama_set_adapters_lora.argtypes = [c_void_p, C.POINTER(c_void_p), c_size_t, C.POINTER(c_float)]
        L.llama_set_adapters_lora.restype = c_int32
        L.llama_set_adapter_cvec.argtypes = [c_void_p, C.POINTER(c_float), c_size_t, c_int32, c_int32, c_int32]
        L.llama_set_adapter_cvec.restype = c_int32

        GB.ggml_get_name.argtypes = [c_void_p]
        GB.ggml_get_name.restype = c_char_p
        GB.ggml_nbytes.argtypes = [c_void_p]
        GB.ggml_nbytes.restype = c_size_t
        GB.ggml_backend_tensor_get.argtypes = [c_void_p, c_void_p, c_size_t, c_size_t]
        GB.ggml_backend_tensor_get.restype = None

        G.ggml_backend_load_all_from_path(str(self.bin_dir).encode("utf-8"))
        L.llama_backend_init()

        G.ggml_backend_dev_count.argtypes = []
        G.ggml_backend_dev_count.restype = c_size_t
        G.ggml_backend_dev_get.argtypes = [c_size_t]
        G.ggml_backend_dev_get.restype = c_void_p
        # device accessors live in ggml-base.dll; the registry (count/get) lives in ggml.dll
        for fn in ("ggml_backend_dev_name", "ggml_backend_dev_description"):
            getattr(GB, fn).argtypes = [c_void_p]
            getattr(GB, fn).restype = c_char_p
        GB.ggml_backend_dev_type.argtypes = [c_void_p]
        GB.ggml_backend_dev_type.restype = C.c_int
        GB.ggml_backend_dev_memory.argtypes = [c_void_p, C.POINTER(c_size_t), C.POINTER(c_size_t)]
        GB.ggml_backend_dev_memory.restype = None
        L.llama_supports_gpu_offload.argtypes = []
        L.llama_supports_gpu_offload.restype = c_bool
        self.gpu_offload: dict = {}

        self.model = None
        self.ctx = None
        self.vocab = None
        self._wanted: dict[str, object] = {}
        self._captured: dict[str, np.ndarray] = {}
        self._seen_names: list[str] = []
        self._record_names = False
        self._cb_error: str | None = None
        self._cb = EVAL_CB(self._eval_cb)
        self._adapters: list[c_void_p] = []

    # ---- layout validation -------------------------------------------------------------
    def check_layout(self) -> dict:
        mp = self.lib.llama_model_default_params()
        cp = self.lib.llama_context_default_params()
        got = {
            "sizeof_model_params": C.sizeof(ModelParams), "sizeof_context_params": C.sizeof(ContextParams),
            "mp.n_gpu_layers": mp.n_gpu_layers, "mp.main_gpu": mp.main_gpu, "mp.vocab_only": mp.vocab_only,
            "mp.use_extra_bufts": mp.use_extra_bufts, "mp.check_tensors": mp.check_tensors,
            "cp.n_ctx": cp.n_ctx, "cp.n_batch": cp.n_batch, "cp.n_ubatch": cp.n_ubatch, "cp.n_seq_max": cp.n_seq_max,
            "cp.n_threads": cp.n_threads, "cp.n_threads_batch": cp.n_threads_batch,
            "cp.rope_scaling_type": cp.rope_scaling_type, "cp.pooling_type": cp.pooling_type,
            "cp.yarn_ext_factor": cp.yarn_ext_factor, "cp.yarn_orig_ctx": cp.yarn_orig_ctx,
            "cp.cb_eval_is_null": not bool(C.cast(cp.cb_eval, c_void_p).value),
            "cp.type_k": cp.type_k, "cp.type_v": cp.type_v,
            "cp.embeddings": cp.embeddings, "cp.offload_kqv": cp.offload_kqv, "cp.no_perf": cp.no_perf,
            "cp.op_offload": cp.op_offload, "cp.swa_full": cp.swa_full, "cp.kv_unified": cp.kv_unified,
            "cp.n_samplers": cp.n_samplers,
        }
        problems = []
        if cp.n_ctx != 512: problems.append(f"n_ctx default {cp.n_ctx} != 512")
        if cp.n_batch != 2048: problems.append(f"n_batch default {cp.n_batch} != 2048")
        if cp.n_ubatch != 512: problems.append(f"n_ubatch default {cp.n_ubatch} != 512")
        if cp.rope_scaling_type != -1: problems.append("rope_scaling_type default != -1 (UNSPECIFIED)")
        if cp.pooling_type != -1: problems.append("pooling_type default != -1 (UNSPECIFIED)")
        if abs(cp.yarn_ext_factor - (-1.0)) > 1e-6: problems.append("yarn_ext_factor default != -1.0")
        if not got["cp.cb_eval_is_null"]: problems.append("cb_eval default not NULL")
        if cp.type_k != 1 or cp.type_v != 1: problems.append("type_k/type_v default != F16")
        if cp.n_samplers != 0: problems.append("n_samplers default != 0")
        if mp.vocab_only or mp.check_tensors: problems.append("model bool defaults unexpected")
        if problems:
            raise RuntimeError("llama.h struct layout mismatch: " + "; ".join(problems) + f" | {got}")
        return got

    # ---- model / context ---------------------------------------------------------------
    def devices(self) -> list[dict]:
        out = []
        for i in range(self.ggml.ggml_backend_dev_count()):
            dev = self.ggml.ggml_backend_dev_get(i)
            free, total = c_size_t(0), c_size_t(0)
            self.ggml_base.ggml_backend_dev_memory(dev, C.byref(free), C.byref(total))
            out.append({
                "name": (self.ggml_base.ggml_backend_dev_name(dev) or b"").decode("utf-8", "replace"),
                "description": (self.ggml_base.ggml_backend_dev_description(dev) or b"").decode("utf-8", "replace"),
                "type": int(self.ggml_base.ggml_backend_dev_type(dev)),   # 0 = CPU, non-zero = GPU class
                "free_mib": round(free.value / 2**20, 1), "total_mib": round(total.value / 2**20, 1),
            })
        return out

    def load(self, model_path: str, n_gpu_layers: int = 99, n_ctx: int = 2048, n_batch: int = 2048,
             n_threads: int = 8, require_gpu: bool = True, n_seq_max: int = 1):
        import re as _re
        devs_before = self.devices()
        gpu_before = [dv for dv in devs_before if dv["type"] != 0]
        want_gpu = require_gpu and n_gpu_layers != 0
        if want_gpu and (not self.lib.llama_supports_gpu_offload() or not gpu_before):
            raise RuntimeError(f"no GPU backend device registered (devices={devs_before}); refusing CPU fallback")
        mp = self.lib.llama_model_default_params()
        mp.n_gpu_layers = n_gpu_layers
        n_log0 = len(self._log_lines)
        self.model = self.lib.llama_model_load_from_file(str(model_path).encode("utf-8"), mp)
        m = _re.search(r"offloaded (\d+)/(\d+) layers to GPU", "".join(self._log_lines[n_log0:]))
        self.gpu_offload = {
            "devices_before_load": devs_before,
            "offloaded_layers": int(m.group(1)) if m else None,
            "offloadable_layers": int(m.group(2)) if m else None,
        }
        if self.model and want_gpu and (not m or int(m.group(1)) != int(m.group(2)) or int(m.group(1)) == 0):
            raise RuntimeError("model was not fully offloaded to the GPU: " + (m.group(0) if m else "no offload line in log"))
        if not self.model:
            raise RuntimeError("model load failed:\n" + "".join(self._log_lines[-30:]))
        self.vocab = self.lib.llama_model_get_vocab(self.model)
        cp = self.lib.llama_context_default_params()
        cp.n_ctx = n_ctx
        cp.n_batch = n_batch
        cp.n_ubatch = n_batch          # one ubatch per sequence -> captured tensors cover every position
        cp.n_seq_max = n_seq_max
        self.n_seq_max = n_seq_max
        cp.n_threads = n_threads
        cp.n_threads_batch = n_threads
        cp.cb_eval = self._cb
        cp.cb_eval_user_data = None
        cp.no_perf = True
        self.ctx = self.lib.llama_init_from_model(self.model, cp)
        if not self.ctx:
            raise RuntimeError("context init failed:\n" + "".join(self._log_lines[-30:]))
        self.n_embd = self.lib.llama_model_n_embd(self.model)
        self.n_layer = self.lib.llama_model_n_layer(self.model)
        self.n_vocab = self.lib.llama_vocab_n_tokens(self.vocab)
        devs_after = self.devices()
        self.gpu_offload["devices_after_context"] = devs_after
        ga = [dv for dv in devs_after if dv["type"] != 0]
        if gpu_before and ga:
            taken = round(gpu_before[0]["free_mib"] - ga[0]["free_mib"], 1)
            self.gpu_offload["vram_taken_mib"] = taken
            if want_gpu and taken < 1024:
                raise RuntimeError(f"GPU free memory dropped by only {taken} MiB after load; weights are not on the GPU")
        return self

    def close(self):
        if self.ctx:
            self.lib.llama_free(self.ctx); self.ctx = None
        if self.model:
            self.lib.llama_model_free(self.model); self.model = None

    # ---- tokens ------------------------------------------------------------------------
    def tokenize(self, text: str, add_special: bool = False, parse_special: bool = True) -> list[int]:
        b = text.encode("utf-8")
        cap = len(b) + 16
        buf = (c_int32 * cap)()
        n = self.lib.llama_tokenize(self.vocab, b, len(b), buf, cap, add_special, parse_special)
        if n < 0:
            raise RuntimeError(f"tokenize failed ({n})")
        return list(buf[:n])

    def piece(self, tok: int) -> str:
        buf = C.create_string_buffer(256)
        n = self.lib.llama_token_to_piece(self.vocab, tok, buf, 256, 0, True)
        return buf.raw[:max(n, 0)].decode("utf-8", "replace")

    # ---- eval callback -----------------------------------------------------------------
    def _eval_cb(self, t, ask, _ud):
        try:
            name = self.ggml_base.ggml_get_name(t)
            name = name.decode("utf-8", "replace") if name else ""
            if ask:
                if self._record_names:
                    self._seen_names.append(name)
                return name in self._wanted
            spec = self._wanted.get(name)
            if spec is None:
                return True
            head = C.cast(t, C.POINTER(TensorHead)).contents
            if head.type != GGML_TYPE_F32:
                raise RuntimeError(f"tensor {name} has ggml type {head.type}, expected F32")
            ne0, ne1 = int(head.ne[0]), int(head.ne[1])
            if int(head.ne[2]) != 1 or int(head.ne[3]) != 1:
                raise RuntimeError(f"tensor {name} is not 2D: ne={list(head.ne)}")
            rows = spec if spec != "all" else list(range(ne1))
            out = np.empty((len(rows), ne0), dtype=np.float32)
            row_bytes = ne0 * 4
            for i, r in enumerate(rows):
                if not (0 <= r < ne1):
                    raise RuntimeError(f"row {r} out of range for {name} with {ne1} rows")
                self.ggml_base.ggml_backend_tensor_get(t, out[i].ctypes.data_as(c_void_p), r * row_bytes, row_bytes)
            self._captured[name] = out
            self._shapes[name] = (ne1, ne0)
            return True
        except Exception as e:  # never let an exception cross the C boundary
            self._cb_error = f"{type(e).__name__}: {e}"
            return False

    def decode(self, tokens: list[int], want: dict[str, object], out_positions: list[int],
               record_names: bool = False) -> dict[str, np.ndarray]:
        """Run one fresh sequence. `want` maps tensor name -> list of row indices (or "all").

        Rows index token positions, except for tensors computed after the last layer's output
        crop (see qwen35.cpp crop_before_nextn), where rows index `out_positions` in order.
        Returns {name: float32 array (n_rows, dim)} plus "__logits__" (len(out_positions), n_vocab)
        when `want` contains "__logits__".
        """
        n = len(tokens)
        self.lib.llama_memory_clear(self.lib.llama_get_memory(self.ctx), True)
        batch = self.lib.llama_batch_init(n, 0, 1)
        try:
            outs = set(out_positions)
            for i, tk in enumerate(tokens):
                batch.token[i] = tk
                batch.pos[i] = i
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = 0
                batch.logits[i] = 1 if i in outs else 0
            batch.n_tokens = n
            self._wanted = {k: v for k, v in want.items() if not k.startswith("__")}
            self._captured, self._shapes = {}, {}
            self._cb_error = None
            self._record_names = record_names
            if record_names:
                self._seen_names = []
            rc = self.lib.llama_decode(self.ctx, batch)
            if self._cb_error:
                raise RuntimeError("eval callback failed: " + self._cb_error)
            if rc != 0:
                raise RuntimeError(f"llama_decode returned {rc}:\n" + "".join(self._log_lines[-12:]))
            res = dict(self._captured)
            missing = [k for k in self._wanted if k not in res]
            if missing:
                raise RuntimeError(f"tensors never observed: {missing}")
            if "__logits__" in want:
                lg = np.empty((len(out_positions), self.n_vocab), dtype=np.float32)
                for j, p in enumerate(sorted(out_positions)):
                    ptr = self.lib.llama_get_logits_ith(self.ctx, p)
                    lg[j] = np.ctypeslib.as_array(ptr, shape=(self.n_vocab,))
                res["__logits__"] = lg
            return res
        finally:
            self.lib.llama_batch_free(batch)

    def decode_multi(self, parts: list[tuple[int, list[int], int, bool]], clear: bool = True) -> dict[int, np.ndarray]:
        """One llama_decode over several sequences. parts = [(seq_id, tokens, start_pos, want_last_logits)].

        Tokens are laid out sequence after sequence in the batch. Returns {seq_id: logits at that
        sequence's last submitted token} for the sequences that asked for them. With clear=False the
        context memory is kept, so a sequence can be continued across calls (start_pos > 0).
        """
        if clear:
            self.lib.llama_memory_clear(self.lib.llama_get_memory(self.ctx), True)
        n = sum(len(t) for _, t, _, _ in parts)
        batch = self.lib.llama_batch_init(n, 0, 1)
        try:
            i = 0
            last_index = {}
            for sid, toks, p0, want_l in parts:
                for j, tk in enumerate(toks):
                    batch.token[i] = tk
                    batch.pos[i] = p0 + j
                    batch.n_seq_id[i] = 1
                    batch.seq_id[i][0] = sid
                    is_last = (j == len(toks) - 1)
                    batch.logits[i] = 1 if ((is_last and want_l) or want_l == "all") else 0
                    if is_last and want_l:
                        last_index[sid] = i
                    i += 1
            batch.n_tokens = n
            self._wanted, self._captured, self._shapes = {}, {}, {}
            self._cb_error, self._record_names = None, False
            rc = self.lib.llama_decode(self.ctx, batch)
            if rc != 0:
                raise RuntimeError(f"llama_decode returned {rc}: " + "".join(self._log_lines[-12:]))
            out = {}
            for sid, bi in last_index.items():
                ptr = self.lib.llama_get_logits_ith(self.ctx, bi)
                out[sid] = np.ctypeslib.as_array(ptr, shape=(self.n_vocab,)).astype(np.float32).copy()
            return out
        finally:
            self.lib.llama_batch_free(batch)

    # ---- adapters ----------------------------------------------------------------------
    def load_lora(self, path: str) -> int:
        h = self.lib.llama_adapter_lora_init(self.model, str(path).encode("utf-8"))
        if not h:
            raise RuntimeError(f"lora load failed: {path}\n" + "".join(self._log_lines[-10:]))
        self._adapters.append(c_void_p(h))
        return len(self._adapters) - 1

    def set_loras(self, scales: dict[int, float]):
        idx = [i for i, s in sorted(scales.items()) if s != 0.0]
        n = len(idx)
        arr = (c_void_p * max(n, 1))(*[self._adapters[i] for i in idx])
        sc = (c_float * max(n, 1))(*[float(scales[i]) for i in idx])
        rc = self.lib.llama_set_adapters_lora(self.ctx, arr, n, sc)
        if rc != 0:
            raise RuntimeError(f"llama_set_adapters_lora returned {rc}")

    def set_cvec(self, layer_vectors: dict[int, np.ndarray] | None):
        """Add a constant vector to the residual stream after the given layers (0-based `il`).

        The C API takes an n_embd x n_layers buffer 'starting from layer 1' (index il-1 for il>=1).
        """
        if not layer_vectors:
            rc = self.lib.llama_set_adapter_cvec(self.ctx, None, 0, self.n_embd, 0, 0)
        else:
            if min(layer_vectors) < 1:
                raise ValueError("control vectors cannot target il=0 in this API")
            hi = max(layer_vectors)
            data = np.zeros((hi, self.n_embd), dtype=np.float32)
            for il, v in layer_vectors.items():
                data[il - 1] = np.asarray(v, dtype=np.float32)
            self._cvec_keep = data
            rc = self.lib.llama_set_adapter_cvec(
                self.ctx, data.ctypes.data_as(C.POINTER(c_float)), data.size, self.n_embd,
                min(layer_vectors), hi)
        if rc != 0:
            raise RuntimeError(f"llama_set_adapter_cvec returned {rc}")
