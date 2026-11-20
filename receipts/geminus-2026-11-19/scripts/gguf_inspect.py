"""Print architecture metadata and the tensor inventory of selected blocks of a GGUF model."""
import sys, collections
from gguf import GGUFReader

path = sys.argv[1]
blocks = [int(b) for b in sys.argv[2].split(",")] if len(sys.argv) > 2 else []
r = GGUFReader(path, "r")

def val(f):
    t = f.types[0].name if f.types else "?"
    try:
        if t == "STRING":
            return bytes(f.parts[f.data[0]]).decode("utf-8", "replace")
        if t == "ARRAY":
            return f"<array len={len(f.data)}>"
        return f.parts[f.data[0]].tolist()[0]
    except Exception as e:  # noqa
        return f"<unreadable {e}>"

print("== metadata (non-tokenizer) ==")
for k, f in r.fields.items():
    if k.startswith("tokenizer.") or k.startswith("GGUF."):
        continue
    v = val(f)
    if isinstance(v, str) and len(v) > 160:
        v = v[:160] + "..."
    print(f"{k} = {v}")

print("\n== tensor count by quant type ==")
print(collections.Counter(t.tensor_type.name for t in r.tensors))
names = [t.name for t in r.tensors]
n_blk = 1 + max(int(n.split(".")[1]) for n in names if n.startswith("blk."))
print("n tensors", len(names), "n_blk", n_blk)
print("\n== non-block tensors ==")
for t in r.tensors:
    if not t.name.startswith("blk."):
        print(f"{t.name:40s} {t.tensor_type.name:8s} shape={list(map(int, t.shape))}")
for b in blocks:
    print(f"\n== blk.{b} ==")
    for t in r.tensors:
        if t.name.startswith(f"blk.{b}."):
            print(f"{t.name:40s} {t.tensor_type.name:8s} shape={list(map(int, t.shape))}")
# block-structure signature across all blocks (detect hybrid layouts)
sig = collections.OrderedDict()
for b in range(n_blk):
    s = tuple(sorted(n.split(".", 2)[2] for n in names if n.startswith(f"blk.{b}.")))
    sig.setdefault(s, []).append(b)
print("\n== distinct block layouts ==")
for s, bs in sig.items():
    print(len(bs), "blocks", bs[:6], "..." if len(bs) > 6 else "", "->", len(s), "tensors")
