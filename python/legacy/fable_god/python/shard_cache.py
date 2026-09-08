"""One GPU's shard of the expert cache. Launch 8 of these (one per A100,
CUDA_VISIBLE_DEVICES pinned) to cache thousands of datasets in parallel."""
import argparse
import sys

sys.path.insert(0, "/home/ubuntu/fable_god")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)       # shard index (== CVD)
    ap.add_argument("--nshards", type=int, default=8)
    ap.add_argument("--category", default="binary")
    ap.add_argument("--limit", type=int, default=8000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--packed", action="store_true",
                    help="use packed shards (batched inference) instead of CSV dirs")
    ap.add_argument("--bases", default="real_packed,synthetic_packed")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--col-target-k", type=int, default=0)
    args = ap.parse_args()

    from python.fusion import experts as E
    is_cls = args.category == "binary"
    exps = [E.TabPFNv3ClsBinary(device="cuda"), E.TabICLCls(device="cuda")] if is_cls \
        else [E.TabPFNv3Reg(device="cuda"), E.TabICLReg(device="cuda")]

    if args.packed:
        from python.fusion.cache import build_cache_packed
        bases = args.bases.split(",")
        print(f"[gpu {args.gpu}] packed cache bases={bases} -> {args.out}", flush=True)
        build_cache_packed(bases, exps, args.out, shard_index=args.gpu,
                           nshards=args.nshards, max_shards=args.max_shards,
                           col_target_k=args.col_target_k, log_every=20)
    else:
        from python.fusion.data import list_datasets
        from python.fusion.cache import build_cache
        dirs = list_datasets(args.category, limit=args.limit, shuffle_seed=args.seed)
        shard = dirs[args.gpu::args.nshards]
        print(f"[gpu {args.gpu}] caching {len(shard)} datasets -> {args.out}", flush=True)
        build_cache(shard, exps, args.out, max_test=args.max_test, log_every=100)


if __name__ == "__main__":
    main()
