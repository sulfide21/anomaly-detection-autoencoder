# Reproducing the UniAD baseline

The `UniAD/` folder is **not** in this repo (it's the authors' code with its own
license). To reproduce our UniAD numbers, clone it and apply four small fixes that
make the 2022 code run on a modern GPU (torch 2.x / Windows / single GPU).

```bash
git clone https://github.com/zhiyuanyou/UniAD.git
```

Then point it at the dataset (junction/symlink), and apply these four fixes:

1. **numpy 2**: `utils/eval_helper.py` and `utils/vis_helper.py` — replace removed
   `np.int` / `np.float` with `int` / `float`.
2. **Windows backend**: `utils/dist_helper.py` — default the backend to `gloo`
   (no NCCL on Windows) and set single-process env (RANK/WORLD_SIZE/LOCAL_RANK).
3. **Single-GPU**: `tools/train_val.py` — skip `DDP` when `world_size==1` (it
   segfaults with gloo-on-CUDA) and guard every `dist.all_reduce`/`dist.barrier`
   behind `if world_size > 1`.
4. **Checkpoint load**: `utils/misc_helper.py` — `torch.load(..., weights_only=False)`
   (torch 2.6+ changed the default).

Launch (single GPU), from `UniAD/experiments/MVTec-AD/`:

```bash
MASTER_ADDR=127.0.0.1 PYTHONPATH=../../ python ../../tools/train_val.py --config config.yaml
```

We trained 100 epochs (`config_run100.yaml`) → ~0.94 detection. The paper's full
1000-epoch schedule reaches ~0.967.
