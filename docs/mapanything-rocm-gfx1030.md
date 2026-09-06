# MapAnything on RX 6900 XT (gfx1030) / ROCm — gotchas

**Context:** MapAnything densify is wired (PR #9); no NVIDIA CUDA on the RX 6900 XT host — try ROCm for `map-anything-apache`, or use cloud CUDA.

## Realistic expectation
gfx1030 **is** listed in newer AMD multi-arch PyTorch ROCm wheels (`device-gfx1030`) — better odds than Instinct-only stacks. Still: MapAnything is Meta/CUDA-first (DINOv2, AMP bf16, optional flash-attn). Treat ROCm as **best-effort**; budget a cloud CUDA fallback.

---

## Install gotchas

1. **Install the gfx1030 wheel, not generic CUDA torch**  
   Use AMD’s multi-arch index / `torch[device-gfx1030]==…+rocm…` (see https://rocm.docs.amd.com AI ecosystem PyTorch install). A plain `pip install torch` CPU wheel or NVIDIA CUDA wheel will lie about devices.

2. **Verify HIP-as-CUDA**  
   ```bash
   rocminfo | rg -i 'gfx1030|6900'
   python -c "import torch; print(torch.cuda.is_available(), torch.version.hip, torch.cuda.get_device_name(0))"
   ```
   On ROCm, `torch.cuda.*` is the API even though it’s AMD.

3. **ROCm ↔ driver ↔ Ubuntu skew**  
   Mismatch = invisible GPU. Stick to one documented ROCm + amdgpu combo; reboot after driver install. Prefer a **rocm/pytorch** container over host soup when possible.

4. **`HSA_OVERRIDE_GFX_VERSION=10.3.0`**  
   Only if rocminfo sees the card but torch doesn’t (or for close cousins). Don’t set it if `device-gfx1030` already matches.

---

## MapAnything-specific gotchas

| Issue | What to do |
|-------|------------|
| **bf16 on RDNA2** | Recipe uses `amp_dtype="bf16"`. If NaNs/slow/unsupported → `amp_dtype="fp16"` or `use_amp=False` for smoke. |
| **Flash-Attention / xFormers** | Official FA2 CK backend targets MI200/300; RDNA needs **Triton** FA (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`) or **disable** flash-attn and use PyTorch SDPA. First failure mode on consumer AMD is often attention kernels. |
| **VRAM 16 GB** | `memory_efficient_inference=True`, `minibatch_size=1`, `--stride 4+`, short smoke bbox only. Don’t feed full FILM rate. |
| **`ignore_pose_inputs=False`** | Keep locked ENU; unchanged on ROCm. |
| **Apache weights** | `facebook/map-anything-apache` / `--apache` — same compute, friendlier license. |
| **First-view poses** | If any view has poses, view0 must too (already in recipe). |
| **cam2world** | Still OpenCV +X right +Y down +Z forward; ENU C + `camera_rotation_cv` as cam2world. |

---

## Smoke ladder (don’t skip)

1. `torch.cuda.is_available()` + tiny matmul on device.  
2. Load `MapAnything.from_pretrained("facebook/map-anything-apache")` on cuda.  
3. **2–4 views** only, `minibatch_size=1`, `amp_dtype="fp16"`.  
4. Scale up views if stable.  
5. If attention/OOM/illegal memory → disable flash-attn / cut resolution / rent CUDA.

---

## If ROCm fails (plan B)
- Cloud CUDA (Colab/RunPod/etc.) with same export bundle from PR #9.  
- Parallel: start **MASt3R matcher** wiring (also likes CUDA, but matcher can be fewer pairs / smaller).  
- Keep OpenMVS CPU densify path as non-neural fallback.

## Links
- https://github.com/facebookresearch/map-anything  
- https://rocm.docs.amd.com (PyTorch for ROCm / Radeon)  
- ROCm FlashAttention: Triton backend for RDNA  
- In-repo densify recipe: [`mapanything-densify.md`](mapanything-densify.md)
