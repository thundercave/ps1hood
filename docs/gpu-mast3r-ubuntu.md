# GPU playbook: MASt3R on Ubuntu (Radeon 6900 XT / ROCm)

**Audience:** your local Ubuntu box (32 GB RAM + RX 6900 XT) running Grok CLI.
This machine is **not** registered with Grok Bot yet — use it for capture / CPU+OpenCL COLMAP first; register later if you want Bot-driven GPU jobs.

**Honest bottom line (2026-09):** official **naver/mast3r** and **dust3r** target **NVIDIA CUDA**. There is **no supported CUDA path on AMD**. ROCm/HIP can run generic PyTorch on a 6900 XT (`gfx1030`), but MASt3R's CroCo RoPE kernels are CUDA extensions — community ROCm ports exist for *adjacent* projects (e.g. MASt3R-SLAM-ROCm, dust3r on **MI300x** `gfx942`), not a turnkey "pip install and go" story for consumer RDNA2. Treat ROCm MASt3R as **experimental / likely to fail**; do **not** plan on CPU MASt3R (unsupported / useless for a block). Prefer denser Street View capture + `colmap_posed` / flow / OpenCV SIFT here, and **cloud CUDA** (or a registered CUDA machine) when you need MASt3R.

Smoke baseline (same Utrecht bbox): `colmap_posed` ≈ **112** points; flow cloud ≈ **12k**. Dense capture (`smoke-dense`) is meant to feed better stereo baselines into those photo backends — and into MASt3R once weights + a CUDA (or working ROCm) device exist.

---

## 0. What works vs what doesn't on 6900 XT

| Path | Realistic? | Notes |
|------|------------|--------|
| NVIDIA CUDA + official mast3r | **Yes** (elsewhere) | Upstream docs / Docker assume CUDA 12.x |
| PyTorch ROCm on 6900 XT | **Yes for torch** | HIP SDK lists RX 6900 XT (`gfx1030`) as supported; often need `HSA_OVERRIDE_GFX_VERSION=10.3.0` |
| Official MASt3R on ROCm | **No / shaky** | CUDA kernels in `dust3r/croco/models/curope/`; not shipped for HIP |
| dust3r ROCm (community) | **MI datacenter only so far** | Verified recipe on MI300x (`PYTORCH_ROCM_ARCH=gfx942`), not RDNA2 |
| MASt3R-SLAM-ROCm fork | **Partial** | Separate SLAM repo; not the same as `ps1hood reconstruct --backend mast3r` |
| MASt3R on CPU | **Useless** | Hours+ and memory thrash; `ps1_hood.reconstruct.mast3r` already warns |
| COLMAP sparse / posed on AMD | **Yes** | CPU SIFT always; GPU OpenCL if your COLMAP build enables it |
| Register machine → Bot later | **Recommended** | Then Bot can drive capture + CUDA cloud / remote workers |
| Cloud CUDA for MASt3R | **Recommended** | Rent a small CUDA VM; copy `runs/<name>/` keyframes or `recon/mast3r_input/` |

**CUDA N/A on this GPU.** Do not install NVIDIA CUDA drivers for a 6900 XT.

---

## 1. Clone ps1hood + uv

```bash
sudo apt update
sudo apt install -y git curl build-essential colmap  # colmap optional but useful

# uv: https://docs.astral.sh/uv/
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"   # or restart shell

git clone https://github.com/thundercave/ps1hood.git
cd ps1hood
cp .env.example .env

uv sync --extra dev --extra browser
uv run ps1hood setup-browser   # only if you will capture google_web
```

Verify:

```bash
uv run ps1hood --help
uv run pytest -q
```

---

## 2. ROCm / PyTorch (optional experiment only)

Install AMD's ROCm stack for Ubuntu from current [ROCm on Radeon](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/) docs (match your Ubuntu LTS). Then:

```bash
# Example indexes change — check https://pytorch.org for the latest rocmN.N wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.3

export HSA_OVERRIDE_GFX_VERSION=10.3.0   # RDNA2 / 6900 XT
export HIP_VISIBLE_DEVICES=0

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
# On ROCm, torch still uses the 'cuda' API name for the HIP device.
```

If `False` / wrong device: fix groups (`video`, `render`), reboot after `amdgpu-install`, re-check override. **Getting torch.cuda True does not mean MASt3R will run.**

### Trying MASt3R anyway (expect breakage)

```bash
git clone --recursive https://github.com/naver/mast3r ~/mast3r
cd ~/mast3r
# Do NOT conda-install pytorch-cuda — keep the ROCm torch from above
pip install -r requirements.txt
pip install -r dust3r/requirements.txt

# CroCo RoPE: CUDA extension — on AMD you need a HIP-capable setup.py
# (see community notes / naver/croco PRs). On gfx1030 this often fails to
# build or faults at runtime. Skip compile → slow/broken path.
# cd dust3r/croco/models/curope && python setup.py build_ext --inplace

mkdir -p checkpoints
wget -P checkpoints \
  https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
```

Put the checkout on `PYTHONPATH` (or `pip install -e .` if it resolves) so `import mast3r` / `import dust3r` work inside the ps1hood venv. If import or forward pass dies, **stop** and use alternatives below — do not burn a day fighting HIP kernels for this hobby pipeline.

---

## 3. Recommended AMD-local work (no MASt3R)

### A. Denser capture for stereo

```bash
cd ~/ps1hood
# Same Utrecht smoke bbox, denser drive, capped panos (not overnight)
uv run ps1hood init smoke-dense \
  --south 52.0895 --west 5.1192 --north 52.0900 --east 5.1200 \
  --preset smoke --spacing 10 --heading-step 60 --max-panos 22 --steps 3

uv run ps1hood run smoke-dense
# or stage-by-stage: discover → capture → crop → satellite → align → interpolate → reconstruct
```

Then compare:

```bash
# vertex counts
grep 'element vertex' runs/smoke/recon/cloud*.ply runs/smoke-dense/recon/cloud*.ply
uv run ps1hood reconstruct smoke-dense --backend flow
uv run ps1hood reconstruct smoke-dense --backend colmap_posed
```

### B. COLMAP (CPU; OpenCL GPU if available)

```bash
# posed triangulator path (photo cloud) — needs align/cameras.json
uv run ps1hood reconstruct smoke-dense --backend colmap_posed

# Check whether your COLMAP binary has CUDA or OpenCL:
colmap -h 2>&1 | head -5
# Feature extraction: --SiftExtraction.use_gpu 1 may use OpenCL on some builds.
# If GPU feature extract fails, force CPU: use_gpu 0 (ps1hood already falls back).
```

COLMAP **dense** PatchMatch stereo is historically CUDA-centric; do not expect AMD dense MVS. Sparse + `point_triangulator` (what `colmap_posed` runs) is the useful path.

### C. Flow / SIFT (always available)

```bash
uv run ps1hood reconstruct smoke-dense --backend flow
uv run ps1hood reconstruct smoke-dense --backend sift
uv run ps1hood studio
```

---

## 4. Run MASt3R on existing smoke keyframes (when CUDA or working ROCm + weights exist)

Weights file (example):

`~/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`

From the ps1hood repo (after `uv sync`, mast3r/dust3r importable, GPU visible):

```bash
cd ~/ps1hood

# Prefer a finished run with align + crops (smoke already has these on the Bot box;
# on your machine, either re-run smoke or rsync runs/smoke from elsewhere).
uv run ps1hood reconstruct smoke --backend mast3r
```

What that does:

1. Exports keyframes to `runs/smoke/recon/mast3r_input/images/`.
2. Loads MASt3R (local `.pth` if found under `~/mast3r`, cwd, or torch/HF cache; else HF hub id).
3. Sparse global alignment → `runs/smoke/recon/cloud.ply`.

Manual export-only if the package is missing: the same command still writes `mast3r_input/` then raises with install hints — you can feed those JPEGs to an upstream mast3r demo on a CUDA box.

Smoke-dense:

```bash
uv run ps1hood reconstruct smoke-dense --backend mast3r
```

**CPU:** do not bother (`device` falls back to CPU and is not a viable workflow).

---

## 5. Cloud CUDA (practical MASt3R)

1. Pack keyframes (no need for full `raw/`):

```bash
cd ~/ps1hood
tar czf /tmp/smoke-mast3r-in.tgz \
  runs/smoke/project.yaml \
  runs/smoke/align \
  runs/smoke/cropped \
  runs/smoke/recon/mast3r_input 2>/dev/null || true
# Or after a failed local mast3r attempt, mast3r_input/ alone is enough for upstream demos.
```

2. On a CUDA VM: clone ps1hood + mast3r (official CUDA torch), `uv sync`, unpack runs, `uv run ps1hood reconstruct smoke --backend mast3r`.
3. Copy back:

```bash
# on CUDA host
tar czf /tmp/smoke-mast3r-out.tgz runs/smoke/recon/cloud.ply runs/smoke/recon/scene.json

# on your Ubuntu box
scp user@cuda-host:/tmp/smoke-mast3r-out.tgz /tmp/
cd ~/ps1hood && tar xzf /tmp/smoke-mast3r-out.tgz
uv run ps1hood studio
```

---

## 6. Register the machine with Grok Bot (later)

When ready for Bot to drive this box:

1. Install / sign in to **Cursor / Grok CLI** on the Ubuntu host (already planned).
2. Register the computer so agents can use `ListMachines` → Shell/Read with your `machineId`.
3. Prefer Bot for: long `google_web` captures, COLMAP posed jobs, rsync of `runs/`, PR prep.
4. Still route **MASt3R** to a CUDA machine or cloud unless ROCm mast3r is proven on *your* 6900 XT with a short smoke pair test first.

---

## 7. Copy recon artifacts back / open a PR

**Artifacts stay local** (`runs/` is gitignored — do not commit Street View JPEGs).

```bash
# Code + docs only
cd ~/ps1hood
git checkout -b chore/gpu-playbook-smoke-dense
git add docs/gpu-mast3r-ubuntu.md
# plus any other code changes
git status
git commit -m "Document AMD/ROCm MASt3R limits and denser smoke-dense capture path."

git push -u origin HEAD
gh pr create --title "docs: GPU MASt3R Ubuntu playbook (6900 XT / ROCm)" --body "## Summary
- Honest ROCm vs CUDA guidance for MASt3R on RX 6900 XT
- Commands for uv/setup, colmap_posed/flow fallbacks, cloud CUDA MASt3R
- smoke-dense denser-capture recipe (capped panos)

## Test plan
- [ ] \`uv run pytest -q\`
- [ ] Skim playbook on Ubuntu host; torch ROCm detect optional
- [ ] Do not commit \`runs/\` imagery"
```

To share clouds with another machine (not GitHub):

```bash
rsync -avP runs/smoke/recon/cloud.ply runs/smoke/recon/scene.json user@other:~/ps1hood/runs/smoke/recon/
```

---

## 8. Quick decision tree

```
Need denser photo stereo today?
  → smoke-dense capture + reconstruct flow / colmap_posed / sift on this box

Need MASt3R-quality cloud?
  → cloud or registered CUDA machine + weights
  → only then try ROCm experiment on 6900 XT (expect pain)

ROCm torch works but mast3r curope build fails?
  → stop; use COLMAP posed + flow; do not use CPU mast3r
```

---

## References

- https://github.com/naver/mast3r (official CUDA)
- https://github.com/naver/dust3r/issues/243 (ROCm docker on **MI300x**, not RDNA2)
- https://github.com/EmmanuelMess/MASt3R-SLAM-ROCm (SLAM fork ROCm notes)
- https://rocm.docs.amd.com/ (Radeon HIP SDK — 6900 XT listed)
- Repo: `docs/cpu-known-pose-mvs.md` (COLMAP posed / OpenCV stereo digest)
