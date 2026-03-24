#!/usr/bin/env python3
"""Smoke test for all *_cache_latents.py scripts.

Creates dummy images and a dataset config, then runs each root-level
cache_latents script end-to-end with a mock VAE so the full encode-and-save
pipeline is exercised without needing real model weights or a GPU.
"""

import os
import subprocess
import sys
import tempfile
import textwrap

import numpy as np
from PIL import Image


def create_dummy_dataset(tmpdir: str, n_images: int = 3, size: int = 64):
    """Create dummy images, captions, control images, and dataset configs."""
    img_dir = os.path.join(tmpdir, "images")
    ctrl_dir = os.path.join(tmpdir, "controls")
    cache_dir = os.path.join(tmpdir, "cache")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ctrl_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    for i in range(n_images):
        img = Image.fromarray(
            np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        )
        img.save(os.path.join(img_dir, f"img_{i:03d}.png"))
        # control images (same size, for kontext/edit architectures)
        img.save(os.path.join(ctrl_dir, f"img_{i:03d}.png"))
        with open(os.path.join(img_dir, f"img_{i:03d}.txt"), "w") as f:
            f.write(f"test caption {i}")

    # basic config (no control images)
    basic_config = os.path.join(tmpdir, "dataset.toml")
    with open(basic_config, "w") as f:
        f.write(
            f'[general]\nresolution = [{size}, {size}]\n'
            f'caption_extension = ".txt"\nbatch_size = 1\nenable_bucket = false\n\n'
            f'[[datasets]]\nimage_directory = "{img_dir}"\n'
            f'cache_directory = "{cache_dir}"\nnum_repeats = 1\n'
        )

    # config with control images (for kontext)
    ctrl_config = os.path.join(tmpdir, "dataset_ctrl.toml")
    with open(ctrl_config, "w") as f:
        f.write(
            f'[general]\nresolution = [{size}, {size}]\n'
            f'caption_extension = ".txt"\nbatch_size = 1\nenable_bucket = false\n\n'
            f'[[datasets]]\nimage_directory = "{img_dir}"\n'
            f'control_directory = "{ctrl_dir}"\n'
            f'cache_directory = "{cache_dir}"\nnum_repeats = 1\n'
        )

    return basic_config, ctrl_config, cache_dir


# mock VAE helpers reused across scripts
FAKE_3D_VAE_LATENT_DIST = textwrap.dedent("""\
import torch

class _FakeLatentDist:
    def __init__(self, val):
        self._val = val
    def sample(self):
        return self._val
    def mode(self):
        return self._val

class _FakeEncOut:
    def __init__(self, val):
        self.latent_dist = _FakeLatentDist(val)
    def __getitem__(self, idx):
        return self
""")

FAKE_2D_AE = textwrap.dedent("""\
import torch

class FakeAE(torch.nn.Module):
    dtype = torch.float32
    device = torch.device("cpu")
    def encode(self, x):
        B, C, H, W = x.shape
        return torch.randn(B, 16, H // 8, W // 8)
""")


def _mock_cache_latents():
    return FAKE_3D_VAE_LATENT_DIST + textwrap.dedent("""\

class FakeVAE(torch.nn.Module):
    dtype = torch.float32
    device = torch.device("cpu")
    def encode(self, x):
        B, C, F, H, W = x.shape
        return _FakeEncOut(torch.randn(B, 16, F, H // 8, W // 8))
    class config:
        scaling_factor = 1.0

import musubi_tuner.cache_latents as _mod
# patch the local binding of load_vae inside cache_latents module
def _fake_load(**kw):
    return FakeVAE(), None, 8, 4
_mod.load_vae = _fake_load
""")


def _mock_hv_1_5():
    return FAKE_3D_VAE_LATENT_DIST + textwrap.dedent("""\

class FakeVAE(torch.nn.Module):
    dtype = torch.float32
    device = torch.device("cpu")
    def encode(self, x):
        B, C, F, H, W = x.shape
        # encode()[0].mode() — [0] gets the dist, .mode() gets the tensor
        return [_FakeLatentDist(torch.randn(B, 16, F, H // 8, W // 8))]

import musubi_tuner.hv_1_5_cache_latents as _mod
import musubi_tuner.hunyuan_video_1_5.hunyuan_video_1_5_vae as _vae_mod
def _fake_load(*a, **kw):
    return FakeVAE()
_vae_mod.load_vae_from_checkpoint = _fake_load
""")


def _mock_wan():
    return textwrap.dedent("""\
import torch
import musubi_tuner.wan_cache_latents as _mod
import musubi_tuner.wan.modules.vae as _vae_mod

_orig_init = _vae_mod.WanVAE.__init__
def _fake_init(self, *a, **kw):
    self.dtype = torch.bfloat16
    self.device = torch.device("cpu")
_vae_mod.WanVAE.__init__ = _fake_init

def _fake_encode(self, x):
    B, C, F, H, W = x.shape
    return [torch.randn(16, F, H // 8, W // 8) for _ in range(B)]
_vae_mod.WanVAE.encode = _fake_encode
""")


def _mock_flux_2():
    return FAKE_2D_AE + textwrap.dedent("""\

import musubi_tuner.flux_2_cache_latents as _mod
import musubi_tuner.flux_2.flux2_utils as _f2u
def _fake_load(*a, **kw):
    return FakeAE()
_f2u.load_ae = _fake_load
""")


def _mock_flux_kontext():
    return FAKE_2D_AE + textwrap.dedent("""\

import musubi_tuner.flux_kontext_cache_latents as _mod
import musubi_tuner.flux.flux_utils as _fu
def _fake_load(*a, **kw):
    return FakeAE()
_fu.load_ae = _fake_load
""")


def _mock_fpack():
    return FAKE_3D_VAE_LATENT_DIST + textwrap.dedent("""\

class FakeVAE(torch.nn.Module):
    dtype = torch.float32
    device = torch.device("cpu")
    def encode(self, x):
        B, C, F, H, W = x.shape
        return _FakeEncOut(torch.randn(B, 16, F, H // 8, W // 8))
    class config:
        scaling_factor = 1.0

class FakeImageEncoder(torch.nn.Module):
    device = torch.device("cpu")
    def forward(self, **kw):
        return type("O", (), {"last_hidden_state": torch.randn(1, 257, 1152)})()
    def eval(self):
        return self
    def to(self, *a, **kw):
        return self

import musubi_tuner.fpack_cache_latents as _mod
import musubi_tuner.frame_pack.framepack_utils as _fpu
import musubi_tuner.frame_pack.clip_vision as _cv

def _fake_load_vae(*a, **kw):
    return FakeVAE()
_fpu.load_vae = _fake_load_vae
_mod.load_vae = _fake_load_vae

def _fake_load_ie(*a, **kw):
    return None, FakeImageEncoder()
_fpu.load_image_encoders = _fake_load_ie
_mod.load_image_encoders = _fake_load_ie

def _fake_clip_encode(img_np, fe, ie):
    return type("O", (), {"last_hidden_state": torch.randn(1, 257, 1152)})()
_cv.hf_clip_vision_encode = _fake_clip_encode
_mod.hf_clip_vision_encode = _fake_clip_encode

# patch encode_datasets_framepack to set missing fp_1f fields on items
_orig_edf = _mod.encode_datasets_framepack
def _patched_edf(datasets, encode, args, **kw):
    for ds in datasets:
        _orig_retrieve = ds.retrieve_latent_cache_batches
        def _patched_retrieve(*a, **k):
            for key, batch in _orig_retrieve(*a, **k):
                for item in batch:
                    if item.fp_1f_target_index is None:
                        item.fp_1f_target_index = 1
                    if item.fp_1f_clean_indices is None:
                        item.fp_1f_clean_indices = [0]
                    if item.fp_1f_no_post is None:
                        item.fp_1f_no_post = False
                yield key, batch
        ds.retrieve_latent_cache_batches = _patched_retrieve
    _orig_edf(datasets, encode, args, **kw)
_mod.encode_datasets_framepack = _patched_edf
""")


def _mock_qwen_image():
    return textwrap.dedent("""\
import torch

class FakeVAE(torch.nn.Module):
    dtype = torch.float32
    device = torch.device("cpu")
    def encode_pixels_to_latents(self, x):
        B, C, F, H, W = x.shape
        return torch.randn(B, 16, F, H // 8, W // 8)

import musubi_tuner.qwen_image_cache_latents as _mod
import musubi_tuner.qwen_image.qwen_image_utils as _qu
def _fake_load_vae(*a, **kw):
    return FakeVAE()
_qu.load_vae = _fake_load_vae
""")


def _mock_kandinsky5():
    return FAKE_3D_VAE_LATENT_DIST + textwrap.dedent("""\

class FakeVAE(torch.nn.Module):
    dtype = torch.float32
    device = torch.device("cpu")
    config = type("C", (), {"scaling_factor": 1.0, "nabla_force_resize": False})()
    def encode(self, x):
        B, C, F, H, W = x.shape
        return _FakeEncOut(torch.randn(B, 16, F, H // 8, W // 8))

import musubi_tuner.kandinsky5_cache_latents as _mod
import musubi_tuner.kandinsky5.models.vae as _k5v
def _fake_build(*a, **kw):
    return FakeVAE()
_k5v.build_vae = _fake_build
# also patch local binding
_mod.build_vae = _fake_build
""")


def _mock_zimage():
    return textwrap.dedent("""\
import torch

class _FakePosterior:
    def __init__(self, val):
        self._val = val
    def mode(self):
        return self._val

class FakeVAE(torch.nn.Module):
    dtype = torch.float32
    device = torch.device("cpu")
    def encode(self, x):
        B, C, H, W = x.shape
        return _FakePosterior(torch.randn(B, 4, H // 8, W // 8))
    def eval(self):
        return self

import musubi_tuner.zimage_cache_latents as _mod
import musubi_tuner.zimage.zimage_autoencoder as _za
def _fake_load(*a, **kw):
    return FakeVAE()
_za.load_autoencoder_kl = _fake_load
""")


# script name -> (extra args, mock function, needs control images)
SCRIPTS = {
    "cache_latents.py":             ([], _mock_cache_latents, False),
    "hv_1_5_cache_latents.py":      ([], _mock_hv_1_5, False),
    "wan_cache_latents.py":         ([], _mock_wan, False),
    "flux_2_cache_latents.py":      (["--model_version", "dev"], _mock_flux_2, False),
    "flux_kontext_cache_latents.py": ([], _mock_flux_kontext, True),
    "fpack_cache_latents.py":       (["--one_frame", "--image_encoder", "dummy"], _mock_fpack, True),
    "qwen_image_cache_latents.py":  ([], _mock_qwen_image, False),
    "kandinsky5_cache_latents.py":  ([], _mock_kandinsky5, False),
    "zimage_cache_latents.py":      ([], _mock_zimage, False),
}


def run_script(
    script: str, extra_args: list, mock_fn, config_path: str, tmpdir: str
) -> tuple[bool, str]:
    """Run a single cache_latents script with mocked VAE. Returns (success, output)."""
    wrapper = os.path.join(tmpdir, f"wrap_{script}")
    module = script.removesuffix(".py")

    argv_items = [f'"wrap_{script}"', f'"--dataset_config"', f'"{config_path}"',
                  '"--device"', '"cpu"', '"--vae"', '"dummy"']
    argv_items += [f'"{a}"' for a in extra_args]
    argv_str = ", ".join(argv_items)

    mock_code = mock_fn()

    with open(wrapper, "w") as f:
        f.write(f"import sys, os\n")
        f.write(f"sys.argv = [{argv_str}]\n")
        f.write(f'os.chdir("{os.getcwd()}")\n\n')
        f.write(mock_code)
        f.write(f"\nfrom musubi_tuner.{module} import main\n")
        f.write("main()\n")

    result = subprocess.run(
        [sys.executable, wrapper],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=os.getcwd(),
    )
    output = result.stdout + result.stderr
    return result.returncode == 0, output


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        basic_config, ctrl_config, cache_dir = create_dummy_dataset(tmpdir)

        passed = 0
        failed = 0
        total = len(SCRIPTS)

        for script in sorted(SCRIPTS):
            extra_args, mock_fn, needs_ctrl = SCRIPTS[script]
            config = ctrl_config if needs_ctrl else basic_config
            label = f"{script:<40}"

            try:
                ok, output = run_script(script, extra_args, mock_fn, config, tmpdir)
            except subprocess.TimeoutExpired:
                print(f"{label} FAIL (timeout)")
                failed += 1
                continue

            if ok:
                cache_files = [f for f in os.listdir(cache_dir) if f.endswith(".safetensors")]
                if cache_files:
                    print(f"{label} OK ({len(cache_files)} cache files)")
                    passed += 1
                else:
                    print(f"{label} FAIL (no cache files written)")
                    print(output[-500:] if len(output) > 500 else output)
                    failed += 1
                # clean for next script
                for cf in cache_files:
                    os.remove(os.path.join(cache_dir, cf))
            else:
                print(f"{label} FAIL (exit code)")
                lines = output.strip().splitlines()
                print("\n".join(lines[-30:]))
                failed += 1

        print(f"\nResults: {passed}/{total} passed, {failed} failed")
        sys.exit(failed)


if __name__ == "__main__":
    main()
