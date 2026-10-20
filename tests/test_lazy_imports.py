import os
import subprocess
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parent.parent / "src")
HEAVY = ("torch", "torchvision", "timm", "segmentation_models_pytorch",
         "huggingface_hub", "matplotlib")


def _run(code):
    env = dict(os.environ, PYTHONPATH=SRC, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, timeout=300)


def test_import_works_with_heavy_modules_blocked():
    code = (
        f"import sys\n"
        f"for n in {HEAVY!r}: sys.modules[n] = None\n"
        "import zebrafish_measure\n"
        "import zebrafish_measure.length, zebrafish_measure.seg\n"
        "import zebrafish_measure.seg_helper, zebrafish_measure.scalebar\n"
    )
    r = _run(code)
    assert r.returncode == 0, r.stderr


def test_import_does_not_load_heavy_modules():
    code = (
        "import sys\n"
        "import zebrafish_measure\n"
        "import zebrafish_measure.length, zebrafish_measure.seg\n"
        "import zebrafish_measure.seg_helper, zebrafish_measure.scalebar\n"
        f"loaded = [m for m in {HEAVY!r} if m in sys.modules]\n"
        "print(loaded)\n"
        "assert not loaded, loaded\n"
    )
    r = _run(code)
    assert r.returncode == 0, r.stderr + r.stdout


def test_fish_classifier_lazy_class():
    code = (
        "import torch\n"
        "import zebrafish_measure as zm\n"
        "import zebrafish_measure.length as ln\n"
        "from zebrafish_measure import FishClassifier\n"
        "assert isinstance(zm.FishClassifier, type)\n"
        "assert issubclass(zm.FishClassifier, torch.nn.Module)\n"
        "assert ln.FishClassifier is zm.FishClassifier is FishClassifier\n"
        "assert all(hasattr(zm, n) for n in zm.__all__)\n"
        "try:\n"
        "    ln.does_not_exist\n"
        "except AttributeError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit('no AttributeError')\n"
    )
    r = _run(code)
    assert r.returncode == 0, r.stderr
