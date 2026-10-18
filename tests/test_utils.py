import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from advrob.utils import AverageMeter, load_config  # noqa: E402


def test_average_meter():
    m = AverageMeter(); m.update(1.0, 2); m.update(4.0, 2)
    assert m.avg == 2.5


def test_config_overrides(tmp_path):
    f = tmp_path / "c.yaml"; f.write_text("epochs: 30\nmodel_args: {a: 1}\n")
    cfg = load_config(f, ["epochs=1", "model_args.a=2"])
    assert cfg["epochs"] == 1 and cfg["model_args"]["a"] == 2
