from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from pathlib import Path

root = Path(__file__).resolve().parents[1]
with initialize_config_dir(version_base="1.1", config_dir=str(root / "config")):
    for name in ("cycloformer_3p53m", "cycloformer_6p80m", "cycloformer_56p90m"):
        cfg = compose(config_name="base", overrides=[f"experiment={name}"])
        net = instantiate(cfg.network)
        n = sum(p.numel() for p in net.parameters())
        print(f"{name:22s} {n/1e6:.2f}M  ({n:,})")
