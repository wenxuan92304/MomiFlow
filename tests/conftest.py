from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest


@pytest.fixture
def tiny_config():
    from utils.config import load_config
    c = load_config("configs/trajectory_only.yaml")
    c["data"].update(obs_len=3, pred_len=2)
    c["trajectory_encoder"].update(hidden_dim=16, num_layers=1, num_heads=2, feedforward_dim=32)
    c["fusion"].update(hidden_dim=16, output_dim=8)
    c["video_encoder"]["output_dim"] = 8
    c["imf"].update(hidden_dim=16, num_layers=2, aux_head_depth=1, num_heads=2, scalar_embedding_dim=16,
                     num_samples=3)
    c["evaluation"]["K"] = 3
    return c
