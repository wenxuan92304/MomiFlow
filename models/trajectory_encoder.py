from torch import nn
from models.positional_encoding import SinusoidalPositionEncoding


class TrajectoryEncoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        e = c["trajectory_encoder"]
        self.input = nn.Linear(e["input_dim"], e["hidden_dim"])
        self.position = SinusoidalPositionEncoding(e["hidden_dim"], c["data"]["obs_len"])
        layer = nn.TransformerEncoderLayer(e["hidden_dim"], e["num_heads"], e["feedforward_dim"],
                                           dropout=e["dropout"], batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, e["num_layers"], enable_nested_tensor=False)

    def forward(self, history_abs):
        return self.encoder(self.position(self.input(history_abs)))
