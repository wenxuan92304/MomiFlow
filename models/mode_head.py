from torch import nn


class ModeHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_modes):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_modes))

    def forward(self, global_feature):
        return self.mlp(global_feature) # logits; loss uses log_softmax without argmax
