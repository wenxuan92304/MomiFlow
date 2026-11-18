from training.trainer import train
from utils.config import parser, load_config
from utils.io import resolve


if __name__ == "__main__":
    p = parser("Train the configured full / bimodal_no_mode / trajectory_only / trajectory_mode model")
    p.add_argument("--resume")
    args = p.parse_args()
    train(load_config(args.config, args.set), resolve(args.resume) if args.resume else None)
