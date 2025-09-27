import argparse, os
from pathlib import Path

def train(args):
    # ... your training code ...
    # Example: save checkpoints under args.ckpt_dir
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs+1):
        # train step ...
        ckpt_path = Path(args.ckpt_dir)/f"model_epoch_{epoch:03d}.pt"
        # torch.save(model.state_dict(), ckpt_path)
        print(f"[mock] saved {ckpt_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data", type=str)
    ap.add_argument("--ckpt_dir", default="checkpoints/exp1", type=str)
    ap.add_argument("--out_dir",  default="outputs/exp1", type=str)
    ap.add_argument("--epochs",   default=3, type=int)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
