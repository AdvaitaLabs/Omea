import argparse
import torch

def load_checkpoint(path):
    with torch.serialization.safe_globals([argparse.Namespace]):
        return torch.load(path,map_location='cpu',weights_only=True)
