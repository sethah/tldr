import argparse
import random
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default="")
    parser.add_argument('--save-path', type=str, default="")
    parser.add_argument('--test-frac', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    dataset_path = Path(args.dataset_path)
    save_path = Path(args.save_path)

    file_hashes = [f.stem for f in dataset_path.iterdir() if f.suffix == '.hdf5']
    random.shuffle(file_hashes)
    n_train = int(len(file_hashes) * (1 - args.test_frac))
    hashes = {'train': file_hashes[:n_train], 'test': file_hashes[n_train:]}

    for phase in ['train', 'test']:
        with open(save_path / f"{phase}.txt", "w") as f:
            for hash in hashes[phase]:
                f.write(str(dataset_path / Path(hash).with_suffix(".hdf5")) + "\n")


