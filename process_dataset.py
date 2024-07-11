import argparse
from wav2midi.dataset import process_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inpath")
    parser.add_argument("outpath")
    args = parser.parse_args()

    process_dataset(args.inpath, args.outpath)
