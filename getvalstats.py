import sys
import argparse
import pandas as pd
from utils.pdutils import stat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='get stats')
    parser.add_argument('--csvfile', metavar='csvfile', type=str,
                        help='CSVFILE')
    parser.add_argument('--method', metavar='method', type=str,
                        help='which method to get stats from')
    parser.add_argument('--epoch', metavar='epoch', type=int,
                        help='Which epoch to get stats from')
    args = parser.parse_args()
    df = pd.read_csv(args.csvfile)
    if any(["signal" in col for col in list(df)]):
        df = df.rename(columns={"signal":"loss"})

    print(stat(df, args.epoch, args. method))
    sys.exit(0)
