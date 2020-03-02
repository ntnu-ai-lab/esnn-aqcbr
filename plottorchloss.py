import sys
import argparse
import pandas as pd
from utils.pdutils import stat
from utils.plotting import setLateXFonts
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot stats')
    parser.add_argument('--csvfile', metavar='csvfile', type=str,
                        help='CSVFILE')
    args = parser.parse_args()
    df = pd.read_csv(args.csvfile)
    if any(["signal" in col for col in list(df)]):
        df = df.rename(columns={"signal":"loss"})
    setLateXFonts()
    basefilename = args.csvfile
    sns_plot = sns.lineplot(x="epoch", y="loss", hue="label", data=df)
    plt.show()
    fig = sns_plot.get_figure()
    fig.savefig(basefilename+".pdf")
    sys.exit(0)
