import sys
import argparse
import pandas as pd
from utils.pdutils import stat
from utils.plotting import setLateXFonts
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from utils.runutils import str2bool
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot stats')
    parser.add_argument('--csvfile', metavar='csvfile', type=str,
                        help='CSVFILE')
    parser.add_argument('--hueorder', metavar='hueorder',
                    help='hueorder', type=lambda s: [item for item in s.split(',')])
    parser.add_argument('--split', metavar='split',
                    help='split training and validation figures into two pdf files', type=str2bool)
    args = parser.parse_args()
    df = pd.read_csv(args.csvfile)
    if any(["signal" in col for col in list(df)]):
        df = df.rename(columns={"signal":"loss"})
    setLateXFonts()
    basefilename = args.csvfile
    hue_order=None
    if args.hueorder is not None:
        hue_order = args.hueorder

    #else:
#        sns_plot = sns.lineplot(x="epoch", y="loss", hue="label", data=df)
    if args.split is not None:
        sns_plot = sns.lineplot(x="epoch", y="loss", hue="label",
                                data=df[df['label'].str.contains('.train')], hue_order=args.hueorder)
        plt.tight_layout()
        plt.show()
        fig = sns_plot.get_figure()
        fig.savefig(basefilename + "-training.pdf")

        plt.clf()
        plt.cla()
        plt.close()

        sns_plot = sns.lineplot(x="epoch", y="loss", hue="label",
                                data=df[df['label'].str.contains('.val')], hue_order=args.hueorder)
        plt.tight_layout()
        plt.show()
        fig = sns_plot.get_figure()
        fig.savefig(basefilename + "-validation.pdf")
    else:
        sns_plot = sns.lineplot(x="epoch", y="loss", hue="label",
                                data=df, hue_order=args.hueorder)
        plt.tight_layout()
        plt.show()
        fig = sns_plot.get_figure()
        fig.savefig(basefilename+"-all.pdf")

        sys.exit(0)

