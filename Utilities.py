import pandas as pd
import glob
import os
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

def filter_abundance(folder):
    files = glob.glob("%s/*.csv" % folder)


    for i in files:
        if "filtered" in i:
            os.remove(i)
            continue
        # data = pd.read_csv(i, index_col=0)
        # base = os.path.basename(i).split(".")[0]
        #
        # columns = list(data.columns.values).copy()
        # index = list(data.index.values).copy()
        #
        # for col in tqdm(columns, desc="Processing %s" % base):
        #
        #     for idx in index:
        #         if col == idx:
        #             val = data[col].loc[idx]
        #             print(val)
        #             if val < 5:
        #                 data.drop(columns=col, index=idx, inplace=True)
        # data.to_csv("%s/%s_filtered.csv" % (folder, base))


def to_percent(folder):
    files = glob.glob("%s/*.csv" % folder)
    for i in files:
        data = pd.read_csv(i)
        data = data / 10.

        data.to_csv(i)


def to_graph(folder):
    files = glob.glob("%s/*.csv" % folder)
    for i in tqdm(files, desc="Processing to gephi"):
        if "gephi" in i:
            continue
        base = i.split(".")[0]
        data = pd.read_csv(i, index_col=0)

        columns = data.columns.values
        indices = data.index.values

        with open(base+"_graph.csv", "w") as f:
            f.write("Source\tTarget\tType\tWeight\tLabel\n")
            # for j in range(0, 11):
            #     f.write("%.1f\tJaccard Similarity\tUndirected\t%f\n" % (j*.1, j*.1))
            for colour, j in enumerate(columns):
                for k in indices:
                    val = data[j].loc[k]
                    if val > 5:  # only keep directed values that are greater than .1
                        f.write("%s\t%s\tUndirected\t%f\t%.2f\n" % (j, k, val, val))









if __name__ == "__main__":
    # filter_abundance(folder="/Users/timmer/Desktop/Ichno_crawl/out/jaccard_similarity")
    to_graph(folder="/Users/timmer/Desktop/Ichno_crawl/out/jaccard_similarity")
