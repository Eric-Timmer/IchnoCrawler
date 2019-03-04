from tqdm import tqdm
import pandas as pd
import numpy as np
import glob
import numba
from difflib import SequenceMatcher
import seaborn as sns
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
import pickle


def read_data_from_xml(path):
    # search for abstracts (start with <ab>)
    abstracts = list()
    titles = list()

    stop_words = list()
    with open("stop_words.txt") as f:
        for line in f:
            stop_words.append(line.rstrip())
    with open("special_characters.txt") as f:
        for line in f:
            stop_words.append(line.rstrip())

    with open(path) as f:

        for i, line in enumerate(tqdm(f, desc="Loading Abstract data from .XML file")):
            # get lines that contain the abstract (i.e. <ab> is in line)
            filtered_words = list()
            if "<ab>" in line:
                line = line.rstrip()
                line = line.replace("<ab>", "")
                line = line.replace("</ab>", "")
                line = line.replace(".", "")
                line = line.replace(",", "")
                words = line.split()
                for w in words:
                    w = w.lower()
                    if w not in stop_words:
                        filtered_words.append(w)
                abstracts.append(filtered_words)
            elif "<atl>" in line:
                line = line.rstrip()
                line = line.replace("<atl>", "")
                line = line.replace("</atl>", "")
                titles.append(line)

    return abstracts, titles


def abstract_word_distances(abstracts, environments, time_scale, trace_fossils, columns, titles):

    env_list = list(environments.values.flatten())
    env_list = filter(lambda v: v == v, env_list)
    search_words = list(environments.columns) + env_list + time_scale + trace_fossils
    sum_df = pd.DataFrame(0, columns=search_words, index=search_words)
    count_df = pd.DataFrame(0., columns=search_words, index=search_words)
    pa_df = pd.DataFrame(0, columns=search_words, index=search_words)
    for i, ab in enumerate(tqdm(abstracts, desc="Processing abstracts")):
        words_found = dict()
        for w in search_words:
            indices = np.where(np.array(ab) == w.lower())[0]
            if len(indices):
                words_found[w] = indices

        # count distances and presence/absence in words found
        for k1 in words_found:
            indices1 = words_found[k1]
            a = k1  # temp place holders a and b
            if k1 in env_list:
                col = np.where(environments.values == k1)[1][0]
                a = environments.columns[col]
            pa_df.loc[a, a] += 1
            for k2 in words_found:
                b = k2
                if k2 in env_list:
                    col = np.where(environments.values == k2)[1][0]
                    b = environments.columns[col]

                if k1 != k2 and a != b:
                    indices2 = words_found[k2]
                    count = 0
                    d = 0
                    for j1 in indices1:
                        for j2 in indices2:
                            d += abs(j1-j2) - 1  # distance between the two words
                            d = float(d)
                            count += 1.

                    sum_df.loc[a, b] += d
                    count_df.loc[a, b] += float(count)
                    pa_df.loc[a, b] += 1

    np.seterr(divide="ignore")

    avg = sum_df.values / count_df.values
    df = pd.DataFrame(avg, columns=search_words, index=search_words)
    df.drop(env_list, axis=0, inplace=True)
    df.drop(env_list, axis=1, inplace=True)
    df.dropna(axis=0, how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    pa_df.drop(env_list, axis=0, inplace=True)
    pa_df.drop(env_list, axis=1, inplace=True)
    pa_df.dropna(axis=0, how="all", inplace=True)
    pa_df.dropna(axis=1, how="all", inplace=True)

    df.to_csv("word_distances.csv")
    pa_df.to_csv("presence_absence.csv")


def abstract_word_matcher(abstracts, environments, time_scale, trace_fossils, columns):

    df = pd.DataFrame(columns=columns)
    print(df)
    print("Checking abstracts for key words")
    for i, ab in enumerate(tqdm(abstracts)):
        df.loc[i, "Abstract"] = ab
        for group in environments.columns:
            for sub_env in environments[group]:
                sub_env = str(sub_env)

                if str(sub_env.lower()) in ab:
                    df.loc[i, group] = 1
                    break

        for t in time_scale:
            if t.lower() in ab:
                df.loc[i, t] = 1

        for t in trace_fossils:
            if t.lower() in ab:
                df.loc[i, t] = 1
    df.to_csv("matched_abstracts.csv")


def paper_word_matcher(papers, environments, time_scale, trace_fossils, columns):

    df = pd.DataFrame(columns=columns)
    print("Checking papers for key words")
    for i, pp in enumerate(tqdm(papers)):
        with open(pp) as f:
            for line in f:
                words = line.split()

                for w in words:
                    for group in environments.columns:
                        for sub_env in environments[group]:
                            sub_env = str(sub_env)

                            match = SequenceMatcher(None, sub_env.lower(), w.lower()).ratio()
                            if match >= 0.9:
                                df.loc[i, group] = 1
                                break

                    for t in time_scale:
                        match = SequenceMatcher(None, t.lower(), w.lower()).ratio()
                        if match >= 0.9:
                            df.loc[i, t] = 1

                    for t in trace_fossils:
                        match = SequenceMatcher(None, t.lower(), w.lower()).ratio()
                        if match >= 0.9:
                            df.loc[i, t] = 1
    df.to_csv("matched_papers.csv")


def remove_stop_words(ichno_papers_list, stop_words_path, special_characters_path, citations):

    stop_words = list()
    with open(stop_words_path) as f:
        for line in f:
            stop_words.append(line.rstrip())
    special_characters = list()
    with open(special_characters_path) as f:
        for line in f:
            special_characters.append(line.rstrip())

    print(special_characters)
    paper_data = dict()
    for i in tqdm(ichno_papers_list):
        temp = list()

        citations_found = False
        with open(i) as f:
            for line in f:
                # split line into words
                words = line.split()
                # iterate through words
                for j in words:
                    # convert each word to lower case
                    j = j.lower()
                    # if the word is not in the stop words list, append to temp
                    if j in stop_words:
                        continue
                    # if the word is one of the "ending words", break out
                    if j in citations:
                        citations_found = True
                        break
                    # if the word contains one of the 'special characters', examine further
                    found_special = False
                    for k in special_characters:
                        if k in j:
                            if k == "," or k == ".":
                                j = j.split(k)[0]
                                if k in j:
                                    found_special = True
                                    break
                            else:
                                found_special = True
                                break

                    if found_special is True:
                        continue
                    if len(j) <= 2:
                        continue

                    # if the word has passed all of these tests, append to temp
                    temp.append(j)
                if citations_found is True:
                    break
        # these are all the words in here. Now search this for matches


        paper_data[i] = temp
    pickle.dump(paper_data, open("paper_data.pkl", "wb"))



    return


def find_words_and_distances(pickled_papers, environments, time_scale, trace_fossils, columns, titles):
    data = pickle.load(open(pickled_papers, "rb"))
    s = 0
    df = pd.DataFrame(columns=columns, index=data.keys())
    for i in tqdm(data):
        s += (len(data[i]))
        for word in data[i]:
            print(word)





    print(s / float(len(data)))


def process_data(path, time_scale, trace_fossils):
    df = pd.read_csv(path)

    groups = ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]

    # for g in tqdm(groups):
    #     subset_env = df[df[g] == 1]
    #     trace_matrix_g = pd.DataFrame(columns=trace_fossils, index=trace_fossils)
    #     for t in tqdm(time_scale):
    #         subset_time = subset_env[subset_env[t] == 1]
    #         trace_matrix = pd.DataFrame(columns=trace_fossils, index=trace_fossils)
    #
    #         # TODO major bug here...only calculates diagonals
    #         temp = list()
    #         for i in trace_fossils:
    #             if i in subset_time.columns:
    #                 temp.append(i)
    #         trace_fossils = temp
    #         print(trace_fossils)
    #
    #         for t1 in trace_fossils:
    #             for t2 in trace_fossils:
    #                 t = subset_time[subset_time[t1] == subset_time[t2]]
    #
    #                 total = t[t1].sum()
    #                 if total > 0:
    #                     trace_matrix.loc[t1, t2] = total
    #                     trace_matrix.loc[t2, t1] = total
    #                 else:
    #                     print(t1, t2)
    #
    #         p = "out/time/env/%s_%s" % (g, t)
    #         trace_matrix.to_csv(p+".csv")
    #         plot_data(p, trace_matrix)
    #
    #     for t1 in trace_fossils:
    #         for t2 in trace_fossils:
    #             t = subset_env[subset_env[t1] == subset_env[t2]]
    #             total = t[t1].sum()
    #             trace_matrix_g.loc[t1, t2] = total
    #             trace_matrix_g.loc[t2, t1] = total
    #
    #     p = "out/env/%s" % g
    #     trace_matrix_g.to_csv(p+".csv")
    #     plot_data(p, trace_matrix_g)

    for t in tqdm(time_scale):
        subset_time = df[df[t] == 1]
        trace_matrix = pd.DataFrame(columns=trace_fossils, index=trace_fossils)

        for t1 in tqdm(trace_fossils):
            for t2 in trace_fossils:
                total = subset_time[subset_time[t1] == subset_time[t2]]
                total = total[t1].sum()
                trace_matrix.loc[t1, t2] = total
                trace_matrix.loc[t2, t1] = total

        p = "out/time/%s" % t
        trace_matrix.to_csv(p+".csv")
        plot_data(p, trace_matrix)


def process_data(path, time_scale, environments, trace_fossils):
    df = pd.read_csv(path)

    for i in time_scale:
        time_subset = df[df[i] == 1]
        for j in tqdm(environments, desc=i):
            association_df = pd.DataFrame(0, columns=trace_fossils, index=trace_fossils)
            environment_subset = time_subset[time_subset[j] == 1]

            traces_present = environment_subset[trace_fossils]

            for index, row in traces_present.iterrows():
                row.dropna(inplace=True)
                # dropped na values in k
                if len(row) == 0:
                    continue
                for t1 in row.index:  # find the trace names still left
                    for t2 in row.index:  # find the trace names still left
                        association_df.loc[t1, t2] += 1

            # drop empty rows/columns
            association_df.replace(0, np.NaN, inplace=True)
            association_df.dropna(axis=0, how="all", inplace=True)
            association_df.dropna(axis=1, how="all", inplace=True)
            association_df.fillna(0, inplace=True)
            dir = "out/paper_data/%s/" % i
            if not os.path.exists(dir):
                os.makedirs(dir)
            association_df.to_csv(dir+j+".csv")

        # traces_present = df[df[i] == 1]
        # for index, row in traces_present.iterrows():
        #     row.dropna(inplace=True)
        #     # dropped na values in k
        #     if len(row) == 0:
        #         continue
        #     for t1 in row.index:  # find the trace names still left
        #         for t2 in row.index:  # find the trace names still left
        #             if t1 in trace_fossils and t2 in trace_fossils:
        #                 association_df.loc[t1, t2] += 1
        #
        # # drop empty rows/columns
        # association_df.replace(0, np.NaN, inplace=True)
        # association_df.dropna(axis=0, how="all", inplace=True)
        # association_df.dropna(axis=1, how="all", inplace=True)
        # association_df.fillna(0, inplace=True)
        # dir = "out/abstract_data/%s/" % i
        # if not os.path.exists(dir):
        #     os.makedirs(dir)
        # association_df.to_csv(dir + j + ".csv")


def plot_data(path, df):
    print("Plotting: %s" % path)
    df.dropna(inplace=True, axis=1, how="all")
    df.dropna(inplace=True, axis=0, how="all")
    df.fillna(0, inplace=True)

    mask = df.where(df ==0, other=False)
    mask = mask.where(df > 0, other=True)

    ax = sns.heatmap(df,
                     mask=mask,
                     # cmap=sns.light_palette("green", 15, reverse=True),
                     annot=True,
                     # linecolor="gray",
                     # linewidth=.05,
                     square=True,
                     annot_kws={"size": 2},
                     xticklabels=df.columns.values,
                     yticklabels=df.columns.values
                     )
    ax.xaxis.tick_top()
    plt.xticks(rotation="vertical", fontsize=3)
    plt.yticks(fontsize=3)
    plt.tight_layout()
    # plt.title(os.path.basename(path), y=1.1)
    plt.savefig(path+"test.pdf")
    # plt.show()
    plt.clf()


    return


def main(load_abstract_data=False, plot_abstract_data=False, load_paper_data=False, cluster_data=True):
    def load_parameters_list(path):
        par = list()
        with open(path) as f:
            for line in f:
                line = line.rstrip()
                if len(line) > 0:
                    par.append(line)
        return par

    def load_parameters_df(path):
        df = pd.read_csv(path, delimiter="\t")
        print(df)
        return df

    print("Loading parameter files")
    environments = load_parameters_df("environments.txt")
    time_scale = load_parameters_list("time_scale.txt")
    trace_fossils = load_parameters_list("trace_fossils.txt")

    # set up dataframe columns
    columns = ["Paper"]
    columns += list(environments.columns)
    columns += time_scale
    columns += trace_fossils
    if load_abstract_data is True:
        # print("Loading abstract data")
        abstracts, titles = read_data_from_xml("bea1e43b-9c5f-45e8-853b-83c7e77a121c.xml")
        abstract_word_distances(abstracts, environments, time_scale, trace_fossils, columns, titles)
    if plot_abstract_data is True:
        p = "out/abstract_data/"
        dist_df = pd.read_csv("word_distances.csv", index_col=0, header=0, encoding="utf-8")
        pa_df = pd.read_csv("presence_absence.csv", index_col=0, header=0,encoding="utf-8")

        print(dist_df.size)
        for i in pa_df.columns:
            if pa_df.loc[i, i] < 10:
                try:
                    dist_df.drop(index=i, axis=0, inplace=True)
                    dist_df.drop(columns=i, axis=1, inplace=True)
                except:
                    pass
        dist_df = dist_df[dist_df <= 300]
        print(dist_df.size)
        plot_data(p, dist_df)
    if __name__ == '__main__':
        if cluster_data is True:
            dist_df = pd.read_csv("word_distances.csv", index_col=0, header=0, encoding="utf-8")
            pa_df = pd.read_csv("presence_absence.csv", index_col=0, header=0, encoding="utf-8")


            # normalize data

            """
            For each depositional environment, for each time-scale, do this...
            """

            # TODO normalize the data...seprate into one per time_period...normalize there, cluster..

            for i in time_scale:
                temp_d = dist_df[pa_df.loc[i] > 0]
                temp_pa = pa_df[pa_df.loc[i] > 0]

                # remove extra filtered columns
                d = set(dist_df.index) - set(temp_d.index)
                temp_d.drop(d, axis=1, inplace=True)
                temp_pa.drop(d, axis=1, inplace=True)

                # remove environments
                d2 = set(environments.columns) - set(temp_d.index)
                d2 = set(environments.columns) - d2
                temp_d.drop(d2, axis=1, inplace=True)
                temp_pa.drop(d2, axis=1, inplace=True)
                temp_d.drop(d2, axis=0, inplace=True)
                temp_pa.drop(d2, axis=0, inplace=True)


                # remove time scales
                d3 = set(time_scale) - set(temp_d.index)
                d3 = set(time_scale) - d3
                temp_d.drop(d3, axis=1, inplace=True)
                temp_d.drop(d3, axis=0, inplace=True)
                temp_pa.drop(d3, axis=1, inplace=True)
                temp_pa.drop(d3, axis=0, inplace=True)
                normalized_dist = (temp_d - temp_d.min()) / (temp_d.max() - temp_d.min())
                normalized_pa = (temp_pa - temp_pa.min()) / (temp_pa.max() - temp_pa.min())

                plot_data(path="out/abstract_data/%s/distance_"%i, df=normalized_dist)
                plot_data(path="out/abstract_data/%s/pa_" % i, df=normalized_pa)


                # NOW...EVERYTHING IS REMOVED EXCEPT FOR THE IMPORTANT STUFF.


    if load_paper_data is True:
        print("Loading paper data")
        papers = glob.glob("/home/timmer/Desktop/output_ichnoPapers/*.txt")
        citations = ["citations", "references", "work cited", "acknowledgements", "conclusion", "conclusions", "summary"]
        # remove_stop_words(papers, "stop_words.txt", "special_characters.txt", citations)
        pickled_papers = "paper_data.pkl"
        find_words_and_distances(pickled_papers, environments, time_scale, trace_fossils, columns)

        paper_word_matcher(papers, environments, time_scale, trace_fossils, columns)



if __name__ == "__main__":
    main()
