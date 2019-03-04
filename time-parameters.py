import glob
import pandas as pd
import networkx as nx
from networkx.algorithms.community import quality, performance, coverage, girvan_newman
from networkx.algorithms import clique
import numpy as np
import matplotlib.pyplot as plt


def get_time_scale_data(environment_data, jaccard_data, time_scale):
    fist_appearance_dict = get_first_appearance(jaccard_data, time_scale)
    last_appearance_dict = get_last_appearance(jaccard_data, time_scale)

    time_scale = list(reversed(time_scale))

    last_j = None

    all_communities = list()
    all_n_ichnogenera = list()
    all_different_ichnogenera = list()
    all_n_fa = list()
    all_n_la = list()
    all_env_proportion = dict()

    for i, t in enumerate(time_scale):
        env_proportion = dict()

        e_data = None
        j_data = None
        for e in environment_data:
            if t in e and "graph" not in e:
                e_data = pd.read_csv(e, index_col=0)
                break
        for j in jaccard_data:
            if t in j and "graph" not in j:
                j_data = pd.read_csv(j, index_col=0)
                break

        if e_data is None:
            raise UserWarning("Environment data for %s not found" % t)
        if j_data is None:
            raise UserWarning("Jaccard similarity data for %s not found" % t)

        n_ichnogenera = len(j_data.index.values)

        # create relative ratio curve...for environments
        e_data_sum = float(e_data.values.sum())
        for env in e_data.columns.values:
            ratio = e_data[env].sum() / e_data_sum
            env_proportion[env] = ratio

        if i == 0:
            different_ichnogenera = 0
            last_j = j_data.copy(deep=True)
        else:
            different_ichnogenera = len(set(j_data.index.values) - set(last_j.index.values))
            last_j = j_data.copy(deep=True)

        # CALCULATE GRAPH PROPERTIES FOR EACH AGE (e.g. n. communities, clustering...)
        G = nx.from_pandas_adjacency(j_data)
        # remove self loops
        G.remove_edges_from(G.selfloop_edges())

        # # compute clustering
        # avg_clustering = nx.average_clustering(G, count_zeros=False)
        # print("Average Clustering: %f" % avg_clustering)

        # find communities
        communities_generator = girvan_newman(G)
        # top_level_communities = next(communities_generator)
        next_level_communities = next(communities_generator)
        communities = sorted(map(sorted, next_level_communities))

        # # assess the quality of partitions
        # p = performance(G, communities)
        # c = coverage(G, communities)
        # cliques = list(clique.find_cliques(G))

        n_la = len(last_appearance_dict[t])
        n_fa = len(fist_appearance_dict[t])
        print("%s: communities: %i, diversity: %i, different ichnogenera: %i, nfa: %i, nla: %i" % (t, len(communities), n_ichnogenera, different_ichnogenera, n_fa, n_la))
        print(env_proportion)

        all_communities.append(len(communities))
        all_n_ichnogenera.append(n_ichnogenera)
        all_different_ichnogenera.append(different_ichnogenera)
        all_n_fa.append(n_fa)
        all_n_la.append(n_la)

        for j in env_proportion:
            try:
                all_env_proportion[j].append(env_proportion[j])
            except KeyError:
                all_env_proportion[j] = [env_proportion[j]]


    plot_data(time_scale,
              all_communities,
              all_n_ichnogenera,
              all_different_ichnogenera,
              all_n_fa,
              all_n_la,
              all_env_proportion)


def plot_data(time, communities, diversity, different_ichnogenera, n_fa, n_la, env_proportion):

    fig, axes = plt.subplots(nrows=1, ncols=6, sharey='row')
    n_la[-1] = np.NaN
    different_ichnogenera[0] = np.NaN

    color = "darkslategrey"


    print(time, communities)
    axes[0].plot(communities, time, linewidth=3, color=color)
    axes[0].set_xlabel("n. Communities")
    axes[0].grid(True)
    axes[0].set_ylim([0, len(time) - 1])

    axes[1].plot(diversity, time, linewidth=3, color=color)
    axes[1].set_xlabel("n. Ichnogenera")
    axes[1].grid(True)
    axes[1].set_ylim([0, len(time) - 1])



    axes[2].plot(different_ichnogenera, time, linewidth=3, color=color)
    axes[2].set_xlabel("Different Ichnogenera \n (from previous time)")
    axes[2].grid(True)
    axes[2].set_ylim([0, len(time) - 1])



    axes[3].plot(n_fa, time, linewidth=3, color=color)
    axes[3].set_xlabel("n. First Mentions")
    axes[3].grid(True)
    axes[3].set_ylim([0, len(time) - 1])



    axes[4].plot(n_la, time, linewidth=3, color=color)
    axes[4].set_xlabel("n. Last Mentions")
    axes[4].grid(True)
    axes[4].set_ylim([0, len(time) - 1])

    environment_color_dict = {'glacial': "oldlace",
                              'desert': "coral",
                              'lake': "mediumspringgreen",
                              'fluvial': "navy",
                              'estuarine': "sandybrown",
                              'deltaic': "royalblue",
                              'coastal': "gold",
                              'shelf': "seagreen",
                              'continental margin': "slategray",
                              'ocean basin': "darkslategray",
                              'carbonate': "paleturquoise"}

    left = [0]*len(time)
    for env in env_proportion:
        print(env)
        axes[5].barh(time, env_proportion[env], left=left, label=env, color=environment_color_dict[env], height=1.)

        for i, val in enumerate(env_proportion[env]):
            left[i] += val
    axes[5].set_xlim([0, 1])

    # Shrink current axis by 20%
    box = axes[5].get_position()

    # Put a legend to the right of the current axis
    axes[5].legend(loc='center left', bbox_to_anchor=(1, 0.5))




    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()



def get_first_appearance(jaccard_data, time_scale):

    time_trace_dict = dict()

    time_scale = list(reversed(time_scale))

    for i, t in enumerate(time_scale):
        time_trace_dict[t] = []
        traces = []
        for j in jaccard_data:
            if t in j and "graph" not in j:
                j_data = pd.read_csv(j, index_col=0)
                traces = j_data.index.values
                break
        if len(traces) == 0:
            continue
        if t == 0:
            for j in traces:
                time_trace_dict[t].append(j)
        else:
            for j in traces:
                found_trace = False
                for k in time_trace_dict:
                    for j_last in time_trace_dict[k]:
                        if j == j_last:
                            found_trace = True
                if found_trace is False:
                    time_trace_dict[t].append(j)

    return time_trace_dict


def get_last_appearance(jaccard_data, time_scale):
    time_trace_dict = dict()

    # don't reverse time for this function...

    for i, t in enumerate(time_scale):
        time_trace_dict[t] = []
        traces = []
        for j in jaccard_data:
            if t in j and "graph" not in j:
                j_data = pd.read_csv(j, index_col=0)
                traces = j_data.index.values
                break
        if len(traces) == 0:
            continue
        if t == 0:
            for j in traces:
                time_trace_dict[t].append(j)
        else:
            for j in traces:
                found_trace = False
                for k in time_trace_dict:
                    for j_last in time_trace_dict[k]:
                        if j == j_last:
                            found_trace = True
                if found_trace is False:
                    time_trace_dict[t].append(j)

    return time_trace_dict


def changing_environments(environment_data, time_scale):

    """
    Go through each environment sheet, and figure out which trace fossils
    are found in 'new' environments compared to last time, and how many are not found in
    environment anymore, and also how many have changed major environment...
    :param environment_data:
    :param time_scale:
    :return:
    """
    return


if __name__ == "__main__":

    environment_data = glob.glob("/Users/timmer/Desktop/Ichno_crawl/out/environment_similarity/*.csv")
    jaccard_data = glob.glob("/Users/timmer/Desktop/Ichno_crawl/out/jaccard_similarity/*.csv")


    time_scale = ["Neogene", "Paleogene", "Cretaceous", "Jurassic", "Triassic", "Permian", "Carboniferous",
                  "Devonian", "Silurian", "Ordovician", "Cambrian", "Proterozoic"]

    get_time_scale_data(environment_data, jaccard_data, time_scale)
