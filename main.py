from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
from textblob import TextBlob
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import glob


def read_data_from_xml(path):
    # search for abstracts (start with <ab>)
    abstracts = list()
    titles = list()

    with open(path) as f:
        for i, line in enumerate(tqdm(f, desc="Loading Abstract data from .XML file")):
            # get lines that contain the abstract (i.e. <ab> is in line)
            if "<ab>" in line:
                line = line.rstrip()
                line = line.replace("<ab>", "")
                line = line.replace("</ab>", "")
                abstracts.append(line)
            elif "<atl>" in line:
                line = line.rstrip()
                line = line.replace("<atl>", "")
                line = line.replace("</atl>", "")
                titles.append(line)

    return abstracts, titles


def send_to_blob(text, traces, count):
    "Replace these words"
    replace_list = ["isp.", "igen.", "etc.", "fm.", "i.e.", "sp."]
    for i in replace_list:
        text = text.replace(i, i[:-1])
        text = text.replace(i.capitalize(), i[:-1])
        text = text.replace(i.upper(), i[:-1])

    blob = TextBlob(text)
    sentences = blob.sentences

    # if sentence begins with 'abstract', remove it!
    if str(sentences[-1]).startswith("Abstract"):
        sentences.pop(-1)

    summaries = list()
    all_found = list()
    for i in sentences:
        found = list()
        f = False
        for j in traces:
            if j.lower() in str(i).lower():
                found.append(j)
                if f is False:
                    count += 1
                    f = True
        if len(found) == 0:
            continue
        all_found += found
        # noun phrases can tell you what the most important parts of each sentence is
        noun_phrases = i.noun_phrases
        # print(noun_phrases)
        summaries.append(list(noun_phrases))

    if len(all_found) == 0:
        return None, None, count, sentences

    return summaries, all_found, count, sentences


def co_occurrence(word_dict, traces):
    all_traces = list()

    for i in traces:
        for j in i:
            all_traces.append(str(j))
    traces = sorted(list(set(all_traces)))

    df = pd.DataFrame(data=np.zeros((len(traces), len(traces))), columns=traces, index=traces)
    word_dist_df = pd.DataFrame(data=np.zeros((len(traces), len(traces))), columns=traces, index=traces)
    for title in tqdm(word_dict, desc="Calculating co-occurrences"):
        for sentence in word_dict[title]:
            sentence_str = ""
            for word in sentence:
                if "ichnofacies" in word:
                    # exclude words that contain ichnofacies...
                    continue
                for char in word:
                    sentence_str += str(char)
                sentence_str += " "

            for i in traces:
                for j in traces:

                    if i.lower() in sentence_str and j.lower() in sentence_str:

                        df[i].loc[j] += int(1)
                        try:

                            i_index = sentence.index(i.lower())
                            j_index = sentence.index(j.lower())
                        except ValueError:
                            for word in sentence:
                                if i.lower() in word:
                                    i_index = sentence.index(word)
                                if j.lower() in word:
                                    j_index = sentence.index(word)

                        dist = abs(i_index - j_index)
                        word_dist_df[i].loc[j] = int(dist)


    df.to_csv("co_occurrences.csv", encoding="utf-8")

    word_dist_df = word_dist_df / df

    word_dist_df.to_csv("word_distances.csv", encoding="utf-8")


def co_occurrence_time(found, ft=1, graph=True):

    data = pickle.load(open("paragraphs.pkl", "rb"))

    environments = pd.read_excel(r"/Users/timmer/Desktop/Ichno_crawl/parameters/depositional_evironments.xlsx")

    environments_dict = dict()
    for i in environments.columns.values:
        vals = environments[i].copy(deep=True)
        vals.dropna(inplace=True, how='any')
        vals = vals.str.lower()
        vals = list(vals.values)
        environments_dict[i.lower()] = vals

    time_scale = {

        "Neogene": ["Neogene",
                    "Holocene",
                    "Pleistocene",
                    "Pliocene",
                    "Miocene",
                    "Gelasian",
                    "Piacenzian",
                    "Zanclian",
                    "Messinian",
                    "Tortonian",
                    "Serravallian",
                    "Langhian",
                    "Burdigalian",
                    "Aquitanian"
                    ],

        "Paleogene": ["Paleogene",
                      "Oligocene",
                      "Eocene",
                      "Paleocene",
                      "Chattian",
                      "Rupelian",
                      "Priabonian",
                      "Bartonian",
                      "Lutetian",
                      "Ypresian",
                      "Thanetian",
                      "Selandian",
                      "Danian"],

        "Cretaceous": ["Cretaceous",
                       "Maastrichtian",
                       "Campanian",
                       "Santonian",
                       "Coniacian",
                       "Turonian",
                       "Cenomanian",
                       "Albian",
                       "Aptian",
                       "Barremian",
                       "Hauterivian",
                       "Valanginian",
                       "Berriasian"],

        "Jurassic": ["Jurassic",
                     "Tithonian",
                     "Kimmeridgian",
                     "Oxfordian",
                     "Callovian",
                     "Bathonian",
                     "Bajocian",
                     "Aalenian",
                     "Toarcian",
                     "Pliensbachian",
                     "Sinemurian",
                     "Hettangian"],

        "Triassic": ["Triassic",
                     "Rhaetian",
                     "Norian",
                     "Carnian",
                     "Ladinian",
                     "Anisian",
                     "Olenekian",
                     "Induan"],

        "Permian": ["Permian",
                    "Lopingian",
                    "Guadalupian",
                    "Cisuralian",
                    "Changhsingian",
                    "Wuchiapingian",
                    "Capitanian",
                    "Wordian",
                    "Roadian",
                    "Kungurian",
                    "Artinskian",
                    "Sakmarian",
                    "Asselian"],

        "Carboniferous": ["Carboniferous",
                          "Pennsylvanian",
                          "Mississippian",
                          "Gzhelian",
                          "Kasimovian",
                          "Moscovian",
                          "Bashkirian",
                          "Serpukhovian",
                          "Viséan",
                          "Tournaisian"],

        "Devonian": ["Devonian",
                     "Famennian",
                     "Frasnian",
                     "Givetian",
                     "Eifelian",
                     "Emsian",
                     "Pragian",
                     "Lochkovian"],

        "Silurian": ["Silurian",
                     "Llandovery",
                     "Pridoli",
                     "Ludlow",
                     "Wenlock",
                     "Ludfordian",
                     "Gorstian",
                     "Homerian",
                     "Sheinwoodian",
                     "Telychian",
                     "Aeronian",
                     "Rhuddanian"],

        "Ordovician": ["Ordovician",
                       "Hirnantian",
                       "Katian",
                       "Sandbian",
                       "Darriwillian",
                       "Dapingian",
                       "Floian",
                       "Tremadoc"],

        "Cambrian": ["Cambrian",
                     "Furongian",
                     "Terreneuvian",
                     "Dolgellian",
                     "Paiban",
                     "Maentwrogian",
                     "Guzhangian",
                     "Drumian",
                     "Amgan",
                     "Botomian",
                     "Atdabanian",
                     "Tommotian",
                     "Fortunian",
                     "Nemakit-Daldynian",
                     "Manikayan",
                     "Manykajan"],

        "Proterozoic": ["Proterozoic",
                        "Neoproterozoic",
                        "Mesoproterozoic",
                        "Paleoproterozoic",
                        "Ediacaran",
                        "Cryogenian",
                        "Tonian",
                        "Stenian",
                        "Ectasian",
                        "Calymmian",
                        "Statherian",
                        "Orosirian",
                        "Rhyacian",
                        "Siderian"]
    }

    all_traces = list()

    for i in found:
        for j in i:
            all_traces.append(j)

    all_traces = list(set(all_traces))
    age_trace = dict()
    age_paragraphs = dict()

    all_envs = list()

    for paragraph in tqdm(data, desc="Acquiring Trace Fossil/Age Counts"):
        ages = list()
        traces = list()
        envs_list = list()
        found_age = False
        for sentence in paragraph:
            for a in time_scale:
                for sub in time_scale[a]:
                    if sub in sentence:
                        ages.append(a)
                        found_age = True
                        try:
                            age_paragraphs[a].append(paragraph)
                        except KeyError:
                            age_paragraphs[a] = [paragraph]
            for t in all_traces:
                if t in sentence:
                    traces.append(t)

            for envs in environments_dict:
                for e in environments_dict[envs]:
                    if e.lower() in str(sentence).lower():
                        envs_list.append(e)
        if found_age is False:
            continue
        if len(traces) == 0:
            continue

        all_envs += envs_list
        for a in list(set(ages)):
            try:
                age_trace[a] += list(set(traces))  # one trace per abstract age
            except KeyError:
                age_trace[a] = list(set(traces))  # one trace per abstract age

    age_co_occurrences(age_paragraphs, all_traces, all_envs, environments_dict)

    if graph is True:
        plot_trace_through_time(all_traces, time_scale, age_trace, ft)


def reversed_dict(environments_dict):
    reversed = dict()
    for k in environments_dict:
        for v in environments_dict[k]:
            reversed[v] = k
    return reversed


def age_co_occurrences(age_paragraph, all_traces, all_envs, environments_dict):

    rev_env = reversed_dict(environments_dict)
    # very nasty embedded for loops
    for age in age_paragraph:
        df = pd.DataFrame(data=np.zeros((len(all_traces), len(all_traces))), columns=all_traces, index=all_traces)
        envs_df =pd.DataFrame(data=np.zeros((len(all_traces), len(environments_dict.keys()))),
                              columns=environments_dict.keys(), index=all_traces)

        word_dist_df = pd.DataFrame(data=np.zeros((len(all_traces), len(all_traces))), columns=all_traces, index=all_traces)
        for paragraph in tqdm(age_paragraph[age], desc="Processing %s" % age):
            found_envs = list()
            found_traces = list()
            for sentence in paragraph:
                for env in all_envs:
                    if env in sentence.lower():
                        found_envs.append(env)
                        found_envs = list(set(found_envs))
                for i in all_traces:

                    for j in all_traces:
                        if i in sentence and j in sentence:
                            df[i].loc[j] += int(1)
                            # calculate distance
                            i_index = sentence.index(i)
                            j_index = sentence.index(j)
                            dist = abs(i_index - j_index)
                            word_dist_df[i].loc[j] = int(dist)
                            found_traces.append(i)
                            found_traces.append(j)
                            found_traces = list(set(found_traces))

            # match sub-environment to 'global' environment
            for i in found_envs:
                col = rev_env[i]
                for j in found_traces:
                    envs_df[col].loc[j] += 1

        # for t in envs_df.index.values:
        #
        #     total = df[t].loc[t]
        #     envs_df[t] = envs_df[t] / float(total)

        envs_df = envs_df.loc[(envs_df.sum(axis=1) > 0), (envs_df.sum(axis=0) > 0)]

        envs_df.to_csv("out/environment_similarity/%s.csv" % age, encoding="utf-8")

        # remove all empty entries
        df = df.loc[(df.sum(axis=1) > 0), (df.sum(axis=0) > 0)]
        word_dist_df = word_dist_df[df.columns.values]
        word_dist_df = word_dist_df / df
        word_dist_df.to_csv("out/word_distances/%s.csv" % age, encoding="utf-8")
        df = jaccard_similarity(df)
        df.to_csv("out/jaccard_similarity/%s.csv" % age, encoding="utf-8")


def jaccard_similarity(df):

    traces1 = df.columns.values
    traces2 = df.columns.values
    similarity = pd.DataFrame(np.zeros_like(df.values), columns=traces1, index=traces2)
    to_remove = list()
    for i in traces1:
        a = df[i].loc[i]
        # if a < 5:
        #     to_remove.append(i)
        if a == 0:
            continue

        for j in traces2:
            b = df[j].loc[j]
            a_and_b = df[i].loc[j]
            sim = a_and_b / (a + b - a_and_b)
            similarity[i].loc[j] = sim
    # filter out traces that were found less than 5 times
    similarity.drop(index=to_remove, columns=to_remove, inplace=True)
    return similarity


def plot_trace_through_time(all_traces, time_scale, age_trace, ft):
    plot_df = pd.DataFrame(columns=sorted(all_traces), index=list(time_scale.keys()))
    diversity = list()
    for a in tqdm(list(time_scale.keys()), desc="Plotting Mention plots"):
        traces = age_trace[a]
        individuals = sorted(list(set(traces)))
        diversity.append(len(individuals))
        counts = list()
        filtered_traces = list()
        for i in individuals:

            c = traces.count(i)

            if c > ft:
                counts.append(c)
                filtered_traces.append(i)
                plot_df[i].loc[a] = c
        # plt.figure(figsize=(8,10))
        # plt.barh(y=filtered_traces, width=counts, color='black')
        # plt.yticks(range(0, len(filtered_traces)), filtered_traces, fontsize=8)
        # plt.xlabel("Count (n. abstracts mentioning ichnogenera, showing > 1)")
        # plt.ylim([-.5, len(filtered_traces)-.5])
        # plt.title("%s Trace Fossil Mentions, Ichnogenera Diversity = %i" % (a, len(set(individuals))))
        # plt.tight_layout()
        # plt.savefig("count_plots/%s.pdf" % a)
    print("Plotting")
    plot_df[plot_df.values <= 2] = np.NaN
    plot_df.dropna(axis="columns", how="all", inplace=True)
    plot_df.to_csv("Mentions_through_time.csv")
    plot_df.fillna(0, inplace=True)

    # RGB colors taken from ICZN strat chart using OS X digital color meter
    colors = reversed([(253/255., 227/255., 23/255.),
                (248/255., 135/255., 64/255.),
                (111/255., 190/255., 62/255.),
                (45/255., 164/255., 188/255.),
                (108/255., 12/255., 127/255.),
                (233/255., 41/255., 31/255.),
                (84/255., 150/255., 133/255.),
                (184/255., 121/255., 39/255.),
                (164/255., 221/255., 166/255.),
                (30/255., 129/255., 93/255.),
                (110/255., 145/255., 69/255.),
                (240/255., 25/255., 80/255.)])
    plot_df = plot_df.iloc[::-1]  # reverse dataframe so that oldest is on the bottom.
    ax = plot_df.T.plot(kind='bar', stacked=True, color=colors, width=.8)
    ax.set_xticklabels(plot_df.columns.values, fontstyle="italic")
    ax.set_ylabel("Abstract presence count [n]")
    handles, labels = ax.get_legend_handles_labels()

    ax.legend(handles[::-1], labels[::-1], loc='upper left')

    plt.tight_layout()
    plt.show()


def process(abstracts_list, titles_list, trace_fossils):

    found_traces = list()
    processed_dict = dict()
    count = 0
    paragraph = list()
    for i, text in enumerate(tqdm(abstracts_list, desc="Extracting sentences from abstracts and matching")):
        title = titles_list[i]
        # split text into sentences
        noun_phrases, found, count, sentences = send_to_blob(text, trace_fossils, count)
        if noun_phrases is None:
            continue
        processed_dict[title] = noun_phrases
        found_traces.append(found)
        paragraph.append(sentences)

    pickle.dump([processed_dict, found_traces], open("dump.pkl", "wb"))
    pickle.dump(paragraph, open("paragraphs.pkl", "wb"))

    return [processed_dict, found_traces]


def main(load=False, tf=True, graph=False):
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
        return df

    if load is True:
        print("Loading parameter files")
        trace_fossils = load_parameters_list("parameters/trace_fossils.txt")
        abstracts_list, titles_list = read_data_from_xml("bea1e43b-9c5f-45e8-853b-83c7e77a121c.xml")
        process(abstracts_list, titles_list, trace_fossils)
    if tf is True:
        print("Calculating cooccurrences through time")
        data, found_traces = pickle.load(open("dump.pkl", "rb"))
        co_occurrence_time(found_traces)

    return


if __name__ == "__main__":
    main()

