from scipy import sparse
from collections import OrderedDict, defaultdict
import numpy as np
from typing import List, Dict, Tuple

WORD = 0
TAG = 1


class FeatureStatistics:
    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        feature_dict_list = ["f100", "f101", "f102", "f103", "f104", "f105", "f106", "f107", "f108", "f109", "f110",
                             "f111"]  # the feature classes used in the code
        self.feature_rep_dict = {fd: OrderedDict() for fd in feature_dict_list}
        '''
        A dictionary containing the counts of each data regarding a feature class. For example in f100, would contain
        the number of times each (word, tag) pair appeared in the text.
        '''
        self.tags = set()  # a set of all the seen tags
        self.tags.add("~")
        self.tags.add("*")
        self.tags_counts = defaultdict(int)  # a dictionary with the number of times each tag appeared in the text
        self.words_count = defaultdict(int)  # a dictionary with the number of times each word appeared in the text
        self.histories = []  # a list of all the histories seen at the test

    def get_word_tag_pair_count(self, file_path) -> None:
        """
            Extract out of text all word/tag pairs
            @param: file_path: full path of the file to read
            Updates the histories list
        """
        with open(file_path) as file:
            for line in file:
                if line[-1:] == "\n":
                    line = line[:-1]
                split_words = line.split(' ')
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_words[word_idx].split('_')

                    # Set previous values based on current index
                    if word_idx == 0:
                        p_word, p_tag, pp_word, pp_tag = ['*', '*', '*', '*']
                    else:
                        p_word, p_tag = split_words[word_idx - 1].split('_')
                        if word_idx == 1:
                            pp_word, pp_tag = ['*', '*']
                        else:
                            pp_word, pp_tag = split_words[word_idx - 2].split('_')

                    # Set next word based on current index
                    if word_idx == len(split_words) - 1:
                        n_word = "~"
                    else:
                        n_word = split_words[word_idx + 1].split('_')[0]

                    self.tags.add(cur_tag)
                    self.tags_counts[cur_tag] += 1
                    self.words_count[cur_word] += 1

                    # f100:
                    if (cur_word, cur_tag) not in self.feature_rep_dict["f100"]:
                        self.feature_rep_dict["f100"][(cur_word, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f100"][(cur_word, cur_tag)] += 1

                    len_cur_word = len(cur_word)
                    for i in range(1, 5):
                        if len_cur_word > i + 1:
                            suff = cur_word[len_cur_word - i:]
                            pre = cur_word[:i]
                            # f101
                            if (suff, cur_tag) not in self.feature_rep_dict["f101"]:
                                self.feature_rep_dict["f101"][(suff, cur_tag)] = 1
                            else:
                                self.feature_rep_dict["f101"][(suff, cur_tag)] += 1
                            # f102
                            if (pre, cur_tag) not in self.feature_rep_dict["f102"]:
                                self.feature_rep_dict["f102"][(pre, cur_tag)] = 1
                            else:
                                self.feature_rep_dict["f102"][(pre, cur_tag)] += 1
                    # f103:
                    if (pp_tag, p_tag, cur_tag) not in self.feature_rep_dict["f103"]:
                        self.feature_rep_dict["f103"][(pp_tag, p_tag, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f103"][(pp_tag, p_tag, cur_tag)] += 1

                    # f104:
                    if (p_tag, cur_tag) not in self.feature_rep_dict["f104"]:
                        self.feature_rep_dict["f104"][(p_tag, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f104"][(p_tag, cur_tag)] += 1

                    # f105:
                    if (cur_tag,) not in self.feature_rep_dict["f105"]:
                        self.feature_rep_dict["f105"][(cur_tag,)] = 1
                    else:
                        self.feature_rep_dict["f105"][(cur_tag,)] += 1

                    # f106:
                    if (p_word, cur_tag) not in self.feature_rep_dict["f106"]:
                        self.feature_rep_dict["f106"][(p_word, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f106"][(p_word, cur_tag)] += 1

                    # f107:
                    if (n_word, cur_tag) not in self.feature_rep_dict["f107"]:
                        self.feature_rep_dict["f107"][(n_word, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f107"][(n_word, cur_tag)] += 1

                    # f108:
                    if any(char.isdigit() for char in cur_word):
                        if (cur_word, cur_tag) not in self.feature_rep_dict["f108"]:
                            self.feature_rep_dict["f108"][(cur_word, cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f108"][(cur_word, cur_tag)] += 1
                    # f109:
                    if any(char.isupper() for char in cur_word):
                        if (cur_word, cur_tag) not in self.feature_rep_dict["f109"]:
                            self.feature_rep_dict["f109"][(cur_word, cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f109"][(cur_word, cur_tag)] += 1

                    # f110:
                    if (p_tag == 'NN' or p_tag == 'DT' or p_tag == 'VB') and  cur_tag == 'JJ':
                        if (cur_word, cur_tag) not in self.feature_rep_dict["f110"]:
                            self.feature_rep_dict["f110"][(cur_word, cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f110"][(cur_word, cur_tag)] += 1
                    # f111:
                    if (len_cur_word, cur_tag) not in self.feature_rep_dict["f111"]:
                        self.feature_rep_dict["f111"][(len_cur_word, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f111"][(len_cur_word, cur_tag)] += 1




                sentence = [("*", "*"), ("*", "*")]
                for pair in split_words:
                    sentence.append(tuple(pair.split("_")))
                sentence.append(("~", "~"))

                for i in range(2, len(sentence) - 1):
                    history = (
                        sentence[i][0], sentence[i][1], sentence[i - 1][0], sentence[i - 1][1], sentence[i - 2][0],
                        sentence[i - 2][1], sentence[i + 1][0])

                    self.histories.append(history)


class Feature2id:
    def __init__(self, feature_statistics: FeatureStatistics, threshold: int):
        """
        @param feature_statistics: the feature statistics object
        @param threshold: the minimal number of appearances a feature should have to be taken
        """
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        self.feature_to_idx = {
            "f100": OrderedDict(), "f101": OrderedDict(), "f102": OrderedDict(), "f103": OrderedDict(),
            "f104": OrderedDict(), "f105": OrderedDict(), "f106": OrderedDict(), "f107": OrderedDict(),
            "f108": OrderedDict(),  "f109": OrderedDict(), "f110": OrderedDict(), "f111": OrderedDict(),

        }
        self.represent_input_with_features = OrderedDict()
        self.histories_matrix = OrderedDict()
        self.histories_features = OrderedDict()
        self.small_matrix = sparse.csr_matrix
        self.big_matrix = sparse.csr_matrix

    def get_features_idx(self) -> None:
        """
        Assigns each feature that appeared enough time in the train files an idx.
        Saves those indices to self.feature_to_idx
        """

        for feat_class in self.feature_statistics.feature_rep_dict:
            if feat_class not in self.feature_to_idx:
                continue
            for feat, count in self.feature_statistics.feature_rep_dict[feat_class].items():
                if count >= self.threshold:
                    self.feature_to_idx[feat_class][feat] = self.n_total_features
                    self.n_total_features += 1
        print(f"you have {self.n_total_features} features!")

    def calc_represent_input_with_features(self) -> None:
        """
        initializes the matrices used in the optimization process - self.big_matrix and self.small_matrix
        """
        big_r = 0
        big_rows = []
        big_cols = []
        small_rows = []
        small_cols = []
        for small_r, hist in enumerate(self.feature_statistics.histories):

            for c in represent_input_with_features(hist, self.feature_to_idx):
                small_rows.append(small_r)
                small_cols.append(c)

            for r, y_tag in enumerate(self.feature_statistics.tags):
                demi_hist = (hist[0], y_tag, hist[2], hist[3], hist[4], hist[5], hist[6])
                self.histories_features[demi_hist] = []

                for c in represent_input_with_features(demi_hist, self.feature_to_idx):
                    big_rows.append(big_r)
                    big_cols.append(c)
                    self.histories_features[demi_hist].append(c)
                big_r += 1

        self.big_matrix = sparse.csr_matrix((np.ones(len(big_rows)), (np.array(big_rows), np.array(big_cols))),
                                            shape=(len(self.feature_statistics.tags) * len(
                                                self.feature_statistics.histories), self.n_total_features),
                                            dtype=bool)

        self.small_matrix = sparse.csr_matrix(
            (np.ones(len(small_rows)), (np.array(small_rows), np.array(small_cols))),
            shape=(len(
                self.feature_statistics.histories), self.n_total_features), dtype=bool)


def represent_input_with_features(history: Tuple, dict_of_dicts: Dict[str, Dict[Tuple[str, str], int]])\
        -> List[int]:
    """
        Extract feature vector in per a given history
        @param history: tuple{c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word}
        @param dict_of_dicts: a dictionary of each feature and the index it was given
        @return a list with all features that are relevant to the given history
    """
    c_word = history[0]
    c_tag = history[1]
    p_word = history[2]
    p_tag = history[3]
    pp_word = history[4]
    pp_tag = history[5]
    n_word = history[6]
    features = []
    len_cur_word = len(c_word)

    # f100
    if (c_word, c_tag) in dict_of_dicts["f100"]:
        features.append(dict_of_dicts["f100"][(c_word, c_tag)])

    for i in range(1, 5):  # Taking all suffixes and prefixes in length <= 4
        if len_cur_word >= i:  # Calculating suff and pre only if it reasonable.
            suff = c_word[len_cur_word - i:]
            pre = c_word[:i]

            # f101
            if (suff, c_tag) in dict_of_dicts["f101"]:
                features.append(dict_of_dicts["f101"][(suff, c_tag)])

            # f02
            if (pre, c_tag) in dict_of_dicts["f102"]:
                features.append(dict_of_dicts["f102"][(pre, c_tag)])

    # f103
    if (pp_tag, p_tag, c_tag) in dict_of_dicts["f103"]:
        features.append(dict_of_dicts["f103"][(pp_tag, p_tag, c_tag)])

    # f104
    if (p_tag, c_tag) in dict_of_dicts["f104"]:
        features.append(dict_of_dicts["f104"][(p_tag, c_tag)])

    # f105
    if (c_tag, ) in dict_of_dicts["f105"]:
        features.append(dict_of_dicts["f105"][(c_tag,)])

    # f106
    if (p_word, c_tag) in dict_of_dicts["f106"]:
        features.append(dict_of_dicts["f106"][(p_word, c_tag)])

    # f107
    if (n_word, c_tag) in dict_of_dicts["f107"]:
        features.append(dict_of_dicts["f107"][(n_word, c_tag)])

    # f108 - captures numbers:
    if any(char.isdigit() for char in c_word):
        if (c_word, c_tag) in dict_of_dicts["f108"]:
            features.append(dict_of_dicts["f108"][(c_word, c_tag)])

    # f109
    if any(char.isupper() for char in c_word):
        if (c_word, c_tag) in dict_of_dicts["f109"]:
            features.append(dict_of_dicts["f109"][(c_word, c_tag)])

    # f110
    if (c_word, c_tag) in dict_of_dicts["f110"]:
        features.append(dict_of_dicts["f110"][(c_word, c_tag)])
    # f111
    if (len_cur_word, c_tag) in dict_of_dicts["f111"]:
        features.append(dict_of_dicts["f111"][(len_cur_word, c_tag)])





    return features


def preprocess_train(train_path, threshold):
    # Statistics
    statistics = FeatureStatistics()
    statistics.get_word_tag_pair_count(train_path)

    # feature2id
    feature2id = Feature2id(statistics, threshold)
    feature2id.get_features_idx()
    feature2id.calc_represent_input_with_features()
    print(feature2id.n_total_features)

    for dict_key in feature2id.feature_to_idx:
        print(dict_key, len(feature2id.feature_to_idx[dict_key]))
    return statistics, feature2id


def read_test(file_path, tagged=True) -> List[Tuple[List[str], List[str]]]:
    """
    reads a test file
    @param file_path: the path to the file
    @param tagged: whether the file is tagged (validation set) or not (test set)
    @return: a list of all the sentences, each sentence represented as tuple of list of the words and a list of tags
    """
    list_of_sentences = []
    with open(file_path) as f:
        for line in f:
            if line[-1:] == "\n":
                line = line[:-1]
            sentence = (["*", "*"], ["*", "*"])
            split_words = line.split(' ')
            for word_idx in range(len(split_words)):
                if tagged:
                    cur_word, cur_tag = split_words[word_idx].split('_')
                else:
                    cur_word, cur_tag = split_words[word_idx], ""
                sentence[WORD].append(cur_word)
                sentence[TAG].append(cur_tag)
            sentence[WORD].append("~")
            sentence[TAG].append("~")
            list_of_sentences.append(sentence)
    return list_of_sentences
