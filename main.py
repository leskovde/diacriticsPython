#!/usr/bin/env python3

# Team members:
# ccf555c1-cba6-11e8-a4be-00505601122b
# 8e5d388f-c72d-11e8-a4be-00505601122b

import argparse
import lzma
import pickle
import os
import urllib.request
import sys

import numpy as np

import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.neural_network
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing

LETTERS_NODIA = "acdeeinorstuuyz"
LETTERS_DIA = "áčďéěíňóřšťúůýž"
DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())


def split(name):
    with open(name, "r", encoding="utf-8") as dataset_file:
        target = dataset_file.read()
    data = target.translate(DIA_TO_NODIA)

    return data, target


class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2021/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(url + name.replace(".txt", ".LICENSE"),
                                       filename=name.replace(".txt", ".LICENSE"))

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")


# Utility function to report best scores
def report(results, n_top=5):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


model_names = "acdeinorstuyz"


class ModelWrapper:
    left_right_offset = 8
    separators = [" ", "\n", ",", "."]

    one_hot_values = [chr(i) for i in range(97, 123)] + separators
    one_hot_indexed = sorted([ord(i) for i in one_hot_values])

    one_hot_list = []
    for i in range(left_right_offset * 2):
        one_hot_list.append(one_hot_indexed)

    fill = "".join([" " for i in range(left_right_offset)])

    def __init__(self):
        self.dictionary = {}

    @staticmethod
    def scramble_sentences(data):
        words = data.split()
        odd_words = words[0::2]
        even_words = words[1::2]
        scrambled_sentence = " "

        for i in range(max(len(odd_words), len(even_words))):
            scrambled_sentence = scrambled_sentence + even_words[i] + " "
            scrambled_sentence = scrambled_sentence + odd_words[i] + " "

        return scrambled_sentence

    @staticmethod
    def flip_sentences(data):
        data_added = []
        lines = data.split('\n')
        for line in lines:
            parts = line.split()
            only_words = [part for part in parts if part.isalpha()]
            only_words.reverse()
            i = 0
            result = []
            for part in parts:
                if not part.isalpha():
                    result.append(part)
                else:
                    result.append(only_words[i])
                    i += 1
            data_added.append(" ".join(result))

        return "\n".join(data_added)

    def augment(self, data):
        data = data + self.flip_sentences(data)
        #data = data + self.scramble_sentences(data)
        no_dia_data = self.simplify(
            self.fill + data.translate(DIA_TO_NODIA) + self.fill)

        data = self.fill + (data + self.fill).lower()
        train = {}
        target = {}

        for i in range(0, len(model_names)):
            train[model_names[i]] = []
            target[model_names[i]] = []

        for i in range(self.left_right_offset, len(no_dia_data)):
            if no_dia_data[i] in model_names:

                feature_vector = []
                for j in range(-self.left_right_offset + i, self.left_right_offset + 1 + i):
                    if j != i:
                        feature_vector.append(ord(no_dia_data[j]))
                train[no_dia_data[i]].append(feature_vector)
                target[no_dia_data[i]].append(ord(data[i]))
        return train, target

    @staticmethod
    def lf(data):
        return [item for sublist in data for item in sublist]

    def create_dictionary(self, data: str):
        data = [line.rstrip("\n").split() for line in data.splitlines()]
        data = self.lf(data)

        unique_words = set()
        variants = {}

        for word in data:
            if not word.isalnum():
                continue

            unique_words.add(word)
            stripped = word.translate(DIA_TO_NODIA)

            if not (stripped in variants):
                variants[stripped] = {}
            if word in variants[stripped]:
                variants[stripped][word] += 1
            else:
                variants[stripped][word] = 1

        dic = {}
        for stripped in variants:
            right_side = []

            for left_side in variants[stripped]:
                right_side.append((left_side, variants[stripped][left_side]))

            right_side.sort(key=lambda x: x[1], reverse=True)
            dic[stripped] = right_side[0][0]

        with lzma.open("words_file", "wb") as model_file:
            pickle.dump(unique_words, model_file)

        with lzma.open("dictionary_file", "wb") as model_file:
            pickle.dump(dic, model_file)

    def simplify(self, data: str):
        output = []
        for i in range(len(data)):
            if data[i].isupper():
                output.append(data[i].lower())
            elif data[i].isalpha() | (data[i] in self.separators):
                output.append(data[i])
            else:
                output.append(",")

        return "".join(output)

    def fit(self, data):
        train, target = self.augment(data)

        models = {}

        """
        params = [
            {'activation': ['tanh'], 'alpha': [0.0001, 0.1, 0.15, 1, 100], 'hidden_layer_sizes': [(50, 100, 50)], 'learning_rate': ['adaptive'], 'solver': ['adam']},
            {'activation': ['relu'], 'alpha': [0.0001, 0.1, 0.15, 1, 100], 'hidden_layer_sizes': [(175, 75, 50)], 'learning_rate': ['adaptive'], 'solver': ['adam']},
            {'activation': ['relu'], 'alpha': [0.0001, 0.1, 0.15, 1, 100], 'hidden_layer_sizes': [(175, 75, 50)], 'learning_rate': ['adaptive'], 'solver': ['adam']},
            {'activation': ['tanh'], 'alpha': [0.0001, 0.1, 0.15, 1, 100], 'hidden_layer_sizes': [(50, 100, 50)], 'learning_rate': ['adaptive'], 'solver': ['adam']},
            {'activation': ['tanh'], 'alpha': [0.0001, 0.1, 0.15, 1, 100], 'hidden_layer_sizes': [(50, 100, 50)], 'learning_rate': ['adaptive'], 'solver': ['adam']},
            {'activation': ['relu'], 'alpha': [0.0001, 0.1, 0.15, 1, 100], 'hidden_layer_sizes': [(175, 75, 50)], 'learning_rate': ['adaptive'], 'solver': ['adam']},
            {'activation': ['relu'], 'alpha': [0.0001, 0.1, 0.15, 1, 100], 'hidden_layer_sizes': [(175, 75, 50)], 'learning_rate': ['adaptive'], 'solver': ['adam']},
            {'activation': ['relu'], 'alpha': [0.0001, 0.1, 0.15, 1, 100], 'hidden_layer_sizes': [(175, 75, 50)], 'learning_rate': ['adaptive'], 'solver': ['adam']},
            {'activation': ['relu'], 'alpha': [0.0001, 0.1, 0.15, 1, 100], 'hidden_layer_sizes': [(175, 75, 50)], 'learning_rate': ['adaptive'], 'solver': ['adam']},
            {'activation': ['relu'], 'alpha': [0.0001, 0.1, 0.15, 1, 100], 'hidden_layer_sizes': [(175, 75, 50)], 'learning_rate': ['adaptive'], 'solver': ['adam']},
            {'activation': ['relu'], 'alpha': [0.0001, 0.1, 0.15, 1, 100], 'hidden_layer_sizes': [(175, 75, 50)], 'learning_rate': ['adaptive'], 'solver': ['adam']},
            {'activation': ['relu'], 'alpha': [0.0001, 0.1, 0.15, 1, 100], 'hidden_layer_sizes': [(175, 75, 50)], 'learning_rate': ['adaptive'], 'solver': ['adam']},
            {'activation': ['relu'], 'alpha': [0.0001, 0.1, 0.15, 1, 100], 'hidden_layer_sizes': [(175, 75, 50)], 'learning_rate': ['adaptive'], 'solver': ['adam']}
        ]
        """

        params = [
            {'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (175, 75, 50), 'learning_rate': 'adaptive',
             'solver': 'adam'},
            {'activation': 'relu', 'alpha': 0.15, 'hidden_layer_sizes': (175, 75, 50), 'learning_rate': 'adaptive',
             'solver': 'adam'},
            {'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (175, 75, 50), 'learning_rate': 'adaptive',
             'solver': 'adam'},
            {'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (100, 75, 50), 'learning_rate': 'adaptive',
             'solver': 'adam'},
            {'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (175, 75, 50,), 'learning_rate': 'adaptive',
             'solver': 'adam'},
            {'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (175, 75, 50), 'learning_rate': 'adaptive',
             'solver': 'adam'},
            {'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (175, 75, 50), 'learning_rate': 'adaptive',
             'solver': 'adam'},
            {'activation': 'relu', 'alpha': 0.15, 'hidden_layer_sizes': (175, 75, 50), 'learning_rate': 'adaptive',
             'solver': 'adam'},
            {'activation': 'relu', 'alpha': 0.1, 'hidden_layer_sizes': (175, 75, 50), 'learning_rate': 'adaptive',
             'solver': 'adam'},
            {'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (175, 75, 50), 'learning_rate': 'adaptive',
             'solver': 'adam'},
            {'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (175, 75, 50), 'learning_rate': 'adaptive',
             'solver': 'adam'},
            {'activation': 'relu', 'alpha': 1, 'hidden_layer_sizes': (175, 75, 50), 'learning_rate': 'adaptive',
             'solver': 'adam'},
            {'activation': 'relu', 'alpha': 0.1, 'hidden_layer_sizes': (175, 75, 50), 'learning_rate': 'adaptive',
             'solver': 'adam'}
        ]

        for i in range(0, len(model_names)):
            letter = model_names[i]

            """
            A:
            Mean validation score: 0.963 (std: 0.001)
            Parameters: {'activation': 'tanh', 'alpha': 0.01, 'hidden_layer_sizes': (50, 100, 50), 'learning_rate': 'adaptive', 'solver': 'adam'}
            
            C:
            Mean validation score: 0.987 (std: 0.001)
            Parameters: {'activation': 'relu', 'alpha': 0.15, 'hidden_layer_sizes': (175, 75, 50), 'learning_rate': 'constant', 'solver': 'adam'}

            D:
            Mean validation score: 0.999 (std: 0.000)
            Parameters: {'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (175, 75, 50), 'learning_rate': 'adaptive', 'solver': 'adam'}
            
            E:
            Mean validation score: 0.965 (std: 0.002)
            Parameters: {'activation': 'tanh', 'alpha': 0.01, 'hidden_layer_sizes': (50, 100, 50), 'learning_rate': 'adaptive', 'solver': 'adam'}
            
            I:
            Mean validation score: 0.943 (std: 0.002)
            Parameters: {'activation': 'tanh', 'alpha': 0.01, 'hidden_layer_sizes': (50, 100, 50), 'learning_rate': 'adaptive', 'solver': 'adam'}
            
            N:
            Mean validation score: 0.999 (std: 0.000)
            Parameters: {'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (175, 75, 50), 'learning_rate': 'adaptive', 'solver': 'adam'}
            
            O:
            Mean validation score: 0.999 (std: 0.000)
            Parameters: {'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (175, 75, 50), 'learning_rate': 'adaptive', 'solver': 'adam'}
            
            R:
            Mean validation score: 0.990 (std: 0.001)
            Parameters: {'activation': 'relu', 'alpha': 0.15, 'hidden_layer_sizes': (175, 75, 50), 'learning_rate': 'constant', 'solver': 'adam'}

            S:
            Mean validation score: 0.983 (std: 0.001)
            Parameters: {'activation': 'relu', 'alpha': 0.1, 'hidden_layer_sizes': (175, 75, 50), 'learning_rate': 'constant', 'solver': 'adam'}

            T:
            Mean validation score: 0.998 (std: 0.000)
            Parameters: {'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (175, 75, 50), 'learning_rate': 'adaptive', 'solver': 'adam'}
            
            U:
            Mean validation score: 0.964 (std: 0.001)
            Parameters: {'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (175, 75, 50), 'learning_rate': 'adaptive', 'solver': 'adam'}
            
            Y:
            Mean validation score: 0.961 (std: 0.001)
            Parameters: {'activation': 'relu', 'alpha': 1, 'hidden_layer_sizes': (175, 75, 50), 'learning_rate': 'constant', 'solver': 'adam'}

            Z:
            Mean validation score: 0.980 (std: 0.003)
            Parameters: {'activation': 'relu', 'alpha': 0.1, 'hidden_layer_sizes': (175, 75, 50), 'learning_rate': 'constant', 'solver': 'adam'}

            parameter_space = {
                'hidden_layer_sizes': [(50, 100, 50), (175, 75, 50)],
                'activation': ['tanh', 'relu'],
                'solver': ['sgd', 'adam'],
                #'alpha': [0.0001, 0.01, 0.1, 0.15, 1, 100],
                'alpha': [0.01],
                'learning_rate': ['constant', 'adaptive'],
            }
            """

            enc = sklearn.preprocessing.OneHotEncoder(categories=self.one_hot_list, sparse=True)
            mlp_data = np.array(train[letter])
            enc.fit(mlp_data)
            mlp_data_encoded = enc.transform(mlp_data)

            # mlp = sklearn.neural_network.MLPClassifier(alpha=fast_params[i])

            # SLOW PARAMS: Used for training after the grid search is done.

            mlp = sklearn.neural_network.MLPClassifier(random_state=args.seed, max_iter=100)
            mlp.set_params(**params[i])

            """
            #GRIDSEARCH: Used with parameter_space to find the best params.
            
            mlp_search = sklearn.model_selection.GridSearchCV(mlp, params[i], n_jobs=2, cv=3)
            models[letter] = mlp_search.fit(mlp_data_encoded, np.array(target[letter]))

            report(mlp_search.cv_results_)
            """

            models[letter] = mlp.fit(mlp_data_encoded, np.array(target[letter]))

            models[letter]._optimizer = None
            for i in range(len(models[letter].coefs_)): models[letter].coefs_[i] = models[letter].coefs_[i].astype(np.float16)
            for i in range(len(models[letter].intercepts_)): models[letter].intercepts_[i] = models[letter].intercepts_[i].astype(np.float16)

            with lzma.open("model_" + model_names[i], "wb") as model_file:
                pickle.dump(models[letter], model_file)

        self.create_dictionary(data.lower())

    def predict(self, data):
        data_cpy = data
        simple_data = self.simplify(data_cpy)

        models = {}

        with lzma.open("words_file", "rb") as model_file:
            words = pickle.load(model_file)

        with lzma.open("dictionary_file", "rb") as model_file:
            dic = pickle.load(model_file)

        for i in range(0, len(model_names)):
            with lzma.open("model_" + model_names[i], "rb") as model_file:
                models[model_names[i]] = pickle.load(model_file)

        mods = {}
        indices = {}

        simple_data = self.fill + simple_data + self.fill
        result = []

        for i in range(len(model_names)):
            letter = model_names[i]
            mods[letter] = []
            indices[letter] = []

        for i in range(self.left_right_offset, len(simple_data) - self.left_right_offset):
            if simple_data[i] in model_names:
                letter = simple_data[i]

                feature_vector = []
                for j in range(-self.left_right_offset + i, self.left_right_offset + 1 + i):
                    if j != i:
                        feature_vector.append(ord(simple_data[j]))

                mods[letter].append(feature_vector)
                indices[letter].append(i - self.left_right_offset)

            result.append(simple_data[i])

        for i in range(len(model_names)):
            letter = model_names[i]
            model = models[letter]

            if mods[letter]:
                source = np.array(mods[letter])

                enc = sklearn.preprocessing.OneHotEncoder(categories=self.one_hot_list, sparse=True)
                enc.fit(source)
                source_encoded = enc.transform(source)

                predicted = model.predict(source_encoded)

                j = 0
                for index in indices[letter]:
                    result[index] = chr(predicted[j])
                    j += 1

        is_word = False
        current_word = ""
        for i in range(0, len(result)):
            if (not is_word) & result[i].isalpha():
                is_word = True
                current_word = result[i]
            elif is_word & result[i].isalpha():
                current_word += result[i]
            elif is_word & (not result[i].isalpha()):
                is_word = False

                if not (current_word in words):
                    no_dia_curr = current_word.translate(DIA_TO_NODIA)

                    if no_dia_curr in dic:
                        new_word = dic[no_dia_curr]

                        for j in range(-len(current_word), 0):
                            result[i + j] = new_word[len(current_word) + j]

                current_word = ""

        for i in range(0, len(result)):
            if data_cpy[i].isupper():
                result[i] = result[i].upper()

        for i in range(0, len(result)):
            if not result[i].isalpha():
                if result[i] != data_cpy[i]:
                    result[i] = data_cpy[i]

        return "".join(result)


def accuracy(reference, prediction):
    ref_list = reference.split()
    pred_list = prediction.split()
    acc = 0

    if len(ref_list) != len(pred_list):
        print("Different lengths!")
        return acc

    for i in range(0, len(ref_list)):
        if ref_list[i] == pred_list[i]:
            acc += 1

    return acc / len(ref_list)


def main(args):
    if args.predict is None:

        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # TODO: Train a model on the given dataset and store it in `model`.
        model = ModelWrapper()
        model.fit(train.target)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, as either a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions. Specifically,
        # produce a diacritized `str` with exactly the same number of words as `test.data`.
        """
        file_list = (
        "train_data.txt", "clanek_noviny.txt", "reportaz_noviny.txt", "uvaha_sloh.txt", "clanek_internet.txt",
        "proza.txt")
        for filename in file_list:
            filename = "tests/" + filename
            test = split(filename)

            predictions = model.predict(test[0])
            with open(filename + "_out", "w", encoding="utf-8") as output:
                output.write(predictions)
            print(filename, ":\t\t", accuracy(test[1], predictions))

        return predictions
        """

        return model.predict(test.data)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
