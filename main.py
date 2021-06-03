import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
from numpy import savetxt
from numpy import loadtxt
import json
from io import open
import math
import string
import sys
import re


JLPT_READING_LEVELS = 5

# helper for cosine to read the file


def read_file(filename):
    try:
        with open(filename, encoding="utf8") as f:
            data = f.read()
        return data
    except IOError:
        print("Error opening or reading input file: ", filename)
        sys.exit()


# for cosine
translation_table = str.maketrans(string.punctuation+string.ascii_uppercase,
                                  " "*len(string.punctuation)+string.ascii_lowercase)

# helper for cosine to get the list of words per line


def get_words_from_line_list(text):
    text = text.translate(translation_table)
    word_list = text.split()
    # print(word_list)
    return word_list

# helper for cosine


def count_frequency(word_list):
    D = {}
    for new_word in word_list:
        if new_word in D:
            D[new_word] = D[new_word] + 1
        else:
            D[new_word] = 1
    return D

 # helper for cosine to compute word frequencies


def word_frequencies_for_file(filename):
    line_list = read_file(filename)
    word_list = get_words_from_line_list(line_list)
    freq_mapping = count_frequency(word_list)
    return freq_mapping

# helper for cosine to compute dot product


def dotProduct(D1, D2):
    Sum = 0.0
    for key in D1:
        if key in D2:
            Sum += (D1[key] * D2[key])
    return Sum

# helper for cosine to compute angle


def vector_angle(D1, D2):
    numerator = dotProduct(D1, D2)
    denominator = math.sqrt(dotProduct(D1, D1)*dotProduct(D2, D2))
    return math.acos(numerator / denominator)

# cosine similarity


def data_cosine_similarity(input, data_file):

    src = "c:/Users/hylee/OneDrive/Desktop/CS229/translated-texts/translated-texts"
    docs = []
    for filename in os.listdir(src):
        docs.append(filename)
    NUM_TEXTS = len(docs)
    sorted_word_list_1 = word_frequencies_for_file(input)
    sorted_word_list_2 = word_frequencies_for_file(data_file)
    distance = vector_angle(sorted_word_list_1, sorted_word_list_2)
    return distance

# returns the jaccards similarity between 2 documents


def compute_Jaccard_similarity(doc1, doc2):

    # extracts the vocabulary of each document by tokenizing, omitting non-alphanumeric characters, and ignoring repeats
    doc1_vocab = set()
    doc2_vocab = set()
    with open(doc1, encoding="utf8") as a_file:
        for line in a_file:
            stripped_line = line.strip()
            word_list = stripped_line.split(" ")
            for word in range(len(word_list)):
                word_list[word] = re.sub(r'\W+', '', word_list[word])
                if not word_list[word] == "":
                    doc1_vocab.add(word_list[word].lower())

    with open(doc2, encoding="utf8") as a_file:
        for line in a_file:
            stripped_line = line.strip()
            word_list = stripped_line.split(" ")

            for word in range(len(word_list)):
                word_list[word] = re.sub(r'\W+', '', word_list[word])
                if not word_list[word] == "":
                    doc2_vocab.add(word_list[word].lower())
    intersect = doc1_vocab.intersection(doc2_vocab)
    union = doc1_vocab.union(doc2_vocab)
    return (len(intersect) / len(union))

# finds list of best matches in a category


def find_best_match(input, assignments, src, cosine_similarity):
    docs = []
    highest_similarity = 0
    best_match = ""
    similarities = {}
    for filename in os.listdir(src):
        docs.append(filename)

    # computes jaccard similarity for input compared to the rest of the texts
    # finds the filename that is the best match
    if not cosine_similarity:
        # higher score indicates higher similarity for Jaccard
        for doc in docs:
            similarity = compute_Jaccard_similarity(
                input, src + "/" + doc)
            similarities[doc] = similarity
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = doc
    else:
        # lower angle between vectors indicates higher similarity for cosine
        highest_similarity = float('inf')
        for doc in docs:
            similarity = data_cosine_similarity(input, src + "/" + doc)
            similarities[doc] = similarity
            if similarity < highest_similarity:
                highest_similarity = similarity
                best_match = doc

    # finds all the members of the same category as the best match and maps them to their similarity scores
    group_members = {}
    ranking = []
    category = assignments[best_match]
    for key in assignments:
        if assignments[key] == category:
            group_members[key] = similarities[key]

    if not cosine_similarity:
        # sort the group members by score (descending order)
        while len(group_members) != 0:
            best_score = -1
            best_file = ""
            for member in group_members:
                if group_members[member] > best_score:
                    best_score = group_members[member]
                    best_file = member
            ranking.append((best_file, best_score))
            group_members.pop(best_file)
    else:
        # sort the group members by score (ascending order)
        while len(group_members) != 0:
            best_score = float('inf')
            best_file = ""
            for member in group_members:
                if group_members[member] < best_score:
                    best_score = group_members[member]
                    best_file = member
            ranking.append((best_file, best_score))
            group_members.pop(best_file)

    return ranking


def run_kmeans(data, input, docs, cosine_similarity):

    # sets the number of clusters
    n_digits = 5

    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation="nearest",
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired, aspect="auto", origin="lower")

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3,
                color="w", zorder=10)
    plt.title("K-means clustering on the documents dataset (PCA-reduced data)\n"
              "Centroids are marked with white cross")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    # plt.show()
    kmeans.labels_
    kmeans.cluster_centers_

    # dictionary mapping filenames to their category assignments
    assignments = {}
    for i in range(len(kmeans.labels_)):
        assignments[docs[i]] = kmeans.labels_[i]

    return assignments


def print_recommendation_list(match_rankings, reading_level):
    reading_difficulty = {}

    # reading the data from the file
    with open('difficulty-assessment/difficulty_dict.txt') as f:
        reading_difficulty_dict = f.read()

    # reconstructing the data as a dictionary
    reading_difficulty_dict = reading_difficulty_dict.replace('\'', '\"')
    reading_difficulty_dict = json.loads(reading_difficulty_dict)

    print("\n---------------------------------------")
    print("Same cluster and in order of similarity:")
    print("---------------------------------------")
    print("Similarity to input, Corpus text")
    for rankings in match_rankings:
        print(str(round(rankings[1], 3)) + ",               " + rankings[0])

    print("\n------------------------------------------------------------")
    print("Same cluster, same reading level, and in order of similarity:")
    print("------------------------------------------------------------")
    print("Similarity to input, Corpus text")
    for rankings in match_rankings:
        if reading_difficulty_dict[rankings[0]] == int(reading_level):
            print(str(round(rankings[1], 3)) +
                  ",               " + rankings[0])
    return


def main():
    # prompts user for Japanese Language reading level
    print("Please enter your estimated JLPT reading reading level on a 1-5 scale, where 5 is beginner and 1 is advanced.")
    reading_level = int(input("Estimated JLPT reading level: "))

    while reading_level not in range(1, JLPT_READING_LEVELS + 1):
        print("Invalid input. Please enter a number between 1 and 5.")
        reading_level = int(input("Estimated JLPT reading level: "))

    # Get input txt
    print("Please enter a path to a .txt file to find similar texts.")
    input_text_path = input("Path to input text: ")
    input_text = ""
    with open(input_text_path, "rb") as f:
        input_text = f.read().decode("UTF-8")

    src = "translated-texts"
    docs = []
    for filename in os.listdir(src):
        docs.append(filename)

    NUM_TEXTS = len(docs)
    # can set to True or False to compute k-means using either Jaccard or cosine similarity
    COSINE_SIMILARITY = False

    kmeans_matrix = np.zeros((NUM_TEXTS, NUM_TEXTS))
    if COSINE_SIMILARITY:
        kmeans_matrix = loadtxt('cosine_similarities.csv', delimiter=',')
    else:
        kmeans_matrix = loadtxt('kmeans-matrix.csv', delimiter=',')

    # runs k-means on the matrix of similarity scores
    cluster_assignments = run_kmeans(
        kmeans_matrix, input_text_path, docs, COSINE_SIMILARITY)

    # returns list of tuples of best matches with their similarity scores
    match_rankings = find_best_match(
        input_text_path, cluster_assignments, src, COSINE_SIMILARITY)
    print("\nHere are your recommendations:")
    print_recommendation_list(match_rankings, str(reading_level))


if __name__ == "__main__":
    main()
