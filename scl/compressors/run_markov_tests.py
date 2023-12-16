"""File to run Markov rANS tests and create graphs
Author: Avash Shrestha, Autumn 2023
"""
import markov_rANS
import rANS
import alias_rANS
import matplotlib.pyplot as plt
import ast

def main():
    markov_encoding_averages, markov_decoding_averages, markov_empirical_entropies_averages, markov_codelens_averages = markov_rANS.test_markov_rANS_coding()
    standard_encoding_averages, standard_decoding_averages, standard_empirical_entropies_averages, standard_codelens_averages = rANS.test_markov_rANS_coding()
    alias_encoding_averages, alias_decoding_averages, alias_empirical_entropies_averages, alias_codelens_averages = alias_rANS.test_markov_rANS_coding()
    x_vals = [str(2 * data_size) for data_size in [5000 * i for i in range(1, 11)]]

    # encoding averages
    plt.plot(x_vals, markov_encoding_averages, marker="o", label="Markov rANS")
    plt.plot(x_vals, standard_encoding_averages, marker="o", label="Standard rANS")
    plt.plot(x_vals, alias_encoding_averages, marker="o", label="Alias rANS")

    plt.xlabel("Data Block Size (bits)")
    plt.ylabel("Encoding Averages (seconds)")
    plt.title("Encoding Averages vs Data Block Size")
    plt.legend()
    plt.show()

    # decoding averages
    plt.plot(x_vals, markov_decoding_averages, marker="o", label="Markov rANS")
    plt.plot(x_vals, standard_decoding_averages, marker="o", label="Standard rANS")
    plt.plot(x_vals, alias_decoding_averages, marker="o", label="Alias rANS")

    plt.xlabel("Data Block Size")
    plt.ylabel("Decoding Averages (seconds)")
    plt.title("Decoding Averages vs Data Block Size")
    plt.legend()
    plt.show()

    # empirical entropies/codelens averages, Markov rANS
    plt.plot(x_vals, markov_empirical_entropies_averages, marker="o", label="Empirical Entropies")
    plt.plot(x_vals, markov_codelens_averages, marker="o", label="Codelens")
    plt.xlabel("Data Block Size")
    plt.ylabel("Entropy Averages (bits)")
    plt.title("Entropy Averages vs Data Block Size (Markov rANS)")
    plt.legend()
    plt.show()

    # empirical entropies/codelens averages, Standard rANS
    plt.plot(x_vals, standard_empirical_entropies_averages, marker="o", label="Empirical Entropies")
    plt.plot(x_vals, standard_codelens_averages, marker="o", label="Codelens")
    plt.xlabel("Data Block Size")
    plt.ylabel("Entropy Averages (bits)")
    plt.title("Entropy Averages vs Data Block Size (Standard rANS)")
    plt.legend()
    plt.show()

    # empirical entropies/codelens averages, Alias rANS
    plt.plot(x_vals, alias_empirical_entropies_averages, marker="o", label="Empirical Entropies")
    plt.plot(x_vals, alias_codelens_averages, marker="o", label="Codelens")
    plt.xlabel("Data Block Size")
    plt.ylabel("Entropy Averages (bits)")
    plt.title("Entropy Averages vs Data Block Size (Alias rANS)")
    plt.legend()
    plt.show()

    # empirical entropies/codelens averages, all
    plt.plot(x_vals, markov_empirical_entropies_averages, marker="o", label="Empirical Entropies, Markov rANS")
    plt.plot(x_vals, markov_codelens_averages, marker="o", label="Codelens, Markov rANS")
    plt.plot(x_vals, standard_empirical_entropies_averages, marker="o", label="Empirical Entropies, Standard/Alias rANS")
    plt.plot(x_vals, standard_codelens_averages, marker="o", label="Codelens, Standard rANS")
    # plt.plot(x_vals, alias_empirical_entropies_averages, marker="o", label="Empirical Entropies, Alias rANS")
    plt.plot(x_vals, alias_codelens_averages, marker="o", label="Codelens, Alias rANS")
    plt.xlabel("Data Block Size")
    plt.ylabel("Entropy Averages (bits)")
    plt.title("Entropy Averages vs Data Block Size")
    plt.legend()
    plt.show()


# main()

def smarter_way():
    """
    This assumes that you have the data in text files
    and the order of the list of data is
    [encoding_averages, decoding_averages, empirical_entropies_averages, codelens_averages]
    where each element is also a list
    """
    with open("data_markov.txt", "r") as f:
        markov_data = []
        for line in f:
            markov_data.append(ast.literal_eval(line.strip()))

    with open("data_standard.txt", "r") as f:
        standard_data = []
        for line in f:
            standard_data.append(ast.literal_eval(line.strip()))

    with open("data_updated_alias.txt", "r") as f:
        alias_data = []
        for line in f:
            alias_data.append(ast.literal_eval(line.strip()))

    x_vals = [str(2 * data_size) for data_size in [5000 * i for i in range(1, 11)]]

    # encoding averages
    plt.plot(x_vals, markov_data[0], marker="o", label="Markov rANS")
    plt.plot(x_vals, standard_data[0], marker="o", label="Standard rANS", color='red')
    plt.plot(x_vals, alias_data[0], marker="o", label="Alias rANS")

    plt.xlabel("Data Block Size (bits)")
    plt.ylabel("Encoding Averages (seconds)")
    plt.title("Encoding Averages vs Data Block Size")
    plt.legend()
    plt.show()

    # decoding averages
    plt.plot(x_vals, markov_data[1], marker="o", label="Markov rANS")
    plt.plot(x_vals, standard_data[1], marker="o", label="Standard rANS", color='red')
    plt.plot(x_vals, alias_data[1], marker="o", label="Alias rANS")

    plt.xlabel("Data Block Size")
    plt.ylabel("Decoding Averages (seconds)")
    plt.title("Decoding Averages vs Data Block Size")
    plt.legend()
    plt.show()

    # empirical entropies/codelens averages, Markov rANS
    plt.plot(x_vals, markov_data[2], marker="o", label="Empirical Entropies")
    plt.plot(x_vals, markov_data[3], marker="o", label="Codelens")
    plt.xlabel("Data Block Size")
    plt.ylabel("Entropy Averages (bits)")
    plt.title("Entropy Averages vs Data Block Size (Markov rANS)")
    plt.legend()
    plt.show()

    # empirical entropies/codelens averages, Standard rANS
    plt.plot(x_vals, standard_data[2], marker="o", label="Empirical Entropies")
    plt.plot(x_vals, standard_data[3], marker="o", label="Codelens")
    plt.xlabel("Data Block Size")
    plt.ylabel("Entropy Averages (bits)")
    plt.title("Entropy Averages vs Data Block Size (Standard rANS)")
    plt.legend()
    plt.show()

    # empirical entropies/codelens averages, Alias rANS
    plt.plot(x_vals, alias_data[2], marker="o", label="Empirical Entropies")
    plt.plot(x_vals, alias_data[3], marker="o", label="Codelens")
    plt.xlabel("Data Block Size")
    plt.ylabel("Entropy Averages (bits)")
    plt.title("Entropy Averages vs Data Block Size (Alias rANS)")
    plt.legend()
    plt.show()


    # empirical entropies/codelens averages, standard vs alias
    plt.plot(x_vals, standard_data[2], marker="o", label="Empirical Entropies, Standard/Alias rANS")
    plt.plot(x_vals, standard_data[3], marker="o", label="Codelens, Standard rANS", color='red')
    # plt.plot(x_vals, alias_data[2], marker="o", label="Empirical Entropies, Alias rANS")
    plt.plot(x_vals, alias_data[3], marker="o", label="Codelens, Alias rANS")
    plt.xlabel("Data Block Size")
    plt.ylabel("Entropy Averages (bits)")
    plt.title("Entropy Averages vs Data Block Size")
    plt.legend()
    plt.show()


smarter_way()
