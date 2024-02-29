import numpy as np
import sys
from tabulate import tabulate

def count_variables(train_file):
    # read the training data from the file
    with open(train_file, "r") as f:
        data = np.loadtxt(f, dtype=int)

    # define the conditional probability values
    prob_B = np.zeros(2)
    prob_C = np.zeros(2)
    prob_BG = np.zeros((2, 2))
    prob_GCF = np.zeros((2, 2, 2))

    # update the counts for each variable
    for b, g, c, f in data:
        prob_B[b] += 1
        prob_BG[b, g] += 1
        prob_C[c] += 1
        prob_GCF[g, c, f] += 1

    return prob_B, prob_BG, prob_C, prob_GCF



def calculate_probabilities(prob_B, prob_BG, prob_C, prob_GCF):
    # calculate sum of probabilities for each variable
    sum_B = prob_B.sum()
    sum_BG = prob_BG.sum(axis=1, keepdims=True)
    sum_C = prob_C.sum()
    sum_GCF = prob_GCF.sum(axis=2, keepdims=True)
    
    # divide each variable's probability by its sum
    prob_B = prob_B / sum_B
    prob_BG = prob_BG / sum_BG
    prob_C = prob_C / sum_C
    prob_GCF = prob_GCF / sum_GCF
    
    return prob_B, prob_BG, prob_C, prob_GCF



def read_training_data(train_file):
    with open(train_file) as f:
        data = np.array([list(map(int, line.strip().split())) for line in f])

    num_B, num_G, num_C, num_F = np.max(data, axis=0) + 1

    prob_B = np.zeros(num_B)
    prob_C = np.zeros(num_C)
    prob_BG = np.zeros((num_B, num_G))
    prob_GCF = np.zeros((num_G, num_C, num_F))

    for b, g, c, f in data:
        prob_B[b] += 1
        prob_C[c] += 1
        prob_BG[b, g] += 1
        prob_GCF[g, c, f] += 1

    return calculate_probabilities(prob_B, prob_BG, prob_C, prob_GCF)



def print_probabilities(prob_B, prob_BG, prob_C, prob_GCF):
    # P(B)
    print("P(B):")
    headers = ["B", "P(B)"]
    data = [[i, prob_B[i]] for i in range(2)]
    print(tabulate(data, headers=headers, tablefmt="orgtbl"))

    # P(G|B)
    print("\nP(G|B):")
    headers = ["B", "G=0", "G=1"]
    data = [[i, prob_BG[i][0], prob_BG[i][1]] for i in range(2)]
    print(tabulate(data, headers=headers, tablefmt="orgtbl"))

    # P(C)
    print("\nP(C):")
    headers = ["C", "P(C)"]
    data = [[i, prob_C[i]] for i in range(2)]
    print(tabulate(data, headers=headers, tablefmt="orgtbl"))

    # P(F|G,C)
    print("\nP(F|G,C):")
    headers = ["F", "G=0,C=0", "G=0,C=1", "G=1,C=0", "G=1,C=1"]
    data = [[f, prob_GCF[0][0][f], prob_GCF[0][1][f], prob_GCF[1][0][f], prob_GCF[1][1][f]] for f in range(2)]
    print(tabulate(data, headers=headers, tablefmt="orgtbl"))

def input(val):
    if val[-1].lower() == 't':
        return 1
    else:
        return 0

def calc_jpd(prob_B, prob_BG, prob_C, prob_GCF, B, G, C, F):
    jpd = np.prod([prob_B[B], prob_BG[B, G], prob_C[C], prob_GCF[G, C, F]])
    return jpd


def main():
    # Command line arguments
    train_file = sys.argv[1]
    prob_B, prob_BG, prob_C, prob_GCF = read_training_data(train_file)
    if len(sys.argv) > 2:
      
        B = input(sys.argv[2])
        G = input(sys.argv[3])
        C = input(sys.argv[4])
        
        F = input(sys.argv[5])
        jpd = calc_jpd(prob_B, prob_BG, prob_C, prob_GCF, B, G, C, F)
        print(f"Probability of B={B}, G={G}, C={C}, F={F}: {jpd}")

    else:
        print_probabilities(prob_B, prob_BG, prob_C, prob_GCF)

main()
