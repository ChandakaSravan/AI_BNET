Name: Sravan Chandaka
Id: 1002059166

Programming language: Python 3.10.11

Code Structure:

The above code is a Python implementation of a Bayesian Network, which is a probabilistic graphical model used to represent the relationships between random variables in a probabilistic way. The network consists of nodes representing the random variables and edges representing the dependencies between them. The implementation uses a tabular method to represent the conditional probabilities of the variables.

The code consists of several functions, which are explained below:

count_variables(train_file): This function takes a filename as input and reads the training data from the file. It then initializes the conditional probability tables and updates them based on the data. Finally, it returns the updated probability tables.

calculate_probabilities(prob_B, prob_BG, prob_C, prob_GCF): This function takes the probability tables as input and calculates the sum of probabilities for each variable. It then divides each variable's probability by its sum to obtain the normalized probabilities. Finally, it returns the normalized probability tables.

read_training_data(train_file): This function reads the training data from the file and initializes the probability tables based on the maximum value of each variable in the data. It then updates the probability tables based on the data and calculates the normalized probability tables using the calculate_probabilities function. Finally, it returns the normalized probability tables.

print_probabilities(prob_B, prob_BG, prob_C, prob_GCF): This function takes the normalized probability tables as input and prints them in a table format using the tabulate library. It prints the probabilities of each variable and its conditional probabilities given its parents.

input(val): This function takes a string input and converts it to a binary value. If the last character of the string is "t" (for "true"), it returns 1. Otherwise, it returns 0.

calc_jpd(prob_B, prob_BG, prob_C, prob_GCF, B, G, C, F): This function calculates the joint probability distribution of the variables given their values. It takes the normalized probability tables and the values of the variables as input and calculates the joint probability using the product rule of probability.

main(): This function is the main function of the program. It reads the command line arguments and calls the appropriate functions based on the arguments. If the program is called with more than two arguments, it assumes that the user is requesting the probability of a specific configuration of the variables and calls the calc_jpd function to calculate the probability. Otherwise, it calls the print_probabilities function to print the probability tables. 

Compile and run instrunctions:
1. Open the cmd and move to project folder where the python file is situated.
2. Run the command: 
        For Task1: python bnet.py <training_data>
	  For Task2: python bnet.py <training_data> Bt Gf Ct Ff 
	  For Task3: python bnet.py training_data.txt Bt Gf given Ff 	



