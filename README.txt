How to run:

1. Install required libraries to your environment using requirements.txt

2. Run GeneticAlgorithm to train neural network. It will write parameters for it every 100 iterations in parameters.txt
   (Alternatively, there are parameters stored in file best_parameters.txt and in test.py)

3. Run test.py with the parameters of your choosing.

Extras:

Run zad2.py to visualize the data.

Run zad1.py to visualize influence of s parameter to neurons of first layer.

Additional info:

NeuralNetwork:
Network supports any arhitecture with at least 3 layers.
First layer is input layer and does nothing except putting input into second layer.
Second layer is filled with neurons of type 1 which output similarity rate of input to their parameter w.
Third and all subsequent layers are standard fully connected layers with sigmoid neurons.

GeneticAlgorithm:
Algorithm: Generative with k-tournament selection
Crossovers: arithmetic recombination, uniform crossover and better parent selection
Mutations: Gaussian adding and Gaussian replacement
Elitism: 2 by default
