from random import randint
import random
import numpy as np
import neural_network
import dataset

class Chromosome:
    def __init__(self, number_of_genes):
        self.number_of_genes = number_of_genes
        self.genes = np.random.uniform(-1, 1, number_of_genes)
        self.badness = 0

    def set_genes(self, genes):
        self.genes = genes

    def get_genes(self):
        return self.genes

    def get_gene(self, index):
        return self.genes[index]

    def set_gene(self, index, gene):
        self.genes[index] = gene

    def __len__(self):
        return self.number_of_genes



class GeneticAlgorithm:
    def __init__(self, population_size, network, data, pm1=0.01, pm2=0.01, sigma1=0.5, sigma2=1, sigma3=1, k=3, t1=1, t2=1, t3=1, elitism=3):
        self.population_size = population_size
        self.population = []
        for i in range(population_size):
            self.population.append(Chromosome(network.get_number_of_parameters()))

        self.network = network
        self.data = data
        self.pm1 = pm1
        self.pm2 = pm2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.k = k
        t = t1 + t2 + t3
        self.v1 = t1 / t
        self.v2 = t2 / t
        self.v3 = t3 / t
        self.elitism = elitism


    @staticmethod
    def arithmetic_recombination(chromosome1, chromosome2):
        intersection = randint(0, len(chromosome1))
        child1 = Chromosome(len(chromosome1))
        child2 = Chromosome(len(chromosome1))
        for i in range(len(chromosome1)):
            if i < intersection:
                child1.set_gene(i, chromosome1.get_gene(i))
                child2.set_gene(i, (chromosome1.get_gene(i) + chromosome2.get_gene(i)) / 2)
            else:
                child2.set_gene(i, chromosome2.get_gene(i))
                child1.set_gene(i, (chromosome1.get_gene(i) + chromosome2.get_gene(i)) / 2)

        return child1 if random.random() > 0 else child2

    @staticmethod
    def better_parent(chromosome1, chromosome2):
        return chromosome1 if chromosome1.badness < chromosome2.badness else chromosome2

    @staticmethod
    def uniform_crossover(chromosome1, chromosome2):
        child = Chromosome(len(chromosome2))
        for i in range(len(chromosome2)):
            child.set_gene(i, chromosome1.get_gene(i) if random.random() < 0.5 else chromosome2.get_gene(i))

        return child

    @staticmethod
    def mutate_add(chromosome, mutation_chance, sigma):
        mutated = Chromosome(len(chromosome))
        for i in range(len(chromosome)):
            x = chromosome.get_gene(i)
            mutated.set_gene(i, x + np.random.normal(0, sigma) if random.random() < mutation_chance else x)

        return mutated

    def weak_add_mutation(self, chromosome):
        return self.mutate_add(chromosome, self.pm1, self.sigma1)

    def strong_add_mutation(self, chromosome):
        return self.mutate_add(chromosome, self.pm1, self.sigma2)

    def mutate_replace(self, chromosome):
        mutated = Chromosome(len(chromosome))
        for i in range(len(chromosome)):
            x = chromosome.get_gene(i)
            mutated.set_gene(i, np.random.normal(0, self.sigma3) if random.random() < self.pm2 else x)

        return mutated

    def k_tournament(self):
        tournament = []
        for i in range(self.k):
            x = randint(0, self.population_size - 1)
            tournament.append(self.population[x])

        tournament.sort(key=lambda x: x.badness)

        return tournament[0], tournament[1]

    def evaluate_population(self):
        for chromosome in self.population:
            chromosome.badness = self.network.calc_error(chromosome.get_genes(), self.data)

    def randomly_mutate(self, chromosome):
        x = random.random()
        if x < self.v1:
            return self.weak_add_mutation(chromosome)
        elif x < self.v1 + self.v2:
            return self.strong_add_mutation(chromosome)
        else:
            return self.mutate_replace(chromosome)

    def get_best_chromosome(self):
        best = self.population[0]
        for chromosome in self.population:
            if chromosome.badness < best.badness:
                best = chromosome
        return best

    def randomly_crossover(self, parent1, parent2):
        x = random.random()
        if x < 1/3:
            return self.arithmetic_recombination(parent1, parent2)
        elif x < 2/3:
            return self.better_parent(parent1, parent2)
        else:
            return self.uniform_crossover(parent1, parent2)

    def run_algorithm(self, epsilon=0.0000001, max_iterations=50000):
        for i in range(max_iterations):
            self.evaluate_population()
            self.population.sort(key=lambda x: x.badness)
            best = self.population[0]
            error = best.badness
            if i%10 == 0:
                print("error:", error)
                print("generation:", i)

            if i%100 == 0 or error < epsilon:
                file = open('parameters.txt', 'a')
                file.write(str(best.get_genes()))
                file.write('\n')
                file.close()
                if error < epsilon:
                    break
            new_population = []

            for j in range(self.elitism):
                new_population.append(self.population[j])

            while len(new_population) < self.population_size:
                parent1, parent2 = self.k_tournament()
                child = self.randomly_crossover(parent1, parent2)
                child = self.randomly_mutate(child)
                new_population.append(child)

            self.population = new_population






nn = neural_network.NeuralNetwork([2, 8, 3])
ds = dataset.Dataset('zad7-dataset.txt')
ga = GeneticAlgorithm(30, nn, ds)
ga.run_algorithm()