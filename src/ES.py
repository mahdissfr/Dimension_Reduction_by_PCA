import random

import numpy

from Chromosome import Chromosome

########## b andaze Mu
from file_handler import read_from_file
from plot import plot


def generate_initial_population(chromosome_length, min_ab, max_ab, x, y):
    list_of_chromosomes = []
    genes = []
    for i in range(chromosome_length):
        list_of_chromosomes.append(Chromosome(chromosome_length, min_ab, max_ab, x, y))
        genes.append(list_of_chromosomes[i].gene)
    return list_of_chromosomes


def generate_new_seed(Mu):
    lambdaParents = []
    size = len(Mu)
    for i in range(7 * size):
        index = random.randint(0, size - 1)
        lambdaParents.append(Mu[index])

    """
    :return: return lambda selected parents
    """
    # Todo
    return lambdaParents


def crossover(chromosome1, chromosome2, alpha):
    gene1 = chromosome1.gene
    gene2 = chromosome2.gene
    chromosome1.gene[0] = alpha * gene1[0] + (1 - alpha) * gene2[0]
    chromosome2.gene[0] = alpha * gene2[0] + (1 - alpha) * gene1[0]
    chromosome1.gene[1] = alpha * gene1[1] + (1 - alpha) * gene2[1]
    chromosome2.gene[1] = alpha * gene2[1] + (1 - alpha) * gene1[1]
    chromosome1.evaluate()
    chromosome1.evaluate()
    return chromosome1, chromosome1


# def get_sigma(x_sigma, ps, c):
# if ps == 0.2:
#     return x_sigma
# elif ps < 0.2:
#     return c * x_sigma
# else:
#     return x_sigma / c
def get_sigma(sigma_max, sigma_min, t, N):
    return sigma_max + (sigma_min - sigma_max) * t / N


def mutation(chromosome, sigma):
    """
    Don't forget to use Gaussian Noise here !
    :param chromosome:
    :return: mutated chromosome
    """
    GaussianNoise = numpy.random.normal(loc=0.0, scale=1.0, size=None)
    chromosome.gene[0] = chromosome.gene[0] + sigma * GaussianNoise
    chromosome.gene[1] = chromosome.gene[1] + sigma * GaussianNoise
    return chromosome


def evaluate_new_generation(generation):
    # Todo
    """
    Call evaluate method for each new chromosome
    :return: list of chromosomes with evaluated scores
    """
    for chromosome in generation:
        chromosome.evaluate()
    return


def Q_tournament(parents):
    q = 4
    index = random.randint(0, len(parents) - 1)
    best = parents[index]
    for i in range(q - 1):
        index = random.randint(0, len(parents) - 1 - i)
        tmp = parents[index]
        if tmp.fitness > best.fitness:
            best = tmp
    return best, index


def choose_new_generation(Mu, lambdaParent):
    # Todo
    """
    Use one of the discussed methods in class.
    Q-tournament is suggested !
    :return: Mu selected chromosomes for next cycle
    """

    parents = lambdaParent
    parents.extend(Mu)
    newGeneration = []
    for i in range(len(Mu)):
        best, index = Q_tournament(parents)
        newGeneration.append(best)
        parents.pop(index)
    return newGeneration


if __name__ == '__main__':
    MuSize = 10
    crossover_probability = 0.4
    # N = 100
    N = 100
    min_ab = 0
    max_ab = 1
    x, y = read_from_file()
    chromosome_length = len(x)
    # chromosome_length = 10

    # ps = 1
    # c = 0.8

    alpha = 0.5
    Smin = 1
    k = 0.125
    Mu = generate_initial_population(chromosome_length, min_ab, max_ab, x, y)

    max_node = max(Mu, key=lambda node: node.fitness)
    min_node = min(Mu, key=lambda node: node.fitness)
    avg_fitness = sum(c.fitness for c in Mu) / len(Mu)
    print("t=0 best fitness: " + str(max_node.fitness) + " worst: " + str(
        min_node.fitness) + " average fitness: " + str(avg_fitness))
    Smax = k * (max_node.fitness - min_node.fitness)
    print("smax: "+str(Smax))
    for t in range(N):
        lambdaParent = generate_new_seed(Mu)
        for i in range(len(lambdaParent)):
            # lambdaParent[i].sigma = get_sigma(lambdaParent[i].sigma, ps, c)
            lambdaParent[i].sigma = get_sigma(Smax, Smin, t, N)
            mutation(lambdaParent[i], lambdaParent[i].sigma)
        crossovered = []
        toCrossOver = int(crossover_probability * len(lambdaParent) / 2)
        for j in range(toCrossOver):
            index1 = random.randint(0, len(lambdaParent) - 1)
            chromosome1 = lambdaParent.pop(index1)
            index2 = random.randint(0, len(lambdaParent) - 1)
            chromosome2 = lambdaParent.pop(index2)
            crossovered.extend(crossover(chromosome1, chromosome2, alpha))
        lambdaParent.extend(crossovered)
        evaluate_new_generation(lambdaParent)
        Mu = choose_new_generation(Mu, lambdaParent)
        max_node = max(Mu, key=lambda node: node.fitness)
        min_node = min(Mu, key=lambda node: node.fitness)
        avg_fitness = sum(c.fitness for c in Mu) / len(Mu)
        print("t=" + str(t+1) + " ) best fitness: " + str(max_node.fitness) + " worst: " + str(
            min_node.fitness) + " average fitness: " + str(avg_fitness))

    print(str(max_node.get_normal_ab()))

    plot(max_node)
