import math
import random
import statistics


class Chromosome:
    def __init__(self, chromosome_length, min, max, x, y):
        self.x = x
        self.y = y
        # Todo create a random list for genes between min and max below
        self.gene = []
        self.score = 0 #ps
        self.fitness = 0
        # self.x_fitness = 0
        self.sigma = 1
        self.min_gene = min
        self.max_gene = max
        self.chromosome_length = chromosome_length
        self.initialize_gene()
        self.evaluate()

    def initialize_gene(self):
        self.gene.extend([random.uniform(self.min_gene, self.max_gene), random.uniform(self.min_gene, self.max_gene)])

    def evaluate(self):
        # self.x_fitness=self.fitness
        self.fitness = statistics.stdev(self.get_z_array())

    def get_z_array(self):
        z = []
        a,b= self.get_normal_ab()
        for i in range(self.chromosome_length):
            z.append(a * self.x[i] + b * self.y[i])
        return z

    def get_normal_ab(self):
        size = math.sqrt(self.gene[0]**2 + self.gene[1]**2)
        return self.gene[0]/size, self.gene[1]/size



