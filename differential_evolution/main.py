import math
import random
import pandas
import copy
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------- #
#                               eggHolder function                              #
# ---------------------------------------------------------------------------- #

def eggHolder(vector):
  output = -(abs(math.sin(vector[0])*math.cos(vector[1])*math.exp(abs(1-(math.sqrt(vector[0]*vector[0] + vector[1]*vector[1])/math.pi)))))
  return output



# ---------------------------------------------------------------------------- #
#                               Initialization                           #
# ---------------------------------------------------------------------------- #


#Parameters
upperBound = 10      
lowerBound = -10
populationSize = 300
generations = 500
crossoverProbability = 0.8

#Population generation
population = []
for i in range(populationSize):
  x = random.randint(-10,10)
  y = random.randint(-10,10)
  population.append([x,y])

#Vector subtraction
def subtract(a, b):
  difference = []
  for i in range(len(a)):
    difference.append(a[i] - b[i])
  return difference

#Vector multiplication
def mul(arr, int):
  for i in range(len(arr)):
    arr[i] = int*arr[i]
  return arr

#Vector addition
def add(a, b, c):
  sum = []
  for i in range(len(a)):
    sum.append(a[i] + b[i] + c[i])
  return sum
# ---------------------------------------------------------------------------- #
#                                   Algorithm                                  #
# ---------------------------------------------------------------------------- #


#Mutant vector generation
def mutant(vector, population, k, f):
  rand_1 = random.randint(0,len(population) - 1)
  rand_2 = random.randint(0,len(population) - 1)
  rand_3 = random.randint(0,len(population) - 1)
  while rand_1 == vector or rand_2 == vector or rand_3 == vector:
    rand_1 = random.randint(0,len(population) - 1)
    rand_2 = random.randint(0,len(population) - 1)
    rand_3 = random.randint(0,len(population) - 1)
  mutantVector = add(vector,mul((subtract(population[rand_1], vector)),k),mul((subtract(population[rand_2], population[rand_3])),f))
  return mutantVector

sum = 0.0
avg_array = []
gen_optimum = []

k = 0.5
for generation in range(generations):
  f = random.randint(-2,2)
  sum = 0
  for i in range(populationSize):
    trialVector = copy.deepcopy(population[i])
    while(True):
      mutantVector = mutant(population[i], population, k, f)
      for bit in range(len(mutantVector)):
        flag = random.randint(0,1)
        if flag >= 0.8:
          trialVector[bit] = mutantVector[bit]
      if trialVector[0] > 10 or trialVector[0] < -10 or  trialVector[1] > 10 or trialVector[1] < -10:
        continue
      else:
        break
    if eggHolder(trialVector) < eggHolder(population[i]):
      population[i] = trialVector
    sum += eggHolder(population[i])
  avg_array.append(sum/populationSize)
  optimum = 9999
  for candidate in population:
    if eggHolder(candidate) < optimum:
      optimum = eggHolder(candidate)
  gen_optimum.append(optimum)


main_optimum = 9999
for candidate in population:
  if eggHolder(candidate) < main_optimum:
    main_optimum = eggHolder(candidate)

print(main_optimum)

gen_file = open("17XJ1A0537.txt","w+")
for i in range(len(gen_optimum)):
  gen_file.write("%lf\n"%gen_optimum[i])
gen_file.close()

plot_gens = [i for i in range(generations)]
plt.plot(plot_gens,gen_optimum)
plt.plot(plot_gens, avg_array)
plt.show()