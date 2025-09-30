#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 09:22:16 2025

@author: thomas
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import scipy
import scipy.stats as stats
G = nx.Graph()

class next_timestep(object):
   
    def __init__(self, suseptible, infected, recovered, population, infection_rate, recovery_rate, time_step, resuseptibility_rate, other_variables):
    
        self.suseptible = suseptible
        self.infected = infected
        self.recovered = recovered
        self.population = population
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.time_step = time_step
        self.resuseptibility_rate = resuseptibility_rate
        self.other_variables = other_variables
    
    def SIR(self):
        
        new_suseptible = self.suseptible - self.time_step*self.infection_rate*self.suseptible*self.infected
        new_infected = self.infected + self.time_step*self.infection_rate*self.suseptible*self.infected - self.time_step*self.recovery_rate*self.infected
        new_recovered = self.recovered + self.time_step*self.recovery_rate*self.infected
        
        return new_suseptible, new_infected, new_recovered
        
    def SIRS(self):
        
        new_suseptible = self.suseptible - self.time_step*(self.infection_rate + self.other_variables)*self.suseptible*self.infected + self.time_step*self.resuseptibility_rate*self.recovered
        new_infected = self.infected + self.time_step*(self.infection_rate+self.other_variables)*self.suseptible*self.infected- self.time_step*self.recovery_rate*self.infected
        new_recovered = self.recovered + self.time_step*self.recovery_rate*self.infected - self.time_step*self.resuseptibility_rate*self.recovered
        
        return [new_suseptible, new_infected, new_recovered]
   
    def Random_SIRS(self):
        a = max(0,random.normalvariate(1, 1))
        b = max(0,random.normalvariate(1, 1))
        c = max(0,random.normalvariate(1, 1))
        new_suseptible = self.suseptible - a*self.time_step*self.infection_rate*self.suseptible*self.infected + c*self.time_step*self.resuseptibility_rate*self.recovered
        new_infected = self.infected + a*self.time_step*self.infection_rate*self.suseptible*self.infected - b*self.time_step*self.recovery_rate*self.infected
        new_recovered = self.recovered + b*self.time_step*self.recovery_rate*self.infected - c*self.time_step*self.resuseptibility_rate*self.recovered
        
        return new_suseptible, new_infected, new_recovered
    
    def Binomial_SIRS(self):
        
       a = np.random.poisson(self.suseptible*self.time_step*self.infection_rate*(self.infected+self.other_variables))
       b = np.random.poisson(self.infected*self.time_step*self.recovery_rate)
       c = np.random.poisson(self.recovered*self.time_step*self.resuseptibility_rate)
       #a = np.random.normal(self.time_step*self.infection_rate*self.infected*self.suseptible, 0.1000*self.time_step*self.infection_rate*self.infected*self.suseptible)
       #b = np.random.normal(self.time_step*self.recovery_rate*self.infected, 0.1000*self.time_step*self.recovery_rate*self.infected)
       #c = np.random.normal(self.time_step*self.resuseptibility_rate*self.recovered,0.1000*self.time_step*self.resuseptibility_rate*self.recovered)
       #print(a)
       new_suseptible = max(self.suseptible - a+ c,0)
       new_infected = max(self.infected + a - b,1)
       new_recovered = max(self.recovered + b - c,0)
   
       return [new_suseptible, new_infected, new_recovered]
    
    def DUAL_SIR(self):
        
       self.suseptible_2 = self.other_variables[0]
       self.infected_2 = self.other_variables[1]
       self.recovered_2 = self.other_variables[2]
       
       new_suseptible_2 = self.suseptible_2 - self.time_step*self.other_variables[3][0]*self.suseptible*self.infected_2
       new_suseptible_1 = max(0,self.suseptible - self.time_step*self.infection_rate*self.suseptible*self.infected - self.time_step*self.other_variables[3][0]*self.suseptible*self.infected_2)
       new_infected_2 = self.infected_2 + self.time_step*self.other_variables[3][0]*self.suseptible*self.infected_2 - self.time_step*self.other_variables[3][1]*self.infected_2
       new_infected_1 = self.infected + self.time_step*self.infection_rate*self.suseptible*self.infected - self.time_step*self.recovery_rate*self.infected
       new_recovered_2 = self.infected_2 + self.time_step*self.other_variables[3][1]*self.infected_2
       new_recovered_1 = self.recovered + self.time_step*self.recovery_rate*self.infected

       
       return new_suseptible_1, new_infected_1, new_recovered_1, new_suseptible_2, new_infected_2, new_recovered_2
   
    def DUAL_BIN_SIR(self):
        
        self.suseptible_2 = self.other_variables[0]
        self.infected_2 = self.other_variables[1]
        self.recovered_2 = self.other_variables[2]
   
        a = np.random.binomial(self.suseptible, 1 - math.e**(-self.time_step*self.infection_rate*self.infected))
        b = np.random.binomial(self.infected, 1 - math.e**(-self.time_step*self.recovery_rate))
        c = np.random.binomial(self.suseptible_2, 1 - math.e**(-self.time_step*self.other_variables[3][0]*self.infected_2))
        d = np.random.binomial(self.infected_2, 1 - math.e**(-self.time_step*self.other_variables[3][1]))
   
        new_suseptible_2 = self.suseptible_2 -  c
        new_suseptible_1 = max(0,self.suseptible - c - a)
        new_infected_2 = self.infected_2 + c - d
        new_infected_1 = self.infected + a - b
        new_recovered_2 = self.infected_2 + d
        new_recovered_1 = self.recovered + b

        return new_suseptible_1, new_infected_1, new_recovered_1, new_suseptible_2, new_infected_2, new_recovered_2
    
class network_update(object):
 
    def __init__(self, G, infection_rate, recovery_rate, resuseptibility_rate, other_data, time_step):
  
        self.G = G
        self.other_data = other_data
        self.time_step = time_step
        self.infection_rate = infection_rate
        self.resuseptibility_rate = resuseptibility_rate
        self.recovery_rate = recovery_rate
        
    def deterministic_SIRS(self):
        suseptible_data = []
        infected_data = []
        recovered_data = []
        time = []
        new_states = []
        present_state = []
        for i in range(len(G)):
           present_state.append(G.nodes[i]['state'])
        for j in range(100000):
            suseptible = 0
            infected = 0
            recovered = 0
            for i in range(len(G)):
                
                neighbors = list(G.neighbors(i))
                additional_infectives = 0
                for k in neighbors:
                    additional_infectives += present_state[k][1]*self.G[i][k]['weight']
                additional_infectives = float(int(additional_infectives))
                
                internal_population = present_state[i][0]+ present_state[i][1]+ present_state[i][2]
                self_interaction = next_timestep(present_state[i][0], present_state[i][1], present_state[i][2],internal_population , self.infection_rate/internal_population, self.recovery_rate, self.time_step, self.resuseptibility_rate, additional_infectives)
                new_state = self_interaction.Binomial_SIRS()
                new_states.append(new_state.copy())
                suseptible += new_state[0]
                infected += new_state[1]
                recovered += new_state[2]
            suseptible_data.append(suseptible)
            infected_data.append(infected)
            recovered_data.append(recovered)

            time.append(j)
            present_state = new_states.copy()
            new_states = []
        return suseptible_data, infected_data, recovered_data, time
class data_analysis(object):
    
    def __init__(self, suseptible_data, infected_data, recovered_data, guess, time_step, itterations,j, scale):
        
        self.suseptible_data = suseptible_data
        self.infected_data = infected_data
        self.recovered_data = recovered_data
        self.guess = guess
        self.time_step = time_step
        self.itterations = itterations
        self.j = j
        self.scale = scale
        
    def data_filter(self):
       
        a = 1
    
    def binomial(self, i):
        
        if self.j ==0:
        
            change_in_suseptible = self.suseptible_data[i] - self.suseptible_data[i+1]
        
        elif self.j == 1:
            change_in_recovered = self.recovered_data[i+1] - self.recovered_data[i]
        
            chance_of_occourance = stats.binom.pmf(change_in_suseptible, self.infected_data[i], 1 - math.e**(-self.time_step*self.guess[1]))
        elif self.j ==2:
            e = 1
        return chance_of_occourance
    def likihood(self):
        
        likihood = self.scale
        for i in range(len(self.suseptible_data) - 1):
            likihood = likihood + self.binomial(i)

        return -likihood
    def fourier_transform(self):
        fft_result = np.fft.fft(self.suseptible_data)
        freqs = np.fft.fftfreq(len(self.suseptible_data), 1)
        amplitude = np.abs(fft_result) / len(self.suseptible_data)
        #fig = plt.figure()
        #plt.plot(freqs[1:len(freqs)//2], amplitude[1:len(amplitude)//2])
        #plt.show()
        return np.log10(freqs[1:len(freqs)//2]), np.log10(amplitude[1:len(amplitude)//2])
def scipy_mediator(guess, other_data):  
    error = data_analysis(other_data[0], other_data[1], other_data[2], guess, other_data[3], other_data[4], other_data[5], other_data[6])
    return error.likihood()
def initial_guess(base_suseptible, base_infected, base_recovered, time_step, itterations):
    population = base_infected[0] + base_infected[0] + base_suseptible[0]
    guess = [0,0,0]
    minimum_1 = 00000
    bound = [(0,20)]
    best_guess_1 = 0
    for i in range(0,20):
      probability = -scipy_mediator([i,0,0], [base_suseptible, base_infected, base_recovered,time_step,itterations, 0, 1])
      
      if probability > minimum_1:
          minimum_1 = probability
          best_guess_1 = i
   
    temp_guess_1 = 0
    minimum_1 = 0
    for j in range(-10,10):
      probability = -scipy_mediator( [best_guess_1+ j/(10),0,0], [base_suseptible, base_infected, base_recovered,time_step,itterations, 0, 1])

      if probability > minimum_1:
          minimum_1 = probability
          temp_guess_2 = j
    
    inverse_scale = -scipy_mediator([temp_guess_1/(10) + best_guess_1,0,0], [base_suseptible, base_infected, base_recovered,time_step,itterations, 0, 1])
    if inverse_scale == 0:
       scale = 1
    else:
       scale = 1/inverse_scale
    guess[0] = scipy.optimize.minimize(scipy_mediator,[temp_guess_1/(10) + best_guess_1,0,0], [base_suseptible, base_infected, base_recovered,time_step,itterations,0, scale], bounds=bound, method='SLSQP', tol = 0.0000000000000000000000001)
    
    minimum_2 = 00000
    bound = [(0,20)]
    best_guess_2 = 0
    for i in range(1,20):
       probability = -scipy_mediator([0,i,0], [base_suseptible, base_infected, base_recovered,time_step,itterations, 1, 1])
       
       if probability > minimum_2:
           minimum_2 = probability
           best_guess_2 = i
    
    temp_guess_2 = 0
    minimum_2 = 0
    for j in range(-10,10):
       probability = -scipy_mediator([0, best_guess_2 + j/10,0], [base_suseptible, base_infected, base_recovered,time_step,itterations, 1, 1])
       
       if probability > minimum_2:
           minimum_2 = probability
           temp_guess_2 = j
    
    inverse_scale = -scipy_mediator([0, temp_guess_2/10 + best_guess_2,0], [base_suseptible, base_infected, base_recovered,time_step,itterations, 1, 1])
    if inverse_scale == 0:
        scale = 1
    else:
        scale = 1/inverse_scale
    guess[1] = scipy.optimize.minimize(scipy_mediator,[0,temp_guess_2/10 + best_guess_2,0], [base_suseptible, base_infected, base_recovered,time_step,itterations,1, scale], bounds=bound, method='SLSQP', tol = 0.0000000000000000000000001)
    
    
    return [guess[0].x[0], guess[1].x[1]]

number_of_nodes = 10
population_per_node = 1000
for i in range(0,number_of_nodes):
    G.add_node(i, state=[7000, 48000, 647000])
for i in range(0,number_of_nodes):
    for j in range(0,number_of_nodes):
        if i != j:
            G.add_edge(i, j, weight = np.random.uniform(0,0.1))
        else:
            G.add_edge(i, j, weight = 0)
population = 0

for i in range(number_of_nodes,2*number_of_nodes):
    G.add_node(i, state =[7000, 48000, 647000])
for i in range(number_of_nodes,2*number_of_nodes):
    for j in range(number_of_nodes,2*number_of_nodes):
        if i != j:
            G.add_edge(i, j, weight = np.random.uniform(0,0.1))
        else:
            G.add_edge(i, j, weight = 0)
G.add_edge(0, number_of_nodes + 1, weight = 0.4)
for i in G:
    population += sum(G.nodes[i]["state"])

pos = nx.spring_layout(G)
nx.draw(G, pos)

fig = plt.figure()
nx.draw_networkx(G, pos)
plt.show()
update = network_update(G, 1, 0.5, 0.0003, 1, 0.1)
suseptible,infected,recovered, time = update.deterministic_SIRS()
fig = plt.figure()
plt.plot(time,suseptible)
plt.plot(time,infected)
plt.plot(time,recovered)
plt.show()
