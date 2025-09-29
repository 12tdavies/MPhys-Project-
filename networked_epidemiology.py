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
        
        new_suseptible = self.suseptible - self.time_step*self.infection_rate*self.suseptible*self.infected + self.time_step*self.resuseptibility_rate*self.recovered
        new_infected = self.infected + self.time_step*self.infection_rate*self.suseptible*self.infected- self.time_step*self.recovery_rate*self.infected
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
        
       a = np.random.binomial(self.suseptible, 1 - math.e**(-self.time_step*self.infection_rate*self.infected))
       b = np.random.binomial(self.infected, 1 - math.e**(-self.time_step*self.recovery_rate))
       c = np.random.binomial(self.recovered, 1-math.e**(-self.time_step*self.resuseptibility_rate))
       #a = np.random.normal(self.time_step*self.infection_rate*self.infected*self.suseptible, 0.1000*self.time_step*self.infection_rate*self.infected*self.suseptible)
       #b = np.random.normal(self.time_step*self.recovery_rate*self.infected, 0.1000*self.time_step*self.recovery_rate*self.infected)
       #c = np.random.normal(self.time_step*self.resuseptibility_rate*self.recovered,0.1000*self.time_step*self.resuseptibility_rate*self.recovered)
       #print(a)
       new_suseptible = max(self.suseptible - a+ c,0)
       new_infected = max(self.infected + a - b,1)
       new_recovered = max(self.recovered + b - c,0)
   
       return new_suseptible, new_infected, new_recovered
    
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
 
    def __init__(self, G, self_interaction, external_interaction, other_data, time_step):
  
        self.G = G
        self.self_interaction = self_interaction
        self.external_interaction = external_interaction
        self.other_data = other_data
        self.time_step = time_step
  
    def deterministic_SIRS(self):
     
        new_states = []
        present_state = []
        for i in range(len(G)):
           present_state.append(G.nodes[i]['state'])
        for j in range(100):
            for i in range(len(G)):
                self_interaction = next_timestep(present_state[i][0], present_state[i][1], present_state[i][2], present_state[i][0]+ present_state[i][1]+ present_state[i][2], 1, 1, self.time_step, 1, 0)
                new_states.append(self_interaction.SIRS())
            present_state = new_states.copy()
            print(present_state)
            new_states = []
number_of_nodes = 10
for i in range(0,number_of_nodes):
    G.add_node(i, state=[0, 0, 0])
for i in range(0,number_of_nodes):
    for j in range(0,number_of_nodes):
        G.add_edge(i, j, weight = np.random.uniform(0,1))

for i in range(number_of_nodes,2*number_of_nodes):
    G.add_node(i, state=[0, 0, 0])
for i in range(number_of_nodes,2*number_of_nodes):
    for j in range(number_of_nodes,2*number_of_nodes):
        G.add_edge(i, j, weight = np.random.uniform(0,1))
G.add_edge(0, 12, weight = 2)
for i in range(len(G.nodes)):
    G.nodes[i].update({'state':[random.randint(1,5),random.randint(1,5),random.randint(1,5)]}) 

pos = nx.spring_layout(G)
nx.draw(G, pos)

fig = plt.figure()
nx.draw_networkx(G, pos)
plt.show()
update = network_update(G, 1, 1, 1, 0.1)
update.deterministic_SIRS()
