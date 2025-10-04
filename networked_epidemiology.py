
            

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import scipy
import scipy.stats as stats
import time as tn
from scipy.ndimage import gaussian_filter1d

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
        
        new_suseptible = self.suseptible - self.time_step*self.infection_rate*self.suseptible*(self.infected + self.other_variables) + self.time_step*self.resuseptibility_rate*self.recovered
        new_infected = self.infected + self.time_step*self.infection_rate*self.suseptible*(self.infected + self.other_variables)- self.time_step*self.recovery_rate*self.infected
        new_recovered = self.recovered + self.time_step*self.recovery_rate*self.infected - self.time_step*self.resuseptibility_rate*self.recovered
        
        return [new_suseptible, new_infected, new_recovered]
    def Binomial_SIRS(self):
        
       a = np.random.poisson(self.suseptible*self.time_step*self.infection_rate*(self.infected+self.other_variables))
       b = np.random.poisson(self.infected*self.time_step*self.recovery_rate)
       c = np.random.poisson(self.recovered*self.time_step*self.resuseptibility_rate)
       #a = np.random.normal(self.time_step*self.infection_rate*self.infected*self.suseptible, 0.1000*self.time_step*self.infection_rate*self.infected*self.suseptible)
       #b = np.random.normal(self.time_step*self.recovery_rate*self.infected, 0.1000*self.time_step*self.recovery_rate*self.infected)
       #c = np.random.normal(self.time_step*self.resuseptibility_rate*self.recovered,0.1000*self.time_step*self.resuseptibility_rate*self.recovered)
       #print(a)
       new_suseptible = max(self.suseptible - a+ c,0)
       new_infected = max(self.infected + a - b,0)
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
 
    def __init__(self, G, infection_rate, recovery_rate, resuseptibility_rate, other_data, time_step, itterations):
  
        self.G = G
        self.other_data = other_data
        self.time_step = time_step
        self.infection_rate = infection_rate
        self.resuseptibility_rate = resuseptibility_rate
        self.recovery_rate = recovery_rate
        self.itterations = itterations
        
    def deterministic_SIRS(self):
        suseptible_data = []
        infected_data = []
        recovered_data = []
        time = []
        new_states = []
        present_state = []
        for i in range(len(self.G)):
           present_state.append(self.G.nodes[i]['state'])
        for j in range(self.itterations):
            suseptible = 0
            infected = 0
            recovered = 0
            for i in range(len(self.G)):
                
                neighbors = list(self.G.neighbors(i))
                additional_infectives = 0
                for k in neighbors:
                    additional_infectives += present_state[k][1]*self.G[i][k]['weight']
                internal_population = present_state[i][0]+ present_state[i][1]+ present_state[i][2]
                self_interaction = next_timestep(present_state[i][0], present_state[i][1], present_state[i][2],internal_population , self.infection_rate, self.recovery_rate, self.time_step, self.resuseptibility_rate, additional_infectives)
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
    
    def __init__(self, suseptible_data, infected_data, recovered_data, guess, time_step, itterations,j, period_evaluation):
        
        self.suseptible_data = suseptible_data
        self.infected_data = infected_data
        self.recovered_data = recovered_data
        self.guess = guess
        self.time_step = time_step
        self.itterations = itterations
        self.j = j
        self.period_evaluation = period_evaluation
    
    def chi_squared_data(self):
        error = 0
        for i in range(1,len(self.suseptible_data)):
            error += abs((self.suseptible_data[i] - self.suseptible_data[i-1] + self.time_step*self.guess[0]*self.suseptible_data[i-1]*self.infected_data[i-1] - self.time_step*self.guess[2]*self.recovered_data[i-1])/self.suseptible_data[i])**2
            error += abs((self.infected_data[i] - self.infected_data[i-1] - self.time_step*self.guess[0]*self.suseptible_data[i-1]*self.infected_data[i-1] + self.time_step*self.guess[1]*self.infected_data[i-1])/self.suseptible_data[i])**2
            error += abs((self.recovered_data[i] - self.recovered_data[i-1] - self.time_step*self.guess[1]*self.infected_data[i-1] + self.time_step*self.guess[2]*self.recovered_data[i-1])/self.suseptible_data[i])**2
        return error
   
    def networked_chi_squared(self, G, period_evaluation):
        error = 0
        for i in range(1,min(period_evaluation, len(self.suseptible_data))):
            system_update = network_update(G, self.guess[0], self.guess[1], self.guess[2], 1, self.time_step, 1)
            guess = system_update.deterministic_SIRS()
            error += abs((self.suseptible_data[i] - sum(guess[0]))/self.suseptible_data)**2
            error += abs((self.infected_data[i] - sum(guess[1]))/self.suseptible_data[i])**2
            error += abs((self.recovered_data[i] - sum(guess[2]))/self.recovered_data[i])**2
            
            suseptible_rescale = self.suseptible_data[i]/sum(guess[0])
            infected_rescale = self.infected_data[i]/sum(guess[1])
            recovered_rescale = self.recovered_data/sum(guess[2])
            
            for j in G:
                G.nodes[j]['state'] = [guess[0][j]*suseptible_rescale, guess[1][j]*infected_rescale, guess[2]]
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
    return error.chi_squared_data()

def network_initialisation( initial_conditions, number_of_nodes):
    
    G = nx.Graph()
    
    for i in range(0,number_of_nodes):
        
        G.add_node(i, state=initial_conditions)
    
    for i in range(0,number_of_nodes):
        
        for j in range(0,number_of_nodes):
            
            if i != j:
                
                G.add_edge(i, j, weight = 1)
                
            else:
                
                G.add_edge(i, j, weight = 0)
                
    population = 0
    
    for i in G:
        
        population += sum(G.nodes[i]["state"])
        
    return G, population

def data_generator(itterations, number_of_nodes, initial_conditions, infection_rate, recovery_rate, resuseptibility_rate, time_step):
    
    G, population = network_initialisation(initial_conditions, number_of_nodes)
    
    update = network_update(G, infection_rate/sum(G.nodes[0]["state"]), recovery_rate, resuseptibility_rate, 1, time_step, itterations)
    
    suseptible,infected,recovered, time = update.deterministic_SIRS()
    
    fig = plt.figure()
    plt.plot(time[1:], suseptible[1:], label = 'suseptible')
    plt.plot(time[1:], infected[1:], label = 'infected')
    plt.plot(time[1:], recovered[1:], label = 'recovered')
    plt.legend()
    plt.show()
    
    return suseptible, infected, recovered

def data_fitter(suseptible, infected, recovered, itterations, time_step):
    
    P, population = network_initialisation([suseptible[0], infected[0], recovered[0]], 1)
    
    bound= [(0,20)]
    
    estimate = scipy.optimize.minimize(scipy_mediator, [1/population,0.05,0.01], [suseptible, infected, recovered,time_step,1, 0, 1],bounds=bound)
    
    print([estimate.x[0]*population, estimate.x[1], estimate.x[2]])
    print([abs((estimate.x[0]*population-0.1)*100/0.1), abs(estimate.x[1] - 0.005)*100/0.005, abs(estimate.x[2] - 0.001)*100/0.001])
    print(estimate.fun)
    update = network_update(P, estimate.x[0], estimate.x[1], estimate.x[2], 1, time_step, itterations)
    suseptible1,infected1,recovered1, time1 = update.deterministic_SIRS()
    fig = plt.figure()
    plt.plot(time1, suseptible1, label = 'suseptible')
    plt.plot(time1, infected1, label = 'infected')
    plt.plot(time1, recovered1, label = 'recovered')
    plt.legend()
    plt.show()

def main(itterations, number_of_nodes, inital_conditions, infection_rate, recovery_rate, resuseptibility_rate, time_step):
    suseptible, infected, recovered = data_generator(itterations, number_of_nodes, inital_conditions, infection_rate, recovery_rate, resuseptibility_rate, time_step)
    data_fitter(suseptible, infected, recovered, itterations, time_step)

main(2000,10,[1000,10,1000], 0.1,0.005,0.001,1)
    
    
    
