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
    
    def SIRS_with_network_data(self):
        suseptible_data = []
        infected_data = []
        recovered_data = []
        time = []
        new_states = []
        present_state = []
        for i in range(len(self.G)):
           present_state.append(self.G.nodes[i]['state'])
        for j in range(1):
            suseptible = []
            infected = []
            recovered = []
            for i in range(len(self.G)):
                
                neighbors = list(self.G.neighbors(i))
                additional_infectives = 0
                for k in neighbors:
                    additional_infectives += present_state[k][1]*self.G[i][k]['weight']
                
                internal_population = present_state[i][0]+ present_state[i][1]+ present_state[i][2]
                self_interaction = next_timestep(present_state[i][0], present_state[i][1], present_state[i][2],internal_population , self.infection_rate, self.recovery_rate, self.time_step, self.resuseptibility_rate, additional_infectives)
                new_state = self_interaction.SIRS()
                new_states.append(new_state.copy())
                suseptible.append(new_state[0])
                infected.append(new_state[1])
                recovered.append(new_state[2])
        return [suseptible, infected, recovered]
class data_analysis(object):
    
    def __init__(self, suseptible_data, infected_data, recovered_data, guess, time_step, itterations,j, network):
        
        self.suseptible_data = suseptible_data
        self.infected_data = infected_data
        self.recovered_data = recovered_data
        self.guess = guess
        self.time_step = time_step
        self.itterations = itterations
        self.j = j
        self.network = network
        
    def chi_squared_data(self):
        error = 0
        population = self.suseptible_data[0] + self.infected_data[0] + self.recovered_data[0]
        for i in range(1,len(self.suseptible_data)):
            error += abs((self.suseptible_data[i] - self.suseptible_data[i-1] + self.time_step*self.guess[0]*self.suseptible_data[i-1]*self.infected_data[i-1] - self.time_step*self.guess[2]*self.recovered_data[i-1])/self.suseptible_data[i])**(2)
            error += abs((self.infected_data[i] - self.infected_data[i-1] - self.time_step*self.guess[0]*self.suseptible_data[i-1]*self.infected_data[i-1] + self.time_step*self.guess[1]*self.infected_data[i-1])/self.infected_data[i])**(2)
            error += abs((self.recovered_data[i] - self.recovered_data[i-1] +self.time_step*self.guess[2]*self.recovered_data[i-1] - self.time_step*self.guess[1]*self.infected_data[i-1])/self.recovered_data[i])**(2)
        return error
   
    def networked_chi_squared(self):
        error = 0
        for i in range(1,len(self.suseptible_data)):
            
            system_update = network_update(self.network, self.guess[0], self.guess[1], self.guess[2], 1, self.time_step, 1)
            
            guess = system_update.SIRS_with_network_data()

            error += abs((self.suseptible_data[i] - sum(guess[0]))/self.suseptible_data[i])**2
            error += abs((self.infected_data[i] - sum(guess[1]))/self.infected_data[i])**2
            error += abs((self.recovered_data[i] - sum(guess[2]))/self.recovered_data[i])**2
            
            suseptible_correction = self.suseptible_data[i]/sum(guess[0])
            infected_correction = self.infected_data[i]/sum(guess[1])
            recovered_correction = self.recovered_data[i]/sum(guess[2])
            
            for j in self.network:
                self.network.nodes[j]['state'] = [guess[0][j]*suseptible_correction, guess[1][j]*infected_correction, guess[2][j]*recovered_correction]
            
        return error
    def fourier_transform(self):
        fft_result = np.fft.fft(self.suseptible_data)
        freqs = np.fft.fftfreq(len(self.suseptible_data), 1)
        amplitude = np.abs(fft_result) / len(self.suseptible_data)
        #fig = plt.figure()
        #plt.plot(freqs[1:len(freqs)//2], amplitude[1:len(amplitude)//2])
        #plt.show()
        return np.log10(freqs[1:len(freqs)//2]), np.log10(amplitude[1:len(amplitude)//2])
def scipy_mediator(guess, other_data):
    
    error = data_analysis(other_data[0], other_data[1], other_data[2], guess, other_data[3], other_data[4], other_data[5],1)
    return error.chi_squared_data()
def scipy_mediator2(guess, other_data):
    network, pop = network_initialisation2([other_data[0][0]/2, other_data[1][0]/2, other_data[2][0]/2], 2, guess[3])
    error = data_analysis(other_data[0], other_data[1], other_data[2], guess, other_data[3], other_data[4], other_data[5],network)
    return error.networked_chi_squared()

def network_initialisation( initial_conditions, number_of_nodes):
    
    G = nx.Graph()
    
    for i in range(0,number_of_nodes):
        
        G.add_node(i, state=initial_conditions)
    
    for i in range(0,number_of_nodes):
        
        for j in range(0,number_of_nodes):
            
            if i +1 == j:
                G.add_edge(i, j, weight = max(0,np.random.gumbel(0,1,1)))   
                
            else:
                
                G.add_edge(i, j, weight = 0)
    #print(G.get_edge_data(0, 1))           
    population = 0
    
    for i in G:
        
        population += sum(G.nodes[i]["state"])
        
    return G, population

def network_initialisation2( initial_conditions, number_of_nodes, link):

    G = nx.Graph()
    
    for i in range(0,number_of_nodes):
        
        G.add_node(i, state=initial_conditions)
    
    for i in range(0,number_of_nodes):
        
        for j in range(0,number_of_nodes):
            
            if i != j:
                
                G.add_edge(i, j, weight = link)
                
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
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    plt.show()
    
    fig = plt.figure()
    plt.plot(time[1:], suseptible[1:], label = 'suseptible')
    plt.plot(time[1:], infected[1:], label = 'infected')
    plt.plot(time[1:], recovered[1:], label = 'recovered')
    plt.legend()
    plt.show()
    
    
    return suseptible, infected, recovered

def data_fitter(suseptible, infected, recovered, itterations, time_step):
    
    P, population = network_initialisation([suseptible[0], infected[0], recovered[0]], 1)
    
    bound= [(0,0.1), (0,0.1), (0,0.1)]
    
    estimate1 = scipy.optimize.minimize(scipy_mediator, [0.1,0.01,0.0005], [suseptible, infected, recovered,time_step,1, 0, 1, 1],bounds=bound, tol = 0.0000000000001)
    estimate = scipy.optimize.minimize(scipy_mediator, [estimate1.x[0],estimate1.x[1],estimate1.x[2]], [suseptible, infected, recovered,time_step,1, 0, 1, 1],bounds=bound, tol = 0.0000000000001)
    print(estimate.x)
    #print([abs((estimate.x[0]*population-0.1)*100/0.1), abs(estimate.x[1] - 0.01)*100/0.01, abs(estimate.x[2] - 0.0005)*100/0.0005])
    print("First order chi^2: " + str(estimate.fun))
    update = network_update(P, estimate.x[0], estimate.x[1], estimate.x[2], 1, time_step, itterations)
    suseptible1,infected1,recovered1, time1 = update.deterministic_SIRS()
    
    fig = plt.figure()
    plt.plot(time1, suseptible1, label = 'suseptible')
    plt.plot(time1, infected1, label = 'infected')
    plt.plot(time1, recovered1, label = 'recovered')
    plt.legend()
    plt.show()
    bound= [(0,0.1), (0,0.1), (0,0.1), (0,10)]
    s = scipy.optimize.minimize(scipy_mediator2, [0.1, estimate.x[1], estimate.x[2], 0.1], [suseptible, infected, recovered,time_step,1, 0, 1, 2],bounds=bound, tol = 0.0000000000001)
    Second_order_estimate = scipy.optimize.minimize(scipy_mediator2, [s.x[0],s.x[1],s.x[2], s.x[3]], [suseptible, infected, recovered,time_step,1, 0, 1, 2],bounds=bound, tol = 0.0000000000001)
    print("Second order chi^2: " +str(Second_order_estimate.fun))
    print(Second_order_estimate.x)
    
    K, population = network_initialisation([suseptible[0]/2, infected[0]/2, recovered[0]/2], 2)
    suseptible2,infected2,recovered2, time2 = update.deterministic_SIRS()
    second_order = network_update(K, Second_order_estimate.x[0], Second_order_estimate.x[1], Second_order_estimate.x[2], 1, time_step, itterations)

    fig = plt.figure()
    plt.plot(time1, suseptible1, label = 'suseptible')
    plt.plot(time1, infected1, label = 'infected')
    plt.plot(time1, recovered1, label = 'recovered')
    plt.legend()
    plt.show()
    
def main(itterations, number_of_nodes, inital_conditions, infection_rate, recovery_rate, resuseptibility_rate, time_step):
    suseptible, infected, recovered = data_generator(itterations, number_of_nodes, inital_conditions, infection_rate, recovery_rate, resuseptibility_rate, time_step)
    data_fitter(suseptible, infected, recovered, itterations, time_step)

main(1000,1,[10000,10000,10000], 0.1,0.01,0.0005,1)
    

    
