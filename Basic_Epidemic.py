#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 14:13:04 2025

@author: thomas
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 19:30:15 2025

@author: thomas
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 14:53:47 2025

@author: thomas
"""

import numpy as np 
import matplotlib.pyplot as plt
import random
import math
import scipy.stats as stats
import scipy
import time as tn


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
        
        return new_suseptible, new_infected, new_recovered
   
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
       new_suseptible = max(self.suseptible - a+ c,0)
       new_infected = max(self.infected + a - b,0)
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
class simulation(object):
    
   def __init__(self, suseptible, infected, recovered, population, infection_rate, recovery_rate, time_step, itterations,resuseptibility_rate, other_data):
       self.itterations = itterations
       
       self.suseptible = np.zeros(int(self.itterations + 1))
       self.infected = np.zeros(int(self.itterations + 1))
       self.recovered = np.zeros(int(self.itterations + 1))
       
       self.suseptible[0] = suseptible[0]
       self.infected[0] = infected[0]
       self.recovered[0] = recovered[0]
       
       self.population = population
       self.infection_rate = infection_rate
       self.recovery_rate = recovery_rate
       self.time_step = time_step
       self.resuseptibility_rate = resuseptibility_rate
       self.other_variables = other_data
   
   def SIR_RUN(self):
        
        time = np.zeros(self.itterations)
        
        start = tn.time()
        for i in range(self.itterations):
            
            next_time_step = next_timestep(self.suseptible[i], self.infected[i], self.recovered[i], self.population, self.infection_rate, self.recovery_rate, self.time_step, self.resuseptibility_rate,0)
            self.suseptible[i+1],self.infected[i+1],self.recovered[i+1] = next_time_step.SIR()
            
            time[i] = i*self.time_step
        end = tn.time()
        #print(end-start)
        return self.suseptible, self.infected, self.recovered, time
  
   def SIRS_RUN(self):
         
         time = np.zeros(self.itterations)
         start = tn.time()
         for i in range(self.itterations):
             
             next_time_step = next_timestep(self.suseptible[i], self.infected[i], self.recovered[i], self.population, self.infection_rate, self.recovery_rate, self.time_step, self.resuseptibility_rate,0)
             self.suseptible[i+1],self.infected[i+1],self.recovered[i+1] = next_time_step.SIRS()

             time[i] = i*self.time_step
         end = tn.time()
         #print(end-start)
         return self.suseptible, self.infected, self.recovered, time
     
   def Random_SIRS(self):
       start = tn.time()
       time = np.zeros(self.itterations)
       for i in range(self.itterations):
           
           next_time_step = next_timestep(self.suseptible[i], self.infected[i], self.recovered[i], self.population, self.infection_rate, self.recovery_rate, self.time_step, self.resuseptibility_rate,0)
           self.suseptible[i+1],self.infected[i+1],self.recovered[i+1] = next_time_step.Binomial_SIRS()
           time[i] = i*self.time_step
       end = tn.time()
       #print(end-start)
       return self.suseptible, self.infected, self.recovered, time
   
   def dual_SIR(self):
        
        time = np.zeros(self.itterations)
        
        self.suseptible_2 = np.zeros(int(self.itterations + 1))
        self.infected_2 = np.zeros(int(self.itterations + 1))
        self.recovered_2 =np.zeros(int(self.itterations + 1))
        
        self.suseptible_2[0] = self.other_variables[0][0]
        self.infected_2[0] = self.other_variables[1][0]
        self.recovered_2[0] = self.other_variables[2][0]
        
        start = tn.time()
        
        for i in range(self.itterations):
            
            next_time_step = next_timestep(self.suseptible[i], self.infected[i], self.recovered[i], self.population, self.infection_rate, self.recovery_rate, self.time_step, self.resuseptibility_rate, [self.suseptible_2[i], self.infected_2[i], self.recovered_2[i], self.other_variables[3]])
            self.suseptible[i+1], self.infected[i+1], self.recovered[i+1], self.suseptible_2[i+1], self.infected_2[i+1], self.recovered_2[i+1] = next_time_step.DUAL_SIR()            
            time[i] = i*self.time_step
        end = tn.time()
       
        return self.suseptible, self.infected, self.recovered, self.suseptible_2, self.infected_2, self.recovered_2,  time
   def dual_SIR_Binomial(self):
     
             
             time = np.zeros(self.itterations)
             
             self.suseptible_2 = np.zeros(int(self.itterations + 1))
             self.infected_2 = np.zeros(int(self.itterations + 1))
             self.recovered_2 =np.zeros(int(self.itterations + 1))
             
             self.suseptible_2[0] = self.other_variables[0][0]
             self.infected_2[0] = self.other_variables[1][0]
             self.recovered_2[0] = self.other_variables[2][0]
             
             start = tn.time()
             
             for i in range(self.itterations):
                 
                 next_time_step = next_timestep(self.suseptible[i], self.infected[i], self.recovered[i], self.population, self.infection_rate, self.recovery_rate, self.time_step, self.resuseptibility_rate, [self.suseptible_2[i], self.infected_2[i], self.recovered_2[i], self.other_variables[3]])
                 self.suseptible[i+1], self.infected[i+1], self.recovered[i+1], self.suseptible_2[i+1], self.infected_2[i+1], self.recovered_2[i+1] = next_time_step.DUAL_BIN_SIR()            
                 time[i] = i*self.time_step
             end = tn.time()
             
             return self.suseptible, self.infected, self.recovered, self.suseptible_2, self.infected_2, self.recovered_2,  time

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
    
    def chi_squared_error(self):
       
        disease_progress = simulation([self.suseptible_data[0]], [self.infected_data[0]], [self.recovered_data[0]],self.suseptible_data[0] + self.infected_data[0] + self.recovered_data[0], self.guess[0], self.guess[1], self.time_step, self.itterations, self.guess[2])
        suseptible_data, infected_data, recovered_data, time = disease_progress.SIRS_RUN()
        
        total_error_data = suseptible_data + infected_data - self.suseptible_data - self.infected_data
        total_error = 0
        for i in range(len(total_error_data)):
            total_error += abs(total_error_data[i])
        return total_error
    def binomial(self, i):
        
        if self.j ==0:
            
            change_in_suseptible = min(0,self.suseptible_data[i] - self.suseptible_data[i+1])
        
            chance_of_occourance = stats.binom.pmf(change_in_suseptible, self.suseptible_data[i], 1 - math.e**(-self.time_step*self.guess[0]*self.infected_data[i]))
        else:
            change_in_recovered = self.recovered_data[i+1] - self.recovered_data[i]
        
            chance_of_occourance = stats.binom.pmf(change_in_recovered, self.infected_data[i], 1 - math.e**(-self.time_step*self.guess[1]))
        return 10*chance_of_occourance
    def likihood(self):
        
        likihood = self.scale
        for i in range(len(self.suseptible_data) - 1):
            likihood = likihood*self.binomial(i)
        return -likihood
def scipy_mediator(guess, other_data):  
    error = data_analysis(other_data[0], other_data[1], other_data[2], guess, other_data[3], other_data[4], other_data[5], other_data[6])
    return error.likihood()
#def initial_guess():
def initial_guess(base_suseptible, base_infected, base_recovered, time_step, itterations):
    minimum_1 = 00000
    guess = [0,0,0]
    bound = [(0,20)]
    best_guess_1 = 0
    for i in range(1,20):
       probability = -scipy_mediator([i,0,0], [base_suseptible, base_infected, base_recovered,time_step,itterations, 0, 1])
       
       if probability > minimum_1:
           minimum_1 = probability
           best_guess_1 = i
    
    temp_guess_1 = 0
    minimum_1 = 0
    for j in range(-10,10):
       probability = -scipy_mediator([best_guess_1 + j/10, 0,0], [base_suseptible, base_infected, base_recovered,time_step,itterations, 0, 1])
       
       if probability > minimum_1:
           minimum_1 = probability
           temp_guess_1 = j
    
    inverse_scale = -scipy_mediator([best_guess_1 + temp_guess_1/10, 0,0], [base_suseptible, base_infected, base_recovered,time_step,itterations, 0, 1])
    if inverse_scale == 0:
        scale = 1
    else:
        scale = 1/inverse_scale
    guess[0] = scipy.optimize.minimize(scipy_mediator,[temp_guess_1/10 + best_guess_1], [base_suseptible, base_infected, base_recovered,time_step,itterations,0, scale], bounds=bound, method='SLSQP', tol = 0.0000000000000000000000001)
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

def main(itterations):
    population = 26
    infection_rate = 3
    recovery_rate = 10
    time_step = 0.00022
    
    resuseptibility_rate = 00
    suseptible = np.ones(1)
    infected = np.ones(1)
    recovered = np.ones(1)
    
    suseptible = population*suseptible
    infected =  infected
    recovered = 0*population*recovered
    

    
    #print(suseptible_data)
   
    
   

    #random_progress = simulation(suseptible, infected, recovered, population, infection_rate, recovery_rate, time_step, itterations, resuseptibility_rate, 0)
    #base_suseptible, base_infected, base_recovered, time = random_progress.Random_SIRS()
    
    #normal = []
    #for i in range(1000, itterations):
    #    normal.append(base_suseptible[i])
    #fig = plt.figure()
    #stats.probplot(normal, dist="norm", plot=plt)
    #plt.show()
    
    #guess = initial_guess(base_suseptible, base_infected, base_recovered, time_step, itterations)
    #print(guess)
    disease_progress = simulation(suseptible, 7*infected, recovered, population, infection_rate, recovery_rate, time_step, itterations, resuseptibility_rate, [suseptible, infected, recovered, [2,10,0]])
    suseptible_data, infected_data, recovered_data, suseptible_data_2, infected_data_2, recovered_data_2, time = disease_progress.dual_SIR_Binomial()
    #time = time/time_step
    #fig = plt.figure()
    #plt.plot(time, base_recovered[1:])
    #plt.plot(time, base_suseptible[1:])
    #plt.plot(time, base_infected[1:])
    #plt.plot(time, suseptible_data[1:], 'k')
    #plt.plot(time, infected_data[1:], 'k')
    #plt.plot(time, recovered_data[1:],'k')
    #plt.plot(time, suseptible_data_2[1:], 'b')
    #plt.plot(time, infected_data_2[1:], 'b')
    #plt.plot(time, recovered_data_2[1:],'b')
    #plt.show()
    #fig2 = plt.figure()
    #plt.plot(suseptible_data, infected_data, 'k')
    #plt.plot(base_suseptible, base_infected, 'b')
    #plt.show()
    #print(guess[0]/guess[1])
    return [suseptible_data_2, time]
def plot_error():
    
    population = 26
    gratings = 10
    itteration_data = []
    error = []
    time_steps = 1000
    time = main(time_steps)[1]
    itterations = 100000
    colour_plot = []
    colour_row = np.zeros(time_steps)
    reduced_colour_plot = []
    reduced_colour_row  = np.zeros(int(time_steps/gratings))
    for i in range(0,population + 1):
        colour_plot.append(colour_row.copy())
        reduced_colour_plot.append(reduced_colour_row.copy())
    
    for i in range(itterations):
        error = main(time_steps)[0]
        for j in range(len(error) - 1):
            colour_plot[int(error[j])][j] += 1
    plt.imshow(colour_plot,cmap='gray', aspect = 'auto')
plot_error()
#main(10000)
        
        
        
    