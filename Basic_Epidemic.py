import numpy as np 
import matplotlib.pyplot as plt

import random
import math
import scipy.stats as stats
import scipy
import time as tn
from scipy.ndimage import gaussian_filter1d
from scipy.stats import genextreme
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
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
    
    def Binomial_SIRS(self):
        
       a = np.random.poisson(self.time_step*self.infection_rate*self.infected*self.suseptible)
       b = np.random.poisson(self.time_step*self.recovery_rate*self.infected)
       c = np.random.poisson(self.time_step*self.resuseptibility_rate*self.recovered)
       #a = np.random.normal(self.time_step*self.infection_rate*self.infected*self.suseptible, 0.1000*self.time_step*self.infection_rate*self.infected*self.suseptible)
       #b = np.random.normal(self.time_step*self.recovery_rate*self.infected, 0.1000*self.time_step*self.recovery_rate*self.infected)
       #c = np.random.normal(self.time_step*self.resuseptibility_rate*self.recovered,0.1000*self.time_step*self.resuseptibility_rate*self.recovered)
       #print(a)
       new_suseptible = max(self.suseptible - a + c,0)
       new_infected = max(self.infected + a - b,0)
       new_recovered = max(self.recovered + b - c,0)
   
       return new_suseptible, new_infected, new_recovered, a
    
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
    
    def envirometal_stochasticity(self, additional_infection_rate):
         #eigenvalue = -self.infected - self.recovery_rate - self.recovered + self.suseptible - (abs((-self.infected - self.recovery_rate - self.recovered + self.suseptible)**2 - 4*((-self.infected - self.resuseptibility_rate)*(self.suseptible - self.recovery_rate) - self.infected*(-self.suseptible - self.resuseptibility_rate))))**0.5
         #self.time_step = min(0.05,0.01/abs(eigenvalue))
         
         infections = self.suseptible*self.infected*(self.infection_rate*self.time_step + additional_infection_rate*(self.time_step**(1/2)))
         recoveries = self.recovery_rate*self.infected*self.time_step
         immunity_losses = self.resuseptibility_rate*self.recovered*self.time_step
         
         new_suseptible = self.suseptible - infections + immunity_losses
         new_infected = self.infected + infections - recoveries
         new_recovered = self.recovered + recoveries - immunity_losses

         return new_suseptible, new_infected, new_recovered, self.time_step
     
    def SIR_Birth_and_Death(self, additional_infection_rate):
         
         infections = self.infection_rate*self.suseptible*self.infected*self.time_step + additional_infection_rate*(self.time_step**(1/2))*self.suseptible*self.infected
         recoveries = self.recovery_rate*self.infected*self.time_step
         
         new_suseptible = self.suseptible - infections + self.resuseptibility_rate*(self.infected + self.recovered)*self.time_step
         new_infected = self.infected + infections - recoveries -  self.resuseptibility_rate*(self.infected)*self.time_step
         new_recovered = self.recovered + recoveries -  self.resuseptibility_rate*(self.recovered)*self.time_step
         
         return new_suseptible, new_infected, new_recovered
    
    def network_envirometal_stochasticity(self, additional_infection_rate):

         additional_infection_matrix = self.other_variables*(np.roll(self.infected, shift=-1, axis=0) + np.roll(self.infected, shift=1, axis=0) + np.roll(self.infected, shift=-1, axis=1) + np.roll(self.infected, shift=1, axis=1))
         infections = self.infection_rate*self.suseptible*(self.infected+additional_infection_matrix)*(self.time_step + additional_infection_rate*(self.time_step**0.5)*np.random.normal(0,1)*np.ones((len(additional_infection_matrix),len(additional_infection_matrix[0]))))#np.random.normal(0,1,(len(additional_infection_matrix),len(additional_infection_matrix[0]))))
         recoveries = self.infected*self.time_step*self.recovery_rate
         immunity_losses = self.recovered*self.time_step*self.resuseptibility_rate
         
         new_suseptible = self.suseptible - infections + immunity_losses
         new_infected = self.infected +infections - recoveries
         new_recovered = self.recovered + recoveries - immunity_losses
         
         return new_suseptible, new_infected, new_recovered
     
    def SIRS_multiparameter(self, additional_infection_rate, recovery_matrix, immunity_loss_matrix):
       
        infections = self.suseptible*self.infected*(self.time_step + additional_infection_rate*(self.time_step**0.5))
        
        recoveries = self.time_step*self.infected*recovery_matrix
        immunity_losses = self.time_step*self.recovered*immunity_loss_matrix
        
        new_suseptible = self.suseptible - infections + immunity_losses
        new_infected = self.infected +infections - recoveries
        new_recovered = self.recovered + recoveries - immunity_losses
        
        return new_suseptible, new_infected, new_recovered
    
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
       total_infections = []
       total_infections.append(0)
       population = self.infected[0] + self.recovered[0] + self.suseptible[0]
       for i in range(self.itterations):
           
           next_time_step = next_timestep(self.suseptible[i], self.infected[i], self.recovered[i], self.population, self.infection_rate, self.recovery_rate, self.time_step, self.resuseptibility_rate,self.other_variables)
           self.suseptible[i+1],self.infected[i+1],self.recovered[i+1],infections = next_time_step.Binomial_SIRS()
           total_infections.append(total_infections[i] + (infections)/population)
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
     
             
             time = np.zeros(self.itterations + 1)
             
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
   def enviromental_stochastic_SIRS(self, varience_in_noise):
    
      time = np.linspace(0, (self.itterations)*self.time_step,(self.itterations+1))
      start = tn.time()
      noise = varience_in_noise*np.random.normal(0,1,self.itterations)
      recovered = self.recovered[0]
      for i in range(self.itterations):
          
          next_time_step = next_timestep(self.suseptible[i], self.infected[i], recovered, self.population, self.infection_rate, self.recovery_rate, self.time_step, self.resuseptibility_rate,0)
          self.suseptible[i+1],self.infected[i+1],recovered,dt = next_time_step.envirometal_stochasticity(noise[i])

          #time[i+1] = time[i] + dt 
      end = tn.time()
      #print(end-start)
      return self.suseptible, self.infected, self.recovered, time
  
   def SIRBD(self, varience_in_noise):
          
          time = np.zeros(self.itterations)
          start = tn.time()
          noise = varience_in_noise*np.random.normal(0,1,self.itterations)
          for i in range(self.itterations):
              
              next_time_step = next_timestep(self.suseptible[i], self.infected[i], self.recovered[i], self.population, self.infection_rate, self.recovery_rate, self.time_step, self.resuseptibility_rate,0)
              self.suseptible[i+1],self.infected[i+1],self.recovered[i+1] = next_time_step.SIR_Birth_and_Death(noise[i])

              time[i] = i*self.time_step
          end = tn.time()
          #print(end-start)
          return self.suseptible, self.infected, self.recovered, time
   def network_enviromental_stochastic_SIRS(self, varience_in_noise, network_dimentions, external_interaction):
         time = np.linspace(0, (self.itterations)*self.time_step,(self.itterations+1))
         self.network_suseptible = self.suseptible[0]*np.ones((network_dimentions[0],network_dimentions[1]))
         self.network_infected = self.infected[0]*np.ones((network_dimentions[0],network_dimentions[1]))
         self.network_recovered = self.recovered[0]*np.ones((network_dimentions[0],network_dimentions[1]))
         reduced_infection_rate = self.infection_rate/(1+4*external_interaction)
         number_of_nodes = network_dimentions[0]*network_dimentions[1]
         for i in range(0,self.itterations):
             next_time_step=next_timestep(self.network_suseptible, self.network_infected,self.network_recovered, 1, reduced_infection_rate, self.recovery_rate, self.time_step, self.resuseptibility_rate, external_interaction)
             self.network_suseptible,self.network_infected,self.network_recovered = next_time_step.network_envirometal_stochasticity(varience_in_noise)
             #self.suseptible[i+1] = np.sum(self.network_suseptible)/number_of_nodes
             #self.infected[i+1] = np.sum(self.network_infected)/number_of_nodes
             self.suseptible[i+1] = self.network_suseptible[0][0]
             self.infected[i+1] = self.network_infected[0][0]
         self.suseptible = self.suseptible
         self.infected = self.infected
         return self.suseptible, self.infected, self.recovered, time
     

   def total_data_network_enviromental_stochastic_SIRS(self, varience_in_noise, network_dimentions, external_interaction):
           time = np.linspace(0, (self.itterations)*self.time_step,(self.itterations+1))
           self.network_suseptible = np.random.uniform(0,0.5,(network_dimentions[0],network_dimentions[1]))*np.ones((network_dimentions[0],network_dimentions[1]))
           self.network_infected =np.random.uniform(0,0.5,(network_dimentions[0],network_dimentions[1]))*np.ones((network_dimentions[0],network_dimentions[1]))
           self.network_recovered = np.ones((network_dimentions[0],network_dimentions[1])) - self.network_suseptible - self.network_infected # self.recovered[0]*np.ones((network_dimentions[0],network_dimentions[1]))
          
           #self.network_recovered[25][25] = 0
           reduced_infection_rate = self.infection_rate/(1+4*external_interaction)
           number_of_nodes = network_dimentions[0]*network_dimentions[1]
           self.suseptible = []
           self.infected = []
           for i in range(0,self.itterations):
               next_time_step=next_timestep(self.network_suseptible, self.network_infected,self.network_recovered, 1, reduced_infection_rate, self.recovery_rate, self.time_step, self.resuseptibility_rate, external_interaction)
               self.network_suseptible,self.network_infected,self.network_recovered = next_time_step.network_envirometal_stochasticity(varience_in_noise)
               #self.suseptible[i+1] = np.sum(self.network_suseptible)/number_of_nodes
               #self.infected[i+1] = np.sum(self.network_infected)/number_of_nodes
               self.suseptible.append(self.network_suseptible)
               self.infected.append(self.network_infected)
           self.suseptible = self.suseptible
           self.infected = self.infected
           return self.suseptible, self.infected, self.recovered, time
       
   def multi_parameter_SIRS(self, resolution,varience_in_noise):
       itterations = 1000000
       rec = np.linspace(0.5/resolution, 2,resolution)
       
       g = np.tile(rec, (resolution, 1))
       l = np.rot90(g.copy())
       node_i = []
       self.suseptible = g.copy()
       self.infected = l.copy()*(np.ones((resolution,resolution)) - g.copy())/(l.copy()+g.copy())
       self.recovered = np.ones((resolution,resolution)) - self.suseptible - self.infected
       const = ((g*l*(np.ones((resolution,resolution))-g))/(g+l))**2
       for i in range(0,resolution):
           for j in range(0,resolution):
               if g[i][j] > 1:
                   g[i][j] = 0
                   self.suseptible[i][j] = 1
                   self.infected[i][j] = 0
                   self.recovered[i][j] = 0
                   const[i][j] = 0 
       energy = np.zeros((resolution,resolution))
       time_step = 0.005
       noise = varience_in_noise*np.random.normal(0,1,(itterations,resolution,resolution))
       for i in range(1,itterations):
           
           
           forward = next_timestep(self.suseptible, self.infected, self.recovered, 1, 1, g, time_step, l, 0)
           new_s,new_i,new_r,dt = forward.envirometal_stochasticity(noise[i])
           loc_energy = (self.suseptible - new_s)**2 +(self.infected - new_i)**2
           energy = energy*(i-1)/i + 0.5*loc_energy/i
           self.suseptible = new_s
           self.infected = new_i
       
       energy = energy/(time_step*varience_in_noise*varience_in_noise)
       
       for i in range(0,resolution):
           for j in range(0,resolution):
               if self.infected[i][j] == 0:
                   const[i][j] = 0
               if g[i][j] == 0:
                   energy[i][j] = 0
       
       return np.log10(energy/const)
   def multi_parameter_SIRS_freq(self, resolution,varience_in_noise):
       itterations = 500
       rec = np.linspace(0.5/resolution, 2,resolution)
       data_s = []
       data_i = []
       g = np.tile(rec, (resolution, 1))
       l = np.rot90(g.copy())
       node_i = []
       self.suseptible = g.copy()
       self.infected = l.copy()*(np.ones((resolution,resolution)) - g.copy())/(l.copy()+g.copy())
       self.recovered = np.ones((resolution,resolution)) - self.suseptible - self.infected
       const = ((g*l*(np.ones((resolution,resolution))-g))/(g+l))**2
       for i in range(0,resolution):
           for j in range(0,resolution):
               if g[i][j] > 1:
                   g[i][j] = 0
                   self.suseptible[i][j] = 1
                   self.infected[i][j] = 0
                   self.recovered[i][j] = 0
                   const[i][j] = 0 
       energy = np.zeros((resolution,resolution))
       time_step = 0.0005
       noise = varience_in_noise*np.random.normal(0,1,(itterations,resolution,resolution))
       s_data = []
       i_data =[]
       for i in range(1,itterations):
           
           
           forward = next_timestep(self.suseptible, self.infected, self.recovered, 1, 1, g, time_step, l, 0)
           new_s,new_i,new_r,dt = forward.envirometal_stochasticity(noise[i])
          
           loc_energy = (self.suseptible - new_s)**2 +(self.infected - new_i)**2
           energy = energy*(i-1)/i + 0.5*loc_energy/i
           
           self.suseptible = new_s
           self.infected = new_i
           self.recovered = new_r
           
           
           s_data.append(new_s.copy())
           i_data.append(new_i.copy())
       energy = energy/(time_step*varience_in_noise*varience_in_noise)
       
       return s_data,i_data,dt, (energy/const)
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
        
    def fourier_transform(self,data,dt):
        
        fft = np.fft.fft(data)
        freq = np.fft.fftfreq(len(data), d = dt)
        self.freqs = freq[1:int(len(freq)/2)]
        #print(self.freqs)
        self.magnitude = np.abs(fft[1:int(len(freq)/2)])**2
    
    def data_compressor(self, data, resolution):
        range_of_data = max(data) - min(data)
        data = data - np.ones(len(data))*min(data)
        reduced_data = data*resolution/range_of_data
        return reduced_data
    def integer(self,value):
        minimum = math.floor(value) 
        if value - minimum < 0.5:
            return int(minimum)
        else:
            return int(minimum + 1)
        
    def colour_pdf(self, x_axis, y_axis, resolution):
        colour_row = np.zeros(resolution + 1)
        colour_matrix = []
        for i in range(resolution + 1):
            colour_matrix.append(colour_row.copy())
        x_reduced = self.data_compressor(x_axis, resolution)
        y_reduced = self.data_compressor(y_axis, resolution)
        fig = plt.figure()
        for i in range(len(x_reduced)):
            colour_matrix[self.integer(x_reduced[i])][self.integer(y_reduced[i])] += 1
        plt.imshow(colour_matrix, aspect= 'equal')
        
        plt.show()
        plt.figure()
        x_average = sum(x_reduced)/len(x_reduced)
        y_average = sum(y_reduced)/len(y_reduced)
        for i in range(resolution):
            for j in range(resolution):
                
                
                colour_matrix[i][j] = colour_matrix[i][j]*(((i - x_reduced[0])**2)+(j - y_reduced[0])**2)
        
        plt.imshow(colour_matrix, aspect= 'equal')
        
        plt.show()
    
    def period_finder(self,resolution):
       freq_pres = np.zeros((resolution,resolution))
       runs = 100
       max_s = np.zeros((resolution,resolution))
       max_i = np.zeros((resolution,resolution))
       energy = np.zeros((resolution,resolution))
       for q in range(runs):
           s,i,dt, e = simulation([1], [1], [1], 1, 1, 1, 0.01, 1000, 1, 1).multi_parameter_SIRS_freq(resolution,  0.000001)
           
           s_fft_data = np.abs(np.fft.fft(s, axis=0))
           i_fft_data = np.abs(np.fft.fft(i,axis = 0))
           
           red_s_fft_data =  s_fft_data[1:int(len(s_fft_data)/2)]
           red_i_fft_data =  i_fft_data[1:int(len(s_fft_data)/2)]
           
           freq = np.fft.fftfreq(len(s_fft_data), d = dt)
           self.freqs = freq[1:int(len(freq)/2)]
           
           df = self.freqs[1] - self.freqs[0]
           
           max_s = max_s + df*2*3.14*np.argmax(red_s_fft_data,axis = 0)
           max_i = max_i + df*2*3.14*np.argmax(red_i_fft_data,axis = 0)
           energy = energy + e
       max_s = max_s/runs
       max_i = max_i/runs
       energy = energy/runs
       
       for i in range(resolution):
            for j in range(resolution):
                if j>(resolution-1)/2:
                   max_s[i][j] =  0
                   max_i[i][j] =  0
               
       fig, ax = plt.subplots()

       im = ax.imshow(max_s, extent=[0, 2, 0, 2],cmap='viridis')
       ax.set_xlabel("$\gamma$")
       ax.set_ylabel("$\lambda$")

       fig.colorbar(im, label = "peak frequency")
       ax.set_box_aspect(1)

       plt.show()
       fig1, ax1 = plt.subplots()

       im1 = ax1.imshow(max_i, extent=[0, 2, 0, 2],cmap='viridis')
       ax1.set_xlabel("$\gamma$")
       ax1.set_ylabel("$\lambda$")

       fig1.colorbar(im1, label = "peak frequency")
       ax1.set_box_aspect(1)

       plt.show()
       
       fig2, ax2 = plt.subplots()

       im2 = ax2.imshow(energy, extent=[0, 2, 0, 2],cmap='viridis')
       ax2.set_xlabel("$\gamma$")
       ax2.set_ylabel("$\lambda$")

       fig2.colorbar(im2, label = "peak frequency")
       ax2.set_box_aspect(1)

       plt.show()
       
     
    def energy(self,resolution):
        energy = np.zeros((resolution,resolution))
        noise = 0.6
        for i in range(1):
            energy =energy + simulation([1], [1], [1], 1, 1, 1, 1, 100, 0, 0).multi_parameter_SIRS(resolution, 0.6)
        energy = energy/1
        im = plt.imshow(energy, cmap='viridis')

        fig, ax = plt.subplots()
        mini = 2*0.01/resolution
        ax.imshow(energy, extent=[mini, 2, mini, 2])

        im = ax.imshow(energy, extent=[mini, 2, mini, 2])
        ax.set_xlabel("$\gamma$")
        ax.set_ylabel("$\lambda$")
        ax.set_title("$\log_{10}(E_\text{Measured}/E_{Expected})$ at $\sigma = $" + str(noise))
        fig.colorbar(im)
        ax.set_box_aspect(1)
        
        plt.show()
    def varience_effect_plotter(self):
        energy = []
        noise = []
        fit = []
        g = 0.5
        l = 0.05
        for i in np.linspace(0,1,10):
            energy.append(main(100000,g,l,i))
            noise.append(i**2)
            fit.append(i*i*(g*l*(1-g)/(g+l))**2)
        fig, ax = plt.subplots()
        
        plt.plot(noise,energy, label = 'data')
        plt.plot(noise,fit, color = 'k', linestyle='--', label = ' theory')
        txt = ax.text(0.02, 0.8, '$\gamma$ = ' + str(g), transform=ax.transAxes, 
              ha='left', va='top', fontsize=10, color='black')
        txt = ax.text(0.02, 0.75, '$\lambda$ = ' + str(l), transform=ax.transAxes, 
              ha='left', va='top', fontsize=10, color='black')
        plt.xlabel(r'$\sigma^2$') 
        plt.ylabel('E') 
        plt.legend()
        plt.show()
    def integrate(self,data):
        integrated_data = np.ones(len(data))
        integrated_data[0] = data[0]
        for i in range(1,len(data)):
            integrated_data[i] = integrated_data[i-1] + data[i]
        return integrated_data
    
    def fft_error(self, f, freq,damp,A, const):
       
        fft_guess = abs(A)*(self.freqs_squared + self.blank_array*abs(const))/(( freq**2-self.freqs_squared)**2 + self.freqs_squared*damp*damp)
        return fft_guess
    def fft_fit(self, data,dt, recovery_rate, resuseptibility_rate):
      # omega_guess = (resuseptibility_rate*(1-recovery_rate))
       #damp_guess = resuseptibility_rate*(1 + resuseptibility_rate)/(resuseptibility_rate + recovery_rate)
       self.fourier_transform(data, dt)
       #self.magnitude = self.magnitude/sum(self.magnitude)
       #self.freqs_squared = self.freqs*self.freqs
       #self.integrated_magnitude = self.integrate(self.magnitude)
       #self.blank_array = np.ones(len(self.magnitude))
       #guess = [omega_guess**0.5,damp_guess ,1 , (resuseptibility_rate)**2]
       #print(guess)
       #plt.plot(self.freqs,self.magnitude)
       #fit = scipy.optimize.minimize(self.fft_error,guess).x
       #fit, cov = scipy.optimize.curve_fit(self.fft_error, self.freqs, self.magnitude, guess,maxfev=100000)
       #print(fit)
       #plt.plot(self.freqs,self.integrated_magnitude)
       #plt.plot(self.freqs,self.integrate(self.fft_error(1, fit[0], fit[1], fit[2], fit[3])))
       #max_freq = self.freqs[np.argmax(self.fft_error(1,fit[0],fit[1],fit[2], fit[3]))]
       #freq = abs(fit[0])
       #damp =  abs(fit[1])
       #A = abs(fit[2])
       #print(fit)
       
       #best_guess = A*(self.freqs_squared)/(( freq**2-self.freqs_squared)**2 + self.freqs_squared*damp*damp)
       #print(abs(2*3.14/(abs(fit[0])**0.5)))

       #plt.plot(self.freqs,self.integrate(best_guess))
       #print(0.1*2*3.14/(self.freqs[np.argmax(self.magnitude)]))
       return self.freqs[np.argmax(self.magnitude)] 

def scipy_mediator(guess, other_data):  
    error = data_analysis(other_data[0], other_data[1], other_data[2], guess, other_data[3], other_data[4], other_data[5], other_data[6])
    return error.likihood()
def main(itterations, recovery_rate, resuseptibility_rate, varience_in_noise):


    population = 1
    infection_rate = 1/(population)
    time_step = 0.1
    suseptible = np.ones(1)
    infected = np.ones(1)
    recovered = np.ones(1)
    suseptible_number = (recovery_rate)/(infection_rate) 
    infected_number =resuseptibility_rate*(population*infection_rate - recovery_rate)/(infection_rate*(resuseptibility_rate + recovery_rate))
    recovered_number = population - suseptible_number - infected_number
    infected =  infected_number*infected
    recovered = recovered_number*recovered
    suseptible = suseptible_number*suseptible
    #square = ((resuseptibility_rate*( 1+resuseptibility_rate)/(resuseptibility_rate + recovery_rate))**2) - 4*resuseptibility_rate*(1 - recovery_rate)

    random_progress = simulation(suseptible, infected, recovered, population, infection_rate, recovery_rate, time_step, itterations, resuseptibility_rate, [infection_rate, 1, 0])
    
    base_suseptible, base_infected, base_recovered, time = random_progress.total_data_network_enviromental_stochastic_SIRS(varience_in_noise, [100,100], 0)
    inf = []
    inf2 = []
    tot_inf = []
    KE = 0#np.zeros((10,10))
    for i in range(1,len(base_infected)):
        loc_energy = (1/time_step)*(base_infected[i] - base_infected[i-1])**2 + time_step*(base_suseptible[i] - base_suseptible[i-1])**2
        KE = KE*(i-1)/i + loc_energy/i
    #    loc_energy = (1/time_step)*(np.sum(base_infected[i]) - np.sum(base_infected[i-1]))**2 + time_step*(np.sum(base_suseptible[i]) - np.sum(base_suseptible[i-1]))**2

    #    KE = KE*(i-1)/i + loc_energy/i
        inf.append(base_infected[i][0][0])
        inf2.append(base_infected[i][50][50])
        tot_inf.append(np.mean(base_infected[i]))
    #return omega
    #total_sys = []
    #indiv_node = []
    
    #for i in range(len(base_infected)):
    #    total_sys.append(np.sum(infected_number + (base_infected[i])/10000-infected_number))
    #    indiv_node.append(base_infected[i][0][0])
    #data_analysis(1, 1, 1, 1, 1, 1, 1, 1).fourier_transform(base_infected, time_step)
    #plt.plot(time[2:],inf, label = "$\sigma = $" + str(varience_in_noise))
    plt.plot(time[2:],inf,label = 'patch (1,1)')
    plt.plot(time[2:],inf2,label = 'patch (50,50)')
    plt.plot(time[2:],tot_inf,label = 'system average')
    plt.xlabel("Time")
    plt.ylabel("Proportion Infectious")
    plt.title("nodes $= (100,100), \gamma = 0.2,\lambda = 0.5$")
    plt.legend()
    #plt.plot(time[1:],total_sys, label = "Patch Model Total")
    #plt.plot(time[1:],indiv_node, 'k', label = "Patch 1")
    #plt.xlabel("Time")
    #plt.ylabel("Infection Incidence Rate")
    #plt.legend()
    print(inf)
    if inf[int(len(inf) - 1)] == 0:
        KE[0][0] = 0
    return base_infected#KE/100#np.sum(KE)/100
#start = tn.time()
#data_analysis(1, 1, 1, 1, 1, 1, 1, 1).energy(10)#freq_pred(10)

x = np.linspace(0, 2*np.pi,100)               
y = np.linspace(0, 2*np.pi, 100)
X, Y = np.meshgrid(x, y)
frames =2000
#inf = main(100, 0.2, 0.002, 0.01)#data_analysis(1, 1, 1, 1, 1, 1, 1, 1).energy(100) #main(frames*10, 0.2, 0.002, 0.01)
s = tn.time()
#gam =0.5
#lam = 0.5
#data_analysis(1, 1, 1, 1, 1, 1, 1, 1).varience_effect_plotter()
#for i in [1,0.5,0.1,0.05,0.01,0.005,0.001]:
#    main(10000, gam, lam, i)
#plt.legend()
#plt.xlabel("Time")
#plt.ylabel("Fraction of Population Infected")
#plt.title("$\gamma =$" + str(gam) + "$,\lambda = $" + str(lam))
#plt.show
print(tn.time() - s)
inf = main(frames*10, 0.2, 0.005, 0.1)

def data(t):
    return inf[t]

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_aspect('equal')
im = ax.pcolormesh(X, Y, data(0), cmap='viridis')
plt.colorbar(im, ax=ax, label='Infection Incidence Rate')
ax.set_title('Animated Color Plot')
ax.set_xlabel('X')
ax.set_ylabel('Y')

def update(frame):
   frame = frame*10
   Z = data(frame)
   im.set_array(Z.ravel())
   im.set_clim(np.min(inf[frame]), np.max(inf[frame])) 
   ax.set_title(f'Time {frame*0.01}')
   return [im] 
#
ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=False, repeat=True)
plt.show()
    
