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
         
         
         
         infections = self.infection_rate*self.suseptible*self.infected*self.time_step + additional_infection_rate*(self.time_step**(1/2))*self.suseptible*self.infected
         recoveries = self.recovery_rate*self.infected*self.time_step
         immunity_losses = self.resuseptibility_rate*self.recovered*self.time_step
         
         new_suseptible = self.suseptible - infections + immunity_losses
         new_infected = self.infected + infections - recoveries
         new_recovered = self.recovered + recoveries - immunity_losses
         
         return new_suseptible, new_infected, new_recovered
     
    def SIR_Birth_and_Death(self, additional_infection_rate):
         
         infections = self.infection_rate*self.suseptible*self.infected*self.time_step + additional_infection_rate*(self.time_step**(1/2))*self.suseptible*self.infected
         recoveries = self.recovery_rate*self.infected*self.time_step
         
         new_suseptible = self.suseptible - infections + self.resuseptibility_rate*(self.infected + self.recovered)*self.time_step
         new_infected = self.infected + infections - recoveries -  self.resuseptibility_rate*(self.infected)*self.time_step
         new_recovered = self.recovered + recoveries -  self.resuseptibility_rate*(self.recovered)*self.time_step
         
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
   def enviromental_stochastic_SIRS(self, varience_in_noise):
    
      time = np.zeros(self.itterations)
      start = tn.time()
      noise = varience_in_noise*np.random.normal(0,1,self.itterations)
      for i in range(self.itterations):
          
          next_time_step = next_timestep(self.suseptible[i], self.infected[i], self.recovered[i], self.population, self.infection_rate, self.recovery_rate, self.time_step, self.resuseptibility_rate,0)
          self.suseptible[i+1],self.infected[i+1],self.recovered[i+1] = next_time_step.envirometal_stochasticity(noise[i])

          time[i] = i*self.time_step
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
        freq = np.fft.fftfreq(len(data), d=dt)
        self.freqs = freq[1:int(len(freq)/2)]
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
    
    def period_finder(self,data, time_step, pred_period):
        
        derivative_data = data[1:] - data[:-1]
        peaks, properties = scipy.signal.find_peaks(data)
        fig = plt.figure()
        plt.hist(peaks[1:] - peaks[:-1], bins=30, density=True, alpha=0.5)

        x = np.linspace(min(peaks[1:] - peaks[:-1]), max(peaks[1:] - peaks[:-1]), 1)
        shape, loc, scale = genextreme.fit(peaks[1:] - peaks[:-1])
        pdf = genextreme.pdf(x, shape, loc=loc, scale=scale)
        plt.plot(x, pdf, 'r-', lw=2)
       
        plt.title("GEV PDF over Data Histogram")

        plt.show()
        return len(data)*time_step/len(peaks)
    def freq_pred(self,resolution):
            colour_matrix = np.zeros((resolution,resolution))
            colour_matrix2 = np.zeros((resolution,resolution)).copy()
            colour_matrix3 = np.zeros((resolution,resolution))
            for i in range(resolution):
                 for j in range(resolution):
                     recovery_rate = 2*(j+0.01)/resolution
                     resuseptibility_rate = 2*(i+0.01)/resolution
                    
                     square =((resuseptibility_rate*( 1+resuseptibility_rate)/(resuseptibility_rate + recovery_rate))**2) - 4*resuseptibility_rate*(1 - recovery_rate)
                     real = -resuseptibility_rate*(1 + resuseptibility_rate)/(recovery_rate + resuseptibility_rate)
                     #real = - resuseptibility_rate/(resuseptibility_rate + recovery_rate)
                     #square = real**2 - 4*resuseptibility_rate*(1 - recovery_rate - resuseptibility_rate)
                     if square > 0:
                         if real + square**0.5 > 0:
                             colour_matrix[resolution - i - 1][j] = 0
                         else:
                             colour_matrix[resolution - i - 1][j] = 1
                             #colour_matrix[resolution - i - 1][j] = 4
                             #preriod = main(1000, recovery_rate, resuseptibility_rate, 0.0003)
                             #colour_matrix2[resolution - i - 1][j] = math.log10(preriod)
                             #period = main(10000, recovery_rate, resuseptibility_rate, 0.0003)/0.0003**2
                             #period = (recovery_rate)*((1 - recovery_rate)*resuseptibility_rate/(recovery_rate + resuseptibility_rate))
                             
                             omega  = 0
                             for k in range(1):
                                 omega += main(20000, recovery_rate, resuseptibility_rate, 0.000003)
                             colour_matrix2[resolution - i - 1][j] +=  np.log(1/omega)#in(100,(160/(resuseptibility_rate*(1-recovery_rate)))**0.5)#min((omega4 + omega4 + omega4 + omega4)/4,100)#count/ticker#(0.5*(abs(square))**0.5)#(period)**2#math.log10(2*math.pi/freq)#min(period, 100)#math.log10(period)#math.log(1/freq)#math.log10(1/((10**frequency) / time_step))
                             print(colour_matrix2[resolution - i - 1][j])
                     else:
                        if real > 0:
                            colour_matrix[resolution - i - 1][j] = 2
                        else:
                            colour_matrix[resolution - i - 1][j] = 3

                            #freq = 0.5*abs(square)**0.5 
                            #colour_matrix[resolution - i - 1][j] = 4
                            #period = (recovery_rate)*((1 - recovery_rate)*resuseptibility_rate/(recovery_rate + resuseptibility_rate))#(recovery_rate + ((recovery_rate))**2)*(resuseptibility_rate*(1 - recovery_rate)/((resuseptibility_rate + recovery_rate)))**2#main(10000, recovery_rate, resuseptibility_rate, 0.003)/(0.003**2)
                            #period = main(10000, recovery_rate, resuseptibility_rate, 0.0003)/0.0003**2
                            count = 0
                            ticker = 0
                            
                         
                            omega  = 0
                            for k in range(1):
                                omega += main(20000, recovery_rate, resuseptibility_rate, 0.000003)
                           
                            colour_matrix2[resolution - i - 1][j] +=np.log(1/omega)#min(100,(160/(resuseptibility_rate*(1-recovery_rate)))**0.5)#min((omega1 + omega1 + omega1+ omega1)/4,100)#count/ticker#(0.5*(abs(square))**0.5)#(period)**2#math.log10(2*math.pi/freq)#min(period, 100)#math.log10(period)#math.log(1/freq)#math.log10(1/((10**frequency) / time_step))
                            print(colour_matrix2[resolution - i - 1][j])
             #Define discrete colors and colormap
            #colors = ['red', 'blue', 'green']
            #fig, ax = plt.subplots()
            #cmap = ListedColormap(colors)
            #im = ax.imshow(colour_matrix, cmap=cmap, interpolation='none',  extent=[0, 2, 0, 2])
            #solution_type = ['Unstable', 'Stable','Stable Oscillatory '] 
            #ax.set_xlabel("$\gamma$")
            #ax.set_ylabel("$\lambda$")

            #patches = [mpatches.Patch(color=colors[i], label=solution_type[i]) for i in range(len(colors))]
#
            #ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

            #plt.show()
            im = plt.imshow(colour_matrix, cmap='viridis')
            print(colour_matrix2)
            fig, ax = plt.subplots()
            mini = 2*0.01/resolution
            ax.imshow(colour_matrix2, extent=[mini, 2, mini, 2])

            im = ax.imshow(colour_matrix2, extent=[mini, 2, mini, 2])
            ax.set_xlabel("$\gamma$")
            ax.set_ylabel("$\lambda$")

            fig.colorbar(im, label = "$\log_{10}($period}$)$")
            ax.set_box_aspect(1)
            
            plt.show()
    def energy(self,resolution,a,b):
                colour_matrix = np.zeros((resolution,resolution))
                colour_matrix2 = np.zeros((resolution,resolution))
                colour_matrix3 = np.zeros((resolution,resolution))
                s_stable = a
                i_stable = b*(1 - a)/((a + b))
                base_energy = 1#(a*b*(1-a)/(a+b))**2
                for i in range(resolution):
                     for j in range(resolution):
                         #suseptible = s_stable + (-0.1 +0.2*i/resolution)
                         #infected = i_stable + (-0.1 +0.2*j/resolution)*b/a
                         suseptible = i/resolution
                         infected = j/resolution
                         recovered = 1 - suseptible - infected
                         if recovered < 0:
                             colour_matrix2[resolution - i - 1][j] = 0#math.log10(2*math.pi/freq)#min(period, 100)#math.log10(period)#math.log(1/freq)#math.log10(1/((10**frequency) / time_step))
                         else:
                             s,k,r = next_timestep(suseptible, infected, recovered, 1, 1, a, 1, b, 0).SIRS()
                             energy = (suseptible - s)**2 + (infected - k)**2 + (recovered - r)**2
                             colour_matrix2[resolution - i - 1][j] = (energy/base_energy)**0.5
                             colour_matrix[i][j] = (energy/base_energy)**0.5
                 #Define discrete colors and colormap
                colors = ['black', 'blue', 'red', 'green']
                fig, ax = plt.subplots()
                cmap = ListedColormap(colors)
                im = ax.imshow(colour_matrix, cmap=cmap, interpolation='none',  extent=[0, 2, 0, 2])
                solution_type = ['Unstable', 'Stable', 'Unstable Oscilatory','Stable Oscillatory '] 
                ax.set_xlabel("b")
                ax.set_ylabel("a")

                patches = [mpatches.Patch(color=colors[i], label=solution_type[i]) for i in range(len(colors))]


                ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

                plt.show()
                
                #im = plt.imshow(colour_matrix, cmap='viridis')
                #fig, ax = plt.subplots()
                #mini = 2*0.01/resolution
                #ax.imshow(colour_matrix2, extent=[mini, 1, mini, 1])
                #plt.plot(i_stable,  s_stable, marker='*', markersize=15, color='yellow')
                #im = ax.imshow(colour_matrix2, extent=[mini, 1, mini, 1])
                #ax.set_xlabel("Fraction of Population Infected")
                #ax.set_ylabel("Fraction of Population Suseptible")
                #plt.title("Energy")
                #fig.colorbar(im, label = "r$\sigma$")
                #ax.set_box_aspect(1)
                #fig = plt.figure()
                #x = np.linspace(0, 1, resolution)
                #y = np.linspace(0, 1, resolution)
                
                #X, Y = np.meshgrid(x, y)
                #ax.imshow(colour_matrix2, extent=[mini, 2, mini, 2])
                #plt.contour(X, Y, colour_matrix, levels=500)
               
                #plt.plot(i_stable, s_stable, marker='*', markersize=15, color='yellow')
                #plt.show()
    def varience_effect_plotter(self):
        energy = []
        noise = []
        for i in np.linspace(0.0,0.5,20):
            energy.append(main(20000,0.5,0.3,i))
            noise.append(i**2)
        fit = np.polyfit(noise, energy, 1, rcond=None, full=False, w=None, cov=False)
        p = np.poly1d(fit)    
        print(fit)
        fig, ax = plt.subplots()
        
        plt.plot(noise,energy, label = 'data')
        plt.plot(noise,p(noise), color = 'k', linestyle='--', label = 'linear fit')
        txt = ax.text(0.02, 0.8, 'a = 0.5', transform=ax.transAxes, 
              ha='left', va='top', fontsize=10, color='black')
        txt = ax.text(0.02, 0.75, 'b = 0.3', transform=ax.transAxes, 
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
       omega_guess = (resuseptibility_rate*(1-recovery_rate))
       damp_guess = resuseptibility_rate*(1 + resuseptibility_rate)/(resuseptibility_rate + recovery_rate)
       self.fourier_transform(data, dt)
       self.magnitude = self.magnitude/sum(self.magnitude)
       self.freqs_squared = self.freqs*self.freqs
       self.integrated_magnitude = self.integrate(self.magnitude)
       self.blank_array = np.ones(len(self.magnitude))
       guess = [omega_guess**0.5,damp_guess ,1 , (resuseptibility_rate)**2]
       print(guess)
       #plt.plot(self.freqs,self.magnitude)
       #fit = scipy.optimize.minimize(self.fft_error,guess).x
       fit, cov = scipy.optimize.curve_fit(self.fft_error, self.freqs, self.magnitude, guess,maxfev=5000)
       print(fit)
       #plt.plot(self.freqs,self.integrated_magnitude)
       #plt.plot(self.freqs,self.integrate(self.fft_error(1, fit[0], fit[1], fit[2], fit[3])))

       #freq = abs(fit[0])
       #damp =  abs(fit[1])
       #A = abs(fit[2])
       #print(fit)
       
       #best_guess = A*(self.freqs_squared)/(( freq**2-self.freqs_squared)**2 + self.freqs_squared*damp*damp)
       #print(abs(2*3.14/(abs(fit[0])**0.5)))

       #plt.plot(self.freqs,self.integrate(best_guess))
       #print(0.1*2*3.14/(self.freqs[np.argmax(self.magnitude)]))
       return fit[0]
def scipy_mediator(guess, other_data):  
    error = data_analysis(other_data[0], other_data[1], other_data[2], guess, other_data[3], other_data[4], other_data[5], other_data[6])
    return error.likihood()
def main(itterations, recovery_rate, resuseptibility_rate, varience_in_noise):
    #print("the recovery rate is " + str(recovery_rate))
    #print("the immunity loss rate is " + str(resuseptibility_rate))
    #print("the varience in the noise is " + str(resuseptibility_rate))
    #print(2*3.14/((resuseptibility_rate*(1-recovery_rate)**0.5)))
    population = 1
    infection_rate = 1/(population)
    time_step = 1
    suseptible = np.ones(1)
    infected = np.ones(1)
    recovered = np.ones(1)
    suseptible_number = (recovery_rate)/(infection_rate) 
    infected_number = resuseptibility_rate*(population*infection_rate - recovery_rate)/(infection_rate*(resuseptibility_rate + recovery_rate))
    recovered_number = population - suseptible_number - infected_number
    infected =  infected_number*infected
    recovered = recovered_number*recovered
    suseptible = suseptible_number*suseptible
    square = ((resuseptibility_rate*( 1+resuseptibility_rate)/(resuseptibility_rate + recovery_rate))**2) - 2*resuseptibility_rate*(1 - recovery_rate)
    #print(4*3.14/(abs(square)**0.5))
    
    if square < 0:
        print("pred period is " + str(2*3.14/(abs(square)**0.5)))
        pred_omega = 4*3.14/(abs(square)**0.5)
    else:
        pred_omega = 0
    #pred_period= itterations
    #x = np.zeros(itterations)
    #y = np.zeros(itterations)
    #print(suseptible_data)
    #for i in range(itterations):
    #    x[i] = random.randint(0,800)
    #    y[i] = random.randint(0,800)
    #disease_progress = simulation(suseptible, infected, recovered, population, infection_rate, recovery_rate, time_step, itterations, resuseptibility_rate, [suseptible, infected, recovered, [0/population,0,0]])
    #suseptible_data, infected_data, recovered_data, time = disease_progress.SIRS_RUN()
    random_progress = simulation(suseptible, infected, recovered, population, infection_rate, recovery_rate, time_step, itterations, resuseptibility_rate, [infection_rate, 1, 0.05])
    
    base_suseptible, base_infected, base_recovered, time = random_progress.enviromental_stochastic_SIRS(varience_in_noise)
    #data_analysis(1, 1, 1, 1, 1, 1, 1, 1).colour_pdf(base_recovered[1:], base_recovered[1:] - base_recovered[:-1], 500)
    #data_analysis(1, 1, 1, 1, 1, 1, 1, 1).colour_pdf(base_recovered[1:], base_recovered[1:] - base_recovered[:-1], 500)
    #data_analysis(1, 1, 1, 1, 1, 1, 1, 1).colour_pdf(base_suseptible[1:], base_suseptible[1:] - base_suseptible[:-1], 100)
    #freq, amp, inverse_ftt = data_analysis(base_recovered, 1, 1, 1, 1, 1,1, 1).fourier_transform(gaussian_filter1d(base_infected[1:],0.2*pred_period/time_step))
    #frequency = freq[np.argmax(amp)]
    #period = time_step/frequency
    #print("the period is " + str(time_step/frequency))
 
    energy = ((base_infected[1:]-base_infected[:-1])/time_step)**2 + ((base_suseptible[1:]-base_suseptible[:-1])/time_step)**2 + ((base_recovered[1:]-base_recovered[:-1])/time_step)**2
    
    average_energy = []
    average_energy.append(0)
    for i in range(1,len(energy)):
        average_energy.append(0.5*time_step*energy[i]/i + average_energy[i-1]*(i-1)/i)
    omega = data_analysis(1, 1, 1, 1, 1, 1, 1, 1).fft_fit(base_infected,time_step,recovery_rate,resuseptibility_rate)
    #omega = data_analysis(1, 1, 1, 1, 1, 1, 1, 1).fft_fit([1,1], base_infected/infected[0]- np.ones(len(base_infected)), square,time_step)
    #omega = data_analysis(1, 1, 1, 1, 1, 1, 1, 1).fft_fit([1,1], base_recovered/recovered[0]- np.ones(len(base_infected)), square,time_step)
    #plt.plot(time,average_energy)
    #plt.plot(time,gaussian_filter1d(energy,100))
    #plt.plot(time,gaussian_filter1d(base_infected[1:],100))
    ##plt.plot(time,gaussian_filter1d(base_infected[1:],pred_period*0.1/time_step))
    #period = data_analysis(1, 1, 1, 1, 1, 1, 1, 1).period_finder(gaussian_filter1d(base_infected[1:],0.2*pred_period/time_step), time_step,pred_period)
    #print(period)
    #period = data_analysis(1, 1, 1, 1, 1, 1, 1, 1).period_finder(gaussian_filter1d(base_recovered,0.17*pred_period/time_step), time_step, pred_period)
    #print(period)
    #fig, ax = plt.subplots()
    #plt.plot(time,base_infected[1:]/population)
    #plt.plot(time,average_energy,label = 'Average Kinetic Energy ')
    #plt.plot(time,0.5*time_step*energy,linestyle='--', label = 'Kinetic Energy ')
    #txt = ax.text(0.02, 0.95, 'a = 0.5', transform=ax.transAxes, 
    #      ha='left', va='top', fontsize=10, color='black')
    #txt = ax.text(0.02, 0.90, 'b = 0.3', transform=ax.transAxes, 
    #      ha='left', va='top', fontsize=10, color='black')
    #txt = ax.text(0.02, 0.85, '$\sigma^2$ = 0.25', transform=ax.transAxes, 
    #      ha='left', va='top', fontsize=10, color='black')
    #plt.legend()
    #plt.show()
    return 10*abs(omega)#average_energy[len(average_energy) - 1]/time_step #100*abs(period - pred_period)/period
    
    #maximum = np.zeros(int(itterations/5000 - 1))
    #minimum = np.zeros(int(itterations/5000 - 1))
   
    #for i in range(int(itterations/5000 - 1)):
    #    maximum[i] = np.max(base_infected[i*5000:i*5000 + 5000]/population)
    #    minimum[i] = -np.min(base_infected[i*5000:i*5000 + 5000]/population)
    #hist, bin_edges= np.histogram(maximum,100)
    #bin_widths = np.diff(bin_edges)
    #plt.figure()
    #plt.bar(bin_edges[:-1], hist, width=bin_widths, edgecolor='black', align='edge')
    #plt.show()
    
    #hist, bin_edges= np.histogram(minimum,100)
    #bin_widths = np.diff(bin_edges)
    #plt.figure()
    #plt.bar(bin_edges[:-1], hist, width=bin_widths, edgecolor='black', align='edge')
    #plt.show()
    #shape1, loc1, scale1 = genextreme.fit(maximum)
    #shape2, loc2, scale2 = genextreme.fit(minimum)
    
    #print(shape1, loc1, scale1)
    #print(shape2, loc2, scale2)
    
   
    #plt.plot(time, base_suseptible[1:], 'r', label = 'suseptible')
    #fig = plt.figure()
    #plt.plot(time, base_infected[1:])

    #plt.ylabel("Infection Incidence Rate")
    #plt.xlabel("Time")
    #plt.figtext(.7, .8, " varience = 0.3")
    #plt.show()
    #return time_step/frequency, pred_period
    #plt.plot(time, base_infected[1:] - base_infected[:-1], 'b')
    #plt.plot(time, base_recovered[1:],'b', label = 'recovered')
   
    
    #data_analysis(1, 1, 1, 1, 1, 1, 1, 1).colour_pdf(x,y, 200)
    #normal = []
    #for i in range(00, itterations):
    #    normal.append(base_suseptible[i] - suseptible_data[i])
    #fig = plt.figure()
    #stats.probplot(normal, dist="norm", plot=plt)
    #plt.show()
    #print(base_infected[int(len(base_infected) - 1)])
    #guess = initial_guess(base_suseptible, base_infected, base_recovered, time_step, 1000)
    #print(guess)
    #fig = plt.figure()
    #plt.plot(time, base_infected[1:])
    #plt.show()
    #return base_suseptible, base_infected, base_recovered, infection_rate, recovery_rate, resuseptibility_rate, time_step
    #disease_progress_2 =  simulation(suseptible, infected, recovered, population, guess[0], guess[1], time_step, itterations, resuseptibility_rate, [suseptible, infected, recovered, [0/population,0,0]])
    #suseptible_data_2, infected_data_2, recovered_data_2, time = disease_progress_2.SIRS_RUN()
    #time = time/time_step
    
        
    #ft = data_analysis(base_recovered, 1, 1, 1, time_step, itterations, 1, 1)    #return ft.fourier_transform()
    #plt.plot(time, suseptible_data_2[1:], 'b')
    #plt.plot(time, infected_data_2[1:], 'b')
    #plt.plot(time, recovered_data_2[1:],'b')
    #plt.show()
    #fig2 = plt.figure()
    #plt.plot(suseptible_data, infected_data, 'k')
    #plt.plot(base_suseptible, base_infected,, 'b')
    #plt.show()
    #print(guess[0]*population/guess[1])
   
    #smoothed_data = gaussian_filter1d(base_suseptible, sigma=10)
    #smoothed_data = []
    #new_time = []
    #for i in range(0, int(len(suseptible_data) - 100)):
    #    smoothed_data.append(sum(base_suseptible[i:i+100]))
    #    new_time.append(i)
    #fig2 = plt.figure()
    #plt.plot(new_time, smoothed_data)
    #plt.show()
    #fig = plt.figure()
    #lhs = infection_rate*resuseptibility_rate*base_infected*base_recovered + infection_rate*infection_rate*base_suseptible*base_suseptible*base_infected + resuseptibility_rate*resuseptibility_rate*base_recovered
    #rhs =infection_rate*infection_rate*base_suseptible*base_infected*base_infected + infection_rate*base_suseptible*recovery_rate*base_infected + resuseptibility_rate*recovery_rate*base_infected
    #print(lhs)
    #print(rhs)
    #plt.plot(time, rhs[1:] - lhs[1:])
    #plt.plot(time, np.zeros(len(time)), 'k')
    #n =0
    #for i in range(len(time) - 1000):
    #    if rhs[1000+ i] - lhs[1000+ i]> 0:
    #        n +=1
    #        
    #print(n)
    #plt.plot(time,rhs[1:])
    #plt.show()
    #fig = plt.figure()
    #plt.plot(base_suseptible[1:], gaussian_filter1d(-base_suseptible[:-1] + base_suseptible[1:],100))
    #plt.show()
   
    #fig = plt.figure()
    #plt.plot(time,infection_rate*(base_suseptible[1:] + base_infected)/((recovery_rate+resuseptibility_rate)))
    #plt.show()
    #return [base_suseptible, time]
#S,I,R,infection_rate, recovery_rate, resuseptibility_rate, time_step = main(100000,0.5,0.0025)
#data_analysis(1, 1, 1, 1, 1, 1, 1, 1).stochastic_exposure_plotter([100,260000000000],[100,260000000000], 500, S[0] + I[0] + R[0], infection_rate, recovery_rate, resuseptibility_rate, time_step)
#data_analysis(1, 1, 1, 1, 1, 1, 1, 1).stochastic_exposure_plotter([np.min(S),np.max(S)],[np.min(I),np.max(I)], 500, S[0] + I[0] + R[0], infection_rate, recovery_rate, resuseptibility_rate, time_step)
#data_analysis(1, 1, 1, 1, 1, 1, 1, 1).colour_pdf(S[1:], I[1:], 500)
#[np.min(S),np.max(S)],[np.min(I),np.max(I)]
#start = tn.time()
data_analysis(1, 1, 1, 1, 1, 1, 1, 1).freq_pred(50)

#data_analysis(1, 1, 1, 1, 1, 1, 1, 1).freq_pred(200)
#data = []
#data1 = []
#data2 = []
#for i in range(0,20):
#    e = (i+0.01)/20
#    data.append(e)
#    omega, predomega = main(5000, e, 0.5, 0.00003)
#    data1.append(omega)
#    data2.append(predomega)
#plt.plot(data,data1)
#plt.plot(data,data2, color = "k")
 
