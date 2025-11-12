import numpy as np 
import matplotlib.pyplot as plt
import random

def forward_difference(suseptible,infected,recovered, infection_rate, recovery_rate, immunity_loss_rate,time_step, additional_infection_rate):
    
    infections = infection_rate*suseptible*infected*time_step + additional_infection_rate*(time_step**(1/2))*suseptible*infected
    recoveries = recovery_rate*infected*time_step
    immunity_losses = immunity_loss_rate*recovered*time_step
    
    new_suseptible = suseptible - infections + immunity_losses
    new_infected = infected + infections - recoveries
    new_recovered = recovered + recoveries - immunity_losses
    
    return new_suseptible, new_infected, new_recovered

def infection_rate_generator(average_infection_rate,time_step):
    #print(average_infection_rate + (time_step**0.5)*random.normalvariate(0, 1))
    return 0.01*random.normalvariate(0, 1)
    

def main(itterations, recovery_rate, immunity_loss_rate, time_step):
    
    population = 1
    infection_rate = 1/population
    
    suseptible = population*recovery_rate
    infected = immunity_loss_rate*(population*infection_rate - recovery_rate)/(infection_rate*(immunity_loss_rate + recovery_rate))
    recovered = 1 - suseptible - infected
    
    suseptible_data = np.zeros(itterations + 1)
    infected_data = np.zeros(itterations + 1)
    recovered_data = np.zeros(itterations + 1)
    time_data = np.zeros(itterations + 1)
    
    suseptible_data[0] = suseptible
    infected_data[0] = infected
    recovered_data[0] = recovered 
    time_data[0] = 0
    
    for i in range(itterations):
        
        additional_infection_rate = infection_rate_generator(infection_rate, time_step)

        suseptible_data[i+1], infected_data[i+1], recovered_data[i+1] = forward_difference(suseptible_data[i], infected_data[i], recovered_data[i], infection_rate, recovery_rate, immunity_loss_rate, time_step, additional_infection_rate)
        time_data[i+1] = time_data[i] + time_step
    
    fig = plt.figure()
    #plt.plot(time_data, suseptible_data, label = "suseptible")
    plt.plot(time_data, infected_data, label = "infected")
   # plt.plot(time_data, recovered_data, label = "recovered")
    plt.legend()
    plt.show()
    
main(1000000,0.5,0.0003, 0.1)
