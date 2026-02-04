import matplotlib.pyplot as plt

def algorithm(S,I,R,beta,gam):
    gam = gam/beta
    dt = 0.05
    infections = S*I*dt/(S+I+R)
    recoveries = gam*I*dt
    
    S += -infections
    I += infections -recoveries
    R += recoveries
    
    return S,I,R,dt*beta
def simulate(S1,I1,R1,beta,gam):
    S = [S1]
    I = [I1]
    R = [R1]
    time = [0]
    for i in range(1000):
        S_new,I_new,R_new,dt = algorithm(S[i], I[i], R[i], beta, gam)
        
        S.append(S_new)
        I.append(I_new)
        R.append(R_new)
        time.append(time[i] + dt)
        
        if R_new > 0.99*(S[0] + I[0] + R[0]):
            break
        if I_new < 0.01:
            break
    plt.plot(time,S, label = "number susceptible")
    plt.plot(time,I,label = "number infected")
    plt.plot(time,R,label ="number recovered")
    plt.xlabel("Time")
    plt.legend()
simulate(20000,1, 1, 1, 0.4)

        
