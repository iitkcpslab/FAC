
import pickle
import gym
from stable_baselines3 import SAC
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn import preprocessing
from discretization import discretize
from discretization import create_uniform_grid
from sklearn.neighbors import KernelDensity

def loadData():
    # for reading also binary mode is important
   
    dfile = open('dsdrmap_20000.pkl', 'rb') 
    rfile = open('rbmap_20000.pkl', 'rb') 
   
   
    dmap = pickle.load(dfile)
    rmap = pickle.load(rfile)
    #print("dmap  ",len(dmap.keys()))
    #print("rmap  ",len(rmap.keys()))

    # for i in range(49):
    #     dfile = open('dsdrmap_'+str(20*(i+1))+'000.pkl', 'rb') 
    #     dmap = pickle.load(dfile)
    #     print("i is ",str(i)," len dmap keys  ",len(dmap.keys()))
    
    # y = np.array([])
    # yk = []
    
    # for i in range(40):
    #     dfile = open('dsdrmap_'+str(20*(i+1))+'000.pkl', 'rb') 
    #     dmap = pickle.load(dfile)
    #     #lvs = [len(dmap[k]) for k in dmap.keys()]
    #     lvs = []
    #     ks = []
    #     mx=0
    #     for k in dmap.keys():
    #         if len(dmap[k])>mx:
    #             lvs = len(dmap[k])
    #             ks = k


    #     y = np.append(y,lvs)
    #     yk.append(ks)

    # print(yk)
    #exit()
    # #y = y/1000
    # x=np.arange(len(y))
    # plt.plot(x,y,  label = "max #rewards growth",alpha=0.45, color='red')
    # #plt.plot(x,np.exp(y/1e5),  label = "exp",alpha=0.45, color='green')
    # #plt.plot(x,x,  label = "identity",alpha=0.45, color='blue')
    

    # ax = plt.axes()
    # ax.set_facecolor("white")
    # ax.yaxis.grid(color='gray', linestyle='dashed')
    # ax.xaxis.grid(color='gray', linestyle='dashed')

    # plt.legend(['Def'])
    # plt.xlabel("Iterations ",fontsize=18)
    # plt.ylabel("Max #Rewards",fontsize=18)
    # plt.title("Max #Rewards")
    # plt.savefig('rewards_vs_states_def.pdf')
    # plt.close()

    y = np.array([])
    yk = []
    
    pmass = []
    pmass2 = []
    pmasx = []
    pmasx2 = []

    num = []
    num2 = []
    for i in range(50):
        dfile = open('dsdrmap_'+str(20*(i+1))+'000.pkl', 'rb') 
        dmap = pickle.load(dfile)
        #lvs = [len(dmap[k]) for k in dmap.keys()]
        
        mx=0
        ks = 0
        for k in dmap.keys():
            if len(dmap[k])>mx:
                mx = len(dmap[k])
                ks = k


        y = np.append(y,mx)
        yk.append(ks)

    
        # print(yk)
        
        # for i in range(len(yk)):
        data = dmap[ks]
        x = np.arange(len(data))
        # plt.scatter(x,data,  label = "rewards",alpha=0.45, color='red')
        # ax = plt.axes()
        # ax.set_facecolor("white")
        # ax.yaxis.grid(color='gray', linestyle='dashed')
        # ax.xaxis.grid(color='gray', linestyle='dashed')
        # #plt.legend(['Def'])
        # #plt.xlabel("#Rewards ",fontsize=18)
        # #plt.ylabel("pmass",fontsize=18)
        # plt.title("Rewards")
        # plt.savefig('rewards_points_'+str(i)+'.pdf')
        # plt.close()
        #print(data)
        #exit()
        data1 = data
        data2 = [[x] for x in np.linspace(np.min(data),np.max(data),len(data))]
        print(len(data1))
        print(len(data2))
        kd = KernelDensity(kernel='epanechnikov', bandwidth=0.3).fit(data1)
        beta = 0.2  #default
        # Get probability for range of values
        plist = []
        for rw in data1: 
            #rw = data[-1]
            start = rw[0]-beta  # Start of the range
            end = rw[0]+beta    # End of the range
            N = 5     # Number of evaluation points
            step = (end - start) / (N - 1)  # Step size
            x = np.linspace(start,end , N)[:, np.newaxis]  # Generate values in the range
            kd_vals = np.exp(kd.score_samples(x))  # Get PDF values for each x
            p = np.sum(kd_vals * step)  # Approximate the integral of the PDF
            plist.append(p)
        print("#rewards is ",len(data1),"  min  prob mass is : ",np.min(plist)," +/- ",np.std(plist))
        pmass.append(np.min(plist))
        pmasx.append(np.max(plist))
        num.append(len(data1))


        kd2 = KernelDensity(kernel='epanechnikov', bandwidth=0.3).fit(data2)
        beta = 0.2  #default
        # Get probability for range of values
        plist2 = []
        for rw in data2: 
            #rw = data[-1]
            start = rw[0]-beta  # Start of the range
            end = rw[0]+beta    # End of the range
            N = 5     # Number of evaluation points
            step = (end - start) / (N - 1)  # Step size
            x = np.linspace(start,end , N)[:, np.newaxis]  # Generate values in the range
            kd2_vals = np.exp(kd2.score_samples(x))  # Get PDF values for each x
            p = np.sum(kd2_vals * step)  # Approximate the integral of the PDF
            plist2.append(p)
        print("#rewards is ",len(data2),"  min  prob mass is : ",np.min(plist2)," +/- ",np.std(plist2))
        pmass2.append(np.min(plist2))
        pmasx2.append(np.max(plist2))
        num2.append(len(data2))
    
    plt.plot(num,pmass, label = "pmass vs #rewards",alpha=0.45, color='red')
    plt.plot(num2,pmass2, label = "uniform pmass vs #rewards",alpha=0.45, color='green')
    

    ax = plt.axes()
    ax.set_facecolor("white")
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')

    #plt.legend(['Def'])
    plt.xlabel("#Rewards ",fontsize=18)
    plt.ylabel("pmass",fontsize=18)
    plt.title("pmass vs #Rewards")
    plt.savefig('rewards_vs_pmass1.pdf')
    plt.close()

    #     kd = KernelDensity(kernel='epanechnikov', bandwidth=0.3).fit(data) #FAC
        
    #     beta = 0.2  #default
    #     # Get probability for range of values
    #     plist = []
    #     for rw in data: 
    #         #rw = data[-1]
    #         start = rw[0]-beta  # Start of the range
    #         end = rw[0]+beta    # End of the range
    #         N = 5     # Number of evaluation points
    #         step = (end - start) / (N - 1)  # Step size
    #         x = np.linspace(start,end , N)[:, np.newaxis]  # Generate values in the range
    #         kd_vals = np.exp(kd.score_samples(x))  # Get PDF values for each x
    #         p = np.sum(kd_vals * step)  # Approximate the integral of the PDF
    #         plist.append(p)
    #     print("#rewards is ",len(data),"  min  prob mass is : ",np.min(plist)," +/- ",np.std(plist))
    #     pmass.append(np.min(plist)*1e4)
    #     num.append(len(data))
    
    with open("num.pkl", 'wb') as file:
        pickle.dump(num, file)
    with open("num2.pkl", 'wb') as file:
        pickle.dump(num2, file)
    
    with open("pmass.pkl", 'wb') as file:
        pickle.dump(pmass, file)
    with open("pmass2.pkl", 'wb') as file:
        pickle.dump(pmass2, file)
    
    with open("pmasx.pkl", 'wb') as file:
        pickle.dump(pmasx, file)
    with open("pmasx2.pkl", 'wb') as file:
        pickle.dump(pmasx2, file)
    
    
    # #x=np.arange(len(y))
    # plt.plot(num,pmass,  label = "pmass vs #rewards",alpha=0.45, color='red')
    

    # ax = plt.axes()
    # ax.set_facecolor("white")
    # ax.yaxis.grid(color='gray', linestyle='dashed')
    # ax.xaxis.grid(color='gray', linestyle='dashed')

    # #plt.legend(['Def'])
    # plt.xlabel("#Rewards ",fontsize=18)
    # plt.ylabel("pmass",fontsize=18)
    # plt.title("pmass vs #Rewards")
    # plt.savefig('rewards_vs_pmass.pdf')
    # plt.close()
    
            
   
    



  

    #dbfile.close()
  
if __name__ == '__main__':
    loadData()

