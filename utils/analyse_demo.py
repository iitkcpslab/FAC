
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

    y = np.array([])
    yk = []
    
    pmins = []
    pmaxs = []
    num = []
    for i in range(1):
        # dfile = open('dsdrmap_'+str(20*(i+1))+'000.pkl', 'rb') 
        # dmap = pickle.load(dfile)
        # #lvs = [len(dmap[k]) for k in dmap.keys()]
        
        # mx=0
        # ks = 0
        # for k in dmap.keys():
        #     if len(dmap[k])>mx:
        #         mx = len(dmap[k])
        #         ks = k


        # y = np.append(y,mx)
        # yk.append(ks)

    
        # print(yk)
        '''
        # for i in range(len(yk)):
        data = [[i] for i in np.arange(i+1)]
        x = np.arange(len(data))
        plt.scatter(x,data,  label = "rewards",alpha=0.45, color='red')
        ax = plt.axes()
        ax.set_facecolor("white")
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.xaxis.grid(color='gray', linestyle='dashed')
        #plt.legend(['Def'])
        #plt.xlabel("#Rewards ",fontsize=18)
        #plt.ylabel("pmass",fontsize=18)
        plt.title("Rewards")
        plt.savefig('rewards_points_'+str(i)+'.pdf')
        plt.close()
        #print(data)
        #exit()
        #data1 = [[x] for x in np.random.uniform(np.min(data),np.max(data),len(data))]
        '''

        #data1=[[i] for i in np.arange(i+1)]
        #data1 = [ [1] for i in np.arange(10000)] + [ [2] for i in np.arange(1000)] + [ [3] for i in np.arange(100)]
        np.random.seed(0)
        data1 = [[x] for x in np.random.uniform(0,1,10)]
        ukd = KernelDensity(kernel='epanechnikov', bandwidth=0.3).fit(data1)
        plist = []
        start = 0.4  # Start of the range
        end = 0.6    # End of the range
        N = 10       # Number of evaluation points
        step = (end - start) / (N - 1)  # Step size
        x = np.linspace(start,end , N)[:, np.newaxis]  # Generate values in the range
        ukd_vals = np.exp(ukd.score_samples(x))  # Get PDF values for each x
        p = np.sum(ukd_vals * step)  # Approximate the integral of the PDF
        print("#rewards is ",len(data1),"  min  prob mass is : ",np.min(p),"  max prob mass is : ",np.max(p))
       
        
    # plt.plot(num,pmins,  label = "pmass vs #rewards",alpha=0.45, color='red')
    # #plt.plot(num,pmaxs,  label = "pmass vs #rewards",alpha=0.45, color='green')
    

    # ax = plt.axes()
    # ax.set_facecolor("white")
    # ax.yaxis.grid(color='gray', linestyle='dashed')
    # ax.xaxis.grid(color='gray', linestyle='dashed')

    # #plt.legend(['Def'])
    # plt.xlabel("#Rewards ",fontsize=18)
    # plt.ylabel("pmass",fontsize=18)
    # plt.title("pmass vs #Rewards")
    # plt.savefig('rewards_vs_pmass1.pdf')
    # plt.close()

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
    
    # # with open("num.pkl", 'wb') as file:
    # #     pickle.dump(num, file)
    # # with open("pmass.pkl", 'wb') as file:
    # #     pickle.dump(pmass, file)
    
    
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
