
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
   
    # nfile = open('num.pkl', 'rb') 
    # pfile = open('pmass.pkl', 'rb') 
   
   
    # num = pickle.load(nfile)
    # pmass = pickle.load(pfile)

    
    # #x=np.arange(len(y))
    # pmass = np.array(pmass)*1e3
    # y = np.exp(np.array(num)/1e5)
    # y1=0.2/y
    # plt.plot(num, pmass,  label = "pmass vs #rewards", alpha=0.45, color='red')
    # plt.plot(num, y, label = "FAC-t vs #rewards", alpha=0.45, color='blue')
    # plt.plot(num, y1, label = "ep/t vs #rewards", alpha=0.45, color='green')


    # N = len(num)
    # Xu = np.linspace(-5, 10, N)[:, np.newaxis]
    # kde = KernelDensity(kernel="epanechnikov", bandwidth=0.3).fit(Xu)
    # log_dens = kde.score_samples(X_plot)
    # X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

    # ax.plot(
    #         X_plot[:, 0],
    #         np.exp(log_dens),
    #         color="green",
    #         lw=lw,
    #         linestyle="-",
    #         label="uniform epanechnikov",
    #     )

    
    
    nfile = open('num.pkl', 'rb') 
    pfile = open('pmass.pkl', 'rb') 
    nfile2 = open('num2.pkl', 'rb') 
    pfile2 = open('pmass2.pkl', 'rb') 
    px = open('pmasx.pkl', 'rb') 
    px2 = open('pmasx2.pkl', 'rb') 
   
   
    num = pickle.load(nfile)
    pmass = pickle.load(pfile)
    num2 = pickle.load(nfile2)
    pmass2 = pickle.load(pfile2)
    px = pickle.load(px)
    px2 = pickle.load(px2)

    
    #x=np.arange(len(y))
    pmass = np.array(pmass) 
    pmass2 = np.array(pmass2) 
    px = np.array(px) 
    px2 = np.array(px2) 
    
    plt.plot(num, pmass,  label = "pmass vs #rewards", alpha=0.45, color='red')
    plt.plot(num, pmass2, label = "uniform pmass #rewards", alpha=0.45, color='blue')
    plt.plot(num, px, label = "max pmass #rewards", alpha=0.45, color='green')
    plt.plot(num, px2, label = "max uniform pmass #rewards", alpha=0.45, color='brown')
    
    ax = plt.axes()
    ax.set_facecolor("white")
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')
    ax.legend(loc="upper left")

    #plt.legend(['Def'])
    plt.xlabel("#Rewards ",fontsize=18)
    plt.ylabel("pmass",fontsize=18)
    plt.title("pmass vs #Rewards")
    plt.savefig('rewards_vs_pmass_comp.pdf')
    plt.close()
    
            
   
    



  

    #dbfile.close()
  
if __name__ == '__main__':
    loadData()
