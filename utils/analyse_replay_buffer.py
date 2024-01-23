
import pickle
import gym
from stable_baselines3 import SAC
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn import preprocessing
from discretization import discretize
from discretization import create_uniform_grid

def loadData():
    # for reading also binary mode is important
    #dbfile1 = open('sac_Ant-v3_replay_buffer_sss_316.pkl', 'rb')     
    #dbfile2 = open('sac_Ant-v3_replay_buffer_sss_324.pkl', 'rb')   
    #dbfile3 = open('sac_Ant-v3_replay_buffer_sss_3241.pkl', 'rb')     

    #dbfile1 = open('sac_Pendulum-v1_replay_buffer_default_basic.pkl', 'rb')     
    #dbfile1 = open('sac_Walker2d-v3_replay_buffer_sss_58.pkl', 'rb')    
    #name = "Walker2d"
    #name = "Ant"
    name = "Hopper"
    #name = "HalfCheetah"
    #name = "Swimmer"
    #name = "Pendulum"
    print(name)
    dbfile1 = open('sac_'+name+'-v3_replay_buffer_fac_1.pkl', 'rb') 
    dbfile2 = open('sac_'+name+'-v3_replay_buffer_default_1.pkl', 'rb') 
    dbfile3 = open('sac_'+name+'-v3_replay_buffer_default_1.pkl', 'rb')     
    

    rb1 = pickle.load(dbfile1)
    rb2 = pickle.load(dbfile2)
    rb3 = pickle.load(dbfile3)
    print("pruned  ",rb1.pos)
    print("full st ",rb2.pos)
    print("default  ",rb3.pos)
    
    #ofile = open('dsdrmap_hop_11k.pkl', 'rb') 
    #omap = pickle.load(ofile)
    
    
    if name=="Hopper":
        low =  np.array([0.0,-1.0,-2.0,-1.0,-1.0,-2.0,-3.0,-8.0,-8.0,-8.0,-7.0])
        high = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 7.0, 7.0, 6.0, 8.0])
        sbins = (50,50,50,50)
        p = [7, 10, 9, 0]
        rlow =  np.array([-1])
        rhigh = np.array([7])
        rbins = (40,)
    elif name=="Walker2d":
        low = np.array([0.0,-1.0,-2.0,-3.0,-2.0,-2.0,-3.0,-2.0,-3.0,-6.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0])
        high = np.array([2.0,1.0,1.0,1.0,2.0,1.0,1.0,2.0,2.0,2.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0])
        sbins = (50,50,50,50,50)
        p = [13, 16, 14, 12, 15]
        rlow =  np.array([-4])
        rhigh = np.array([11])
        rbins = (75,)
    state_grid = create_uniform_grid(low, high, sbins)

    
    r_grid = create_uniform_grid(rlow, rhigh, rbins)
    
    
    # obs1 = rb1.observations
    # obs2 = rb2.observations
    # obs3 = rb3.observations

    # s = 10000
    # t = 11000
    # data1 = obs1[s:t][:,:,7]
    # data2 = obs2[s:t][:,:,7]
    # data3 = obs3[s:t][:,:,7]

    # data1 = rb3.rewards
    # print(np.max(data1))
    # print(np.min(data1))
    # exit()
    # obs2 = rb2.rewards
    #obs3 = rb3.rewards
 
    # obs1 = rb1.observations
    # obs2 = rb2.observations
    # obs3 = rb3.observations

    od1 = rb1.observations
    od2 = rb2.observations
    od3 = rb3.observations
    rd1 = rb1.rewards
    rd2 = rb2.rewards
    rd3 = rb3.rewards
    #print(od1[1][0][p])
    
    # print(discretize(od1[10000][0][p],state_grid))
    # print(tuple(discretize(od1[10000][0][p],state_grid)))
    # #dss = discretize(obs[0], state_grid)
    # print(omap[tuple(discretize(od1[10000][0][p],state_grid))])
    #print(omap.keys())


    #dobs2 = [omap[tuple(discretize(od2[i][0][p],state_grid))] for i in np.arange(s,rb2.pos)]
    #dobs3 = [omap[tuple(discretize(od3[i][0][p],state_grid))] for i in np.arange(s,rb3.pos)]
    '''
    data1 = rb1.rewards[s:rb1.pos]
    data2 = rb2.rewards[s:rb2.pos]
    data3 = rb3.rewards[s:rb3.pos]
    
    # Creating histogram
    #plt.hist(data1, bins=[-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3,3.2,3.4,3.6,3.8,4,4.2,4.4,4.6,4.8,5], alpha=0.45, color='red')
    #plt.hist(data2, bins=[-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3,3.2,3.4,3.6,3.8,4,4.2,4.4,4.6,4.8,5], alpha=0.45, color='blue')
    #plt.hist(data3, bins=[-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3,3.2,3.4,3.6,3.8,4,4.2,4.4,4.6,4.8,5], alpha=0.45, color='green')
    
    plt.hist(data1, bins=[-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3], alpha=0.45, color='red')
    #plt.hist(data2, bins=[-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3], alpha=0.45, color='blue')
    #plt.hist(data3, bins=[-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3], alpha=0.45, color='green')
    
    #plt.hist(data1, bins=20, alpha=0.45, color='red')
    #plt.hist(data2, bins=20, alpha=0.45, color='blue')
    #plt.hist(data3, bins=20, alpha=0.45, color='green')
    
    plt.xlabel("Abstract states",fontsize=18)
    plt.ylabel("No. of Samples",fontsize=18)
    plt.ylim(0, 300)
    
    plt.legend(['pruned'])
    plt.savefig('pruned_hopper_till_11k_new.pdf')
    
    # plt.legend(['full st'])
    # plt.savefig('full_st_hopper_till_11k_new.pdf')
    
    # plt.legend(['default'])
    # plt.savefig('default_hopper_till_11k_new.pdf')
    '''
    # dobs1 = []
    # domap = {}
    # for i in np.arange(s,rb1.pos-1):
    #     dobs1.append(len(omap[tuple(discretize(od1[i][0][p],state_grid))]))
    #     domap[tuple(discretize(od1[i][0][p],state_grid))] = i
    
    # print(dobs1)
    # #exit()
    # plt.plot(dobs1, alpha=0.45, color='red')
    #plt.xlabel("Abstract states",fontsize=18)
    #plt.ylabel("No. of Samples",fontsize=18)
    # plt.legend(['pruned'])
    # plt.savefig('pruned_distinct_obs_hop_till_11k.pdf')

    # dobs2 = []
    # domap = {}
    # for i in np.arange(s,rb2.pos-1):
    #     print(i)
    #     dobs2.append(len(omap[tuple(discretize(od2[i][0][p],state_grid))]))
    #     domap[tuple(discretize(od2[i][0][p],state_grid))] = i    
    # print(dobs2)
    # plt.plot(dobs2, alpha=0.45, color='blue')
    # plt.xlabel("Abstract states",fontsize=18)
    # plt.ylabel("No. of Samples",fontsize=18)
    # plt.legend(['full st'])
    # plt.savefig('full_st_distinct_obs_hop_till_11k.pdf')
    

    
    print(r_grid)
    print(rd1[0])
    ll = discretize(rd1[0],r_grid)
    print(ll[0])
    print(r_grid[0][ll])
    #exit()
    
    #map to count number of unique states
    cnt_map1 = {}
    cnt_map2 = {}
    cnt_map3 = {}
    
    s = 10000
    #t = 300000

    #######################################################
    dr1 = []
    drmap = {}
    for i in np.arange(s,rb1.pos-1):
        #print(i)
        #print(rd1[i])
        #print(discretize(rd1[i],r_grid))
        dobi = tuple(discretize(od1[i][0][p],state_grid))
        if dobi not in cnt_map1.keys():
            cnt_map1[dobi] = 0
        else:
            cnt_map1[dobi] += 1
        dr1.append(tuple(discretize(rd1[i],r_grid)))
        if tuple(discretize(rd1[i],r_grid)) not in drmap.keys():
            drmap[tuple(discretize(rd1[i],r_grid))] = [tuple(discretize(od1[i][0][p],state_grid))]
        else:
            #if tuple(discretize(od1[i][0][p],state_grid)) not in drmap[tuple(discretize(rd1[i],r_grid))]:
            drmap[tuple(discretize(rd1[i],r_grid))].append(tuple(discretize(od1[i][0][p],state_grid))) 

   
    for k in drmap.keys():
        drmap[k] = len(drmap[k])
    print(drmap.keys())
    print(drmap.values())
   
    rKeys = list(drmap.keys())
    rKeys.sort()
    rdict1 = {i: drmap[i] for i in rKeys}

    #######################################
    dr1 = []
    drmap = {}
    for i in np.arange(s,rb2.pos-1):
        #print(i)
        #print(rd1[i])
        #print(discretize(rd1[i],r_grid))
        dobi = tuple(discretize(od2[i][0][p],state_grid))
        if dobi not in cnt_map2.keys():
            cnt_map2[dobi] = 0
        else:
            cnt_map2[dobi] += 1

        dr1.append(tuple(discretize(rd2[i],r_grid)))
        if tuple(discretize(rd2[i],r_grid)) not in drmap.keys():
            drmap[tuple(discretize(rd2[i],r_grid))] = [tuple(discretize(od2[i][0][p],state_grid))]
        else:
            #if tuple(discretize(od2[i][0][p],state_grid)) not in drmap[tuple(discretize(rd2[i],r_grid))]:
            drmap[tuple(discretize(rd2[i],r_grid))].append(tuple(discretize(od2[i][0][p],state_grid))) 

   
    for k in drmap.keys():
        drmap[k] = len(drmap[k])
    print(drmap.keys())
    print(drmap.values())
    rKeys = list(drmap.keys())
    rKeys.sort()
    rdict2 = {i: drmap[i] for i in rKeys}
    ############################################
    dr1 = []
    drmap = {}
    for i in np.arange(s,len(rd3)-1):
        #print(i)
        #print(rd1[i])
        #print(discretize(rd1[i],r_grid))
        dobi = tuple(discretize(od3[i][0][p],state_grid))
        if dobi not in cnt_map3.keys():
            cnt_map3[dobi] = 0
        else:
            cnt_map3[dobi] += 1

        dr1.append(tuple(discretize(rd3[i],r_grid)))
        if tuple(discretize(rd3[i],r_grid)) not in drmap.keys():
            drmap[tuple(discretize(rd3[i],r_grid))] = [tuple(discretize(od3[i][0][p],state_grid))]
        else:
            #if tuple(discretize(od3[i][0][p],state_grid)) not in drmap[tuple(discretize(rd3[i],r_grid))]:
            drmap[tuple(discretize(rd3[i],r_grid))].append(tuple(discretize(od3[i][0][p],state_grid))) 

   
    for k in drmap.keys():
        drmap[k] = len(drmap[k])
    print(drmap.keys())
    print(drmap.values())
    rKeys = list(drmap.keys())
    rKeys.sort()
    rdict3 = {i: drmap[i] for i in rKeys}
    ##############################################
     
    print("no of distinct obs in FAC",len(cnt_map1.keys()))
    print("no of distinct obs in FAC full st",len(cnt_map2.keys()))
    print("no of distinct obs in SAC",len(cnt_map3.keys()))
    
    print(np.array(list(cnt_map1.values())))
    x1=np.arange(len(list(cnt_map1.values())))
    x2=np.arange(len(list(cnt_map2.values())))
    x3=np.arange(len(list(cnt_map3.values())))
    plt.plot(x1,list(cnt_map1.values()),  label = "pruned",alpha=0.45, color='red')
    ax = plt.axes()
    ax.set_facecolor("white")
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')
    plt.xlabel("Abstract States idx",fontsize=18)
    plt.ylabel("# Duplicates",fontsize=18)
    plt.title("FAC Unique States")
    plt.savefig('FAC_num_unique_states.pdf')
    plt.close()
    plt.plot(x2,list(cnt_map2.values()),  label = "full",alpha=0.45, color='green')
    ax = plt.axes()
    ax.set_facecolor("white")
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')
    plt.xlabel("Abstract States idx",fontsize=18)
    plt.ylabel("# Duplicates",fontsize=18)
    plt.title("FAC_full_state Unique States")
    plt.savefig('FAC_full_num_unique_states.pdf')
    plt.close()
    plt.plot(x3,list(cnt_map3.values()),  label = "def",alpha=0.45, color='blue')
    ax = plt.axes()
    ax.set_facecolor("white")
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')
    plt.xlabel("Abstract States idx",fontsize=18)
    plt.ylabel("# Duplicates",fontsize=18)
    plt.title("SAC Unique States")
    plt.savefig('SAC_num_unique_states.pdf')
    plt.close()
    #ax = plt.axes()
    #ax.set_facecolor("white")
    # ax.yaxis.grid(color='gray', linestyle='dashed')
    # ax.xaxis.grid(color='gray', linestyle='dashed')
    
    exit()
    print(list(rdict1.keys()))
    dk1 = np.array([ r_grid[0][k[0]] for  k in rdict1.keys()])
    dk2 = np.array([ r_grid[0][k[0]] for k in rdict2.keys()])
    dk3 = np.array([ r_grid[0][k[0]] for k in rdict3.keys()])

    print(dk1)
    print(dk2)
    print(dk3)
    #exit()
    plt.bar(dk1-0.1,list(rdict1.values()), width=0.2, alpha=0.45, color='red')
    plt.xlabel("Rewards",fontsize=18)
    plt.ylabel("No. of Abstract states",fontsize=18)
    
    # plt.xticks(dk1, rotation ='vertical')
    # plt.margins(0.2)
    
    plt.xlim(-1, 3)
    plt.ylim(0, 300)
    plt.legend(['pruned'])
    plt.savefig('pruned_reward_absstate_hop_till_11k.pdf')
    plt.close()

    plt.bar(dk2-0.1,list(rdict2.values()),  width=0.2, alpha=0.45, color='blue')
    plt.xlabel("Rewards",fontsize=18)
    plt.ylabel("No. of Abstract states",fontsize=18)
    plt.xlim(-1, 3)
    plt.ylim(0, 300)
    plt.legend(['full state'])
    plt.savefig('fullst_reward_absstate_hop_till_11k.pdf')
    plt.close()
    plt.bar(dk3-0.1,list(rdict3.values()),  width=0.2, alpha=0.45, color='green')
    plt.xlabel("Rewards",fontsize=18)
    plt.ylabel("No. of Abstract states",fontsize=18)
    plt.xlim(-1, 3)
    plt.ylim(0, 300)
    plt.legend(['default'])
    plt.savefig('default_reward_absstate_hop_till_11k.pdf')
   
    
    # plt.bar(list(rdict1.keys()),list(rdict1.values()), alpha=0.45, color='red')
    # plt.bar(list(rdict2.keys()),list(rdict2.values()), alpha=0.45, color='blue')
    # plt.bar(list(rdict3.keys()),list(rdict3.values()), alpha=0.45, color='green')
    # plt.xlabel("Rewards",fontsize=18)
    # plt.ylabel("No. of Abstract states",fontsize=18)
    # plt.legend(['pruned','full state','default'])
    # plt.savefig('reward_absstate_hop_till_11k.pdf')

    # plt.plot(dr1, alpha=0.45, color='blue')
    # plt.xlabel("Abstract states",fontsize=18)
    # plt.ylabel("No. of Samples",fontsize=18)
    # plt.legend(['full st'])
    # plt.savefig('full_st_distinct_obs_hop_till_11k.pdf')
    
    #plt.hist(data2, bins=20, alpha=0.45, color='blue')
    #plt.hist(data3, bins=20, alpha=0.45, color='green')
    
    #plt.title("Comparison")
    
    # plt.xlabel("Abstract states",fontsize=18)
    # plt.ylabel("No. of Samples",fontsize=18)
    
    # plt.legend(['default'])
    # plt.savefig('default_rew_pend_hist_till_20k.pdf')
    
    exit()
    
    # print("pruned mean ",np.mean(data1)," std  ",np.std(data1))
    # print("full st mean ",np.mean(data2)," std  ",np.std(data2))
    # print("def mean ",np.mean(data3)," std  ",np.std(data3))
    # exit()
    # plt.plot(data1[:,:,7],  label = "pruned", color='green')
    # plt.plot(data2[:,:,7],  label = "full", color='red')
    # plt.plot(data3[:,:,7],  label = "def", color='orange')
    
    # ax = plt.axes()
    # ax.set_facecolor("white")
    # ax.yaxis.grid(color='gray', linestyle='dashed')
    # ax.xaxis.grid(color='gray', linestyle='dashed')
    # plt.xlabel("Timesteps",fontsize=18)
    # plt.ylabel("Magnitude",fontsize=18)
    # plt.title("Full state v/s MD")
    
    # # Adding legend, which helps us recognize the curve according to it's color
    # plt.legend(prop={'size': 10})
   
    # rws = rb1.rewards[st:ed]
    # print(st,"  to  ",ed)
    # #print(len(rws))
    # print("mean ",np.mean(rws))
    # print("std ",np.std(rws))
    # print("min ",np.min(rws))
    # print("max ",np.max(rws))
    # print(len(rws))

    # sbw = 1.059*np.std(rws,ddof=1)*len(rws)**(-0.2)
    # print(" scott bw ",sbw)
    # exit()
    # print(len(acts))
    # print(acts[0][0])
    # print("length of action vector  ",len(acts[0][0]))
    # print(len(acts[0]))
    # print(acts[0])
    # print(len(acts[0][0]))
    # print(acts[0][0])
    # print(acts[1][0])
    # print(acts[2][0])
    # x = []
    # z = 100 #sample size
    # for i in range(len(acts[0:z])):
    #     x.append(acts[i][0])    
    # print(x)
    






    exit()
    p = [21, 23, 19, 25, 24, 20, 26, 22]
    q = [3, 1, 4, 7, 0, 5, 2, 6]
    #print("p is : ",p)
    #print("q is : ",q)

        
    acts=rb1.actions
    obs=rb1.observations
    print(len(acts))
    print(acts[0][0])
    print("length of action vector  ",len(acts[0][0]))
    print(len(acts[0]))
    print(acts[0])
    print(len(acts[0][0]))
    print(acts[0][0])
    print(acts[1][0])
    print(acts[2][0])
    x = []
    z = 100 #sample size
    for i in range(len(acts[0:z])):
        x.append(acts[i][0])    
    print(x)
    
    t = np.arange(len(x))
    #t = np.arange(1,z,10)
    
    id = 0 #action dim
    x0 = [x[i][id] for i in range(z)]
    plt.plot(t, x0, label = str(id), color='red')
    # id = 1 #action dim
    # x1 = [x[i][id] for i in range(z)]
    # plt.plot(t, x1, label = str(id), color='blue')
    # id = 2 #action dim
    # x2 = [x[i][id] for i in range(z)]
    # plt.plot(t, x2, label = str(id), color='green')
    # #id = 3 #action dim
    #x3 = [x[i][id] for i in range(z)]
    #plt.plot(t, x3, label = str(id), color='yellow')
    
    plt.legend()
    plt.savefig('acts.png')
    plt.show()
    exit()
    



  

    dbfile.close()
  
if __name__ == '__main__':
    loadData()
