import matplotlib.pyplot as plt
import csv
import numpy as np

x = []
y = []
name = 'hop_default.csv'
name = 'run-TD3_LABER_hum.csv'

with open(name, mode ='r')as file:
        csvFile = csv.reader(file)
        next(csvFile)
        
        for lines in csvFile:
            #print(lines)
            x.append(float(lines[1])*0.001)     
            y.append(float(lines[2]))

mx = np.max(y)

mi = np.min(y)
if mx<0:
    y = [x + abs(mi)  for x in y] 
      
rev_y = y[::-1]
rev_x = x[::-1]

mx = np.max(y)
print(mx,0.9*mx)
res = next(x for x, val in enumerate(rev_y) if val < 0.9*mx)
     
print(res)
print(rev_x[res])

plt.plot(x,y, label = "rw",color="purple")

plt.savefig("conv.png")
#plt.show()

