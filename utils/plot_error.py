# import require modules
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
# defining our function
#x = np.array([0.1,0.3,0.5,0.7,1])
#y = np.array([18.35,17.25,16.33,14.75,13.75])
#y_error = np.array([0.06,0.07,0.03,0.06,0.09])
#y = np.array([1559.09,1231.04,1724.42,1548.83,1277.20])
#y_error = np.array([19.78,19.29,33.49,26.60,31.95])

x = np.array([10,50,100,200,300,400,500])
#y = np.array([1869.61,1396.36,1013.41,949.05,1231.04,960.24,1556.75])
#y_error = np.array([37.64,23.77,25.48,13.16,19.29,43.17,159.59])
y = np.array([18.62,16.33,18.40,17.17,17.25,16.95,16.56])
y_error = np.array([18.62,0.06,0.04,0.02,0.07,0.47,1.85])

# plotting our function and
# error bar
#fig.suptitle('Control Cost with $\mu$=0.3 ', fontsize=20)
plt.xlabel(r'$\eta$', fontsize=18)
#plt.xlabel(r'$\mu$', fontsize=18)
plt.ylabel('Distance Covered', fontsize=16)
#plt.ylabel('Control Cost', fontsize=16)
plt.plot(x, y)

plt.errorbar(x, y, yerr = y_error, fmt ='o')

plt.savefig("dc_beta.pdf")
#plt.savefig("cc_mu.pdf")
