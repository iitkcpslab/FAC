
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from sklearn.neighbors import KernelDensity




# ----------------------------------------------------------------------
# Plot a 1D density example
N = 40
np.random.seed(1)
X = np.concatenate(
    (np.random.normal(0, 1, int(0.3 * N)), np.random.normal(5, 1, int(0.7 * N)))
)[:, np.newaxis]

 
# Xu = np.random.uniform(-5, 10, N)[:, np.newaxis]
# Xu1 = np.linspace(-5, 10, N)[:, np.newaxis]

# plt.scatter(np.arange(len(Xu1)),Xu1)
# plt.savefig('demo.pdf')
# exit()



X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

true_dens = 0.3 * norm(0, 1).pdf(X_plot[:, 0]) + 0.7 * norm(5, 1).pdf(X_plot[:, 0])

fig, ax = plt.subplots()
#ax.fill(X_plot[:, 0], true_dens, fc="black", alpha=0.2, label="input distribution")
colors = ["navy", "cornflowerblue", "darkorange"]
kernels = ["gaussian", "tophat", "epanechnikov"]
lw = 2

for color, kernel in zip(colors, kernels):
    kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(X)
    log_dens = kde.score_samples(X_plot)
    ax.plot(
        X_plot[:, 0],
        np.exp(log_dens),
        color=color,
        lw=lw,
        linestyle="-",
        label="'{0}'".format(kernel),
    )

#ax.text(6, 0.38, "N={0} points".format(N))

#generating uniform data
#Xu = np.random.uniform(-5, 10, N)[:, np.newaxis]
Xu = np.linspace(-5, 10, N)[:, np.newaxis]

kde = KernelDensity(kernel="epanechnikov", bandwidth=0.5).fit(Xu)
log_dens = kde.score_samples(X_plot)
ax.plot(
        X_plot[:, 0],
        np.exp(log_dens),
        color="green",
        lw=lw,
        linestyle="-",
        label="uniform epanechnikov",
    )


ax.legend(loc="upper left")
ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), "+k")
ax.plot(Xu[:, 0], -0.005 - 0.01 * np.random.random(Xu.shape[0]), "+r")
ax.set_xlim(-4, 9)
ax.set_ylim(-0.02, 0.4)
plt.savefig('kernels.pdf')
