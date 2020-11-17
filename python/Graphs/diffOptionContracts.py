"""
///////////////////////// TOP OF FILE COMMENT BLOCK ////////////////////////////
//
// Title:           Plot of European call and European put pricing in B-S Model
// Course:          Master's thesis, 2020
//
// Author:          Peter Pommergård Lind
// Email:           ppl_peter@protonmail.com
// Encoding:        utf-8
///////////////////////////////// CITATIONS ////////////////////////////////////
//
// Options, Futures, and Other Derivatives by John C. Hull 10th edition
//
/////////////////////////////// 80 COLUMNS WIDE ////////////////////////////////
"""
import numpy as np

def EuroCall(K,S):
    return ( max(S-K, 0))

def EuroPut(K,S):
    return ( max(K-S, 0))


from matplotlib import pyplot as plt

plt.style.use('ggplot')
fig, ax = plt.subplots(1, 2)
fig.set_size_inches(6, 4)
x = np.arange(0, 200, 0.5)
y = [EuroCall(100, S) for S in x]
ax[0].set_title('European call')
ax[0].set_xlabel("S(T)")
ax[0].set_ylabel("Payoff")
ax[0].plot(x, y, 'r', linewidth=1)
x1 = np.arange(0, 200, 0.5)
y1 = [EuroPut(100, S) for S in x]
ax[1].set_title('European put')
ax[1].set_xlabel("S(T)")
ax[1].plot(x1, y1, 'c', linewidth=1)
plt.savefig("/home/ppl/Documents/Universitet/KUKandidat/Speciale/DeepPricing/latex/Figures/contractfct.pdf")
plt.show()
