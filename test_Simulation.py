import pandas as pd
import matplotlib.pyplot as plt
import pyLeaf

if __name__ == "__main__":
    Weather = pd.read_excel("Input.xlsx")
    Parameters = {'vcmax25':50,'go':0.08,'g1':3}
    C4Leaf = pyLeaf.Leaf(Parameters)
    C4Leaf.SeriesSolver(Weather)

plt.subplot(131)
plt.plot(Weather['PAR'], C4Leaf.LeafMassFlux['aNet'])
plt.xlabel(r'PAR [W m$^{-2}$]', fontsize=14)
plt.ylabel(r'$A_{net}$ [$\mu$ mol m$^{-2}$ s$^{-1}$]', fontsize=14)

plt.subplot(132)
plt.plot(Weather['PAR'], C4Leaf.LeafState['tLeaf'])
plt.xlabel(r'PAR [W m$^{-2}$]', fontsize=14)
plt.ylabel(r'$T_{leaf}$ [$^{o}$C]', fontsize=14)

plt.subplot(133)
plt.plot(Weather['PAR'], C4Leaf.LeafMassFlux['transpiration'])
plt.xlabel(r'PAR [W m$^{-2}$]', fontsize=14)
plt.ylabel(r'$\tau$ [$\mu$ mol m$^{-2}$ s$^{-1}$]', fontsize=14)

plt.show()