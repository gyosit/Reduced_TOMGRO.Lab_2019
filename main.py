#Reference
 #Jones(1999) "Reduced state-variable tomato growth model"
 #Jones(1991) "A dynamix tomato growth and yield model(TOMGRO)"
 #Dimokas(2009) "Calibration and validation of a biological model..."
 #Heuvelink(1994) "Dry-matter partitioning in a tomato crop:Comparison of two simulation models"

import numpy as np

##########
# dN/dt : The rate of node development
def fN(T):
  #Heuvelink(1994)
  if(T > 12 and T <= 28):
    return 1.0 + 0.0281 * (T - 28)
  else if(T > 28 and T < 50):
    return 1.0 - 0.0455 * (T - 28)
  else
    return 0
  return 0

def dNdt(T):
  Nm = 0.5 #P Jones(1991)
  return Nm * fN(T)

##########
# d(LAI)/dt : The rate of LAI(Leaf Area Index) development

def lambda(T):
  return 0.5

def dLAIdt(dens, T, Td , N, lambda_, dNdt_):
  max_LA_exp = 3.10 #P (rho)
  beta = 0.169 #P coefficient
  Nb = 16.0 #P LAI vs N, horizontal axis coefficient
  a = np.exp(beta * (N - Nb))
  return dens * max_LA_exp * lambda_ * a * dNdt_ / (1 + a)

##########
# dWfdt : The rate of Fruit dry weight

def fR(N):
  #root phenology-dependent fraction; Jones(1991)
  return -0.0046 * N + 0.2034

def LFmax(CO2):
  #maximum leaf photosyntehstic rate;Jones(1991)
  #CO2[ppm]?
  CO2_ef = 0.0693 #P (tau) carbon dioxide use efficiency; Jones(1991)
  return CO2_ef * CO2

def PGRED(T):
  #function to modify Pg under suboptimal daytime temperatures; Jones(1991)
  if(T > 0 and T <= 12):
    return 1.0 / 12.0 * T
  else if(T > 12 and T <= 28):
    return 1.0
  else if(T > 28 and T <= 35):
    return 1.0 - 1.0 / 7.0 * (T - 28.0)
  else:
    return 0
  return 0

def Pg(LFmax_, T, PPFD, LAI):
  D = 2.593 #P coefficient to convert Pg from CO2 to CH2O; Jones(1991)
  K = 0.58 #P light extinction coefficient; Jones(1991)
  m = 0.1 #P leaf light transmission coefficient; Jones(1991)
  Qe = 0.0645 #P leaf quantum efficiency; Jones(1991)
  a = D * LFmax_ * PGRED(T) / K
  b = np.log(((1-m) * LFmax_ + Qe * K * PPFD) / 
    ((1-m) * LFmax_ + Qe * K * PPFD * np.exp(-1 * K * LAI)))
  return  a * b

def Rm(T, W, Wm):
  #Jones(1999)
  #Hourly Data!
  Q10 = 1.4 #Jones(1991)
  rm = 0.016 #Jones(1999)
  return Q10 ** ((T-20)/10) * rm * (W - Wm)

def GRnet(Pg_, Rm_, fR_):
  E = 0.717 #P convert efficiency; Dimokas(2009)
  return E * (Pg_ - Rm_) * (1 - fR_)

def g(T_daytime):
  T_CRIT = 24.4 #P mean daytime temperature above which fruits abortion start; Jones(1999)
  return 1.0 - 0.154 * (T_daytime - T_CRIT)

def dWfdt(GRnet_)
#rfがわからない