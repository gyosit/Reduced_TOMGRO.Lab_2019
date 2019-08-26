#Reference
 #Jones(1999) "Reduced state-variable tomato growth model"
 #Jones(1991) "A dynamix tomato growth and yield model(TOMGRO)"
 #Dimokas(2009) "Calibration and validation of a biological model..."
 #Heuvelink(1994) "Dry-matter partitioning in a tomato crop:Comparison of two simulation models"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

##########
# dN/dt : The rate of node development
def fN(T):
  #Heuvelink(1994) & Jones(1991)
  if(T > 12 and T <= 28):
    return 1.0 + 0.0281 * (T - 28)
  elif(T > 28 and T < 50):
    return 1.0 - 0.0455 * (T - 28)
  else:
    return 0
  return 0

def dNdt(fN_):
  Nm = 0.5 #P Jones(1991)
  #("dNdt:", Nm * fN_)
  return Nm * fN_

##########
# d(LAI)/dt : The rate of LAI(Leaf Area Index) development

def lambdas(Td):
  return 1.0

def dLAIdt(LAI, dens, N, lambda_, dNdt_):
  sigma = 0.038 #P Maximum leaf area expansion per node, coefficient in expolinear equation; Jones(1999)
  beta = 0.169 #P Coefficient in expolinear equation; Jones(1999)
  Nb = 16.0 #P Coefficient in expolinear equation, projection of linear segment of LAI vs N to horizontal axis; Jones(1999)
  LAImax = 4.0 #P Jones(1999)

  if(LAI > LAImax):
    return 0
    
  a = np.exp(beta * (N - Nb))
  return dens * sigma * lambda_ * a * dNdt_ / (1 + a)

##########
# dW/dt
def dWdt(LAI, dWfdt_, GRnet_, dens, dNdt_):
  LAImax = 4.0 #P Jones(1999)
  if(LAI >= LAImax):
    p1 = 2.0 #P Jones(1999)
  else:
    p1 = 0
  Vmax = 8.0 #P Jones(1999)

  a = dWfdt_ + (Vmax - p1) * dens * dNdt_
  b = GRnet_ - p1 * dens * dNdt_
  return min(a, b)

# dWdt_max

##########
# dWmdt
def Df(T):
  # The rate of development or aging of fruit at temperature T; Jones(1991)
  if(T > 9 and T <= 28):
    return 0.0017 * T - 0.015
  elif(T > 28 and T <= 35):
    return 0.032
  else:
    return 0
  
def dWmdt(Df_, Wf, Wm, N):
  NFF = 22.0 #P Jones(1999)
  kF = 5.0 #P Jones(1999)

  if(N <= NFF + kF):
    return 0
  return Df_ * (Wf - Wm)

##########
# dWfdt : The rate of Fruit dry weight

def fR(N):
  #root phenology-dependent fraction; Jones(1991)
  if(N >= 30):
    return 0.07
  return -0.0046 * N + 0.2034

def LFmax(CO2):
  #maximum leaf photosyntehstic rate;Jones(1991)
  #CO2[ppm]?
  tau = 0.0693 #P carbon dioxide use efficiency; Jones(1991)
  return tau * CO2

def PGRED(T):
  #function to modify Pg under suboptimal daytime temperatures; Jones(1991)
  if(T > 0 and T <= 12):
    return 1.0 / 12.0 * T
  elif(T > 12 and T < 35):
    return 1.0
  else:
    return 0
  return 0

def Pg(LFmax_, PGRED_, PPFD, LAI):
  D = 2.593 #P coefficient to convert Pg from CO2 to CH2O; Jones(1991)
  K = 0.58 #P light extinction coefficient; Jones(1991)
  m = 0.1 #P leaf light transmission coefficient; Jones(1991)
  Qe = 0.0645 #P leaf quantum efficiency; Jones(1991)
  a = D * LFmax_ * PGRED_ / K
  b = np.log(((1-m) * LFmax_ + Qe * K * PPFD) / 
    ((1-m) * LFmax_ + Qe * K * PPFD * np.exp(-1 * K * LAI)))
  return  a * b

def Rm(T, W, Wm):
  #Jones(1999)
  #Hourly Data!
  Q10 = 1.4 #P Jones(1991)
  rm = 0.016 #P Jones(1999)
  return Q10 ** ((T-20)/10) * rm * (W - Wm)

def GRnet(Pg_, Rm_, fR_):
  E = 0.717 #P convert efficiency; Dimokas(2009)
  #print("GRnet", Pg_, Rm_, fR_)
  return max(0, E * (Pg_ - Rm_) * (1 - fR_))

def fF(Td):
  #Jones(1991)
  if(Td > 8 and Td <= 28):
    return 0.0017 * Td - 0.0147
  elif(Td > 28):
    return 0.032
  else:
    return 0
  return 0

def g(T_daytime):
  T_CRIT = 24.4 #P mean daytime temperature above which fruits abortion start; Jones(1999)
  return 1.0 - 0.154 * (T_daytime - T_CRIT)

def dWfdt(GRnet_, fF_, N, g_):
  NFF = 22.0 #P Nodes per plant when first fruit appears; Jones(1999)
  alpha_F = 0.80 #P Maximum partitioning of new growth to fruit; Jones(1999)
  v = 0.135 #P Transition coefficient between vegetative and full fruit growth; Jones(1999)
  fF_ = 0.5 #P ORIGINAL
  if(N <= NFF):
    return 0
  #print(GRnet_, fF_, 1 - np.exp(v*(NFF-N)), g_)
  return GRnet_ * alpha_F * fF_ * (1 - np.exp(v*(NFF-N))) * g_

def createGraph(bots, filename):
  fig, ax = plt.subplots(figsize=(10, 5))
  norm_ = plt.Normalize(vmin=100, vmax=170)
  cm_ = plt.cm.rainbow
  plt.title(filename)
  ax.set_xlabel("Temperature [°C]")
  ax.set_ylabel("PPFD [μmol m^-2 s^-1]")
  im = ax.scatter(bots.T[1][0], bots.T[2][0], c=bots.T[0][1], cmap=cm_, norm=norm_)
  sm = plt.cm.ScalarMappable(cmap=cm_, norm=norm_)
  sm.set_array([])
  plt.colorbar(sm)
  plt.xlim(15, 40)
  plt.ylim(20, 60)
  plt.savefig(filename+".png")

def calc(inT, inPPFD):
  #T_max = np.array([32.8, 34.4, 32.2, 30.6, 28.3, 28.3, 30.1, 30.6, 30.6, 30.3, 31.0, 29.7, 29.7, 30.7, 30.1, 30.9 ,28.8, 26.8, 29.5, 29.7, 29.0])
  #T_min = np.array([22.5, 21.4, 21.4, 20.8, 19.7, 15.7, 15.5, 15.8, 17.0, 18.1, 18.4, 13.3, 18.0, 18.6, 18.8, 20.6, 20.0, 17.0, 17.2, 16.9, 18.5])
  #Tdata = (T_max + T_min) / 2
  Tdata = np.array([inT]*90)
  PPFD = 40 #Jones(1991)
  PPFD = inPPFD

  #Initial Variables
  N = 6.0
  LAI = 0.006
  W = 0
  Wm = 0
  Wf = 0
  CO2 = 350

  for T in Tdata:
    # Reset varialbes
    dNdt_ = 0
    Td = 0 # Change!

    # for daily tempereture
    for h in range(24):
      Td += T
      fN_ = fN(T)
    Td /= 24 # Dialy temperature
    dNdt_ += dNdt(fN_)

    # dN/dt
    #fN_ = fN(Td)
    #dNdt_ += dNdt(fN_

    # d(LAI)/dt
    lambda_ = lambdas(Td)
    dLAIdt_ = dLAIdt(LAI=LAI, dens=3.10, N=N, lambda_=lambda_, dNdt_=dNdt_)

    # dWfdt
    fR_ = fR(N)
    LFmax_ = LFmax(CO2)
    PGRED_ = PGRED(Td)
    Pg_ = Pg(LFmax_, PGRED_, PPFD, LAI)
    Rm_ = Rm(Td, W, Wm)
    GRnet_ = GRnet(Pg_, Rm_, fR_)
    fF_ = fF(Td)
    g_ = g(Td)
    dWfdt_ = dWfdt(GRnet_, fF_, N, g_)

    # dWdt
    dWdt_ = dWdt(LAI=LAI, dWfdt_=dWfdt_, GRnet_=GRnet_, dens=3.10, dNdt_=dNdt_)

    # dWmdt
    Df_ = Df(Td)
    dWmdt_ = dWmdt(Df_, Wf, Wm, N)

    # Reload
    N += dNdt_
    LAI += dLAIdt_
    Wf += dWfdt_
    W += dWdt_
    Wm += dWmdt_

  return {"stem":N, "LAI":LAI, "Wf":Wf, "W":W, "Wm":Wm}

def main1():
  ##########
  # Main Loop
  res = []

  # Initial Value
  targT = 34.0
  targPPFD = 44.89
  deltaT = 0.1
  deltaPPFD = 0.1

  resT = [0, 0]
  resPPFD = [0, 0]
  slopeT = 0
  slopePPFD = 0

  for i in range(10000):
    for j in range(2):
      targT += deltaT * j
      targPPFD += deltaPPFD * j

      res = calc(targT, targPPFD)
      N = res["stem"]
      LAI = res["LAI"]
      W = res["W"]
      Wf = res["Wf"]
      Wm = res["Wm"]

      targT -= deltaT * j
      targPPFD -= deltaPPFD * j
      resT[j] = W
      resPPFD[j] = W

    print(W, targT, targPPFD)
    slopeT = resT[1] - resT[0]
    slopePPFD = resPPFD[1] - resPPFD[0]

    targT += 0.1 * slopeT
    targPPFD += 0.1 * slopePPFD

def main2():
  ##########
  # Main Loop
  bots = []
  best = [0, 30, 40]

  # Initial Value
  for i in range(100):
    T = random.random() * 50
    PPFD =random.random() * 40
    v = [0, 0, 0]
    xy = [0, T, PPFD]
    opt = [0, T, PPFD]
    bots.append([xy, opt, v])
  bots = np.array(bots)

  for i in range(10):
    for bot in bots:
      T = bot[0][1]
      PPFD = bot[0][2]
      res = calc(T, PPFD)
      N = res["stem"]
      LAI = res["LAI"]
      W = res["W"]
      Wf = res["Wf"]
      Wm = res["Wm"]
      
      bot[0][0] = W
      if(W > bot[1][0]):
        bot[1][0] = W
        bot[1][1] = T
        bot[1][2] = PPFD

      if(W > best[0]):
        best[0] = W
        best[1] = T
        best[2] = PPFD

    for bot in bots:
      delT1 = bot[1][1] - bot[0][1]
      delT2 = best[1] - bot[0][1]
      delPPFD1 = bot[1][2] - bot[0][2]
      delPPFD2 = best[2] - bot[0][2]
      v = bot[2]
      v = [0, v[0]+0.5*delT1+0.3*delT2, v[1]+0.5*delPPFD1+0.3*delPPFD2]
      bot[2] = v
      bot[0][1] += v[1]
      bot[0][2] += v[2]

    createGraph(bots, str(i)+"turn")
    print(best)
    
  #plt.show()
  #for bot in bots:
    #print(bot[0])

main2()