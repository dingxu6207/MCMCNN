# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 14:51:36 2021

@author: dingxu
"""
import numpy as np
import matplotlib.pylab as plt
import time
from tensorflow.keras.models import load_model
import emcee
import corner


def calculater(ydata, caldata):
    res_ydata  = np.array(ydata) - np.array(caldata)
    stdres = np.std(res_ydata)
    ss_res     = np.sum(res_ydata**2)
    ss_tot     = np.sum((ydata - np.mean(ydata))**2)
    r_squared  = 1 - (ss_res / ss_tot)
    return stdres,r_squared


def quantile(x, q, weights=None): 
 
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    if weights is None:
        return np.percentile(x, list(100.0 * q))
    else:
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x)")
        idx = np.argsort(x)
        sw = weights[idx]
        cdf = np.cumsum(sw)[:-1]
        cdf /= cdf[-1]
        cdf = np.append(0, cdf)
        return np.interp(q, cdf, x[idx]).tolist()
    
     
mpath = ''
model10mc = load_model(mpath+'model10mc.hdf5')
l3model10mc = load_model(mpath+'model10l3mc.hdf5')


path = ''
fileone = 'KIC 6431545.txt'
data = np.loadtxt(path+fileone)

phrase = data[:,0]
datay = data[:,1]-np.mean(data[:,1])
x = np.linspace(0,1,100) 
sigma = np.diff(datay,2).std()/np.sqrt(6) #Estimated observation noise values


###########MCMCparameters
nwalkers = 30
niter = 500
nburn = 200 #Retain the last number of points for calculation
index = 0

#initial space[T/5850，incl/90,q,f,t2t1,l3,offset1, offset2]
init_dist = [(5459/5850-0.0001, 5459/5850+0.0001), 
             (54.40/90-6/90, 54.40/90+10/90), 
             (0.5, 3), 
             (0.05, 0.9), 
             (0.8, 1.2),
             #(0, 1),
             (-10,10),
             (-0.01,0.01)
             ]

priors=init_dist.copy()
ndim = len(priors) #Number of dimensions

def predict(allpara):
    
    arraymcn = np.array(allpara)
    
    if index == 0:
        arraymc = arraymcn[0:5]
        mcinput = np.reshape(arraymc,(1,5))
        lightdata = model10mc(mcinput)
        return lightdata[0]+arraymcn[6]
        
        
    if index == 1:
        arraymc = arraymcn[0:6]
        mcinput = np.reshape(arraymc,(1,6))
        lightdata = l3model10mc(mcinput)
        return lightdata[0]+arraymcn[7]
        
        
    

def getdata(allpara):
    arraymc = np.array(allpara)
    if index == 0:
        offset = int(arraymc[5])
        dataym = np.hstack((datay[offset:], datay[:offset]))
        
    if index == 1:
        offset = int(arraymc[6])
        dataym = np.hstack((datay[offset:], datay[:offset]))
    
    noisy = np.interp(x,phrase,dataym) #y轴
    
    return noisy

def rpars(init_dist):#In the ndim dimension, spread the ndim points evenly over the initial range
    return [np.random.rand() * (i[1]-i[0]) + i[0] for i in init_dist] 


def lnprior(priors, values):#Determine if the new MCMC point is inside the initial area
    
    lp = 0.
    for value, prior in zip(values, priors):
        if value >= prior[0] and value <= prior[1]:
            lp+=0
        else:
            lp+=-np.inf 
    return lp


def lnprob(z): #Calculating the posterior probability
    
    lnp = lnprior(priors,z)#Determine if the new MCMC point is inside the initial area

    if not np.isfinite(lnp):
            return -np.inf


    output = predict(z)
    
    noisy = getdata(z)
    
    lnp = -0.5*np.sum(np.log(2 * np.pi * sigma ** 2)+(output-noisy)**2/(sigma**2)) #Calculating the likelihood function
      
    return lnp


def run(init_dist, nwalkers, niter,nburn):
    
    ndim = len(init_dist)
    # Generate initial guesses for all parameters for all chains
    p0 = [rpars(init_dist) for i in range(nwalkers)] 

    sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob) 
    pos, prob, state = sampler.run_mcmc(p0, niter, progress=True) 
    emcee_trace = sampler.chain[:, -nburn:, :].reshape(-1, ndim).T 

    return emcee_trace 

t1 = time.time()
emcee_trace  = run(priors, nwalkers, niter,nburn) #run mcmc
print('time=',time.time()-t1) 
    
mu = []
sigma_1 = []
sigma_2 = []
    
for mi, x1 in enumerate(emcee_trace):
    q_16, q_50, q_84 = quantile(x1, [0.16, 0.5, 0.84])          
    q_m, q_p = q_50 - q_16, q_84 - q_50
  
    mu.append(q_50) #median value
    sigma_1.append(q_m) #high limitation
    sigma_2.append(q_p) #low
 
sigma_1 = np.array(sigma_1)
sigma_2 = np.array(sigma_2) 
   



####################

if index == 1:
    emcee_trace[1,:] = emcee_trace[1,:]*90
    figure = corner.corner(emcee_trace.T[:,1:6],bins=100,
                           labels=[r"$incl$", r"$q$", r"$f$", r"$T_2/T_1$", r"$l3$", r"$offset1$", r"$offset2$"],
                           label_kwargs={"fontsize": 15},title_fmt='.4f',show_titles=True, title_kwargs={"fontsize": 15}, color ='blue',
                           fill_contours=True,smooth=0.3,smooth1d=0.3)

if index == 0:
    emcee_trace[1,:] = emcee_trace[1,:]*90
    figure = corner.corner(emcee_trace.T[:,1:5],bins=100,
                           labels=[r"$incl$", r"$q$", r"$f$", r"$T_2/T_1$", r"$offset1$", r"$offset2$"],
                           label_kwargs={"fontsize": 15},title_fmt='.4f',show_titles=True, title_kwargs={"fontsize": 15}, color ='blue',
                           fill_contours=True,smooth=0.3,smooth1d=0.3)
    
plt.savefig('corner.png')
#------------------------------------------------------------

pre=predict(mu)
plt.figure()
ax = plt.gca()

if index == 0:
    offset = int(mu[5])
else :
    offset = int(mu[6])

datay = np.hstack((datay[offset:], datay[:offset])) 
noisy = np.interp(x, phrase, datay)  

 
ax.plot(phrase, datay, '.', c = 'b')

ax.plot(x, pre,'-r') 
ax.yaxis.set_ticks_position('left') 
ax.invert_yaxis() #y-axis reversed
plt.xlabel('phase',fontsize=18)
plt.ylabel('mag',fontsize=18)

print('T1 = '+str(mu[0]*5850))
print('incl = '+str(mu[1]*90))
print('q = '+str(mu[2]))
print('f = '+str(mu[3]))
print('t2t1 = '+str(mu[4]))

if index == 1:
    print('l3 = '+str(mu[5]))