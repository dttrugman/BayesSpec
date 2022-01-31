### Example PyMC3 Model for Bayesian Source Spectral Inference ####
### Daniel Trugman (dtrugman@jsg.utexas.edu), 2022
###  See: https://docs.pymc.io/en/v3/
###    for more details on the PyMC3 probabalistic Programming Environment

##### Import statements #####
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az
import theano.tensor as tt

#### Define Run Parameters ####

def get_params():

    # parameter dictionary
    params = {
        "data_file": "data.csv", # specify path to your spectral ratio data
        "event_file": "events.csv", # specify path to your event list (event 0 = target, 1+ = egfs)
        "out_file": "pymc3-out.netcdf", # specify output file
        "gamma": 1.0, # Brune or Boatwright sharpness (1.0 or 2.0)
        "kS": 0.26, # Kaneko/Shearer for S-wave, Vr=0.9*beta
        "beta": 3000.0, # shear wave speed, m/s
        "nsamp": 2500, # number of samples per thread after tuning
        "ncore": 4, # number of cores to use for parallel threads
        "ntune": 1500, # number of tuning samples (excluded on output)
    }

#### Define Priors ####

def get_priors():

    # define prior parameters in a dictionary
    priors = {
        "strikeP": 90.0, # strike prior (from moment tensor, for example)
        "stdSTK": 3.0, # standard deviation of normal distribution
        "meanLDS": np.log10(5.0), # mean stress drop, log10 MPa
        "stdLDS": 0.40,  # stdev of stress drop, log10 MPa
        "stdMw": 0.10, # stdev of Mw
        "lmuNFALL": np.log(2.0), # lognormal mean for nfall
        "lsigNFALL": 0.10, # lognormal sigma for nfall
        "alphaVR": 12.0, # alpha for vr beta distribution
        "betaVR": 3.0, # beta for vr beta distribution
    }

    # derived parameters
    priors["stdLM0"] = 1.5*priors["stdMw"] # converted to log10 M0
    priors["stdLFC"] = priors["stdLDS"] / 3.0 # converted to log10 FC

    # return results
    return priors

#### Define Auxiliary Functions ####

# Compute corner frequency from Moment (Nm), Stress Drop (MPa), k-factor, shear speed (m/s)
def get_fc(M0,DS,kF,beta):
    return kF * beta * ((16.0/7.0)*(1.0e6*DS/M0))**(1.0/3.0)

# Compute M0 (Nm) from Mw
def get_M0(Mw):
    return 10.0**(1.5*Mw + 9.1)

# Compute log10 of theoretical spectrum at frequencies ff.
#   - Assumes a zero-f level of M0, corner fc, sharpness gamma, falloff nfall
def tspec10(ff,M0,fc,gamma,nfall): 
    return np.log10(M0) - (1.0/gamma) *np.log10(
        1.0 + (ff/fc)**(nfall*gamma))

# Sine and Cosine in degrees
def cosd(x):
    return np.cos(np.pi*x/180.0)
def sind(x):
    return np.sin(np.pi*x/180.0)

###############################################################
###############################################################

#################### Main function ###########################

if __name__ == "__main__":

    ### Load parameters
    params = get_params()

    ### Load priors
    priors = get_priors()

    ### Load datasets

    # spectral ratios
    data = pd.read_csv(params["data_file"])
    ndata = len(data)

    # event information: **
    # *** note: assumes first (0th) event is the target
    events = pd.read_csv(params["event_file"])
    nev = len(events)

    ### Extract Data and Define Event-specific priors

    # compute moment for each event
    preM0 = get_M0(events["mag"].values)
    preLM0 = np.log10(preM0) # in log10 units

    # use this to compute priors for fc for each event
    preDS = (10.0**priors["meanLDS"])*np.ones(nev) # in MPa
    preFC = get_fc(preM0,preDS,params["kS"],params["beta"]) # Hz
    preLFC = np.log10(preFC) # convert to log10 units

    # extract arrays from data df
    ffSR = data["ff"].values # frequency for each data point (Hz)
    l10SR = data["l10SR"].values # log10 of spectral ratio for each data point
    eID = data["eID"].values # serial eventID, with 0 being the target event and 1+ being egfs
    aziSR = data["azi"].values # azimuth in degrees from source to station
    toaSR = data["toa"].values # takeoff angle in degrees from vertical upward

    # define angles for calculation of directivity factor normalization
    #   (simple array sampling focal sphere)
    psivec = np.linspace(0.0,180.0,361)
    cospsivec = cosd(psivec)

    ###############################

    #### Define Bayesian Model ####

    print("Instantiating model, please wait...")
    t00 = time.time()

    # define model context: normal Nfall + Tdist Noise
    with pm.Model() as bayspec:

        # prior for log10 FC
        logFC = pm.Normal("logFC",mu=preLFC,sigma=priors["stdLFC"],shape=nev) 

        # prior for log10 dMR
        dlogMR = pm.Normal("dlogMR",mu=0.0,sigma=priors["stdLM0"],shape=nev)

        # prior for falloff rate
        nfalls = pm.Lognormal("nfalls",mu=priors["lmuNFALL"],
            sigma=priors["lsigNFALL"],shape=nev)

        # prior for rupture strike angle, dip direction
        #   (optional but sometimes helpful to bound rdip to avoid singularity)
        strike = pm.Normal("strike",mu=priors["strikeP"],sigma=priors["stdSTK"])
        rdip = pm.Bound(pm.Normal, lower=20.0, upper=160.0)("rdip",mu=90.0,sigma=7.5)

        # prior for normalized vrup and directivity ratio
        vrup = pm.Beta("vrup",alpha=priors["alphaVR"],beta=priors["betaVR"])
        edir = pm.Uniform("edir",lower=-1.0,upper=1.0) # either direction

        # compute correction factor to normalize directivity
        zeta00 = vrup*cospsivec
        fcorr = tt.mean( # averaged across focal sphere
            tt.sqrt((1.0+edir**2)*(1.0+tt.square(zeta00))+4.0*edir*zeta00) / (
            tt.sqrt(2.0)*(1.0-tt.square(zeta00)))     )
        
        # deterministic calculation of fC for Target
        #   zeta = vrup/c * cosine of angle between rupture and ray
        #   fdir is the directivity function, normalized to 1 over focal sphere
        zeta = vrup*(cosd(aziSR-strike)*sind(toaSR)*sind(rdip)+cosd(toaSR)*cosd(rdip))
        fdir = tt.sqrt((1.0+edir**2)*(1.0+tt.square(zeta))+4.0*edir*zeta) / (
            tt.sqrt(2.0)*(1.0-tt.square(zeta))) / fcorr # normalized across focal sphere
        fcT = (10.0**(logFC[0])) * fdir # modulate by directivity function

        # deterministic calculation of fC for  EGF E
        fcE = 10.0**(logFC[eID]) # EGFs are easy w/out directivity

        # prior for tsig and tdof (uninformative)
        tsig = pm.HalfNormal("tsig",sd=0.5)
        tdof = pm.HalfNormal("tdof",sd=10.0)

        # theoretical spectral ratio: deterministic variable
        theoSR = (preLM0[0] - preLM0[eID]) + (dlogMR[0]-dlogMR[eID]) + \
             (1.0/params["gamma"])*np.log10(1.0+((ffSR)/fcE)**(nfalls[eID]*params["gamma"])) - \
             (1.0/params["gamma"])*np.log10(1.0+((ffSR)/fcT)**(nfalls[0]*params["gamma"]))

        # observed data likelihood
        logSR = pm.StudentT("logSR",nu=tdof,mu=theoSR,sigma=tsig,observed=l10SR)

    print("Done, elapsed seconds:",time.time()-t00)
    print()

    #### Sample ####

    print("Sampling, please wait...")
    t00 = time.time()
    with bayspec: # sampling parameters are specified above
        idata = pm.sample(params["nsamp"],tune=params["ntune"],cores=params["ncore"],
            return_inferencedata=True) # inference data object

    print("Done, elapsed minutes:",(time.time()-t00)/60.0)
    print()

    ### Summarize results and saving to file
    print("Saving file and summary.")
    with bayspec:

        # arviz summary file
        azsum = az.summary(idata)
        print(azsum)

        # save sampling file (netcdf in this example)
        az.to_netcdf(idata,params["out_file"])

    ### DONE!
    print("\n\nDONE WITH CALCULATIONS")