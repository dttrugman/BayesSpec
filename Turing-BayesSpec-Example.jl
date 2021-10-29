### Example Turing Models for Bayesian Source Spectral Inference ####
### Daniel Trugman (dtrugman@jsg.utexas.edu), 2021
###  See: https://github.com/TuringLang/Turing.jl 
###    for more details on Turing usage and syntax. 


#### Initial imports ####
using Turing
using MCMCChains, Serialization
using CSVFiles, DataFrames
using Random

#### Define Turing Models ####
#  Note: some of the priors have hard-coded parameters, but these can be adjusted or
#  changed to input parameters based on user needs. Truncation of model distributions 
#  may improve stability and sampling efficiency while enforcing physical constraints,
#  but is optional in many problems.

#### Model for Bayesian spectral ratio (Laplace Likelihood)
#   logSR: data, log10 of spectral ratio
#      ff: frequency at each data point [Hz]
#     azi: azimuthal angle for each data point (mainshock)
#     toa: takeoff angle for each data point (mainshock) ***from vertical upward***
#     eid: EGF index for each data point
#  preLM0: log10 M0 prior for each earthquake (negf+1)
#  preLFC: log10 fc prior for each earthquake (negf+1)
#  strike00: strike prior
#  stdSTK: standard deviation for prior
# stdLM0: standard deviation in moment
# stdLFC: standard deviation in corner frequency
# stdNFALL: standard deviation in falloff rate



#### Brune Model, Unilateral
@model function brunespecUL(
    logSR::Vector{Float64},ff::Vector{Float64},azi::Vector{Float64},toa::Vector{Float64},
    eid::Vector{Int64},preLM0::Vector{Float64},preLFC::Vector{Float64},strike00::Float64, 
    stdSTK::Float64,stdLM0::Float64,stdLFC::Float64,stdNFALL::Float64)

    # get sizes
    nev = length(preFC)::Int64
    
    # prior for FCs
    dlogFC ~ filldist(truncated(Normal{Float64}(0.0,stdLFC),-4.0*stdLFC,+4.0*stdLFC),nev)

    # prior for dMR
    dlogMR ~ MvNormal(zeros(Float64,nev),stdLM0) # note! use sdev and not variance if scalar
    
    # prior for nfall
    nfalls ~ filldist(truncated(Normal{Float64}(2.0,stdNFALL),1.5,3.0),nev)
    
    # prior for rupture strike angle
    strike ~ truncated(Normal{Float64}(strike00,stdSTK),
        strike00-4.0*stdSTK,strike00+4.0*stdSTK)

    # rupture dip angle from vertical
    rdip ~ truncated(Normal{Float64}(90.0,10.0),20.0,160.0)     

    # priors for directivity
    chi ~ Uniform{Float64}(-0.99,0.99) # controls strength of directivity

    # prior for scale (uniformative)
    scale ~ truncated(Normal{Float64}(0.0, 0.5),0.005,2.0)

    # compute actual FCs for each event
    allFCs = 10.0.^(preLFC .+ dlogFC)

    # likelihood for logSR data, sharpness gamma = 1.0 # 
    for ii in eachindex(logSR)

       # compute target event fC
       fC = allFCs[nev] / ( # denominator accounts for unilateral directivity
            1.0 - chi*(cosd(azi[ii]-strike)*sind(toa[ii])*sind(rdip)+cosd(toa[ii])*cosd(rdip) ) )
       
       # observe SR
       logSR[ii] ~ Laplace(
         preLM0[nev]+dlogMR[nev]-preLM0[eid[ii]]-dlogMR[eid[ii]] +
        log10(1.0 + (ff[ii]/allFCs[eid[ii]])^(nfalls[eid[ii]])) -
        log10(1.0 + (ff[ii]/fC)^(nfalls[nev]) ),
        scale)
    end
    
end

#### Brune Model, Bilateral
@model function brunespecBL(
    logSR::Vector{Float64},ff::Vector{Float64},azi::Vector{Float64},toa::Vector{Float64},
    eid::Vector{Int64},preLM0::Vector{Float64},preLFC::Vector{Float64},strike00::Float64, 
    stdSTK::Float64,stdLM0::Float64,stdLFC::Float64,stdNFALL::Float64)

    # get sizes
    nev = length(preFC)::Int64
    
    # prior for FCs
    dlogFC ~ filldist(truncated(Normal{Float64}(0.0,stdLFC),-4.0*stdLFC,+4.0*stdLFC),nev)

    # prior for dMR
    dlogMR ~ MvNormal(zeros(Float64,nev),stdLM0) # note! use sdev and not variance if scalar
    
    # prior for nfall
    nfalls ~ filldist(truncated(Normal{Float64}(2.0,stdNFALL),1.5,3.0),nev)
    
    # prior for rupture strike angle
    strike ~ truncated(Normal{Float64}(strike00,stdSTK),
        strike00-4.0*stdSTK,strike00+4.0*stdSTK)

    # rupture dip angle from vertical
    rdip ~ truncated(Normal{Float64}(90.0,10.0),20.0,160.0)     

    # priors for directivity
    chi ~ Uniform{Float64}(0.0,0.95) # controls strength of directivity

    # prior for scale (uniformative)
    scale ~ truncated(Normal{Float64}(0.0, 0.5),0.005,2.0)

    # compute actual FCs for each event
    allFCs = 10.0.^(preLFC .+ dlogFC)

    # likelihood for logSR data, sharpness gamma = 1.0 # 
    for ii in eachindex(logSR)

       # compute target event fC
       fC = allFCs[nev] / ( # denominator accounts for bilateral directivity
            1.0 - (chi^2)*(cosd(azi[ii]-strike)*sind(toa[ii])*sind(rdip)+cosd(toa[ii])*cosd(rdip))^2  )
       
       # observe SR
       logSR[ii] ~ Laplace(
         preLM0[nev]+dlogMR[nev]-preLM0[eid[ii]]-dlogMR[eid[ii]] +
        log10(1.0 + (ff[ii]/allFCs[eid[ii]])^(nfalls[eid[ii]])) -
        log10(1.0 + (ff[ii]/fC)^(nfalls[nev]) ),
        scale)
    end
    
end

