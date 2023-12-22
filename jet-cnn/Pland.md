Plan

1. Modifcations to be done to KerasCNN_bootstrap
 - Reading of our HepMass data, st it shares the structure of the code
 - Replace their model with ours. 
 - The name of the sets to be saved.  

- This will give us a a result the bootstrap sets that will be used for the construction of 
  the pdf estimation

2. Modifications to be done in bootstrap_analysis
  - Changes the model in verbose mode so that how much it has advanced is shown.
  - The results are important to be observed, they will talk us about the variation in the metrics.
  - The name of the densitites to be saved.

- This will create the average pdf from the bootstrap sets calculated before, that we will finally use in llr
  calculation. 

3. Modifications to be done in the jet_llr code
  - The cross sections of the processes, luminosity and efficiency.

4. Make one that is with the signal in general.  

---------------------------
4. Notes of things done differently

 - Tha same datasets use to bootstrap in the constuction of the pdf are the same used to evaluate the llr.
 - Instead of sampling all sets and then construct an estimation of the pdf, each set is used to estimate the pdf
   and latter a mean value is calculated (note that bins are not chosen dinamically). 
 - Possion distributions are used for the sampling 
 - The number of bins is set to 1000
 - Gaussians are not strictly  gaussians, this is of special importance when they use them to estimate the deviation 
   corresponding with the cut point. 
 - They select the data for which it was possible to obtain a good training algorithm, this should be supposedly justify in the
   freedom to choose your model that adjust better the data. 
