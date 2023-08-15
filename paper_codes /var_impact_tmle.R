library(SuperLearner)
library(fastDummies)
library(tictoc)
library(tidyr)
library(tmle)

# data
tested_patients = read.csv(file = "Data/sinolave_03112021_meds_for_estimation_rapid.csv")
patients = tested_patients[tested_patients$covid19_lab == 1,] # this is the example of overall; changes for each phase see prediction_sl.R for an example

covariates = names(patients)[!names(patients) %in%
                               c("Oleada", "Nationality", "death",
                                 "RangoEdad", "covid19_lab", "year_month_clinic_admission")]
covariates
A_covs = covariates[c(5:17, 19, 20, 21, 22)]
A_covs
bg_covs = covariates[!covariates %in% A_covs]

# super learner specificatiions
Q_lib <- c("SL.mean", "SL.glmnet", 
           "SL.ranger", 
           "SL.xgboost",
           "SL.gam", 
           "SL.bayesglm", 
           "SL.earth")
g_lib <- c("SL.mean", "SL.glmnet", "SL.xgboost")

## only one example for hypertension
set.seed(813222)
hyp_res = tmle(patients$death, patients$Preexisting_Hypertension, # changes for each pre-existing condition
               W=tested_patients[,bg_covs], V=3, 
               Q.SL.library = Q_lib, g.SL.library = g_lib, family="binomial"
               )

apply(hyp_res$Qstar,2,mean)

