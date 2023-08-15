# load libraries
library(sl3)
library(SuperLearner)
library(fastDummies)
library(tictoc)
library(tidyr)
library(dplyr)

library(origami)
library(pROC)
library(caret)
library(cvAUC)
library(dplyr)

# functions for prediction summary
get_pred_summary <- function(y_test, preds){
  res_roc <- roc(as.vector(y_test), as.vector(preds))
  best_threshold = coords(res_roc, "best", input="threshold", best.method="youden")$threshold
  best_preds_class = factor(ifelse(as.vector(preds) > best_threshold, 1, 0))
  cm = confusionMatrix(data=best_preds_class, reference=as.factor(y_test))
  return(list('cm' = cm,
              'res_roc' = res_roc))
}

# calculate cross validated AUC
cv_AUC_res <- function(y_holdout, predsY) {
  Y = y_holdout
  set.seed(1)
  V = 10
  folds = make_folds(y_holdout, 
                     fold_fun = folds_vfold,
                     strata_ids = y_holdout,
                     V = 10)
  n=length(predsY)
  fold=rep(NA,n)
  for (i in 1:10) {
    fold[folds[[i]]$validation_set] = i
  }
  
  ### Get the CI for the x-validated AUC
  ciout=ci.cvAUC(predsY, Y, folds = fold)
  txt=paste(round(ciout$cvAUC,3)," (",round(ciout$ci[1],3),"-",round(ciout$ci[2],3),")",sep="")
  return(txt)}


# data
tested_patients = read.csv(file = "Data/sinolave_03112021_meds_for_estimation_rapid.csv")
dim(tested_patients) # 1423720
patients = tested_patients[tested_patients$covid19_lab == 1,] # 1423720

# for different phases; we subset the data based on the Oleada variable
# for example: phase 1 
# patients = patients[patients$Oleada == 1, ]

# we used the full data to do the prediction (training/ testing)
# for each phase as well
# variable importance for all time is done on n = 3e5
# but full data used for each phase variable importance

# set.seed(12345)
# row_patients = sample(x = 1:nrow(patients), size = 3e5) # small version
# patients = patients[row_patients, ]
# dim(patients)

# variable specifications
covariates = names(patients)[!names(patients) %in%
                               c("Nationality", "covid19_lab", "death", 
                                 'RangoEdad', 
                                 "Oleada",
                                 "Hypertension_or_meds","Diabetes_or_meds")]

outcome = 'death'
set.seed(12834897)
foldid <-
  sample(1:5, size = length(patients %>% pull(outcome)), 
         replace = TRUE)

# training on 1-4; testing on 5 (80-20% split)
train = patients[foldid != 5, c(covariates, outcome)]
testing = patients[foldid == 5, c(covariates, outcome)]

# generate the sl3 task
outcome <- outcome
covariates <- covariates 
data <- train
n <- nrow(data)


############## super learner ################

# learners used for SL
suggested_learners <- list(
  mean = make_learner(Lrnr_mean),
  glm = make_learner(Lrnr_glm_fast),
  xgb = make_learner(Lrnr_xgboost, nrounds = 20, maxdepth = 6),
  ranger_small = make_learner(Lrnr_ranger, num.trees=500),
  lasso_fast = make_learner(Lrnr_glmnet, nfold=3),
  ridge_fast = make_learner(Lrnr_glmnet, nfold=3, alpha=0),
  enet_fast = make_learner(Lrnr_glmnet, nfold=3, alpha=0.5),
  earth = make_learner(Lrnr_earth),
  bayesglm = make_learner(Lrnr_bayesglm),
  gam = Lrnr_pkg_SuperLearner$new(SL_wrapper = "SL.gam"),
  bart = Lrnr_dbarts$new(ndpost = 1000, verbose = FALSE),
  xgb_SL = make_learner(Lrnr_xgboost, nrounds = 1000, 
                        max_depth = 4, eta = 0.1)
)

folds <- make_folds(n, V=3, strata_ids = train$death)
task <- make_sl3_Task(data, covariates, outcome, folds=folds)
options(sl3.verbose = TRUE)
sl <- make_learner(Lrnr_sl, suggested_learners) #suggested learners

# fit the super learner
tictoc::tic()
sl_fit <- sl$train(task)
tictoc::toc() 

# generate test task
test_task <- make_sl3_Task(testing, covariates, outcome)

# prediction results
sl_train_preds = sl_fit$predict(task)
pred_sl_summary_train = get_pred_summary(y_test=data %>% pull(outcome), 
                                         preds = sl_train_preds)

pred_sl_summary_train$res_roc

sl_test_preds = sl_fit$predict(test_task)
pred_sl_summary = get_pred_summary(y_test=testing %>% pull(outcome), 
                                   preds = sl_test_preds)


pred_sl_summary$cm
pred_sl_summary$res_roc 
plot(pred_sl_summary$res_roc)

# training cross validated AUC
cv_AUC_res(data %>% pull(outcome), sl_train_preds)
# testing
cv_AUC_res(testing %>% pull(outcome), sl_test_preds)


# variable importance
varimp <- importance(sl_fit,type = "permute",
                     eval_fun = loss_loglik_binomial)

# # plot variable importance
# nvar = 10
# varimp %>%
#   importance_plot(nvar = nvar) + theme_minimal() +
#   theme(
#     axis.text.x = element_text(size = 10),
#     axis.text.y = element_text(size = 10),
#     axis.title.x = element_text(size = 12),
#     axis.title.y = element_text(size = 12),
#     plot.title = element_text(size = 14)
#   ) +
#   theme(panel.background = element_rect(fill = "white"),
#         axis.text.x = element_text(color = "black"),
#         axis.text.y = element_text(color = "black"),
#         panel.border = element_rect(fill = NA, color = "black"),
#         plot.background = element_blank(),
#         legend.background = element_blank(),
#         legend.key = element_blank()) +
#   scale_x_discrete(limits=rev)


############## age only log reg ################

age_only_glm = glm("death ~ Age_in_Years", family=binomial, data=data)
age_only_train = (predict(age_only_glm, data,  type="response"))
pred_age_only_train = get_pred_summary(y_test=data %>% pull(outcome), 
                                       preds = age_only_train)

age_only_test = (predict(age_only_glm, testing,  type="response"))

pred_age_only_test = get_pred_summary(y_test=testing %>% pull(outcome), 
                                      preds = age_only_test)

plot(pred_age_only_test$res_roc)


# training
cv_AUC_res(data %>% pull(outcome), sl_train_preds)
# testing
cv_AUC_res(testing %>% pull(outcome), sl_test_preds)

# training
cv_AUC_res(data %>% pull(outcome), age_only_train)
# testing
cv_AUC_res(testing %>% pull(outcome), age_only_test)
