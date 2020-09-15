rm(list=ls())

##
## This is all my code to analyze the IMDB database
##

## Libraries I Need
library(tidyverse)
library(caret)
library(earth)
library(DataExplorer)
library(randomForest)
library(xgboost)

## Read in the data
imdb <- read_csv("CleanedIMDBData.csv")

##
## Some Variable Transformations which may help
## models predict better
##

## Dummary (Indicator) Variables
IVTrans <- dummyVars(imdb_score~.-movie_title-Set, data=imdb)
imdb.iv <- predict(IVTrans, newdata=imdb)  %>% as.data.frame() %>%
  bind_cols(., imdb %>% select(movie_title, Set, imdb_score))

## Principal Components Transformation
pcTrans <- preProcess(x=imdb %>% select(-imdb_score), method="pca")
imdb.pca <- predict(pcTrans, newdata=imdb)
plot_correlation(imdb.pca, type="continuous", cor_args=list(use="pairwise.complete.obs"))

## Center and Scaling
trans.cs <- preProcess(x=imdb.iv %>% select(-imdb_score), method=c("center", "scale"))
imdb.cs <- predict(trans.cs, newdata=imdb.iv)
trans01 <- preProcess(x=imdb.iv %>% select(-imdb_score), method="range",
                      rangeBounds=c(0,1))
imdb.01 <- predict(trans01, newdata=imdb.iv)

####################################
## Fit some models for prediction ##
####################################

## Split the test and training data
imdb.train <- imdb.cs %>% filter(!is.na(imdb_score))
imdb.test <- imdb.cs %>% filter(is.na(imdb_score))

## Fit a linear regression model
linreg <- train(form=imdb_score~.,
                data=(imdb.train %>% select(-Set, -movie_title)),
                method="lm",
                trControl=trainControl(method="repeatedcv",
                                       number=10, #Number of pieces of your data
                                       repeats=3) #repeats=1 = "cv"
)
linreg$results
linreg.preds <- data.frame(Id=imdb.test$movie_title, Predicted=predict(linreg, newdata=imdb.test))
write_csv(x=linreg.preds, path="./LinearRegressionPredictions.csv")

## Fit an Elastic Net model
elnet <- train(form=imdb_score~.,
               data=(imdb.train %>% select(-Set, -movie_title)),
               method="glmnet",
               trControl=trainControl(method="repeatedcv",
                                      number=10, #Number of pieces of your data
                                      repeats=3) #repeats=1 = "cv"
)
plot(elnet)

## Tune the elastic net
elnet.grid <- expand.grid(alpha=seq(0.4, 0.8, length=10),
                         lambda=seq(0, 0.02, length=10))
elnet <- train(form=imdb_score~.,
               data=(imdb.train %>% select(-Set, -movie_title)),
               method="glmnet",
               trControl=trainControl(method="repeatedcv",
                                      number=10, #Number of pieces of your data
                                      repeats=3), #repeats=1 = "cv"
               tuneGrid=elnet.grid
)
plot(elnet)
elnet.preds <- data.frame(Id=imdb.test$movie_title, Predicted=predict(elnet, newdata=imdb.test))
write_csv(x=elnet.preds, path="./ElasticNetPredictions.csv")
elnet.preds[which(elnet.preds$Predicted == max(elnet.preds$Predicted)),]

## Fit a Random Forest model
# Initial try
rf_test <- randomForest(formula=imdb_score~.,
                        data=imdb.train,
                        mtry=rf_tuned)
plot(rf_test)
rf_test

# Tuning tests
rf_tuned <- tuneRF(x=(imdb.train %>% select(-Set, -movie_title)),
                   y=imdb.train$imdb_score,
                   ntreeTry=500,
                   mtryStart=7,
                   stepFactor=1.5,
                   improve=0.01,
                   trace=TRUE)
plot(rf_tuned, type="l")

# Make prediction
rf.preds <- data.frame(Id=imdb.test$movie_title, Predicted=predict(rf_test, newdata=imdb.test))
write_csv(x=rf.preds, path="./RandomForestPredictions.csv")
hist(rf.preds$Predicted)



## Fit a MARS model
mars_tunegrid <- expand.grid(degree=1:3, 
                             nprune=seq(2,100, length.out = 10) %>% floor())
tuned_mars <- train(form=imdb_score~.,
                    data=(imdb.train %>% select(-Set, -movie_title)),
                    method = "earth",
                    metric = "RMSE",
                    trControl = trainControl(method="cv", number=10),
                    tuneGrid = mars_tunegrid)
# Best model
tuned_mars$bestTune
plot(tuned_mars)

# Fit the model
mars.preds <- data.frame(Id=imdb.test$movie_title, Predicted=predict(tuned_mars, newdata=imdb.test))
write_csv(x=mars.preds, path="./MARSPredictions.csv")

## Fit a Boosted Gradient model
## Baseline model
grid_default <- expand.grid(
  nrounds = 100,
  max_depth = 6,
  eta = 0.3,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

train_control <- caret::trainControl(
  method = "none",
  verboseIter = FALSE, # no training log
  allowParallel = TRUE # FALSE for reproducible results 
)

xgb_base <- train(form=imdb_score~.,
                  data=(imdb.train %>% select(-Set, -movie_title)),
                  method="xgbTree",
                  trControl=train_control,
                  tuneGrid=grid_default,
                  verbose=TRUE
)

## Next, start tuning hyperparameters
nrounds <- 1000
tune_grid <- expand.grid(
  nrounds = seq(from = 200, to = nrounds, by = 50),
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

tune_control <- caret::trainControl(
  method = "cv", # cross-validation
  
  number = 3, # with n folds 
  #index = createFolds(tr_treated$Id_clean), # fix the folds
  verboseIter = FALSE, # no training log
  allowParallel = TRUE # FALSE for reproducible results 
)

xgb_tune <-train(form=imdb_score~.,
                 data=(imdb.train %>% select(-Set, -movie_title)),
                 method="xgbTree",
                 trControl=tune_control,
                 tuneGrid=tune_grid,
                 verbose=TRUE
)

# helper function for the plots
tuneplot <- function(x, probs = .90) {
  ggplot(x) +
    coord_cartesian(ylim = c(quantile(x$results$RMSE, probs = probs), min(x$results$RMSE))) +
    theme_bw()
}

tuneplot(xgb_tune)
xgb_tune$bestTune

## Next round of tuning
tune_grid2 <- expand.grid(nrounds = seq(from = 50, to = nrounds, by = 50),
                          eta = xgb_tune$bestTune$eta,
                          max_depth = ifelse(xgb_tune$bestTune$max_depth == 2,
                          c(xgb_tune$bestTune$max_depth:4),
                          xgb_tune$bestTune$max_depth - 1:xgb_tune$bestTune$max_depth + 1),
                          gamma = 0,
                          colsample_bytree = 1,
                          min_child_weight = c(1, 2, 3),
                          subsample = 1
)

xgb_tune2 <- caret::train(
  form=imdb_score~.,
  data=(imdb.train %>% select(-Set, -movie_title)),
  method="xgbTree",
  trControl=tune_control,
  tuneGrid=tune_grid2,
  verbose=TRUE
)

tuneplot(xgb_tune2)
xgb_tune2$bestTune
min(xgb_tune$results$RMSE)
min(xgb_tune2$results$RMSE)

## Next tuning round
tune_grid3 <- expand.grid(
  nrounds = seq(from = 50, to = nrounds, by = 50),
  eta = xgb_tune$bestTune$eta,
  max_depth = xgb_tune2$bestTune$max_depth,
  gamma = 0,
  colsample_bytree = c(0.4, 0.6, 0.8, 1.0),
  min_child_weight = xgb_tune2$bestTune$min_child_weight,
  subsample = c(0.5, 0.75, 1.0)
)

xgb_tune3 <- caret::train(
  form=imdb_score~.,
  data=(imdb.train %>% select(-Set, -movie_title)),
  method="xgbTree",
  trControl=tune_control,
  tuneGrid=tune_grid3,
  verbose=TRUE
)

tuneplot(xgb_tune3, probs = .95)
xgb_tune3$bestTune
min(xgb_tune$results$RMSE)
min(xgb_tune2$results$RMSE)
min(xgb_tune3$results$RMSE)

## Tuning the Gamma
tune_grid4 <- expand.grid(
  nrounds = seq(from = 50, to = nrounds, by = 50),
  eta = xgb_tune$bestTune$eta,
  max_depth = xgb_tune2$bestTune$max_depth,
  gamma = c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0),
  colsample_bytree = xgb_tune3$bestTune$colsample_bytree,
  min_child_weight = xgb_tune2$bestTune$min_child_weight,
  subsample = xgb_tune3$bestTune$subsample
)

xgb_tune4 <- caret::train(
  form=imdb_score~.,
  data=(imdb.train %>% select(-Set, -movie_title)),
  method="xgbTree",
  trControl=tune_control,
  tuneGrid=tune_grid4,
  verbose=TRUE
)

tuneplot(xgb_tune4)
xgb_tune4$bestTune
min(xgb_tune$results$RMSE)
min(xgb_tune2$results$RMSE)
min(xgb_tune3$results$RMSE)
min(xgb_tune4$results$RMSE)

## Reduce learning rate
tune_grid5 <- expand.grid(
  nrounds = seq(from = 100, to = 10000, by = 100),
  eta = c(0.01, 0.015, 0.025, 0.05, 0.1),
  max_depth = xgb_tune2$bestTune$max_depth,
  gamma = xgb_tune4$bestTune$gamma,
  colsample_bytree = xgb_tune3$bestTune$colsample_bytree,
  min_child_weight = xgb_tune2$bestTune$min_child_weight,
  subsample = xgb_tune3$bestTune$subsample
)

xgb_tune5 <- caret::train(
  form=imdb_score~.,
  data=(imdb.train %>% select(-Set, -movie_title)),
  method="xgbTree",
  trControl=tune_control,
  tuneGrid=tune_grid5,
  verbose=TRUE
)

tuneplot(xgb_tune5)
xgb_tune5$bestTune
min(xgb_tune$results$RMSE)
min(xgb_tune2$results$RMSE)
min(xgb_tune3$results$RMSE)
min(xgb_tune4$results$RMSE)
min(xgb_tune5$results$RMSE)

## Fit the model and predict
final_grid <- expand.grid(
  nrounds = xgb_tune5$bestTune$nrounds,
  eta = xgb_tune5$bestTune$eta,
  max_depth = xgb_tune5$bestTune$max_depth,
  gamma = xgb_tune5$bestTune$gamma,
  colsample_bytree = xgb_tune5$bestTune$colsample_bytree,
  min_child_weight = xgb_tune5$bestTune$min_child_weight,
  subsample = xgb_tune5$bestTune$subsample
)

xgb_model <- caret::train(
  form=imdb_score~.,
  data=(imdb.train %>% select(-Set, -movie_title)),
  method="xgbTree",
  trControl=tune_control,
  tuneGrid=final_grid,
  verbose=TRUE
)

xgb.preds <- data.frame(Id=imdb.test$movie_title, Predicted=predict(xgb_model, newdata=imdb.test))
head(xgb.preds, 25)
write_csv(x=xgb.preds, path="./XGBPredictions.csv")
