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
trans.cs <- preProcess(x=imdb %>% select(-imdb_score), method=c("center", "scale"))
imdb.cs <- predict(trans.cs, newdata=imdb)
trans01 <- preProcess(x=imdb %>% select(-imdb_score), method="range",
                      rangeBounds=c(0,1))
imdb.01 <- predict(trans01, newdata=imdb)

####################################
## Fit some models for prediction ##
####################################

## Split the test and training data
imdb.train <- imdb %>% filter(!is.na(imdb_score))
imdb.test <- imdb %>% filter(is.na(imdb_score))

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
# Training the model
xgboost_model <- xgboost(data=as.matrix(imdb.train %>% select(-Set, -movie_title)),
                         label=as.numeric(imdb.train$imdb_score),
                         max_depth=3,
                         nrounds = 10,
                         prediction=TRUE)

## Fit a Convolutional NN 