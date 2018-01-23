# Reading required libraries
library(xgboost)
library(data.table)
library(dummies)

# Reading data
# To read -1 in Label column as NA, na.strings is used as shown below
df_train <- fread("Train.csv", na.strings = c("NA", "-1"))
df_test <- fread("Test.csv", na.strings = c("NA", "-1"))

# To check the missing values in each column
# sapply(df_train, function(x) sum(is.na(x)))
# sapply(df_test, function(x) sum(is.na(x)))

# Removing records with NA as the label
df_train <- df_train[!is.na(df_train$cancel), ]

# cleaning raw features
# Feature selection was performed based on cross validation
df_train <- df_train[, ":="(id = NULL,
                            tenure = NULL,
                            claim.ind = as.numeric(claim.ind),
                            n.adults  = as.numeric(n.adults),
                            n.children = as.numeric(n.children),
                            ni.gender = NULL,
                            ni.marital.status = as.numeric(ni.marital.status),
                            premium = NULL,
                            sales.channel = as.numeric(as.factor(sales.channel)),
                            coverage.type = NULL,
                            dwelling.type = NULL,
                            len.at.res = as.numeric(len.at.res),
                            credit = as.numeric(as.factor(credit)),
                            house.color = as.numeric(as.factor(house.color)),
                            ni.age = NULL,
                            year = NULL,
                            zip.code = as.numeric(zip.code))]

df_test <- df_test[, ":="(tenure = NULL,
                          claim.ind = as.numeric(claim.ind),
                          n.adults  = as.numeric(n.adults),
                          n.children = as.numeric(n.children),
                          ni.gender = NULL,
                          ni.marital.status = as.numeric(ni.marital.status),
                          premium = NULL,
                          sales.channel = as.numeric(as.factor(sales.channel)),
                          coverage.type = NULL,
                          dwelling.type = NULL,
                          len.at.res = as.numeric(len.at.res),
                          credit = as.numeric(as.factor(credit)),
                          house.color = as.numeric(as.factor(house.color)),
                          ni.age = NULL,
                          year = NULL,
                          zip.code = as.numeric(zip.code))]

# Saving the target variable and test ids
target <- as.numeric(df_train$cancel)
df_test_ids <- df_test$id

# Removing the target variable "cancel" from train data
df_train <- df_train[, ":="(cancel = NULL)]
# Removing the id variable "cancel" from test data, id variable was already removed in train
df_test <- df_test[, ":="(id = NULL)]

# Building xgboost
# Initializing hyper parameters
params <- list(objective="binary:logistic", eta=0.05, max_depth=2, 
               subsample=0.75, colsample_bytree=0.8, 
               min_child_weight=1, eval_metric="auc")

# Seed is set to reproduce the results
# Note that seed has to be run each time the cv is performed to reproduce the same results
seed <- 235
set.seed(seed)

# Performing 10 fold cross validation
model_xgb_cv <- xgb.cv(data=as.matrix(df_train), label=as.matrix(target),
                       nfold=10, params = params, nrounds = 1000,
                       early_stopping_rounds = 20, print_every_n = 30)

# Taking the best number of rounds obtained from cross validation (nrounds)
model_xgb <- xgboost(data=as.matrix(df_train), label=as.matrix(target),
                     nrounds = model_xgb_cv$best_iteration, params = params, print_every_n = 30)

# predicting on test data using xgboost model built
xgb_pred <- predict(model_xgb, as.matrix(df_test))

# Writing xgboost predicitions
# xgb_submit <- data.frame("id"=df_test_ids, "cancel"=xgb_pred)
# write.csv(xgb_submit, "xgb_submit.csv", row.names=F)

# Saving back the target variable to build logistic regression
df_train$cancel <- target

# Custom function for mode
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# Filling missing values from train data using mode and mean appropriately
df_train[is.na(df_train$claim.ind), "claim.ind"] <- getmode(df_train$claim.ind)
df_train[is.na(df_train$n.adults), "n.adults"] <- mean(df_train$n.adults, na.rm = TRUE)
df_train[is.na(df_train$n.children), "n.children"] <- mean(df_train$n.children, na.rm = TRUE)
df_train[is.na(df_train$ni.marital.status), "ni.marital.status"] <- getmode(df_train$ni.marital.status)
df_train[is.na(df_train$sales.channel), "sales.channel"] <- getmode(df_train$sales.channel)
df_train[is.na(df_train$len.at.res), "len.at.res"] <- mean(df_train$len.at.res, na.rm = TRUE)
df_train[is.na(df_train$credit), "credit"] <- getmode(df_train$credit)
df_train[is.na(df_train$house.color), "house.color"] <- getmode(df_train$house.color)
df_train[is.na(df_train$zip.code), "zip.code"] <- getmode(df_train$zip.code)

df_test[is.na(df_test$claim.ind), "claim.ind"] <- getmode(df_train$claim.ind)
df_test[is.na(df_test$n.adults), "n.adults"] <- mean(df_train$n.adults, na.rm = TRUE)
df_test[is.na(df_test$n.children), "n.children"] <- mean(df_train$n.children, na.rm = TRUE)
df_test[is.na(df_test$ni.marital.status), "ni.marital.status"] <- getmode(df_train$ni.marital.status)
df_test[is.na(df_test$sales.channel), "sales.channel"] <- getmode(df_train$sales.channel)
df_test[is.na(df_test$len.at.res), "len.at.res"] <- mean(df_train$len.at.res, na.rm = TRUE)
df_test[is.na(df_test$credit), "credit"] <- getmode(df_train$credit)
df_test[is.na(df_test$house.color), "house.color"] <- getmode(df_train$house.color)
df_test[is.na(df_test$zip.code), "zip.code"] <- getmode(df_train$zip.code)

# Converting categorical columns with more than 2 labels to dummies
df_train <- dummy.data.frame(df_train, names=c("sales.channel", "credit", "house.color"), sep="_")
df_test <- dummy.data.frame(df_test, names=c("sales.channel", "credit", "house.color"), sep="_")

# Building Logistic regression model
model_glm <- glm(cancel ~ ., data = df_train)

# Predicting on test data using glm model built
glm_pred <- predict(model_glm, df_test)

# Writing Logistic regression predictions
# glm_submit <- data.frame("id"=df_test_ids, "cancel"=glm_pred)
# write.csv(glm_submit, "glm_submit.csv", row.names=F)

# Simple average ensembling of both the models built
pred <- xgb_pred*0.5 + glm_pred*0.5

# Writing ensemble predictions
ens_submit <- data.frame("id"=df_test_ids, "cancel"=pred)
write.csv(ens_submit, "ens_submit.csv", row.names=F)