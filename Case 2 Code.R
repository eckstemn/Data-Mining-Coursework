# libraries
library(MASS)
library(rpart)
library(rpart.plot)
library(dplyr)
library(ROCR)
library(ipred)
library(randomForest)
library(gbm)
library(mgcv)
library(nnet)
library(neuralnet)


set.seed(13023260)

# Load Boston data
data(Boston)
# create training and testing set
sampleIndex <- sample(nrow(Boston),nrow(Boston)*0.75)
BostonTrain <- Boston[sampleIndex,]
BostonTest <- Boston[-sampleIndex,]


# Linear Model
full <- lm(medv~., BostonTrain)
best <- step(full)
summary(best)
pred.lm <- predict(best)
MSE.lm <- mean((pred.lm-BostonTrain$medv)^2)
MSE.lm
os.pred.lm <- predict(best, BostonTest)
MSPE.lm <- mean((os.pred.lm-BostonTest$medv)^2)
MSPE.lm

# Tree Models
boston.tree<- rpart(best, data = BostonTrain)
plot(boston.tree, uniform=TRUE, 
     main="Regression Tree")
text(boston.tree, use.n=TRUE, all=TRUE, cex=.7)
prp(boston.tree,digits = 4, extra = 1)
boston.tree.train.pred <- predict(boston.tree)
boston.tree.pred<- predict(boston.tree, newdata = BostonTest)
MSE.tree <- mean((BostonTrain$medv - boston.tree.train.pred)^2)
MSE.tree
MSPE.tree <- mean((BostonTest$medv-boston.tree.pred)^2)
MSPE.tree

## Bagging
boston.bag<- bagging(medv~crim+zn+chas+nox+rm+dis+rad+tax+ptratio+black+lstat, 
                     data = BostonTrain, nbagg=100)
boston.bag
boston.bag.pred.train <- predict(boston.bag, newdata = BostonTrain)
in_samp_err <- mean((BostonTrain$medv-boston.bag.pred.train)^2)
in_samp_err
boston.bag.pred.test <- predict(boston.bag, newdata = BostonTest)
out_samp_err <- mean((BostonTest$medv-boston.bag.pred.test)^2)
out_samp_err

## OOB error
boston.bag.oob<- bagging(medv~crim+zn+chas+nox+rm+dis+rad+tax+ptratio+black+lstat,
                         data = BostonTrain, coob=T, nbagg=100)
boston.bag.oob

## Random Forests
boston.rf<- randomForest(medv~., 
                         data = BostonTrain, importance=TRUE, mtry = sqrt(13))
boston.rf
boston.rf$importance

boston.rf.pred.train<- predict(boston.rf, BostonTrain)
RF.MSE <- mean((BostonTrain$medv-boston.rf.pred.train)^2)
RF.MSE
boston.rf.pred.test <- predict(boston.rf, BostonTest)
RF.MSPE <- mean((BostonTest$medv-boston.rf.pred.test)^2)
RF.MSPE

# RF OOB Error
oob.err<- rep(0, 13)
test.err<- rep(0, 13)
for(i in 1:13){
  fit<- randomForest(medv~.,
                     data = BostonTest, mtry=i)
  oob.err[i]<- fit$mse[500]
  test.err[i]<- mean((BostonTest$medv-predict(fit, BostonTest))^2)
  cat(i, " ")
}
matplot(cbind(test.err, oob.err), pch=15, col = c("red", "blue"), type = "b", ylab = "MSE", xlab = "mtry")
legend("topright", legend = c("test Error", "OOB Error"), pch = 15, col = c("red", "blue"))

## Boosting Tree
boston.boost<- gbm(medv~., data = BostonTrain, distribution = "gaussian", n.trees = 10000, shrinkage = 0.01, interaction.depth = 8)
summary(boston.boost)

# prediction on training
boston.boost.pred.train<- predict(boston.boost, BostonTrain, n.trees = 10000)
ise <- mean((BostonTrain$medv-boston.boost.pred.train)^2)
ise
# prediction on testing
boston.boost.pred.test<- predict(boston.boost, BostonTest, n.trees = 10000)
ose <- mean((BostonTest$medv-boston.boost.pred.test)^2)
ose

# Generalized Additive Models
Boston.gam <- gam(medv ~ s(crim)+zn+s(indus)+chas
                  +s(nox)+s(rm)
                  +s(dis)+rad+s(tax)+s(ptratio)+s(black)+s(lstat), data=BostonTrain)
summary(Boston.gam)

plot(fitted(Boston.gam), residuals(Boston.gam), xlab = 'fitted', ylab = 'residuals', 
     main = 'Residuals by Fitted from GAM')
plot(Boston.gam,pages=1)

Boston.gam.predict.test <- predict(Boston.gam, BostonTest) #Boston.gam built on training data
Boston.gam.mse.test <- mean((Boston.gam.predict.test - BostonTest$medv)^2) ## out-of-sample
Boston.gam.mse.test
Boston.gam.mse.train <- Boston.gam$dev/Boston.gam$df.res
Boston.gam.mse.train

# Neural Networks
set.seed(13023260)
maxs <- apply(Boston, 2, max) 
mins <- apply(Boston, 2, min)

scaled <- as.data.frame(scale(Boston, center = mins, scale = maxs - mins))
index <- sample(1:nrow(Boston),round(0.75*nrow(Boston)))

train_ <- scaled[index,]
test_ <- scaled[-index,]

n <- names(train_)
f <- as.formula(paste("medv ~", paste(n[!n %in% "medv"], collapse = " + ")))
nn <- neuralnet(f,data=train_,hidden=c(5,3),linear.output=T)
plot(nn)

pr.nn <- compute(nn,test_[,1:13])

pr.nn_ <- pr.nn$net.result*(max(Boston$medv)-min(Boston$medv))+min(Boston$medv)
test.r <- (test_$medv)*(max(Boston$medv)-min(Boston$medv))+min(Boston$medv)
# MSE of testing set
MSPE.nn <- sum((test.r - pr.nn_)^2)/nrow(test_)
MSPE.nn

pr.nn <- compute(nn,train_[,1:13])

pr.nn_ <- pr.nn$net.result*(max(Boston$medv)-min(Boston$medv))+min(Boston$medv)
train.r <- (train_$medv)*(max(Boston$medv)-min(Boston$medv))+min(Boston$medv)
# MSE of training set
MSE.nn <- sum((train.r - pr.nn_)^2)/nrow(test_)
MSE.nn




# Bankruptcy data


cost <- function(r, pi){
  weight1 = 35
  weight0 = 1
  c1 = (r==1)&(pi==0) #logical vector - true if actual 1 but predict 0
  c0 = (r==0)&(pi==1) #logical vector - true if actual 0 but predict 1
  return(mean(weight1*c1+weight0*c0))
}

set.seed(13023260)
bankruptcy <- read.csv("/Users/MeganEckstein/Documents/2019SpringSemester/DataMiningII/bankruptcy.csv", sep = ",", header = T)
str(bankruptcy)
bankrupt <- select(bankruptcy, -c(CUSIP, FYEAR))
str(bankrupt)
index <- sample(nrow(bankrupt),nrow(bankrupt)*0.75)
bank.train = bankrupt[index,]
bank.test = bankrupt[-index,]
bank.train$DLRSN <- as.factor(bank.train$DLRSN)
bank.test$DLRSN <- as.factor(bank.test$DLRSN)
bank.glm1<- glm(DLRSN~., family=binomial, data=bank.train)
summary(bank.glm1)
bank.glm1$deviance
AIC(bank.glm1)
BIC(bank.glm1)
hist(predict(bank.glm1))
hist(predict(bank.glm1,type="response"))

nullmodel=glm(DLRSN~1, family = binomial, data = bank.train)
fullmodel=glm(DLRSN~., family = binomial, data = bank.train)
n <- nrow(bank.train)
model_step_f_BIC_DLRSN<- step(nullmodel, scope=list(lower=nullmodel, upper=fullmodel), direction='forward',k = log(n))
summary(model_step_f_BIC_DLRSN)


nullmodel=glm(DLRSN~1, family = binomial, data = bank.train)
fullmodel=glm(DLRSN~., family = binomial, data = bank.train)
model_step_f_AIC_DLRSN<- step(nullmodel, scope=list(lower=nullmodel, upper=fullmodel), direction='forward')
summary(model_step_f_AIC_DLRSN)

pred.glm.bank.train<- (predict(model_step_f_AIC_DLRSN, type="response"))
pred <- ROCR::prediction(pred.glm.bank.train, bank.train$DLRSN)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values"))

# Misclassification Rate Table
table(predict(model_step_f_AIC_DLRSN,type="response") > (1/36))

pcut1<- 1/36
class.glm.train<- (pred.glm.bank.train>pcut1)*1
MR_train<- mean(bank.train$DLRSN!=class.glm.train)
MR_train
FPR_train<- sum(bank.train$DLRSN==0 & class.glm.train==1)/sum(bank.train$DLRSN==0)
FPR_train
FNR_train<- sum(bank.train$DLRSN==1 & class.glm.train==0)/sum(bank.train$DLRSN==1)
FNR_train

asym_MR <- cost(r = bank.train$DLRSN, pi = class.glm.train) 
asym_MR
pcut <- 1/36
pred.glm.bank.test<- predict(model_step_f_AIC_DLRSN, newdata = bank.test, type="response")
class.pred.test<- (pred.glm.bank.test>pcut)*1
table(bank.test$DLRSN, class.pred.test, dnn = c("True", "Predicted"))
cost.test <- cost(r = bank.test$DLRSN, pi = class.pred.test) 
cost.test

pred <- ROCR::prediction(pred.glm.bank.test, bank.test$DLRSN)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values"))

pcut1<- 1/36
class.glm.test<- (pred.glm.bank.test>pcut1)*1
MR_test<- mean(bank.test$DLRSN!=class.glm.test)
MR_test
FPR_test<- sum(bank.test$DLRSN==0 & class.glm.test==1)/sum(bank.test$DLRSN==0)
FPR_test
FNR_test<- sum(bank.test$DLRSN==1 & class.glm.test==0)/sum(bank.test$DLRSN==1)
FNR_test

# classification tree
bank.rpart0 <- rpart(formula = DLRSN~., data =bank.train, method = "class")

pred0<- predict(bank.rpart0, type="class")
table(bank.train$DLRSN, pred0, dnn = c("True", "Pred"))

bank.rpart1 <- rpart(formula =  DLRSN~., data = bank.train, method = "class", parms = list(loss=matrix(c(0,35,1,0), nrow = 2)))
bank.rpart1
prp(bank.rpart1, extra = 1)

pred0<- predict(bank.rpart1, type="class")
table(bank.train$DLRSN, pred0, dnn = c("True", "Pred"))
# in sample prediction
bank.train.pred.tree1<- predict(bank.rpart1, bank.train, type="class")
table(bank.train$DLRSN, bank.train.pred.tree1, dnn=c("Truth","Predicted"))
prob.ct.in<-predict(bank.rpart1,bank.train,type="class")
pred.ct.in<-((as.numeric(prob.ct.in)-1)>=pcut)*1
table(bank.train$DLRSN,pred.gam.in,dnn=c("Observed","Predicted"))
mean(ifelse(bank.train$DLRSN != bank.train.pred.tree1, 1, 0))

# out of sample
bank.train.pred.tree1<- predict(bank.rpart1, bank.test, type="class")
table(bank.test$DLRSN, bank.train.pred.tree1, dnn=c("Truth","Predicted"))
prob.ct.in<-predict(bank.rpart1,bank.test,type="class")
pred.ct.in<-(prob.ct.in>=pcut)*1
table(bank.train$DLRSN,pred.gam.in,dnn=c("Observed","Predicted"))
mean(ifelse(bank.test$DLRSN != bank.test.pred.tree1, 1, 0))

# in sample 
mean(ifelse(bank.train$DLRSN != pred.ct.in, 1, 0))
ct.pred.train <- predict(bank.rpart1, bank.train, type = "class")
pcut1<- 1/36
class.ct.train<- (ct.pred.train>pcut1)*1
MR_train<- mean(bank.train$DLRSN!=class.ct.train)
MR_train




# out of sample
bank.test.pred.tree1<- predict(bank.rpart1, bank.test, type="class")
table(bank.test$DLRSN, bank.test.pred.tree1, dnn=c("Truth","Predicted"))

bank.test$DLRSN <- as.numeric(bank.test$DLRSN)
asym_MR <- cost(r = bank.train$DLRSN, pi = pred.ct.in) 
asym_MR
pcut <- 1/36
ct.pred.test<- predict(bank.rpart1, newdata = bank.test, type="class")
class.pred.test<- ((as.numeric(ct.pred.test)-1)>pcut)*1

table(bank.test$DLRSN, class.pred.test, dnn = c("True", "Predicted"))
cost.test <- cost(r = bank.test$DLRSN, pi = class.pred.test) 
cost.test


# generalized additive models
bank.gam <- gam(formula = DLRSN~s(R1)+s(R2)+s(R3)+R4+s(R6)+R7+s(R8)+s(R9)+s(R10), family=binomial,data=bank.train)
summary(bank.gam)
plot(bank.gam, shade=TRUE,seWithMean=TRUE,scale=0, pages = 1)
pcut.gam <- 1/36
prob.gam.in<-predict(bank.gam,bank.train,type="response")
pred.gam.in<-(prob.gam.in>=pcut.gam)*1
table(bank.train$DLRSN,pred.gam.in,dnn=c("Observed","Predicted"))
# in sample 
mean(ifelse(bank.train$DLRSN != pred.gam.in, 1, 0))
gam.pred.train <- predict(bank.gam, bank.train, type = "response")
pcut1<- 1/36
class.gam.train<- (gam.pred.train>pcut1)*1
MR_train<- mean(bank.train$DLRSN!=class.gam.train)
MR_train

asym_MR <- cost(r = bank.train$DLRSN, pi = pred.gam.in) 
asym_MR
gam.pred.test <- predict(bank.gam, bank.test, type = "response")
pcut1<- 1/36
class.gam.test<- (gam.pred.test>pcut1)*1
cost(bank.test$DLRSN, class.gam.test)

# out of sample 
mean(ifelse(bank.train$DLRSN != pred.gam.in, 1, 0))
gam.pred.test <- predict(bank.gam, bank.test, type = "response")
pcut1<- 1/36
class.gam.test<- (gam.pred.test>pcut1)*1
MR_test<- mean(bank.test$DLRSN!=class.gam.test)
MR_test

# NEURAL NETWORK
bank.train$DLRSN <- as.factor(bank.train$DLRSN)

bank.nnet <- nnet(formula = DLRSN~R1+R2+R3+R4+R5+R6+R7+R8+R9+R10, 
                  data=bank.train, size=1, maxit=500)
prob.nnet= predict(bank.nnet,bank.test)
pred.nnet = as.numeric(prob.nnet > (1/36))
table(bank.test$DLRSN,pred.nnet, dnn=c("Observed","Predicted"))
# mr testing
mean(ifelse(bank.test$DLRSN != pred.nnet, 1, 0))

# MR training
prob.nnet= predict(bank.nnet,bank.train)
pred.nnet = as.numeric(prob.nnet > (1/36))
table(bank.train$DLRSN,pred.nnet, dnn=c("Observed","Predicted"))
mean(ifelse(bank.train$DLRSN != pred.nnet, 1, 0))

cost(bank.train$DLRSN, pred.nnet)

prob.nnet= predict(bank.nnet,bank.test)
pred.nnet = as.numeric(prob.nnet > (1/36))
cost(bank.test$DLRSN, pred.nnet)



