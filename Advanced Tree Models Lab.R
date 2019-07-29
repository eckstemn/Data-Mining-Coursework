# load Boston data
library(MASS)
data(Boston)
index <- sample(nrow(Boston),nrow(Boston)*0.75)
boston.train <- Boston[index,]
boston.test <- Boston[-index,]

# load credit card data
credit.data <- read.csv("/Users/MeganEckstein/Documents/2019SpringSemester/DataMiningII/credit_default.csv", header=T)
# convert categorical variables
credit.data$SEX<- as.factor(credit.data$SEX)
credit.data$EDUCATION<- as.factor(credit.data$EDUCATION)
credit.data$MARRIAGE<- as.factor(credit.data$MARRIAGE)
# random splitting
index <- sample(nrow(credit.data),nrow(credit.data)*0.75)
credit.train = credit.data[index,]
credit.test = credit.data[-index,]

library(ipred)
boston.bag<- bagging(medv~., data = boston.train, nbagg=100)
boston.bag
boston.bag.pred<- predict(boston.bag, newdata = boston.test)
mean((boston.bag.pred-boston.test$medv)^2)

# compare with single tree
library(rpart)
boston.tree<- rpart(medv~., data = boston.train)
boston.tree.pred<- predict(boston.tree, newdata = boston.test)
mean((boston.tree.pred-boston.test$medv)^2)

# how many trees are good?
ntree<- c(1, 3, 5, seq(10, 200, 10))
MSE.test<- rep(0, length(ntree))
for(i in 1:length(ntree)){
  boston.bag1<- bagging(medv~., data = boston.train, nbagg=ntree[i])
  boston.bag.pred1<- predict(boston.bag1, newdata = boston.test)
  MSE.test[i]<- mean((boston.bag.pred1-boston.test$medv)^2)
}
plot(ntree, MSE.test, type = 'l', col=2, lwd=2, xaxt="n")
axis(1, at = ntree, las=1)


# out of bag
boston.bag.oob<- bagging(medv~., data = boston.train, coob=T, nbagg=100)
boston.bag.oob

credit.bag<- bagging(as.factor(default.payment.next.month)~., data = credit.train, nbagg=100)
credit.bag.pred<- predict(credit.bag, newdata = credit.train, type="prob")[,2]
credit.bag.pred.test<- predict(credit.bag, newdata = credit.test, type="prob")[,2]
library(ROCR)

pred = prediction(credit.bag.pred.test, credit.test$default.payment.next.month)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values"))

#ROC for training set
pred = prediction(credit.bag.pred, credit.train$default.payment.next.month)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
unlist(slot(performance(pred, "auc"), "y.values"))
table(credit.train$default.payment.next.month, credit.bag.pred, dnn = c("True", "Pred"))

credit.bag.pred.test<- predict(credit.bag, newdata = credit.test, type="class")
table(credit.test$default.payment.next.month, credit.bag.pred.test, dnn = c("True", "Pred"))

predtrain = prediction(credit.bag.pred, credit.train$default.payment.next.month)
perftrain = performance(predtrain, "tpr", "fpr")
plot(perftrain, colorize=TRUE)
unlist(slot(performance(predtrain, "auc"), "y.values"))

credit.bag.pred.train <- predict(credit.bag, newdata = credit.train, type="class")
table(credit.train$default.payment.next.month, credit.bag.pred.train, dnn = c("True", "Pred"))

credit.rpart <- rpart(formula = default.payment.next.month ~ ., data = credit.train, method = "class")
credit.test.pred.tree1 <- predict(credit.rpart, credit.test, type="class")
table(credit.test$default.payment.next.month, credit.test.pred.tree1, dnn=c("Truth","Predicted"))

credit.test.pred.tree1 <- predict(credit.rpart, credit.test, type="prob")

pred = prediction(credit.test.pred.tree1[,2], credit.test$default.payment.next.month)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)

unlist(slot(performance(pred, "auc"), "y.values"))



# RANDOM FORESTS
library(randomForest)
boston.rf<- randomForest(medv~., data = boston.train, importance=TRUE)
boston.rf
boston.rf$importance
plot(boston.rf$mse, type='l', col=2, lwd=2, xlab = "ntree", ylab = "OOB Error")

boston.rf.pred<- predict(boston.rf, boston.test)
mean((boston.rf.pred-boston.test$medv)^2)

oob.err<- rep(0, 13)
test.err<- rep(0, 13)
for(i in 1:13){
  fit<- randomForest(medv~., data = boston.train, mtry=i)
  oob.err[i]<- fit$mse[500]
  test.err[i]<- mean((predict(fit, boston.test)-boston.test$medv)^2)
  cat(i, " ")
}

matplot(cbind(test.err, oob.err), pch=15, col = c("red", "blue"), type = "b", ylab = "MSE", xlab = "mtry")
legend("topright", legend = c("test Error", "OOB Error"), pch = 15, col = c("red", "blue"))

credit.rf <- randomForest(as.factor(default.payment.next.month)~., data = credit.train)
credit.rf

plot(credit.rf, lwd=rep(2, 3))
legend("right", legend = c("OOB Error", "FPR", "FNR"), lwd=rep(2, 3), lty = c(1,2,3), col = c("black", "red", "green"))

credit.rf.pred<- predict(credit.rf, type = "prob")[,2]
costfunc = function(obs, pred.p, pcut){
  weight1 = 5   # define the weight for "true=1 but pred=0" (FN)
  weight0 = 1    # define the weight for "true=0 but pred=1" (FP)
  c1 = (obs==1)&(pred.p<pcut)    # count for "true=1 but pred=0"   (FN)
  c0 = (obs==0)&(pred.p>=pcut)   # count for "true=0 but pred=1"   (FP)
  cost = mean(weight1*c1 + weight0*c0)  # misclassification with weight
  return(cost) # you have to return to a value when you write R functions
} 
p.seq = seq(0.01, 0.5, 0.01)
cost = rep(0, length(p.seq))  
for(i in 1:length(p.seq)){ 
  cost[i] = costfunc(obs = credit.train$default.payment.next.month, pred.p = credit.rf.pred, pcut = p.seq[i])  
}
plot(p.seq, cost)

library(ROCR)
pred <- prediction(credit.rf.pred, credit.train$default.payment.next.month)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)

#Get the AUC
unlist(slot(performance(pred, "auc"), "y.values"))

## out-of-sample
optimal.pcut= p.seq[which(cost==min(cost))]
credit.rf.pred.test<- predict(credit.rf, newdata=credit.test, type = "prob")[,2]
credit.rf.class.test<- (credit.rf.pred.test>optimal.pcut)*1
table(credit.test$default.payment.next.month, credit.rf.class.test, dnn = c("True", "Pred"))


# BOOSTING 
library(gbm)
?gbm
# 8 means we grow until the 8th split
boston.boost<- gbm(medv~., data = boston.train, distribution = "gaussian", n.trees = 10000, shrinkage = 0.01, interaction.depth = 8)
summary(boston.boost)
par(mfrow=c(1,2))
plot(boston.boost, i="lstat")
plot(boston.boost, i="rm")
boston.boost.pred.test<- predict(boston.boost, boston.test, n.trees = 10000)
mean((boston.boost.pred.test-boston.test$medv)^2) 

par(mfrow=c(1,1))
ntree<- seq(100, 10000, 100)
predmat<- predict(boston.boost, newdata = boston.test, n.trees = ntree)
err<- apply((predmat-boston.test$medv)^2, 2, mean)
plot(ntree, err, type = 'l', col=2, lwd=2, xlab = "n.trees", ylab = "Test MSE")
abline(h=min(test.err), lty=2)

library(adabag)
credit.train$default.payment.next.month= as.factor(credit.train$default.payment.next.month)
credit.boost= boosting(default.payment.next.month~., data = credit.train, boos = T)
save(credit.boost, file = "credit.boost.Rdata")

# Training AUC
pred.credit.boost= predict(credit.boost, newdata = credit.train)
pred <- prediction(pred.credit.boost$prob[,2], credit.train$default.payment.next.month)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
#Get the AUC
unlist(slot(performance(pred, "auc"), "y.values"))

pred.credit.boost= predict(credit.boost, newdata = credit.test)
# Testing AUC
pred <- prediction(pred.credit.boost$prob[,2], credit.test$default.payment.next.month)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
#Get the AUC
unlist(slot(performance(pred, "auc"), "y.values"))

