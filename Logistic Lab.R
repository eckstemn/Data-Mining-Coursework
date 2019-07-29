cred.dat <- read.csv("/Users/MeganEckstein/Documents/2019SpringSemester/DataMining/credit_default.csv",
                     header = T)
colnames(cred.dat)

mean(cred.dat$default.payment.next.month)
library(dplyr)
cred.dat<- rename(cred.dat, default=default.payment.next.month)

str(cred.dat)    # structure - see variable type
summary(cred.dat) # summary statistics

cred.dat$SEX<- as.factor(cred.dat$SEX)
cred.dat$EDUCATION<- as.factor(cred.dat$EDUCATION)
cred.dat$MARRIAGE<- as.factor(cred.dat$MARRIAGE)



table.edu<- table(cred.dat$EDUCATION, cred.dat$default)
table.edu
chisq.test(table.edu)

index <- sample(nrow(cred.dat),nrow(cred.dat)*0.75)
credit.train = cred.dat[index,]
credit.test = cred.dat[-index,]

credit.glm0<- glm(default~., family=binomial, data=credit.train)
summary(credit.glm0)

credit.glm0$deviance
AIC(credit.glm0)
BIC(credit.glm0)
hist(predict(credit.glm0))
hist(predict(credit.glm0,type="response"))
table(predict(credit.glm0,type="response") > 0.5)
table(predict(credit.glm0,type="response") > 0.4)
table(predict(credit.glm0,type="response") > 0.2)
table(predict(credit.glm0,type="response") > 0.0001)


pred.glm0.train<- predict(credit.glm0, type="response")

library(ROCR)
pred <- prediction(pred.glm0.train, credit.train$default)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)

#Get the AUC
unlist(slot(performance(pred, "auc"), "y.values"))

pcut = 0.5
# symmetric 
cost1 <- function(r, pi) {
  mean(((r = 0)&(pi>pcut)) | (r==1)&(pi<pcut))
}

#asymmetric cost
cost2 <- function(r, pi) {
  weight1 = 5
  weight0 = 1
  c1 = (r==1)&(pi<pcut)
  c0 = (r==0)&(pi>pcut)
  return(mean(weight1*c1 + weight0*c0))
}

# (1/36) for the bankruptcy data


# cross validation
library(boot)
library(glmnet)
credit.glm1 <- glm(default~., family = binomial, data = cred.dat)
cv.result = cv.glm(data=cred.dat, glmfit = credit.glm1, cost = cost2, K = 10)
cv.result$delta[2]
cv.result$delta




library(PRROC)
score1= pred.glm0.train[credit.train$default==1]
score0= pred.glm0.train[credit.train$default==0]
roc= roc.curve(score1, score0, curve = T)
roc$auc

pr= pr.curve(score1, score0, curve = T)
pr

plot(pr)

pred.glm0.test<- predict(credit.glm0, newdata = credit.test, type="response")
pred <- prediction(pred.glm0.test, credit.test$default)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)

#Get the AUC
unlist(slot(performance(pred, "auc"), "y.values"))

score1.test= pred.glm0.test[credit.test$default==1]
score0.test= pred.glm0.test[credit.test$default==0]
roc.test= roc.curve(score1.test, score0.test, curve = T)
roc.test$auc

pr.test= pr.curve(score1.test, score0.test, curve = T)
pr.test

plot(pr.test)

table((pred.glm0.train > 0.9)*1)
table((pred.glm0.train > 0.5)*1)
table((pred.glm0.train > 0.2)*1)
table((pred.glm0.train > 0.0001)*1)

pcut1<- mean(credit.train$default)
# get binary prediction
class.glm0.train<- (pred.glm0.train>pcut1)*1
# get confusion matrix
table(credit.train$default, class.glm0.train, dnn = c("True", "Predicted"))

# (equal-weighted) misclassification rate
MR<- mean(credit.train$default!=class.glm0.train)
# False positive rate
FPR<- sum(credit.train$default==0 & class.glm0.train==1)/sum(credit.train$default==0)
# False negative rate (exercise)
FNR <- sum(credit.train$default==1 & class.glm0.train==0)/sum(credit.train$default==0)

# define a cost function with input "obs" being observed response 
# and "pi" being predicted probability, and "pcut" being the threshold.
costfunc = function(obs, pred.p, pcut){
  weight1 = 5   # define the weight for "true=1 but pred=0" (FN)
  weight0 = 1    # define the weight for "true=0 but pred=1" (FP)
  c1 = (obs==1)&(pred.p<pcut)    # count for "true=1 but pred=0"   (FN)
  c0 = (obs==0)&(pred.p>=pcut)   # count for "true=0 but pred=1"   (FP)
  cost = mean(weight1*c1 + weight0*c0)  # misclassification with weight
  return(cost) # you have to return to a value when you write R functions
} # end of the function

# define a sequence from 0.01 to 1 by 0.01
p.seq = seq(0.01, 1, 0.01) 

# write a loop for all p-cut to see which one provides the smallest cost
# first, need to define a 0 vector in order to save the value of cost from all pcut
cost = rep(0, length(p.seq))  
for(i in 1:length(p.seq)){ 
  cost[i] = costfunc(obs = credit.train$default, pred.p = pred.glm0.train, pcut = p.seq[i])  
} # end of the loop

# draw a plot with X axis being all pcut and Y axis being associated cost
plot(p.seq, cost)

# find the optimal pcut
optimal.pcut.glm0 = p.seq[which(cost==min(cost))]

# step 1. get binary classification
class.glm0.train.opt<- (pred.glm0.train>optimal.pcut.glm0)*1
# step 2. get confusion matrix, MR, FPR, FNR
table(credit.train$default, class.glm0.train.opt, dnn = c("True", "Predicted"))
MR<- mean(credit.train$default!= class.glm0.train.opt)
FPR<- sum(credit.train$default==0 & class.glm0.train.opt==1)/sum(credit.train$default==0)
FNR<- sum(credit.train$default==1 & class.glm0.train.opt==0)/sum(credit.train$default==1)
cost<- costfunc(obs = credit.train$default, pred.p = pred.glm0.train, pcut = optimal.pcut.glm0) 

credit.glm.back <- step(credit.glm0) # backward selection (if you don't specify anything)
summary(credit.glm.back)
credit.glm.back$deviance
AIC(credit.glm.back)
BIC(credit.glm.back)

credit.glm.back.BIC <- step(credit.glm0, k=log(nrow(credit.train))) 
summary(credit.glm.back.BIC)
credit.glm.back.BIC$deviance
AIC(credit.glm.back.BIC)
BIC(credit.glm.back.BIC)

dummy<- model.matrix(~ ., data = cred.dat)
# look at first few rows of data
head(dummy)
credit.data.lasso<- data.frame(dummy[,-1])
#index <- sample(nrow(credit.data),nrow(credit.data)*0.80)
credit.train.X = as.matrix(select(credit.data.lasso, -default)[index,])
credit.test.X = as.matrix(select(credit.data.lasso, -default)[-index,])
credit.train.Y = credit.data.lasso[index, "default"]
credit.test.Y = credit.data.lasso[-index, "default"]

library(glmnet)
credit.lasso<- glmnet(x=credit.train.X, y=credit.train.Y, family = "binomial")
credit.lasso.cv<- cv.glmnet(x=credit.train.X, y=credit.train.Y, family = "binomial", type.measure = "class")
plot(credit.lasso.cv)

coef(credit.lasso, s=credit.lasso.cv$lambda.min)
coef(credit.lasso, s=credit.lasso.cv$lambda.1se)
# in-sample prediction
pred.lasso.train<- predict(credit.lasso, newx=credit.train.X, s=credit.lasso.cv$lambda.1se, type = "response")
# out-of-sample prediction
pred.lasso.test<- predict(credit.lasso, newx=credit.test.X, s=credit.lasso.cv$lambda.1se, type = "response")

