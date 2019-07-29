library(factoextra)
library(fpc)
library(gridExtra)
library(readxl)
library(arules)
library(arulesViz)

set.seed(13023260)
europe_emp <- read.table("/Users/MeganEckstein/Documents/2019SpringSemester/DataMiningII/europeanJobs.txt", header = T)

index <- sample(nrow(europe_emp),nrow(europe_emp)*0.90)
ee_train <- europe_emp[index,]
ee_test <- europe_emp[-index,]

ee_train1 <- europe_emp[index,]
ee_test1 <- europe_emp[-index,]
ee_train2 <- ee_train1[,-1]
ee_test2 <- ee_test1[,-1]


# KMEANS CLUSTERING
ee_train <- scale(ee_train[,-1])
ee_test <- scale(ee_test[,-1])

mean(ee_train2[,"Agr"])
mean(ee_train2[,"Min"])
mean(ee_train2[,"Man"])
mean(ee_train2[,"PS"])
mean(ee_train2[,"Con"])
mean(ee_train2[,"SI"])
mean(ee_train2[,"Fin"])
mean(ee_train2[,"SPS"])
mean(ee_train2[,"TC"])

fit2 <- kmeans(ee_train2, 2, nstart = 20)
fit2$centers
p2 <- fviz_cluster(fit2, data = ee_train) + ggtitle("k = 2")

fit3 <- kmeans(ee_train2, 3, nstart = 20)
fit3$centers
p3 <- fviz_cluster(fit3, data = ee_train) + ggtitle("k = 3")

fit4 <- kmeans(ee_train2, 4, nstart = 20)
fit4$centers
p4 <- fviz_cluster(fit4, data = ee_train) + ggtitle("k = 4")

fit5 <- kmeans(ee_train2, 5, nstart = 20)
fit5$centers
p5 <- fviz_cluster(fit5, data = ee_train) + ggtitle("k = 5")

fit6 <- kmeans(ee_train2, 6, nstart = 20)
fit6$centers
p6 <- fviz_cluster(fit6, data = ee_train) + ggtitle("k = 6")

fit7 <- kmeans(ee_train2, 7, nstart = 20)
fit7$centers
p7 <- fviz_cluster(fit7, data = ee_train) + ggtitle("k = 7")

grid.arrange(p2,p3,p4,p5,p6,p7, nrow = 3)

wss <- (nrow(ee_train)-1)*sum(apply(ee_train,2,var))
for (i in 2:12) wss[i] <- sum(kmeans(ee_train,
                                     centers=i)$withinss)
plot(1:12, wss, type="b", xlab="Number of Clusters",ylab="Within groups sum of squares")

#prediction.strength(ee_train, Gmin=2, Gmax=15, M=10,cutoff=0.8)

d = dist(ee_train2, method = "euclidean")
result = matrix(nrow = 14, ncol = 3)
for (i in 2:15){
  cluster_result = kmeans(ee_train2, i)
  clusterstat=cluster.stats(d, cluster_result$cluster)
  result[i-1,1]=i
  result[i-1,2]=clusterstat$avg.silwidth
  result[i-1,3]=clusterstat$dunn   
}
plot(result[,c(1,2)], type="l", ylab = 'silhouette width', xlab = 'number of clusters')
plot(result[,c(1,3)], type="l", ylab = 'dunn index', xlab = 'number of clusters')

# HIERARCHICAL CLUSTERING

ee_train_dist <- dist(ee_train2)
ee_train_hclust=hclust(ee_train_dist, method="ward")
plot(ee_train_hclust)

ee_train_2clust = cutree(ee_train_hclust, k=2)
ee_train_3clust = cutree(ee_train_hclust, k=3)
ee_train_4clust = cutree(ee_train_hclust, k=4)
ee_train_5clust = cutree(ee_train_hclust, k=5)
ee_train_6clust = cutree(ee_train_hclust, k=6)
ee_train_7clust = cutree(ee_train_hclust, k=7)

plotcluster(ee_train, ee_train_2clust)
plotcluster(ee_train, ee_train_3clust)
plotcluster(ee_train, ee_train_4clust)
plotcluster(ee_train, ee_train_5clust)
plotcluster(ee_train, ee_train_6clust)
plotcluster(ee_train, ee_train_7clust)


# ASSOCIATION RULES



TransFood <- read.csv('https://xiaoruizhu.github.io/Data-Mining-R/data/food_4_association.csv')
TransFood <- TransFood[, -1]
# Find out elements that are not equal to 0 or 1 and change them to 1.
Others <- which(!(as.matrix(TransFood) ==1 | as.matrix(TransFood) ==0), arr.ind=T )
TransFood[Others] <- 1
TransFood <- as(as.matrix(TransFood), "transactions")

itemFrequencyPlot(TransFood, support = 0.1, cex.names=0.8)
basket_rules <- apriori(TransFood,parameter = list(sup = 0.003, conf = 0.9,target="rules"))
summary(basket_rules)
inspect((basket_rules))
inspect(subset(basket_rules, size(basket_rules)>2))
plot((sort(basket_rules, by="lift")), method = "graph")

plot(basket_rules, method="grouped")


