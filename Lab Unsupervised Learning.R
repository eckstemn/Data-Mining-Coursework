seed = read.table('http://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt', header=F)
seed = seed[,1:7]
colnames(seed) = c("area", "perimeter","campactness", "length", "width", "asymmetry", "groovelength")

seed <- scale(seed)

library(factoextra)
distance <- get_dist(seed)
fviz_dist(distance, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))

# K-Means Cluster Analysis
fit <- kmeans(seed, 2) #2 cluster solution
#Display number of clusters in each cluster
table(fit$cluster)

fviz_cluster(fit, data = seed)

k3 <- kmeans(seed, centers = 3, nstart = 25)
k4 <- kmeans(seed, centers = 4, nstart = 25)
k5 <- kmeans(seed, centers = 5, nstart = 25)
# nstart can change the results

# plots to compare
p1 <- fviz_cluster(fit, geom = "point", data = seed) + ggtitle("k = 2")
p2 <- fviz_cluster(k3, geom = "point",  data = seed) + ggtitle("k = 3")
p3 <- fviz_cluster(k4, geom = "point",  data = seed) + ggtitle("k = 4")
p4 <- fviz_cluster(k5, geom = "point",  data = seed) + ggtitle("k = 5")

library(gridExtra)
grid.arrange(p1, p2, p3, p4, nrow = 2)

library(fpc)
plotcluster(seed, fit$cluster)

#See exactly which item are in 1st group
seed[fit$cluster==1,]
#get cluster means for scaled data
aggregate(seed,by=list(fit$cluster),FUN=mean)

#or alternatively, use the output of kmeans
fit$centers


# Determine number of clusters
wss <- (nrow(seed)-1)*sum(apply(seed,2,var))
for (i in 2:12) wss[i] <- sum(kmeans(seed,
                                     centers=i)$withinss)
plot(1:12, wss, type="b", xlab="Number of Clusters",ylab="Within groups sum of squares")

prediction.strength(seed, Gmin=2, Gmax=15, M=10,cutoff=0.8)

d = dist(seed, method = "euclidean")
result = matrix(nrow = 14, ncol = 3)
for (i in 2:15){
  cluster_result = kmeans(seed, i)
  clusterstat=cluster.stats(d, cluster_result$cluster)
  result[i-1,1]=i
  result[i-1,2]=clusterstat$avg.silwidth
  result[i-1,3]=clusterstat$dunn   
}
plot(result[,c(1,2)], type="l", ylab = 'silhouette width', xlab = 'number of clusters')

plot(result[,c(1,3)], type="l", ylab = 'dunn index', xlab = 'number of clusters')

#Wards Method or Hierarchical clustering
#Calculate the distance matrix
seed.dist=dist(seed)
#Obtain clusters using the Wards method
seed.hclust=hclust(seed.dist, method="ward")
plot(seed.hclust)

#Cut dendrogram at the 3 clusters level and obtain cluster membership
seed.3clust = cutree(seed.hclust,k=3)
#See exactly which item are in third group
seed[seed.3clust==3,]
#get cluster means for raw data
#Centroid Plot against 1st 2 discriminant functions
#Load the fpc library needed for plotcluster function
library(fpc)
#plotcluster(ZooFood, fit$cluster)
plotcluster(seed, seed.3clust)

library(mclust)
mclust_result = Mclust(seed)
summary(mclust_result)
plot(mclust_result)

library(arules)
data("Groceries")
#run summary report
summary(Groceries)

x = Groceries[size(Groceries) > 30]
inspect(x)

#
itemFrequencyPlot(Groceries, support = 0.1, cex.names=0.8)

# Run the apriori algorithm
basket_rules <- apriori(Groceries,parameter = list(sup = 0.003, conf = 0.5,target="rules"))

summary(basket_rules)

inspect(head(basket_rules))

#Basket rules of size greater than 4
inspect(subset(basket_rules, size(basket_rules)>2))

inspect(subset(basket_rules, lift>3))

yogurt.rhs <- subset(basket_rules, subset = rhs %in% "yogurt" & lift>0.4)
inspect(yogurt.rhs)

meat.lhs <- subset(basket_rules, subset = lhs %in% "meat" & lift>1.5)
inspect(meat.lhs)

library('arulesViz')
plot(basket_rules)


plot(basket_rules, interactive=TRUE)


plot(head(sort(basket_rules, by="lift"), 10), method = "graph")

plot(basket_rules, method="grouped")

