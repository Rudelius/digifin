library(dplyr)
library(tree)
library(randomForest)
library(utf8)
set.seed(1)

# Plot two plots at a time.
par(mfrow= c(1,2))

# Clean environment.
rm(list = ls())

# Import loandata.
loandata <- read.csv(file="/Users/johan/Dropbox/Handels/digifin/data/loandata.csv", header = TRUE, sep=",")

# Import prepared municipality -> region dataset.
# https://www.dropbox.com/s/04puwtchxprr1w3/lan.csv?dl=0
municipalities <- read.csv(file="/Users/johan/Dropbox/Handels/digifin/data/lan.csv", header = TRUE, sep=";")



######################################## Data cleaning part #######################################

# Drop key attribute.
loandata$key <- NULL

# Reformat dates into year numerics.
loandata[,1] = 2000+(unclass(as.Date(loandata$creationdateuserloan))-unclass(as.Date("2000/01/01")))/365 #creationdateuserloan
loandata[,25] = 2000+(unclass(as.Date(loandata$lastucrequest))-unclass(as.Date("2000/01/01")))/365 #lastucrequest
v <- loandata$lastaddresschange #add day "01" to lastaddresschange dates which are in format: "YYYYMM"; avoid pasting into NAs.
for (i in 1:length(v)){
  if(!is.na(v[i])){
    v[i] <- paste(v[i],"01",sep="") 
  }
}
loandata$lastaddresschange <- 2000+(unclass(as.Date(v, format="%Y%m%d"))-unclass(as.Date("2000/01/01")))/365 #lastaddresschange

# Replace "None", "NA", "Other" and empty string with NA. Requires dplyr package.
for (i in 1:length(loandata)){
  loandata[,i] <- na_if(loandata[,i], "None")
  loandata[,i] <- na_if(loandata[,i], "")
  loandata[,i] <- na_if(loandata[,i], "NA")
  loandata[,i] <- na_if(loandata[,i], "Other")
}

# Clean faulty jobtype data.
x <- loandata$jobtype
x <- replace(x, x == "Agricul", "Agriculture")
x <- replace(x, x == "Custome", "CustomerServ")
x <- replace(x, x == "Economi", "Economics")
x <- replace(x, x == "Educati", "Educational")
x <- replace(x, x == "ImportE", "ImportExport")
x <- replace(x, x == "Logisti", "Logistics")
x <- replace(x, x == "Managem", "Management")
x <- replace(x, x == "Product", "Production")
x <- replace(x, x == "Psychol", "Psychology")
x <- replace(x, x == "Publish", "Publishing")
x <- replace(x, x == "Realest", "Realestate")
x <- replace(x, x == "Researc", "Research")
x <- replace(x, x == "Securit", "Security")
loandata$jobtype <- factor(x)

# Clean faulty jobstatus data.
x <- loandata$jobstatus
x <- replace(x, x == "TemporaryEmployme", "TemporaryEmployment")
loandata$jobstatus <- factor(x)

# Fix encoding of Swedish special characters for municipality.
a = as.character(loandata$municipality)
Encoding(a) <- "latin1"
loandata$municipality <- as.factor(utf8_encode(a))

# VLOOKUP municipality -> region to reduce number of geographical data factor levels.
loandata <- (merge(municipalities, loandata, by = 'municipality'))



####################################### Machinelearning part ######################################

# Create a vector to randomly split the dataset into training and test ratio 9:1
trainingRows <- sample(1:nrow(loandata), 0.9*length(loandata[,1]))

# Split loandata into training and test sets
loandata.train <- loandata[trainingRows, ]
loandata.test <- loandata[-trainingRows, ]

# Generate a tree on the training data. Nate that there seems to be a greediness problem, with the tree improving when we remove category.
tree.loandata <- tree(formula = newpayingremark ~ .-municipality-newpayingremark-category-jobtype, data = loandata, subset = trainingRows)

# Cross validation
cv.loandata <- cv.tree(tree.loandata)

# print results
#cv.loandata 

# Plot cv results
plot(cv.loandata$size, cv.loandata$dev, type = "b")
plot(cv.loandata$k, cv.loandata$dev, type = "b")

# Prune to 2 leaf nodes, which usually gives the best result depending on random draw
tree.loandata.pruned <- prune.tree(tree.loandata, best = 2)

# Plot trees
plot(tree.loandata)
text(tree.loandata, pretty = 0)
title(main = "Unpruned")

plot(tree.loandata.pruned)
text(tree.loandata.pruned, pretty = 0)
title(main = "Pruned")


####################################### Testing part ######################################

# Calculate percentage of correct predictions for unpruned tree.
pred.tree.loandata <- predict(tree.loandata, loandata.test, type = "vector")
correctPredictions = 0
for (i in 1:length(pred.tree.loandata)){
  if(abs(pred.tree.loandata[i]-loandata.test$newpayingremark[i]) < 0.5){
    correctPredictions = correctPredictions + 1
  } 
}
correctPredictions.tree.loandata <- correctPredictions / length(loandata.test[,1])

# Calculate the mean error of predictions for unpruned tree.
meanError.tree.loandata <- mean(abs(pred.tree.loandata-loandata.test$newpayingremark))


# Calculate percentage of correct predictions for pruned tree.
pred.tree.loandata.pruned <- predict(tree.loandata.pruned, loandata.test, type = "vector")
correctPredictions = 0
for (i in 1:length(pred.tree.loandata.pruned)){
  if(abs(pred.tree.loandata.pruned[i]-loandata.test$newpayingremark[i]) < 0.5){
    correctPredictions = correctPredictions + 1
  } 
}
correctPredictions.tree.loandata.pruned <- correctPredictions / length(loandata.test[,1])

# Calculate the mean error of predictions for pruned tree.
meanError.tree.loandata.pruned <- mean(abs(pred.tree.loandata.pruned-loandata.test$newpayingremark))

stats <- matrix(c(meanError.tree.loandata, correctPredictions.tree.loandata,
                  meanError.tree.loandata.pruned, correctPredictions.tree.loandata.pruned)
                ,ncol = 2)
colnames(stats) <- c('unpruned', 'pruned')
rownames(stats) <- c('meanError', 'correctPredictions')
as.table(stats)



#################################### Bagging part ######################################

# Inpute values for the NA:s with random forests and proximity, takes a while.
#loandata.naInpute <- rfImpute(x = loandata[,2:27], y = loandata$newpayingremark, iter=1, ntree=10)
#colnames(loandata.naInpute)[1] <- "newpayingremark"

# Make a bagging model.
#bag.loandata = randomForest(x = loandata.naInpute[,2:26], y = loandata$newpayingremark, 
#                            subset = trainingRows, mtry = 25, importance = TRUE, ntree = 10)

#pred.bag.loandata <- predict(bag.loandata, newdata = loandata.test)
