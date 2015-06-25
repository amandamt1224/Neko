data <- read.csv("~/Desktop/coords/july_coords_all_8.csv", header=FALSE)
predictions <- (data$V3)*100
summary(predictions)
hist(predictions,
     main="July. Prediction Values (Step of 8)",
     xlab="Prediction Value",
     las=0,
     breaks=c(0, seq(5, 100, 5)),
     labels=TRUE
     )

dataHits <- read.csv("~/Desktop/coords/july_coords_hits_8.csv", header=FALSE)
hits <- (dataHits$V3)*100
summary(hits)
hist(hits,
     main="July. Hits (Step of 8)",
     xlab="Value",
     labels=TRUE,
     las=0
     )

