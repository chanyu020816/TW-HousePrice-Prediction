library(tidyverse)
library(party)
df = read.csv("/home/chanyu/Desktop/school/DataMining/project/model/df.csv")
df$車位類別 = as.factor(df$車位類別)
df$city = as.factor(df$city)
df$有無管理組織 = as.factor(df$有無管理組織)
df$主要用途 = as.factor(df$主要用途)
df$建物型態 = as.factor(df$建物型態)
df$電梯 = as.factor(df$電梯)
df$有無管理組織 = as.factor(df$有無管理組織)
df$主要用途 = as.factor(df$主要用途)
df$建物型態 = as.factor(df$建物型態)
mod_bas = mob(
  總價元 ~ 土地移轉總面積平方公尺 + 建物移轉總面積平方公尺 + 建物現況格局_房 |
    # 建物現況格局_廳 + 建物現況格局_衛 + 建物現況格局_隔間 + 單價元平方公尺 + 
    # 車位移轉總面積_平方公尺 + 車位總價元 + 主建物面積 + 附屬建物面積 + 陽台面積 |
    電梯 + 交易標的 + 建物型態 + 主要用途 + 有無管理組織 + 車位類別 + city,
  data = df,
  model = linearModel)



# Import Required packages
set.seed(500)
library(neuralnet)
library(MASS)

# Boston dataset from MASS
data <- Boston

# Normalize the data
maxs <- apply(data, 2, max) 
mins <- apply(data, 2, min)
scaled <- as.data.frame(scale(data, center = mins, 
                              scale = maxs - mins))

# Split the data into training and testing set
index <- sample(1:nrow(data), round(0.75 * nrow(data)))
train_ <- scaled[index,]
test_ <- scaled[-index,]


# Build Neural Network
nn <- neuralnet(medv ~ crim + zn + indus + chas + nox 
                + rm + age + dis + rad + tax + 
                  ptratio + black + lstat, 
                data = train_, hidden = c(5, 3), 
                linear.output = TRUE)

# Predict on test data
pr.nn <- compute(nn, c[,1:13])

# Compute mean squared error
pr.nn_ <- pr.nn$net.result * (max(data$medv) - min(data$medv)) 
+ min(data$medv)
test.r <- (test_$medv) * (max(data$medv) - min(data$medv)) + 
  min(data$medv)
MSE.nn <- sum((test.r - pr.nn_)^2) / nrow(test_)

# Plot the neural network
plot(nn)
train_
a = c(0.00632, 18, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296, 15.3, 396.9, 4.98)
compute(nn, test_[1, 1:13])


a
