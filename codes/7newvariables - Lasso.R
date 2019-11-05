library(glmnet)
library(data.table)

## use the dataset base on shuoge + new variables
train2 = fread('test100k_7new.csv')
test2 = fread('train100k_7new.csv')

train = train2
test = test2


x6 = model.matrix(HasDetections~.,train)[,-1]
y6 = as.numeric(train$HasDetections>0)   ### transfer it to 0 and 1


cv.fit6 = cv.glmnet(x6,y6,family = 'binomial',alpha=1, type.measure = 'auc')
best6 = cv.fit6$lambda.min

plot(cv.fit6)


test.x6 = model.matrix(HasDetections~.,test)[,-1]
test.y6 = as.numeric(test$HasDetections>0)

# use the whole dataset to train the model
lasso6 = glmnet(x6,y6,lambda = best6,family = 'binomial',alpha=1)
coefi = coef(lasso6, s=best6)
co = as.matrix(coefi)
co = data.frame(co)
write.csv(co,'coefficients_lasso.csv')


## AUC
library(ROCR)
prob6 = predict(lasso6,newx = test.x6, s= best6, type = 'response')
p6 = prediction(prob6, as.numeric(test.y6))
perf6 = performance(p6,'tpr','fpr')
plot(perf6)
performance(p6,'auc')  #0.6898781


auc(test.y6,prob6)

op <- par(mfrow=c(1, 2))
plot(cv.fit6$glmnet.fit, "norm",   label=TRUE)
plot(cv.fit6$glmnet.fit, "lambda", label=TRUE)


### train error 
pred.train6 = predict(lasso6, newx = x6, s= best6, type = 'class')
mean(y6 == pred.train6)


## test error 
pred6 = predict(lasso6, newx = test.x6, s= 'lambda.min', type = 'class')
table(test.y6,pred6)
mean(test.y6 == pred6)






