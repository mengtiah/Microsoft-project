library(data.table)
library(keras)
library(caret)
library(xgboost)
library(tidyverse)
library(glmnet)

### processed data
train.data=fread("train_100k1.csv")
test.data=fread("test_100k1.csv")

#add 7 new variables

trainnew=train.data%>%
  mutate(primary_drive_c_ratio=Census_SystemVolumeTotalCapacity/Census_PrimaryDiskTotalCapacity,
         non_primary_drive_MB=Census_PrimaryDiskTotalCapacity-Census_SystemVolumeTotalCapacity,
         aspect_ratio=Census_InternalPrimaryDisplayResolutionHorizontal/Census_InternalPrimaryDisplayResolutionVertical,
         MegaPixels=Census_InternalPrimaryDisplayResolutionHorizontal*Census_InternalPrimaryDisplayResolutionVertical,
         ram_per_processor=Census_TotalPhysicalRAM/Census_ProcessorCoreCount,
         new_num_0=Census_InternalPrimaryDiagonalDisplaySizeInInches/Census_ProcessorCoreCount,
         new_num_1=Census_ProcessorCoreCount*Census_InternalPrimaryDiagonalDisplaySizeInInches)


testnew=test.data%>%
  mutate(primary_drive_c_ratio=Census_SystemVolumeTotalCapacity/Census_PrimaryDiskTotalCapacity,
         non_primary_drive_MB=Census_PrimaryDiskTotalCapacity-Census_SystemVolumeTotalCapacity,
         aspect_ratio=Census_InternalPrimaryDisplayResolutionHorizontal/Census_InternalPrimaryDisplayResolutionVertical,
         MegaPixels=Census_InternalPrimaryDisplayResolutionHorizontal*Census_InternalPrimaryDisplayResolutionVertical,
         ram_per_processor=Census_TotalPhysicalRAM/Census_ProcessorCoreCount,
         new_num_0=Census_InternalPrimaryDiagonalDisplaySizeInInches/Census_ProcessorCoreCount,
         new_num_1=Census_ProcessorCoreCount*Census_InternalPrimaryDiagonalDisplaySizeInInches)

write.csv(trainnew, 'train100k_7new.csv', row.names = FALSE) 
write.csv(testnew, 'test100k_7new.csv', row.names = FALSE)


### after feature selection -dataset

selected_variable=read.csv("selected_variable.csv")
train1=trainnew[,colnames(trainnew) %in% selected_variable$variables]
test1=testnew[,colnames(testnew) %in% selected_variable$variables]

train2=cbind(train1,trainnew[,66])
test2=cbind(test1,testnew[,66])


write.csv(train2, 'train100k_2.csv', row.names = FALSE) 
write.csv(test2, 'test100k_2.csv', row.names = FALSE)


### xgboost

labels=as.matrix(train2[,45])

tr=as.matrix(train2[,-45])

ts_labels=as.matrix(test2[,45])
ts=as.matrix(test2[,-45])

dtrain <- xgb.DMatrix(data = tr, label= labels)
dtest <- xgb.DMatrix(data = ts, label= ts_labels)


model=xgboost(data=dtrain,
              max.depth=8,
              nround=50,
              early_stopping_rounds=3,
              objective="binary:logistic",
              gamma=1,
              num_parallel_tree =10, 
              subsample = 0.8,
              colsample_bytree =0.3)

pred.train=predict(model,dtrain)
pred=predict(model,dtest)

# training error
trainerr=mean(as.numeric(pred.train>0.5)!=labels)
# testing error
err=mean(as.numeric(pred>0.5)!=ts_labels)

# auc - test
roc_obj=roc(as.numeric(ts_labels),as.numeric(pred),algorithm =2)
plot(roc_obj)
pROC::auc(roc_obj)

# roc curve
pred=prediction(pred,as.numeric(ts_labels))
perf=performance(pred,"tpr","fpr")
plot(perf)
