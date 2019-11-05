## USe the old dataset
# train = readRDS('train5-basic.RDS')
# test = readRDS('test5-basic.RDS')

## use the new dataset runing the DNN
# train1 = fread('train_100k1.csv')
# test1 = fread('test_100k1.csv')

# use shuoge dataset + new variables
library(data.table)
# train2 = fread('test100k_7new.csv')
# test2 = fread('train100k_7new.csv')

#train = train2
#test = test2

### Use the dataset after the best subset

train4 = fread('train100k_2.csv')
test4 = fread('test100k_2.csv')

train = train4
test = test4

train.x = model.matrix(HasDetections~.,train)[,-1]
train.y = train$HasDetections

test.x = model.matrix(HasDetections~.,test)[,-1]
test.y = test$HasDetections

library(keras)
create_model_logi <- function() {
  model<-keras_model_sequential() %>%
    layer_dense(units=30, input_shape=c(44), activation="relu")%>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units=20, activation="relu")%>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units=10, activation="relu")%>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units=1, activation="sigmoid")
  model %>% compile(
    optimizer=optimizer_adam(lr=0.0005),
    loss="binary_crossentropy",
    metrics=c("accuracy")
  )
  model
}

model_logi <- create_model_logi()
model_logi %>% summary()

checkpoint_dir <- "checkpoints"
dir.create(checkpoint_dir, showWarnings = FALSE)
filepath <- file.path(checkpoint_dir, "weights.{epoch:02d}-{val_loss:.2f}.hdf5")
checkpointer=callback_model_checkpoint(filepath, monitor='val_loss', verbose=0, save_best_only=T, save_weights_only=F, mode='auto', period=1)

history <- model_logi %>% fit(
  train.x, train.y, 
  epochs = 100, batch_size = 1000, 
  validation_split = 0.2,
  callbacks=checkpointer
)

list.files(checkpoint_dir)


fresh_model <- create_model_logi()
fresh_model %>% load_model_weights_hdf5(
  file.path(checkpoint_dir, 'weights.68-0.69.hdf5') # "weights.91-7.69.hdf5")  ## select the least lost
)

model_logi %>% evaluate(test.x, test.y)
fresh_model %>% evaluate(test.x, test.y)  ## 0.51948
fresh_model %>% evaluate(train.x, train.y)


yhat_keras_class_vec <- predict_classes(object = fresh_model, x = test.x) %>%
  as.vector()
yhat_keras_prob_vec  <- predict_proba(object = fresh_model, x = test.x) %>%
  as.vector()

library(tibble)
library(tidyverse)
estimates_keras_tbl <- tibble(
  truth      = as.factor(test.y) %>% fct_recode(yes = "1", no = "0"),
  estimate   = as.factor(yhat_keras_class_vec) %>% fct_recode(yes = "1", no = "0"),
  class_prob = yhat_keras_prob_vec
)

library(caret)
library(yardstick)
estimates_keras_tbl %>% roc_auc(truth, class_prob)
plot(history)

