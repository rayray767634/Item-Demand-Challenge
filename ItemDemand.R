library(vroom)
library(tidyverse)
library(dplyr)
library(patchwork)
library(tidymodels)
library(glmnet)
library(stacks)
library(embed)
library(discrim)
library(naivebayes)
library(kknn)
library(kernlab)
library(themis)
library(bonsai)
library(lightgbm)
library(dbarts)
library(timetk)
library(modeltime)

item.train <- vroom("demandtrain.csv")

item.test <- vroom("demandtest.csv")

storeItem <- item.train %>%
  filter(store == 5, item == 38) 

storeItem2 <- item.train %>%
  filter(store == 2) %>%
  filter(item == 7)

storeItem3 <- item.train %>%
  filter(store == 3) %>%
  filter(item == 4)

storeItem4 <- item.train %>%
  filter(store == 4) %>%
  filter(item == 33)

ACF1 <- storeItem1 %>%
  pull(sales) %>%
  forecast::ggAcf(., lag.max = 2*365)

ACF2 <- storeItem2 %>%
  pull(sales) %>%
  forecast::ggAcf(., lag.max = 2*365)

ACF3 <- storeItem3 %>%
  pull(sales) %>%
  forecast::ggAcf(., lag.max = 2*365)

ACF4 <- storeItem4 %>%
  pull(sales) %>%
  forecast::ggAcf(., lag.max = 2*365)


ACF1 + ACF2 + ACF3 + ACF4


# ML for Time Series

my_recipe <- recipe(sales~date , data = storeItem) %>%
  step_date(date, features = c("dow","month","decimal"))


prep <- prep(my_recipe)
baked <- bake(prep, new_data = storeItem)


# knn model
knn_model <- nearest_neighbor(neighbors = tune()) %>%
  set_mode("regression") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)

# fit or tune model

tuning_grid_knn <- grid_regular(neighbors(),
                                levels = 5)

# Set up K-fold CV
folds <- vfold_cv(storeItem, v = 5, repeats = 1)
# Run the CV
CV_results_knn <-knn_wf  %>%
  tune_grid(resamples = folds,
            grid = tuning_grid_knn,
            metrics = metric_set(smape))

bestTune_knn <- CV_results_knn %>%
  select_best("smape")

collect_metrics(CV_results_knn) %>%
  filter(.metric == "smape") %>%
  pull(mean)

# time series
cv_split <- time_series_split(storeItem, assess = "3 months", cumulative = TRUE)
cv_split %>%
  tk_time_series_cv_plan() %>% # Puth into a data frame
  plot_time_series_cv_plan(date,sales, .interactive = FALSE)

es_model <- exp_smoothing() %>%
  set_engine("ets") %>%
  fit(sales~date, data = training(cv_split))

# cross-validate to tune model
cv_results <- modeltime_calibrate(es_model,
                                  new_data = testing(cv_split))

# visualize CV results
p1 <- cv_results %>%
  modeltime_forecast(
    new_data = testing(cv_split),
    actual_data = storeItem
  ) %>%
  plot_modeltime_forecast(.interactive = TRUE)

# evaluate the accuracy
cv_results %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = FALSE
  )

# refit to all data then forecast
es_fullfit <- cv_results %>%
  modeltime_refit(data = storeItem)

es_preds <- es_fullfit %>%
  modeltime_forecast(h = "3 months") %>%
  rename(date = .index, sales = .value) %>%
  select(date,sales) %>%
  full_join(., y = item.test, by = "date") %>%
  select(id, sales)

p2 <- es_fullfit %>%
  modeltime_forecast(h = "3 months", actual_data = storeItem) %>%
  plot_modeltime_forecast(.interactive = FALSE)




cv_split2 <- time_series_split(storeItem2, assess = "3 months", cumulative = TRUE)
cv_split2 %>%
  tk_time_series_cv_plan() %>% # Puth into a data frame
  plot_time_series_cv_plan(date,sales, .interactive = FALSE)

es_model2 <- exp_smoothing() %>%
  set_engine("ets") %>%
  fit(sales~date, data = training(cv_split2))

# cross-validate to tune model
cv_results2 <- modeltime_calibrate(es_model2,
                                  new_data = testing(cv_split2))

# visualize CV results
p3 <- cv_results2 %>%
  modeltime_forecast(
    new_data = testing(cv_split2),
    actual_data = storeItem2
  ) %>%
  plot_modeltime_forecast(.interactive = TRUE)

# evaluate the accuracy
cv_results2 %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = FALSE
  )

# refit to all data then forecast
es_fullfit2 <- cv_results2 %>%
  modeltime_refit(data = storeItem2)

es_preds2 <- es_fullfit %>%
  modeltime_forecast(h = "3 months") %>%
  rename(date = .index, sales = .value) %>%
  select(date,sales) %>%
  full_join(., y = item.test, by = "date") %>%
  select(id, sales)

p4 <- es_fullfit2 %>%
  modeltime_forecast(h = "3 months", actual_data = storeItem2) %>%
  plot_modeltime_forecast(.interactive = FALSE)


plotly::subplot(p1,p3,p2,p4, nrows = 2)

# ARIMA

storeItem_train <- item.train %>%
  filter(store == 5, item == 38)
storeItem_test <- item.test %>%
  filter(store == 5, item == 38)

arima_recipe <- recipe(sales~date , data = storeItem) %>%
  step_date(date, features = c("dow","month","decimal"))

arima_model <- arima_reg(seasonal_period = 365,
                         non_seasonal_ar = 5, # default max p to tune
                         non_seasonal_ma = 5, # default max q to tune
                         seasonal_ar = 2, # default max P to tune
                         seasonal_ma = 2, # default max Q to tune
                         non_seasonal_differences = 2, # default max D to tune
                         seasonal_differences = 2) %>% # default max D to tune
  set_engine("auto_arima")

arima_wf <- workflow() %>%
  add_recipe(arima_recipe) %>%
  add_model(arima_model) %>%
  fit(data = training(cv_split))

cv_results <- modeltime_calibrate(arima_wf,
                                  new_data = testing(cv_split))

p1 <- cv_results %>%
  modeltime_forecast(
    new_data = testing(cv_split),
    actual_data = storeItem
  ) %>%
  plot_modeltime_forecast(.interactive = TRUE)

# evaluate the accuracy
cv_results %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = FALSE
  )

# refit to all data then forecast
es_fullfit <- cv_results %>%
  modeltime_refit(data = storeItem_train)

es_preds <- es_fullfit %>%
  modeltime_forecast(h = "3 months") %>%
  rename(date = .index, sales = .value) %>%
  select(date,sales) %>%
  full_join(., y = item.test, by = "date") %>%
  select(id, sales)

p2 <- es_fullfit %>%
  modeltime_forecast(h = "3 months", actual_data = storeItem) %>%
  plot_modeltime_forecast(.interactive = FALSE)


cv_results2 <- modeltime_calibrate(arima_wf,
                                  new_data = testing(cv_split2))

p3 <- cv_results2 %>%
  modeltime_forecast(
    new_data = testing(cv_split2),
    actual_data = storeItem2
  ) %>%
  plot_modeltime_forecast(.interactive = TRUE)

# evaluate the accuracy
cv_results2 %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = FALSE
  )

# refit to all data then forecast
es_fullfit <- cv_results2 %>%
  modeltime_refit(data = storeItem2)

es_preds <- es_fullfit %>%
  modeltime_forecast(h = "3 months") %>%
  rename(date = .index, sales = .value) %>%
  select(date,sales) %>%
  full_join(., y = item.test, by = "date") %>%
  select(id, sales)

p4 <- es_fullfit %>%
  modeltime_forecast(h = "3 months", actual_data = storeItem) %>%
  plot_modeltime_forecast(.interactive = FALSE)

plotly::subplot(p1,p3,p2,p4, nrows = 2)

# prophet model

prophet_model <- prophet_reg() %>%
  set_engine(engine = "prophet") %>%
  fit(sales ~ date, data = training(cv_split))

## calibrate workflow

cv_results <- modeltime_calibrate(prophet_model,
                                  new_data = testing(cv_split))

# visualize CV results
p1 <- cv_results %>%
  modeltime_forecast(
    new_data = testing(cv_split),
    actual_data = storeItem
  ) %>%
  plot_modeltime_forecast(.interactive = TRUE)

# evaluate the accuracy
cv_results %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = FALSE
  )

# refit to all data then forecast
prophet_fullfit <- cv_results %>%
  modeltime_refit(data = storeItem)

prophet_preds <- prophet_fullfit %>%
  modeltime_forecast(h = "3 months") %>%
  rename(date = .index, sales = .value) %>%
  select(date,sales) %>%
  full_join(., y = item.test, by = "date") %>%
  select(id, sales)

p2 <- prophet_fullfit %>%
  modeltime_forecast(h = "3 months", actual_data = storeItem) %>%
  plot_modeltime_forecast(.interactive = FALSE)


cv_results2 <- modeltime_calibrate(prophet_model,
                                  new_data = testing(cv_split2))

# visualize CV results
p3 <- cv_results2 %>%
  modeltime_forecast(
    new_data = testing(cv_split2),
    actual_data = storeItem2
  ) %>%
  plot_modeltime_forecast(.interactive = TRUE)

# evaluate the accuracy
cv_results2 %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = FALSE
  )

# refit to all data then forecast
prophet_fullfit2 <- cv_results2 %>%
  modeltime_refit(data = storeItem2)

prophet_preds2 <- prophet_fullfit2 %>%
  modeltime_forecast(h = "3 months") %>%
  rename(date = .index, sales = .value) %>%
  select(date,sales) %>%
  full_join(., y = item.test, by = "date") %>%
  select(id, sales)

p4 <- prophet_fullfit2 %>%
  modeltime_forecast(h = "3 months", actual_data = storeItem2) %>%
  plot_modeltime_forecast(.interactive = FALSE)

plotly::subplot(p1,p3,p2,p4, nrows = 2)


# final model

# read in data
item <- vroom::vroom("demandtrain.csv")
itemTest <- vroom::vroom("demandtest.csv")
n.stores <- max(item$store)
n.items <- max(item$item)

# model
item_recipe <- recipe(sales~., data=item) %>%
  step_date(date, features=c("dow", "month", "decimal", "doy", "year")) %>%
  step_range(date_doy, min=0, max=pi) %>%
  step_mutate(sinDOY=sin(date_doy), cosDOY=cos(date_doy)) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(sales)) %>%
  step_rm(date, item, store) %>%
  step_normalize(all_numeric_predictors())
boosted_model <- boost_tree(tree_depth=2, #Determined by random store-item combos
                            trees=1200,
                            learn_rate=0.01) %>%
  set_engine("lightgbm") %>%
  set_mode("regression")
boost_wf <- workflow() %>%
  add_recipe(item_recipe) %>%
  add_model(boosted_model)

## Double Loop over all store-item combos
for(s in 1:n.stores){
  for(i in 1:n.items){
    
    ## Subset the data
    train <- item %>%
      filter(store==s, item==i)
    test <- itemTest %>%
      filter(store==s, item==i)
    
    ## Fit the data and forecast
    fitted_wf <- boost_wf %>%
      fit(data=train)
    preds <- predict(fitted_wf, new_data=test) %>%
      bind_cols(test) %>%
      rename(sales=.pred) %>%
      select(id, sales)
    
    ## Save the results
    if(s==1 && i==1){
      all_preds <- preds
    } else {
      all_preds <- bind_rows(all_preds,
                             preds)
    }
    
  }
}

vroom_write(x=all_preds, "./submission.csv", delim=",")
