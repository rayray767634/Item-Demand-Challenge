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
