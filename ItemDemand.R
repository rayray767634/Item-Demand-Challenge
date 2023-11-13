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

storeItem1 <- item.train %>%
  filter(store == 1) %>%
  filter(item == 24)

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
