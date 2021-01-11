## Wilson Tan
## HarvardX: PH125.9x - Data Science: Capstone
## Soccer - Goal Difference Prediction
## https://github.com/wilsontanws14

###############
# Preparation
###############

# Install the required libraries if not already present
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")

# Load installed libraries
library(tidyverse)
library(caret)
library(data.table)
library(recosystem)

# Access to soccer dataset
# Credits to SAIF UDDIN (Source: football-data.co.uk)
soccer <- read.csv("soccer.csv") 

head(soccer)

##################
# Data Processing
##################
soccer <- 
  soccer %>%
  mutate(Match = paste(HomeTeam, AwayTeam, sep = "-", collapse = NULL),
         # assigning indices to the HomeTeam and AwayTeam
         HomeTeamId = group_indices(soccer, .dots="HomeTeam"),
         AwayTeamId = group_indices(soccer, .dots="AwayTeam"),
         goal = FTHG + FTAG,
         # goal difference of the specific match (target value that the model will be predicting)
         goal_diff = FTHG - FTAG,
         # assigning points to HTFormPtsStr and ATFormPtsStr (3 points for W, 1 point for D, 0 point for L)
         Form_HTMatch = str_count(HTFormPtsStr,"W") + str_count(HTFormPtsStr, "D") + str_count(HTFormPtsStr, "L"),
         Form_ATMatch = str_count(HTFormPtsStr,"W") + str_count(HTFormPtsStr, "D") + str_count(HTFormPtsStr, "L"),
         Form_HTPts = 3 * str_count(HTFormPtsStr,"W") + 1 * str_count(HTFormPtsStr, "D"),
         Form_ATPts = 3 * str_count(ATFormPtsStr,"W") + 1 * str_count(ATFormPtsStr, "D"),
         Form_HT = round(if_else(Form_HTMatch <= 2, 0, Form_HTPts / Form_HTMatch), digits = 1),
         Form_AT = round(if_else(Form_ATMatch <= 2, 0, Form_ATPts / Form_ATMatch), digits = 1),
         # compare the form (in terms of points) between the two teams
         PD = round(Form_HT - Form_AT, 1),
         # converting the continuous form (points) to discrete categories
         PD_Category = case_when( 
           PD >= 2.5 ~ "7 - Best Form",
           PD < 2.5 & PD >= 1.5 ~ "6 - Better Form",
           PD < 1.5 & PD >= 0.5 ~ "5 - Good Form",
           PD < 0.5 & PD > -0.5 ~ "4 - Neutral",
           PD <= -0.5 & PD > -1.5 ~ "3 - Poor Form",
           PD <= -1.5 & PD > -2.5 ~ "2 - Worse Form",
           PD <= -2.5 ~ "1 - Worst Form"),
         PD_Cat_Abb = case_when( 
           PD >= 2.5 ~ "P7",
           PD < 2.5 & PD >= 1.5 ~ "P6",
           PD < 1.5 & PD >= 0.5 ~ "P5",
           PD < 0.5 & PD > -0.5 ~ "P4",
           PD <= -0.5 & PD >= -1.5 ~ "P3",
           PD <= -1.5 & PD >= -2.5 ~ "P2",
           PD <= -2.5 ~ "P1"),
         # compare the form (in terms of goals scored and conceded) between the two teams
         GD = HTGD - ATGD,
         # converting the continuous form (goals) to discrete categories
         GD_Category = case_when( 
           GD >= 3.5 ~ "9 - Overwhelming Advantage",
           GD < 3.5 & GD >= 2.5 ~ "8 - Huge Advantage",
           GD < 2.5 & GD >= 1.5 ~ "7 - Advantage",
           GD < 1.5 & GD >= 0.5 ~ "6 - Slight Advantage",
           GD < 0.5 & GD > -0.5 ~ "5 - Neutral",
           GD <= -0.5 & GD > -1.5 ~ "4 - Slight Disadvantage",
           GD <= -1.5 & GD > -2.5 ~ "3 - Disadvantage",
           GD <= -2.5 & GD > -3.5 ~ "2 - Huge Disadvantage",
           GD <= -3.5 ~ "1 - Overwhelming Disadvantage"),
         GD_Cat_Abb = case_when( 
           GD >= 3.5 ~ "G9",
           GD < 3.5 & GD >= 2.5 ~ "G8",
           GD < 2.5 & GD >= 1.5 ~ "G7",
           GD < 1.5 & GD >= 0.5 ~ "G6",
           GD < 0.5 & GD > -0.5 ~ "G5",
           GD <= -0.5 & GD > -1.5 ~ "G4",
           GD <= -1.5 & GD > -2.5 ~ "G3",
           GD <= -2.5 & GD > -3.5 ~ "G2",
           GD <= -3.5 ~ "G1"),
         GDPD = paste(GD_Cat_Abb, PD_Cat_Abb, sep=""),
         # extracting month and year from Date
         Month = month(Date),
         Year = year(Date)) %>%
  select(MatchId, Match, 
         Date, Year, Month, 
         HomeTeamId, HomeTeam, AwayTeamId, AwayTeam, 
         FTR, FTHG, FTAG, goal, goal_diff, 
         GD, GD_Category, GD_Cat_Abb, PD, PD_Category, PD_Cat_Abb, GDPD)

head(soccer)

##################################
# Create model set, validation set
##################################

# Create validation set which will be set to 10% of soccer data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = soccer$goal, times = 1, p = 0.1, list = FALSE)
model <- soccer[-test_index,]
temp <- soccer[test_index,]

# Make sure HomeTeam, AwayTeam and GDPD in validation set are also in model set
validation <- temp %>% 
  semi_join(model, by = "HomeTeam") %>%
  semi_join(model, by = "AwayTeam") %>%
  semi_join(model, by = "GDPD")

# Add rows removed from validation set back into model set
removed <- anti_join(temp, validation)
model <- rbind(model, removed)

########################
# Create train-test set
########################

# Further splitting the model set to train and test sets
set.seed(1, sample.kind="Rounding")

# Create test set which will be set to 10% of model set
test_index <- createDataPartition(y = model$goal, times = 1, p = 0.1, list = FALSE)
train <- model[-test_index,]
temp <- model[test_index,]

# Make sure HomeTeam, AwayTeam and GDPD in test set are also in train set
test <- temp %>% 
  semi_join(train, by = "HomeTeam") %>%
  semi_join(train, by = "AwayTeam") %>%
  semi_join(train, by = "GDPD")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test)
train <- rbind(train, removed)

# Remove unnecessary datasets
rm(removed, temp, test_index) 

################
# Data Analysis
################

# Head
head(model)
head(validation)

# Summary
summary(model)
sapply(model, class)
mean(model$goal_diff)

# Distribution of goal_diff by match
model %>%
  ggplot(aes(goal_diff)) +
  geom_histogram(bins = 12, color = "orange") + 
  labs(x = "Matchday Goal Difference", 
       y = "Number of Matches", 
       title = "Histogram - Matches distribution through Matchday Goal Difference")

# Conclusion: The histogram is a unimodal distribution with a single peak at 0 matchday goal difference. Another point to note is that 
# the distribution is skewed left, meaning the majority of the observations above 0, with only a handful of observations  
# being much larger than the rest. 

# Distribution of goal difference in Typical Matchups (HomeTeam VS AwayTeam)

p <- model %>%
  group_by(Match) %>%
  summarize(HomeTeam = HomeTeam,
            AwayTeam = AwayTeam,
            goal_diff = mean(goal_diff))


p <- distinct(p)

p %>% ggplot(aes(HomeTeam, 
                 AwayTeam, 
                 fill = goal_diff)) +
  geom_tile() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.3)) +
  scale_fill_gradient2(low = "red",
                       high = "green", 
                       midpoint = 0) +
  labs(title = "Heat Map - Avg. No. of Goals Scored in Typical Matchups")

# Conclusion: From the heat map distributions, it can be observed that certain matchups tend to have more goals then the others.
# One major contributing factor is the difference in strengths between the HomeTeam and the AwayTeam. To determine this 
# difference in strengths, it is necessary to take a closer look at goals conceded and goals scored by the individual team 
# to better understand the significance of the strength of a team

# Distribution of goals scored and goals conceded based on HomeTeam

model %>% 
  group_by(HomeTeam) %>%
  summarize(FTHG = mean(FTHG),
            FTAG = mean(FTAG)) %>%
  ggplot(aes(FTAG, 
             FTHG, 
             label = HomeTeam,
             color = HomeTeam)) +
  geom_label(size=2.5, label.padding = unit(0.1, "lines")) +
  theme(legend.position = "none") +
  labs(x = "Number of Goals Conceded in Home Games",
       y = "Number of Goals Scored in Home Games",
       title = "Distribution - Avg Goals Scored VS Avg Goals Conceded by Home Team")

# Distribution of goals scored and goals conceded based on AwayTeam

model %>% 
  group_by(AwayTeam) %>%
  summarize(FTAG = mean(FTAG),
            FTHG = mean(FTHG)) %>%
  ggplot(aes(FTHG, 
             FTAG, 
             label = AwayTeam,
             color = AwayTeam)) +
  geom_label(size=2.5, label.padding = unit(0.1, "lines")) +
  theme(legend.position = "none") +
  labs(x = "Number of Goals Conceded in Away Games",
       y = "Number of Goals Scored in Away Games",
       title = "Distribution - Avg Goals Scored VS Avg Goals Conceded by Away Team")

# Conclusion: Other than the few outliers, there is a general inverse relationship between Goals Scored and Goals Conceded at 
# Home Ground. Normally, stronger teams tend to score more and concede less when playing in their home ground whereas weaker
# teams tend to find it more difficult to score and concede more. 

# Distribution of goal_diff by HomeTeamGD and AwayTeamGD
model %>%
  ggplot(aes(GD, 
             goal_diff,
             fill = count)) +
  stat_bin2d(aes(fill = after_stat(count)), 
             binwidth = c(0.5,1)) +
  scale_fill_gradient(low = "grey",
                      high = "blue") + 
  labs(x = "Form - Normalised Goal Difference (Leading to Game)",
       y = "Matchday Goal Difference",
       title = "Heat Map - Matchday Goal Difference VS Form (Normalised Goal Difference)")


# Conclusion: Normalized Goal Difference (GD) is determined by taking the difference between the normalized goal difference of
# HomeTeam and AwayTeam, which are in turn calculated from the Number of Goals Scored and Conceded by the respective team 
# throughout the season. From the tile plot, GD appears to have a positive correlation with the goal_diff of a match. Across 
# the season, there could be many unique scenarios that could result in very distinct values of GD in a matchup. As such, 
# it is expected that certain GDs calculated in the model dataset may be unrepeatable (distinct) in the validation dataset 
# and vice versa. Therefore, we would propose convert this CONTINUOUS value to a discrete categorical value using a range 
# to avoid any errors in subsequent modeling (included in data processing). 

max(model$GD)
min(model$GD)

# Distribution of goal_diff by GD Category
model %>% 
  group_by(GD_Category) %>%
  summarise(count = n()) %>%
  ggplot(aes(count, GD_Category)) +
  geom_col(color = "blue") + 
  labs(x = "Number of Matches",
       y = "Form - Normalised Goal Difference Category",
       title = "Distribution - Form (Normalised Goal Difference Category) VS Number of Matches")

model %>%
  ggplot(aes(GD_Category, goal_diff)) + 
  geom_boxplot(color = "blue") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Form - Normalised Goal Difference Category",
       y = "Matchday Goal Difference",
       title = "Box Plot - Matchday Goal Difference VS Form (Normalised Goal Difference Category)")

# The distribution of these categories are summarized in the following charts. These observations are aligned with the initial 
# understanding of the relationship between GD and matchday goal difference. 

# Distribution of goal_diff by Form Difference (PD)
model %>%
  ggplot(aes(PD, 
             goal_diff,
             fill = count)) +
  stat_bin2d(aes(fill = after_stat(count)), 
             binwidth = c(1/2, 1)) +
  scale_fill_gradient(low = "grey",
                      high = "orange") + 
  labs(x = "Form - Normalised Point Difference (Leading to Game)",
       y = "Matchday Goal Difference",
       title = "Heat Map - Matchday Goal Difference VS Form (Normalised Point Difference)")

# Conclusion: Similar trend as observed in the heat map between goal diff and GD. 

model %>% 
  group_by(PD_Category) %>%
  summarise(count = n()) %>%
  ggplot(aes(count, PD_Category)) +
  geom_col(color = "orange") + 
  labs(x = "Number of Matches",
       y = "Form - Normalised Point Difference Category",
       title = "Distribution - Form (Normalised Point Difference Category) VS Number of Matches")

model %>%
  ggplot(aes(PD_Category, goal_diff)) + 
  geom_boxplot(color = "orange") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Form - Normalised Point Difference Category",
       y = "Matchday Goal Difference",
       title = "Box Plot - Matchday Goal Difference VS Form (Normalised Point Difference Category)")

# Comment on the range

# Combining GD and PD effect

# Facet Grid - goal_diff VS Form (GD + PD)
p <- model %>% 
  ggplot(aes(PD_Cat_Abb, goal_diff)) + 
  geom_count()

p + facet_grid(. ~ GD_Cat_Abb) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
  labs(x = "Form - Goal Difference + Point Difference Category",
       y = "Matchday Goal Difference",
       title = "Facet Grid - Matchday Goal Difference VS Form (Goal Difference + Point Difference Category)")

# Distribution - Average goal_diff VS Form (GDPD)
model %>%
  group_by(GDPD) %>%
  summarize(goal_diff = mean(goal_diff),
            count = n()) %>%
  ggplot(aes(GDPD, 
             goal_diff, 
             fill = count)) + 
  geom_col() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
  labs(x = "Form - Combined Goal Difference + Point Difference Category",
       y = "Average Matchday Goal Difference",
       title = "Distribution - Average Matchday Goal Difference VS Form (Combined GDPD Category)")

# Box Plot - goal_diff VS Form (GDPD)
model %>% 
  ggplot(aes(GDPD, goal_diff, fill = GDPD)) +
  geom_boxplot(alpha = 0.3) + 
  stat_summary(fun = mean, geom = "point", shape = 23, size = 2) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        legend.position = "None") +
  labs(x = "Form - Combined Goal Difference + Point Difference Category",
       y = "Matchday Goal Difference",
       title = "Box Plot - Matchday Goal Difference VS Form (Combined Goal Difference + Point Difference Category)")

# Distribution of goal by Year & Month
p <- model %>% 
  mutate(Month = case_when( 
    Month == 8 ~ "01 - Aug",
    Month == 9 ~ "02 - Sep",
    Month == 10 ~ "03 - Oct",
    Month == 11 ~ "04 - Nov",
    Month == 12 ~ "05 - Dec",
    Month == 1 ~ "06 - Jan",
    Month == 2 ~ "07 - Feb",
    Month == 3 ~ "08 - Mar",
    Month == 4 ~ "09 - Apr",
    Month == 5 ~ "10 - May")) 

p %>% group_by(Month) %>%
  summarize(goal_diff = mean(goal_diff)) %>%
  ggplot(aes(Month, goal_diff)) +
  geom_col(color = "green") +
  labs(x = "Months (Season starts in Aug)",
       y = "Average Matchday Goal Difference",
       title = "Distribution - Avg. Matchday Goal Difference VS Month")



model %>% 
  group_by(Year) %>%
  summarize(goal_diff = mean(goal_diff)) %>%
  ggplot(aes(Year, goal_diff)) +
  geom_col(color = "green") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.3)) +
  labs(x = "Years",
       y = "Average Matchday Goal Difference",
       title = "Distribution - Avg. Matchday Goal Difference VS Year")

#####################
# Modeling Approach
#####################

## Naive Baseline Model (Simple Average)##

# Compute the dataset's mean rating
mu <- mean(train$goal_diff)

# Test results based on simple prediction
rmse_baseline <- RMSE(test$goal_diff, mu)
accuracy_baseline <- mean(if_else(mu > 0.5, "H", "NH") == test$FTR)

# Check results
# Save prediction in data frame
rmse_results <- data_frame(Model = "[Test] Naive Baseline (Mean) Model", 
                           RMSE = rmse_baseline,
                           Accuracy = accuracy_baseline)
rmse_results %>% knitr::kable()


## Matchup Effect Model ##

# Model taking into account the HomeTeam effect, b_h
b_h <- train %>%
  group_by(HomeTeam) %>%
  summarize(b_h = mean(goal_diff - mu))

# Model taking into account the AwayTeam effect, b_a
b_a <- train %>%
  left_join(b_h, by='HomeTeam') %>%
  group_by(AwayTeam) %>%
  summarize(b_a = mean(goal_diff - mu - b_h))

# predict all unknown goal_diff with mu, b_h and b_a
predicted_goaldiff <- test %>% 
  left_join(b_h, by='HomeTeam') %>%
  left_join(b_a, by='AwayTeam') %>%
  mutate(pred = mu + b_h + b_a) %>%
  pull(pred)

# calculate RMSE of Matchup effect
rmse_matchupeffect <- RMSE(test$goal_diff, predicted_goaldiff)

# calculate Accuracy of Matchup effect
accuracy_matchupeffect <- mean(if_else(predicted_goaldiff > 0.5, "H", "NH") == test$FTR)

# Test and save rmse results 
rmse_results <- bind_rows(rmse_results,
                          data_frame(Model="[Test] MatchUp Effect Model",  
                                     RMSE = rmse_matchupeffect,
                                     Accuracy = accuracy_matchupeffect))

# Consolidate results
rmse_results %>% knitr::kable()

## Matchup & GDPD Model ##

# Model taking into account the GDPD effect, b_f
b_f <- train %>%
  left_join(b_h, by='HomeTeam') %>%
  left_join(b_a, by='AwayTeam') %>%
  group_by(GDPD) %>%
  summarize(b_f = mean(goal_diff - mu - b_h - b_a))

# predict all unknown goals with mu, b_h,  b_a and b_f
predicted_goaldiff <- test %>% 
  left_join(b_h, by='HomeTeam') %>%
  left_join(b_a, by='AwayTeam') %>%
  left_join(b_f, by="GDPD") %>%
  mutate(pred = mu + b_h + b_a + b_f) %>%
  pull(pred)

# calculate RMSE of Matchup + GDPD effect
rmse_matchupXGDPDeffect <- RMSE(test$goal_diff, predicted_goaldiff)

# calculate Accuracy of Matchup + GDPD effect
accuracy_matchupXGDPDeffect <- mean(if_else(predicted_goaldiff > 0.5, "H", "NH") == test$FTR)

# Test and save rmse results 
rmse_results <- bind_rows(rmse_results,
                          data_frame(Model="[Test] MatchUp & Form Effect Model",  
                                     RMSE = rmse_matchupXGDPDeffect, 
                                     Accuracy = accuracy_matchupXGDPDeffect))
# Consolidate results
rmse_results %>% knitr::kable()

## Matchup & GDPD & Month Model ##

# Model taking into account the Month effect, b_m
b_m <- train %>%
  left_join(b_h, by='HomeTeam') %>%
  left_join(b_a, by='AwayTeam') %>%
  left_join(b_f, by='GDPD') %>%
  group_by(Month) %>%
  summarize(b_m = mean(goal_diff - mu - b_h - b_a - b_f))

# predict all unknown goals with mu, b_h, b_a, b_f and b_m
predicted_goaldiff <- test %>% 
  left_join(b_h, by='HomeTeam') %>%
  left_join(b_a, by='AwayTeam') %>%
  left_join(b_f, by="GDPD") %>%
  left_join(b_m, by='Month') %>%
  mutate(pred = mu + b_h + b_a + b_f + b_m) %>%
  pull(pred)

# calculate RMSE of Matchup & GDPD & Month effect
rmse_matchupXGDPDXmontheffect <- RMSE(test$goal_diff, predicted_goaldiff)

# calculate Accuracy of Matchup & GDPD & Month effect
accuracy_matchupXGDPDXmontheffect <- mean(if_else(predicted_goaldiff > 0.5, "H", "NH") == test$FTR)

# Test and save rmse results 
rmse_results <- bind_rows(rmse_results,
                          data_frame(Model="[Test] MatchUp & GDPD & Month Effect Model",  
                                     RMSE = rmse_matchupXGDPDXmontheffect, 
                                     Accuracy = accuracy_matchupXGDPDXmontheffect))
# Consolidate results
rmse_results %>% knitr::kable()


## Matchup & GDPD Effect + Regularization Model

# The use of regularization penalizes on matchups with very few occurences. 
# The tuning parameter, lambda, resulting in the smallest RMSE will be used to shrink the HomeTeam, AwayTeam and Form effect 
# for the test set. 

# Determining the lambda with the lowest RMSE
lambdas <- seq(from=80, to=140, by=0.5)

# output RMSE of each lambda, repeat earlier steps (with regularization)
rmses <- sapply(lambdas, function(l){
  # calculate average goal_diff across training data
  mu <- mean(train$goal_diff)
  # compute regularized HomeTeam bias term
  b_h <- train %>% 
    group_by(HomeTeam) %>%
    summarize(b_h = sum(goal_diff - mu)/(n()+l))
  # compute regularize AwayTeam bias term
  b_a <- train %>% 
    left_join(b_h, by="HomeTeam") %>%
    group_by(AwayTeam) %>%
    summarize(b_a = sum(goal_diff - b_h - mu)/(n()+l))
  # compute regularize GDPD bias term
  b_f <- train %>%
    left_join(b_h, by='HomeTeam') %>%
    left_join(b_a, by='AwayTeam') %>%
    group_by(GDPD) %>%
    summarize(b_f = sum(goal_diff - b_h - b_a - mu)/(n()+l))
  # compute predictions on test set based on these above terms
  predicted_goaldiff <- test %>% 
    left_join(b_h, by = "HomeTeam") %>%
    left_join(b_a, by = "AwayTeam") %>%
    left_join(b_f, by = "GDPD") %>%
    mutate(pred = mu + b_h + b_a + b_f) %>%
    pull(pred)
  # output RMSE of these predictions
  return(RMSE(predicted_goaldiff, test$goal_diff))
})

# quick plot of RMSE vs lambdas
qplot(lambdas, rmses)
# print minimum RMSE 
min(rmses)

# The linear model with the minimizing lambda
lam <- lambdas[which.min(rmses)]
lam

# compute regularize HomeTeam bias term
b_h <- train %>% 
  group_by(HomeTeam) %>%
  summarize(b_h = sum(goal_diff - mu)/(n()+lam))
# compute regularize AwayTeam bias term
b_a <- train %>% 
  left_join(b_h, by="HomeTeam") %>%
  group_by(AwayTeam) %>%
  summarize(b_a = sum(goal_diff - mu - b_h)/(n()+lam))
# compute regularize AwayTeam bias term
b_f <- train %>%
  left_join(b_h, by='HomeTeam') %>%
  left_join(b_a, by='AwayTeam') %>%
  group_by(GDPD) %>%
  summarize(b_f = sum(goal_diff - b_h - b_a - mu)/(n()+lam))
# compute predictions on test set based on these above terms
predicted_goaldiff <- test %>% 
  left_join(b_h, by = "HomeTeam") %>%
  left_join(b_a, by = "AwayTeam") %>%
  left_join(b_f, by = "GDPD") %>%
  mutate(pred = mu + b_h + b_a + b_f) %>%
  pull(pred)
# output RMSE of these predictions
rmse_regularizedXmatchupXGDPDeffect <- RMSE(predicted_goaldiff, test$goal_diff)

# calculate Accuracy of these predictions
accuracy_regularizeXmatchupXGDPDeffect <- mean(if_else(predicted_goaldiff > 0.5, "H", "NH") == test$FTR)

# Test and save RMSE results 
rmse_results <- bind_rows(rmse_results,
                          data_frame(Model="[Test] Matchup & Form Effect + Regularization Model",  
                                     RMSE = rmse_regularizedXmatchupXGDPDeffect,
                                     Accuracy = accuracy_regularizeXmatchupXGDPDeffect))

# Consolidate results
rmse_results %>% knitr::kable()

## Final Model - Matchup + GDPD Effect + Regularization Model

# Matchup & GDPD Effect + Regularization Model, with the lowest RMSE, is selected as the Final Model to run on the validation 
# set. 

# compute regularize HomeTeam bias term
b_h <- model %>% 
  group_by(HomeTeam) %>%
  summarize(b_h = sum(goal_diff - mu)/(n()+lam))
# compute regularize AwayTeam bias term
b_a <- model %>% 
  left_join(b_h, by="HomeTeam") %>%
  group_by(AwayTeam) %>%
  summarize(b_a = sum(goal_diff - mu - b_h)/(n()+lam))
# compute regularize GDPD bias term
b_f <- model %>%
  left_join(b_h, by='HomeTeam') %>%
  left_join(b_a, by='AwayTeam') %>%
  group_by(GDPD) %>%
  summarize(b_f = sum(goal_diff - b_h - b_a - mu)/(n()+lam))
# compute predictions on test set based on these above terms
predicted_goaldiff <- validation %>% 
  left_join(b_h, by = "HomeTeam") %>%
  left_join(b_a, by = "AwayTeam") %>%
  left_join(b_f, by = "GDPD") %>%
  mutate(pred = mu + b_h + b_a + b_f) %>%
  pull(pred)
# output RMSE of these predictions
rmse_regularizedXmatchupXGDPDeffect <- RMSE(predicted_goaldiff, validation$goal_diff)

# calculate Accuracy of these predictions
accuracy_regularizeXmatchupXGDPDeffect <- mean(if_else(predicted_goaldiff > 0.5, "H", "NH") == validation$FTR)

# Test and save RMSE results 
rmse_results <- bind_rows(rmse_results,
                          data_frame(Model="[Validation] Matchup & Form Effect + Regularization Model",  
                                     RMSE = rmse_regularizedXmatchupXGDPDeffect,
                                     Accuracy = accuracy_regularizeXmatchupXGDPDeffect))

# Consolidate results
rmse_results %>% knitr::kable()

