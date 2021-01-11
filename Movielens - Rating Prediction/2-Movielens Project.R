## Wilson Tan
## HarvardX: PH125.9x - Data Science: Capstone
## MovieLens Rating Prediction
## https://github.com/wilsontanws14

##############
# Preparation
##############

# Operating System
version

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

# Access to Movielens 10M dataset (http://files.grouplens.org/datasets/movielens/ml-10m.zip)
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

################################
# Create edx set, validation set
################################

# Create validation set which will be set to 10% of Movielens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Remove unnecessary datasets
rm(dl, ratings, movies, test_index, temp, movielens, removed)

################################
# Data Processing
################################

# Extract year and month of reviews from timestamp in both datasets
edx$date <- as.POSIXct(edx$timestamp, origin="1970-01-01")
validation$date <- as.POSIXct(validation$timestamp, origin="1970-01-01")

edx$year <- format(edx$date,"%Y")
edx$month <- format(edx$date,"%m")

validation$year <- format(validation$date,"%Y")
validation$month <- format(validation$date,"%m")

# Extract year and month of movie release from title in both datasets
edx <- edx %>%
  mutate(title = str_trim(title)) %>%
  extract(title,
          c("titleTemp", "release"),
          regex = "^(.*) \\(([0-9 \\-]*)\\)$",
          remove = F) %>%
  mutate(release = if_else(str_length(release) > 4,
                           as.integer(str_split(release, "-",
                                                simplify = T)[1]),
                           as.integer(release))
  ) %>%
  mutate(title = if_else(is.na(titleTemp),
                         title,
                         titleTemp)
  ) %>%
  select(-titleTemp)

validation <- validation %>%
  mutate(title = str_trim(title)) %>%
  extract(title,
          c("titleTemp", "release"),
          regex = "^(.*) \\(([0-9 \\-]*)\\)$",
          remove = F) %>%
  mutate(release = if_else(str_length(release) > 4,
                           as.integer(str_split(release, "-",
                                                simplify = T)[1]),
                           as.integer(release))
  ) %>%
  mutate(title = if_else(is.na(titleTemp),
                         title,
                         titleTemp)
  ) %>%
  select(-titleTemp)

# Splitting movies with multiple genres and store it into a new dataset for genre analysis.
# A separate dataset is used to analyse the significance of genre to the rating as splitting muliple genres within the edx dataset unintentionally duplicate reviews. 
edx_genre <- edx %>%
  mutate(genre = fct_explicit_na(genres,
                                 na_level = "(no genres listed)")
  ) %>%
  separate_rows(genre,
                sep = "\\|")

################################
# Train-Test Split
################################

# Further splitting the edx set to train and test sets
set.seed(1, sample.kind="Rounding")

# Create test set which will be set to 10% of edx set
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in test set are also in train set
test <- temp %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test)
train <- rbind(train, removed)

# Remove unnecessary datasets
rm(removed, temp, test_index) 


################################
# Data Analysis
################################

# Head
head(edx)
head(validation)

# Summary
summary(edx)
sapply(edx, class)

# Distinct movies and users 
edx %>% group_by(movieId) %>% summarize(count = n()) %>% nrow()
edx %>% group_by(userId) %>% summarize(count = n()) %>% nrow()

# Distribution of reviews through rating score
edx %>%
  ggplot(aes(rating)) +
  theme_classic() +
  geom_histogram(binwidth = 0.25) +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
  labs(x = "Star Rating",
       y = "Number of Reviews",
       title = "Reviews distribution through Star Rating")
# Conclusion: Half star ratings are less common than whole star ratings

# Distribution of reviews through movieId

users <- sample(unique(edx$userId), 100)
edx %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="MovieID", ylab="UserID")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")

# Conclusion: There are movies with more reviews than the others.

edx %>% 
  group_by(movieId) %>%
  summarize(count = n()) %>%
  mutate(rank = rank(-count)) %>%
  arrange(rank) %>%
  mutate(cum_count = cumsum(count),
         percent = cum_count/max(cum_count)*100) %>%
  ggplot(aes(rank, percent)) +
  geom_line(color = "blue") + 
  geom_hline(yintercept = 80, linetype = "dashed", color = "red") +
  geom_vline(xintercept = 1700, linetype = "dashed", color = "red") +
  labs(x = "Movies Ranking (by Number of Reviews)",
       y = "Cumulative Percentage",
       title = "Pareto - No. of Reviews VS MovieId") +
  theme(plot.title=element_text(size=10),
        axis.title=element_text(size=8))
# Conclusion: Reviews for 1700 most critic movies (out of 10,677) made up ~80% of the total number of reviews.

# Distribution of Ratings and Number of Reviews through MovieId
p <- edx %>% 
  group_by(movieId) %>%
  summarize(rating = mean(rating),
            review_count = n()) 

p %>%
  ggplot(aes(review_count, rating)) + 
  geom_point(alpha = 0.2, color = "blue") +
  labs(x = "Number of Reviews",
       y = "Average Rating",
       title = "Avg. Rating & No. of Reviews by MovieId") + 
  theme(plot.title=element_text(size=10))


# Conclusion: Most critic movies (with more reviews) tend to have better average rating


# Conclusion: Most critic movies (with more reviews) tend to have better average rating

# Histogram of Number of Reviews VS UserID

edx %>% 
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "orange") + 
  scale_x_log10() + 
  labs(x = "Number of Reviews",
       y = "Number of Users",
       title = "Histogram - Users")

# Distribution of Ratings and Number of Reviews through UserId

p <- edx %>%
  group_by(userId) %>%
  summarize(rating = mean(rating),
            review_count = n())

p %>% 
  ggplot(aes(review_count, rating)) +
  geom_point(alpha = 0.2, color = "orange") + 
  labs(x = "Number of Reviews",
       y = "Average Rating",
       title = "Distribution - Average Rating & Number of Reviews through UserId")

# Conclusion: Certain users rate more often than the others. Average rating from users with more reviews tend to mean. 

# Distribution of reviews and ratings through genre

p <-
  edx_genre %>% 
  group_by(genre) %>%
  summarize(count = n(), median_rating = median(rating), rating = mean(rating), release = median(release)) %>%
  arrange(desc(rating))

p

# From the table, it can be noted that the year of the movie release has an impact to the rating. For example, Film-Noir genre, having a larger proportion of classics (all-time great old movies), tends to receive higher ratings.
# At the same time, more recent technologies like the IMAX has also garnered higher ratings. Therefore it is worthwhile to dive deeper to study if the release year of the movie has an impact on the rating. 

p %>%
  filter(genre != "(no genres listed)") %>%
  ggplot(aes(count, rating,
             label = genre,
             color = genre)) +
  geom_label(size=2, label.padding = unit(0.1, "lines")) +
  theme(legend.position = "None") +
  labs(x = "Number of Reviews",
       y = "Rating",
       title = "Reviews and Ratings distribution through Genre")

# Conclusion: The least-rated genres tend to have a higher rating.

# Distribution of reviews through release year
edx %>% 
  group_by(release) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(release, rating)) +
  geom_point() +
  labs(x = "Year Release",
       y = "Rating",
       title = "Ratings distribution through Year Released")

edx %>%
  group_by(release) %>%
  summarize(rating = mean(rating),
            count = n_distinct(movieId)) %>%
  ggplot(aes(release, count, color = rating)) +
  geom_point() +
  labs(x = "Year Release",
       y = "Number of Movies",
       title = "Movies distribution through Year Released")

# Conclusion: Confirmed that movies released in earlier years are more highly rated than movies released in recent years. As explained, this is because audience tends to re-watch classics and are likely to rate these favorably. 
# A further point to note is that the incorporation of reviews into online platforms only started in 1990s. (The age group of the reviewers may explain the sudden drop in rating between 1980s to 1990s)

# Distribution of reviews through review year and month
edx %>% 
  group_by(month) %>%
  summarize(count = n(), rating = mean(rating)) %>%
  ggplot(aes(month, rating)) + 
  geom_point() +
  labs(x = "Month Reviewed",
       y = "Rating",
       title = "Ratings distribution through Review Month")

edx %>% 
  group_by(year) %>%
  summarize(count = n(), rating = mean(rating)) %>%
  ggplot(aes(year, rating)) + 
  geom_point() +
  labs(x = "Year Reviewed",
       y = "Rating",
       title = "Ratings distribution through Review Year")

# Conclusion: Ratings in Oct, Nov and Dec tend to be higher than the other months. An explanation could be that reviews collected in these 3 months coincided with blockbuster (highly anticipated) movies release [scheduled for the winter holidays].
# As for the year of the review, the significance is inconclusive.

#####################
# Modeling Approach
#####################

## Naive Baseline Model (Simple Average)##

# Compute the dataset's mean rating
mu <- mean(train$rating)

# Test results based on simple prediction
rmse_baseline <- RMSE(test$rating, mu)

# Check results
# Save prediction in data frame
rmse_results <- data_frame(Model = "[Test] Naive Baseline (Mean) Model", RMSE = rmse_baseline)
rmse_results %>% knitr::kable()

## Movie Effect Model ##

# Simple model taking into account the movie effect b_i
b_i <- train %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# predict all unknown ratings with mu and b_i
predicted_ratings <- test %>% 
  left_join(b_i, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

# calculate RMSE of movie ranking effect
rmse_movieeffect <- RMSE(test$rating, predicted_ratings)

# plot the distribution of b_i's
qplot(b_i, data = b_i, bins = 15, color = I("blue"))

# Test and save rmse results 
rmse_results <- bind_rows(rmse_results,
                          data_frame(Model="[Test] Movie Effect Model",  
                                     RMSE = rmse_movieeffect ))
# Consolidate results
rmse_results %>% knitr::kable()

## Movie & User Effect Model ##

b_u <- train %>%
  left_join(b_i, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# plot the distribution of b_u's
b_u %>% qplot(b_u, geom ="histogram", bins = 30, data = ., color = I("orange"))

# predict all unknown ratings with mu, b_i and b_u
predicted_ratings <- test %>%
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# calculate RMSE of movie ranking effect
rmse_userXmovieeffect <- RMSE(test$rating, predicted_ratings)

# Test and save rmse results 
rmse_results <- bind_rows(rmse_results,
                          data_frame(Model="[Test] Movie & User Effect Model",  
                                     RMSE = rmse_userXmovieeffect ))

# Consolidate results
rmse_results %>% knitr::kable()

## Movie & User Effect + Regularization Model ##

# The use of regularization penalizes on movies with very few ratings or users who only rated a very small number of movies. 
# The tuning parameter, lambda, resulting in the smallest RMSE will be used to shrink the movie and user effect for the test set. 

# Determining the lambda with the lowest RMSE
lambdas <- seq(from=0, to=10, by=0.25)

# output RMSE of each lambda, repeat earlier steps (with regularization)
rmses <- sapply(lambdas, function(l){
  # calculate average rating across training data
  mu <- mean(train$rating)
  # compute regularized movie bias term
  b_i <- train %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  # compute regularize user bias term
  b_u <- train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  # compute predictions on test set based on these above terms
  predicted_ratings <- test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  # output RMSE of these predictions
  return(RMSE(predicted_ratings, test$rating))
})
  
# quick plot of RMSE vs lambdas
qplot(lambdas, rmses)
# print minimum RMSE 
min(rmses)


# The linear model with the minimizing lambda
lam <- lambdas[which.min(rmses)]

b_i <- train %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lam))
# compute regularize user bias term
b_u <- train %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n()+lam))
# compute predictions on test set based on these above terms
predicted_ratings <- test %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
# output RMSE of these predictions
rmse_regularizedXuserXmovieeffect <- RMSE(predicted_ratings, test$rating)

# Test and save RMSE results 
rmse_results <- bind_rows(rmse_results,
                          data_frame(Model="[Test] Movie & User Effect + Regularization Model",  
                                     RMSE = rmse_regularizedXuserXmovieeffect ))

# Consolidate results
rmse_results %>% knitr::kable()

# Conclusion: There is only a slight improvement in the regularized model.

## Movie & User Effect + Matrix Factorization Model

# For more info on Matrix Factorization: https://www.youtube.com/watch?v=ZspR5PZemcs 
# compute movie effect without regularization
b_i <- train %>% 
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# compute user effect without regularization
b_u <- train %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - b_i - mu))

# compute residuals 
train <- train %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(res = rating - mu - b_i - b_u)

# compute residuals on test set
test <- test %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(res = rating - mu - b_i - b_u)

# create data saved on disk in 3 columns with no headers
train_data <- data_memory(user_index = train$userId, item_index = train$movieId, 
                        rating = train$res, index1 = T)

test_data <- data_memory(user_index = test$userId, item_index = test$movieId, 
                         index1 = T)

recommender <- Reco()

# This is a randomized algorithm
set.seed(1) 

## Warning!!! This may take up to an hour to run
# call the `$tune()` method to select best tuning parameters
res = recommender$tune(
  train_data,
  opts = list(dim = c(10, 20, 30),
              costp_l1 = 0, costq_l1 = 0,
              lrate = c(0.05, 0.1, 0.2), nthread = 2)
)

# show best tuning parameters
print(res$min)

# Train the recommender model
set.seed(1) 
suppressWarnings(recommender$train(train_data, opts = c(dim = 30, costp_l1 = 0,
                                                      costp_l2 = 0.01, costq_l1 = 0,
                                                      costq_l2 = 0.1, lrate = 0.05,
                                                      verbose = FALSE)))

# Apply model on test set
predicted_ratings <- recommender$predict(test_data, out_memory()) + mu + test$b_i + test$b_u 

# Set rating ceiling at 5 and floor at 0.5 stars
ind <- which(predicted_ratings > 5)
predicted_ratings[ind] <- 5

ind <- which(predicted_ratings < 0.5)
predicted_ratings[ind] <- 0.5

# create a results table with this approach
model_MatrixFactorization <- RMSE(test$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          tibble(Model="[Test] Movie & User Effect + Matrix Factorization Model",  
                                 RMSE = model_MatrixFactorization))
rmse_results %>% knitr::kable()

## (Final) Movie & User Effect + Matrix Factorization Model

# compute movie effect without regularization
b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# compute user effect without regularization
b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - b_i - mu))

# compute residuals 
edx <- edx %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(res = rating - mu - b_i - b_u)

# compute residuals on validation set
validation <- validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(res = rating - mu - b_i - b_u)

# create data saved on disk in 3 columns with no headers
edx_data <- data_memory(user_index = edx$userId, item_index = edx$movieId, 
                          rating = edx$res, index1 = T)

validation_data <- data_memory(user_index = validation$userId, item_index = validation$movieId, index1 = T)

recommender <- Reco()

# This is a randomized algorithm
set.seed(1) 

## Warning!!! This may take up to an hour to run
# call the `$tune()` method to select best tuning parameters
res = recommender$tune(
  edx_data,
  opts = list(dim = c(10, 20, 30),
              costp_l1 = 0, costq_l1 = 0,
              lrate = c(0.05, 0.1, 0.2), nthread = 2)
)

# show best tuning parameters
print(res$min)

# Train the recommender model
set.seed(1) 
suppressWarnings(recommender$train(edx_data, opts = c(dim = 30, costp_l1 = 0,
                                                      costp_l2 = 0.01, costq_l1 = 0,
                                                      costq_l2 = 0.1, lrate = 0.05,
                                                      verbose = FALSE)))

# Apply model on validation set
predicted_ratings <- recommender$predict(validation_data, out_memory()) + mu + validation$b_i + validation$b_u 

# Set rating ceiling at 5 and floor at 0.5 stars
ind <- which(predicted_ratings > 5)
predicted_ratings[ind] <- 5

ind <- which(predicted_ratings < 0.5)
predicted_ratings[ind] <- 0.5

# create a results table with this approach
model_MatrixFactorization <- RMSE(validation$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          tibble(Model="[Validation] Movie & User Effect + Matrix Factorization Model",  
                                 RMSE = model_MatrixFactorization))
rmse_results %>% knitr::kable()

# Summary: The Matrix Factorization model is regarded as the optimal model (characterized by the lowest RMSE value) 
# to use for predicting movie ratings. In the earlier data analysis, we could infer that further improvements can be
# made to the model by adding effects like release year, review month and genre. However, due to the limitations of
# the hardware (RAM), these models cannot be validated. 