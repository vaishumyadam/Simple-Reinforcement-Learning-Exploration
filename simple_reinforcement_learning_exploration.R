## Vaishu Myadam (vmyadam1208@gmail.com)
## June, 2020

## Exploring the reinforcement learning package with grid world and tic tac toe (Q-learning)
## Using official documentation

# Necessary libraries:

library(ReinforcementLearning)

# Tic tac toe:

data("tictactoe")
head(tictactoe, 10)
control <- list(alpha = 0.2, gamma = 0.4, epsilon = 0.1)
model <- ReinforcementLearning(tictactoe, s = "State", a = "Action", r = "Reward", s_new = "NextState", iter = 1, control = control)
results <- computePolicy(model)
head(results)
summary(model)

# Grid world:

states <- c("s1", "s2", "s3", "s4")
actions <- c("up", "down", "left", "right")
envie <- gridworldEnvironment
print(envie)
data <- sampleExperience(N = 1000, env = envie, states = states, actions = actions)
head(data)
control <- list(alpha = 0.1, gamma = 0.1, epsilon = 0.1)
model <- ReinforcementLearning(data, s = "State", a = "Action", r = "Reward", s_new = "NextState", control = control)
computePolicy(model)
summary(model)

# Prediction

data_to_be_predicted <- data.frame(State = c("s1", "s2", "s1"), stringsAsFactors = FALSE)
data_to_be_predicted$OptimalAction <- predict(model, data_to_be_predicted$State)
data_to_be_predicted

# See what happens when you add new data

new_data <- sampleExperience(N = 1000, 
                             env = envie, 
                             states = states, 
                             actions = actions, 
                             actionSelection = "epsilon-greedy",
                             model = model, 
                             control = control)
modified_model <- ReinforcementLearning(new_data, 
                                   s = "State", 
                                   a = "Action", 
                                   r = "Reward", 
                                   s_new = "NextState", 
                                   control = control,
                                   model = model)
plot(modified_model)