# Autonomous Driving Using Deep Q Learning


Detailed information about the project can be found in the article [a link](https://github.com/root999/Autonomous-Driving-Using-Deep-Q-Learning/blob/master/Autonomous%20Driving%20Using%20Deep%20Q%20Learning%20Algorithm%20Article.pdf).

This code developed based on gym-torcs [repo](https://github.com/ugo-nama-kun/gym_torcs). Installation and requriements explained in the repo.

## Running the code

To start training your agent you have to run

``` python DeepQLearning.py```



## Running Prediction without Training the Agent

Agent is driving the car while training. If you trained a model and if you want to drive a car using that model you should

  Change ```LOAD_MODEL = False to True ```in [main function](https://github.com/root999/Autonomous-Driving-Using-Deep-Q-Learning/blob/master/DeepQLearning.py).
  You need to change model_name to name the model that you want to run in [main function](https://github.com/root999/Autonomous-Driving-Using-Deep-Q-Learning/blob/master/DeepQLearning.py). In addition to that, you need to specify the model path both in main function and create_model function.
 
