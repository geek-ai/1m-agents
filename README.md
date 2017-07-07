# A preliminary platform for up to 1 million reinforcement learning agents

Our goal is to provide a **Multi-Agent Reinforcement Learning** plateform for up to 1 million agents. 

Now the whole plateform is still on working. We provide two specific setting for demo.

You could see two directories `Population Dynamics` and `Collective Grouping Behaviors`, which correspond to the two setting in the paper(when arXiv version is available) respectively.

### Dependencies

- [Tensorflow](tensorflow.org)
- [opencv2/3](opencv.org)

### **Population Dynamics** setting

#### Usage
    cd Population Dynamics
    ./train.sh

The log is saved at `Population_dynmacis.log`.

### **Collective Grouping Behaviors** setting

####Usage
    cd Collective Grouping Behaviors
    ./train.sh

The log is saved at `Collective_Grouping_Behaviors.log`.

You could also open the bash file to change the parameters, the list above is the specific explanations of the parameters.

- `random_seed default=10` The random seed of the random number generator for generating the obstacles.
- `width default=1000` The width of the map.
- `height default=1000` The height of the map.
- `batch_size default=32` The batch size of the process of training.
- `view_args defalut=2500-5-5-0,2500-5-5-1,2500-5-5-2,2500-5-5-3` Define the view size and face direction of the agents. Four number in each item means the number of agents that has these property, the left view size, the front view size and the face direction, where 0 means north in the map, 1 means east, 2 means south and 3 means west.
- `agent_number defalut=10000` The initial number of agents.
- `pig_max_number default=5000` The initial number of prey-pig.
- `rabbit_max_number default=3000` The initial number of prey-rabbit.
- `agent_increase_rate default=0.001` The birth rate of the agent.
- `pig_increaserate default=0.001` The birth rate of the prey-pig.
- `rabbit_increase_rate default=0.001` The birth rate of the prey-rabbit.
- `reward_radius_pig default=7` The reward radius threshold of the prey-pig.
- `reward_radius_rabbit default=2` The reward radius threshold of the prey-rabbit.
- `reward_threshold_pig default=3` The reward threshold of the prey-pig.
- `agent_emb_dim default=5` The dimension of the agent embedding.
- `damage_per_step default=0.01` The decrease health of agent per step.
- `model_name default=DNN` The category of the model.
- `model_hidden_size default=32, 32` The units number of each layer.
- `activations default=sigmoid, sigmoid` The activation functions of each layer.
- `view_flat_size defalut=32` The input dimension of the neural network, please see the paper for details.
- `num_actions default=9` The number of actions.
- `reward_decay defalut=0.9` The reward decay in the reinforcement learning.
- `savd_dir default=models` The model save path.
- `load_dir default=None` The model load path.
- `round default=100` The training rounds.
- `time_step defalut=500` The training steps in each round.
- `learning_rate default=0.001` The learning rate of the reinforcement learning.
- `log_file default=log.txt` The log file path.

For more details of the parameters, please refer to the papers.
