add_pig_number=500
add_rabbit_number=500
add_every=500

random_seed=10
width=1000
height=1000
batch_size=32
view_args=2500-5-5-0,2500-5-5-1,2500-5-5-2,2500-5-5-3
pig_max_number=5000
pig_min_number=2000
pig_increase_every=1
pig_increase_number=10
pig_increase_policy=1
agent_increase_rate=0.003
pig_increase_rate=0.006
rabbit_increase_rate=0.008
rabbit_max_number=30000
reward_radius_pig=7
reward_threshold_pig=3
reward_radius_rabbit=2
img_length=5
images_dir=images
agent_mortal=1
agent_emb_dim=5
agent_id=1
damage_per_step=0.01

model_name=DNN
model_hidden_size=32,32
activations=sigmoid,sigmoid
view_flat_size=335
num_actions=9
reward_decay=0.9
save_every_round=10
save_dir=models
load_dir=None

video_dir=videos
video_per_round=0
round=100
time_step=500
policy=e_greedy
epsilon=0.1
agent_number=10000
learning_rate=0.001
log_file=log.txt

python main.py --add_pig_number $add_pig_number --add_rabbit_number $add_rabbit_number --add_every $add_every --random_seed $random_seed --width $width --height $height --batch_size $batch_size --view_args $view_args --pig_max_number $pig_max_number --pig_min_number $pig_min_number --pig_increase_every $pig_increase_every --pig_increase_policy $pig_increase_policy --agent_increase_rate $agent_increase_rate --pig_increase_rate $pig_increase_rate --rabbit_increase_rate $rabbit_increase_rate --rabbit_max_number $rabbit_max_number --reward_radius_pig $reward_radius_pig --reward_threshold_pig $reward_threshold_pig --reward_radius_rabbit $reward_radius_rabbit --img_length $img_length --images_dir $images_dir --agent_mortal $agent_mortal --agent_emb_dim $agent_emb_dim --agent_id $agent_id --damage_per_step $damage_per_step --model_name $model_name --model_hidden_size $model_hidden_size --activations $activations --view_flat_size $view_flat_size --num_actions $num_actions --reward_decay $reward_decay --save_every_round $save_every_round --save_dir $save_dir --load_dir $load_dir --video_dir $video_dir --video_per_round $video_per_round --round $round --time_step $time_step --policy $policy --epsilon $epsilon --agent_number $agent_number --learning_rate $learning_rate --log_file $log_file
