random_seed=10
width=10000
height=10000
batch_size=256
view_args=250000-5-5-0,250000-5-5-1,250000-5-5-2,250000-5-5-3
pig_max_number=500000
pig_min_number=200000
pig_increase_every=1
pig_increase_number=10
pig_increase_policy=1
agent_increase_rate=0.004
pig_increase_rate=0.006
reward_radius=5
reward_threshold=3
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
save_every_round=1
save_dir=models
load_dir=models/round_90/model.ckpt

video_dir=videos
video_per_round=0
round=100
time_step=100
policy=e_greedy
epsilon=0.1
agent_number=1000000
learning_rate=0.001
log_file=log.txt

python main.py --random_seed $random_seed --width $width --height $height --batch_size $batch_size --view_args $view_args --pig_max_number $pig_max_number --pig_min_number $pig_min_number --pig_increase_every $pig_increase_every --pig_increase_policy $pig_increase_policy --agent_increase_rate $agent_increase_rate --pig_increase_rate $pig_increase_rate --reward_radius $reward_radius --reward_threshold $reward_threshold  --img_length $img_length --images_dir $images_dir --agent_mortal $agent_mortal --agent_emb_dim $agent_emb_dim --agent_id $agent_id --damage_per_step $damage_per_step --model_name $model_name --model_hidden_size $model_hidden_size --activations $activations --view_flat_size $view_flat_size --num_actions $num_actions --reward_decay $reward_decay --save_every_round $save_every_round --save_dir $save_dir --load_dir $load_dir --video_dir $video_dir --video_per_round $video_per_round --round $round --time_step $time_step --policy $policy --epsilon $epsilon --agent_number $agent_number --learning_rate $learning_rate --log_file $log_file
