0 : leader
1: follower


udevadm info -q all -n /dev/ttyACM0 | egrep 'DEVNAME=|ID_VENDOR_ID=|ID_MODEL_ID=|ID_SERIAL=|ID_SERIAL_SHORT=|ID_MODEL=|ID_VENDOR=|ID_USB_INTERFACE_NUM='


ID_SERIAL_SHORT=685FACDC503059384C2E3120FF070B30

ID_VENDOR_ID=2f5d


udevadm info -q all -n /dev/ttyACM1 | egrep 'DEVNAME=|ID_VENDOR_ID=|ID_MODEL_ID=|ID_SERIAL=|ID_SERIAL_SHORT=|ID_MODEL=|ID_VENDOR=|ID_USB_INTERFACE_NUM='

ID_SERIAL_SHORT=C69AD0E0503059384C2E3120FF08102E
ID_VENDOR_ID=2f5d





# teleop wo cam
lerobot-teleoperate \
--robot.type=omx_follower \
--robot.port=/dev/omx_follower \
--robot.id=omx_follower_arm \
--teleop.type=omx_leader \
--teleop.port=/dev/omx_leader \
--teleop.id=omx_leader_arm

# teleop w cam
lerobot-teleoperate \
--robot.type=omx_follower \
--robot.port=/dev/omx_follower \
--robot.id=omx_follower_arm \
--robot.cameras="{front: {type: opencv, index_or_path: '/dev/video4', width: 640,
height: 480, fps: 30}}" \
--teleop.type=omx_leader \
--teleop.port=/dev/omx_leader \
--teleop.id=omx_leader_arm \
--display_data=true




hf auth login --token [hf_token] --add-to-git-credential

HF_USER=$(hf auth whoami | head -n 1 | sed 's/\x1b\[[0-9;]*m//g' | awk -F': ' '{print $2}' | xargs)

## 데이터 수집 명령어 (예시)
source ~/venv/il/bin/activate
HF_USER=$(hf auth whoami | head -n 1 | sed 's/\x1b\[[0-9;]*m//g' | awk -F': ' '{print $2}' | xargs)

rm -rf ~/.cache/huggingface/lerobot/${HF_USER}/record-test

### cam 1 ea
cd ~/il_ws/src/lerobot && lerobot-record \
--robot.type=omx_follower \
--robot.port=/dev/omx_follower \
--robot.id=omx_follower_arm \
--robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
--teleop.type=omx_leader \
--teleop.port=/dev/omx_leader \
--teleop.id=omx_leader_arm \
--display_data=true \
--dataset.repo_id=${HF_USER}/record-test \
--dataset.single_task="Pick up Doll" \
--dataset.num_episodes=5 \
--dataset.reset_time_s=10




## 카메라 추가 및 에피소드 시간 조절
### cam 2 ea
rm -rf ~/.cache/huggingface/lerobot/${HF_USER}/pick_and_place

cd ~/il_ws/src/lerobot && lerobot-record \
--robot.type=omx_follower \
--robot.port=/dev/omx_follower \
--robot.id=omx_follower_arm \
--robot.cameras="{front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
--teleop.type=omx_leader \
--teleop.port=/dev/omx_leader \
--teleop.id=omx_leader_arm \
--display_data=true \
--dataset.repo_id=${HF_USER}/pick_and_place_omx \
--dataset.single_task="Pick up Doll" \
--dataset.episode_time_s=60 \
--dataset.num_episodes=100 \
--dataset.reset_time_s=10


HF_USER=$(hf auth whoami | head -n 1 | sed 's/\x1b\[[0-9;]*m//g' | awk -F': ' '{print $2}' | xargs)
cd ~/il_ws/src/lerobot && lerobot-record \
--robot.type=omx_follower \
--robot.port=/dev/omx_follower \
--robot.id=omx_follower_arm \
--robot.cameras="{front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
--teleop.type=omx_leader \
--teleop.port=/dev/omx_leader \
--teleop.id=omx_leader_arm \
--display_data=true \
--dataset.repo_id=${HF_USER}/pick_and_place_omx \
--dataset.single_task="Pick up Doll" \
--dataset.episode_time_s=60 \
--dataset.num_episodes=45 \
--dataset.reset_time_s=10 \
--resume=true



## replay
lerobot-replay \
--robot.type=omx_follower \
--robot.port=/dev/ttyACM0 \
--robot.id=omx_follower_arm \
--dataset.repo_id=iamtony-ca/pick_and_place \
--dataset.episode=3


pip install grpcio grpcio-tools

## train (act)
lerobot-train \
--dataset.repo_id=iamtony-ca/record-test \
--policy.type=act \
--output_dir=outputs/train/omx_act_policy \
--job_name=act_record-test \
--policy.device=cuda \
--wandb.enable=true \
--policy.repo_id=iamtony-ca/omx_act_policy


lerobot-train \
--dataset.repo_id=iamtony-ca/pick_and_place \
--policy.type=act \
--output_dir=outputs/train/omx_act_policy \
--job_name=act_pick_and_place \
--policy.device=cuda \
--wandb.enable=true \
--policy.repo_id=iamtony-ca/omx_act_policy

### train w checkpoint
lerobot-train \
--dataset.repo_id=${HF_USER}/record-test \
--policy.type=act \
--output_dir=outputs/train/omx_act_policy \
--job_name=act_record-test \
--policy.device=cuda \
--wandb.enable=true \
--policy.repo_id=${HF_USER}/omx_act_policy \
--batch_size=8 \
--save_checkpoint=true \
--save_freq=10000 \
--steps=100000

lerobot-train \
--dataset.repo_id=${HF_USER}/pick_and_place_omx \
--policy.type=act \
--output_dir=outputs/train/omx_act_policy50 \
--job_name=act_pick_and_place_omx \
--policy.device=cuda \
--wandb.enable=true \
--policy.repo_id=${HF_USER}/omx_act_policy50 \
--batch_size=8 \
--save_checkpoint=true \
--save_freq=10000 \
--steps=100000

rm -rf outputs/train/omx_act_policy50
lerobot-train \
--dataset.repo_id=${HF_USER}/pick_and_place_omx \
--policy.type=act \
--output_dir=outputs/train/omx_act_policy50 \
--job_name=act_pick_and_place_omx \
--policy.device=cuda \
--wandb.enable=true \
--policy.repo_id=${HF_USER}/omx_act_policy50 \
--batch_size=4 \
--policy.use_amp=true \
--save_checkpoint=true \
--save_freq=10000 \
--steps=100000


lerobot-train \
--dataset.repo_id=${HF_USER}/pick_and_place_omx_100 \
--policy.type=act \
--output_dir=outputs/train/omx_act_policy100 \
--job_name=act_pick_and_place_omx_100 \
--policy.device=cuda \
--wandb.enable=true \
--policy.repo_id=${HF_USER}/omx_act_policy100 \
--batch_size=4 \
--policy.use_amp=true \
--save_checkpoint=true \
--save_freq=10000 \
--steps=100000


### resume train for checkpoint
python -m lerobot.scripts.train \
--config_path=outputs/train/omx_act_policy/checkpoints/last/pretrained_model/
train_config.json \
--resume=true

### upload to hf w cli
huggingface-cli upload ${HF_USER}/omx_act_policy100 outputs/train/omx_act_policy100/checkpoints/100000/pretrained_model .


## inference
## aync
v4l2-ctl --list-devices

python -m lerobot.async_inference.policy_server \
--host=127.0.0.1 \
--port=8000 \
--fps=30 \
--inference_latency=0.033 \
--obs_queue_timeout=1


python -m lerobot.async_inference.robot_client \
--robot.type=omx_follower \
--robot.port=/dev/omx_follower \
--robot.id=omx_follower_arm \
--robot.cameras="{front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
--task=ruvit/omx_policy_3 \
--server_address=127.0.0.1:8000 \
--policy_type=act \
--pretrained_name_or_path=ruvit/omx_policy_3 \
--policy_device=cuda \
--actions_per_chunk=100 \
--chunk_size_threshold=0.7 \
--aggregate_fn_name=weighted_average \
--debug_visualize_queue_size=True


python -m lerobot.async_inference.robot_client \
--robot.type=omx_follower \
--robot.port=/dev/omx_follower \
--robot.id=omx_follower_arm \
--robot.cameras="{front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
--task=ruvit/omx_policy_3 \
--server_address=127.0.0.1:8000 \
--policy_type=act \
--pretrained_name_or_path=ruvit/omx_policy_3 \
--policy_device=cuda \
--actions_per_chunk=70 \
--chunk_size_threshold=0.6 \
--aggregate_fn_name=weighted_average \
--debug_visualize_queue_size=True


python -m lerobot.async_inference.robot_client \
--robot.type=omx_follower \
--robot.port=/dev/omx_follower \
--robot.id=omx_follower_arm \
--robot.cameras="{front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
--task="Pick up Doll" \
--server_address=127.0.0.1:8000 \
--policy_type=act \
--pretrained_name_or_path=iamtony-ca/omx_act_policy50 \
--policy_device=cuda \
--actions_per_chunk=70 \
--chunk_size_threshold=0.6 \
--aggregate_fn_name=weighted_average \
--debug_visualize_queue_size=True


python -m lerobot.async_inference.robot_client \
--robot.type=omx_follower \
--robot.port=/dev/omx_follower \
--robot.id=omx_follower_arm \
--robot.cameras="{front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
--task=${HF_USER}/omx_act_policy50_2 \
--server_address=127.0.0.1:8000 \
--policy_type=act \
--pretrained_name_or_path=${HF_USER}/omx_act_policy50_2 \
--policy_device=cuda \
--actions_per_chunk=70 \
--chunk_size_threshold=0.6 \
--aggregate_fn_name=weighted_average \
--debug_visualize_queue_size=True


#### succeed inference
python -m lerobot.async_inference.robot_client \
--robot.type=omx_follower \
--robot.port=/dev/omx_follower \
--robot.id=omx_follower_arm \
--robot.cameras="{front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
--task=${HF_USER}/omx_act_policy100 \
--server_address=127.0.0.1:8000 \
--policy_type=act \
--pretrained_name_or_path=${HF_USER}/omx_act_policy100 \
--policy_device=cuda \
--actions_per_chunk=70 \
--chunk_size_threshold=0.6 \
--aggregate_fn_name=weighted_average \
--debug_visualize_queue_size=True



# memo
performance : svla, pi0, groot nx > act /// groot n1x > svla > act -> but, when it comes to simple single task, act >= svla.
raw dataset is the same but cli is different after record
