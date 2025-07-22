import setproctitle
import logging
import argparse
import torch
import sys
import os

from tensorboardX.writer import SummaryWriter
from framework.utils import set_seed
from framework.trainer import Trainer, TrainerConfig
from framework.utils import get_dim_from_space
from envs.env import Env
from framework.buffer import ReplayBuffer
from framework.rollout import RolloutWorker
from datetime import datetime, timedelta
from models.gpt_model import GPT, GPTConfig


from framework.utils import CPUManager

from envs import config

# args = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='madt')
parser.add_argument('--cuda_id', type=str, default='0',help='指定使用的GPU ID，例如：0,1,2,3')
parser.add_argument('--use_cpu', action='store_true', default=False, help='强制使用CPU训练，默认使用CUDA')
parser.add_argument('--map_name', type=str, default='3s5z', help='地图名称')
parser.add_argument('--rtg_min', type=float, default=5.0, help='最小RTG值')
parser.add_argument('--rtg_max', type=float, default=25.0, help='最大RTG值')
parser.add_argument('--rtg_momentum', type=float, default=0.95, help='RTG平滑因子')
parser.add_argument('--win_rate_threshold', type=float, default=0.6, help='胜率阈值')
parser.add_argument('--return_scale', type=float, default=0.2, help='return影响因子')
# 在现有参数后添加
parser.add_argument('--n_threads', type=int, default=1, help='实际使用的线程数')




parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=1)
parser.add_argument('--model_type', type=str, default='state_only')
parser.add_argument('--eval_episodes', type=int, default=32)
parser.add_argument('--max_timestep', type=int, default=400)
parser.add_argument('--log_dir', type=str, default='./logs/')
parser.add_argument('--save_log', type=bool, default=True)
parser.add_argument('--exp_name', type=str, default='easy_trans')
parser.add_argument('--pre_train_model_path', type=str, default='../offline_model/')

parser.add_argument('--offline_map_lists', type=list, default=['8m','8m','8m','8m','8m'])
parser.add_argument('--offline_episode_num', type=list, default=[60,60,60,60,60])#[200, 200, 200, 200, 200]
parser.add_argument('--offline_data_quality', type=list, default=['good'])
parser.add_argument('--offline_data_dir', type=str, default='../PATH_TO_OFFLINE_DATA/')

parser.add_argument('--offline_epochs', type=int, default=10)
parser.add_argument('--offline_mini_batch_size', type=int, default=128)
parser.add_argument('--offline_lr', type=float, default=1e-4)
parser.add_argument('--offline_eval_interval', type=int, default=1)
parser.add_argument('--offline_train_critic', type=bool, default=True)
parser.add_argument('--offline_model_save', type=bool, default=True)

parser.add_argument('--online_buffer_size', type=int, default=64)
parser.add_argument('--online_epochs', type=int, default=5000)
parser.add_argument('--online_ppo_epochs', type=int, default=5)
parser.add_argument('--online_lr', type=float, default=1e-4)
parser.add_argument('--online_eval_interval', type=int, default=1)
parser.add_argument('--online_train_critic', type=bool, default=True)
parser.add_argument('--online_pre_train_model_load', type=bool, default=False)
parser.add_argument('--online_pre_train_model_id', type=int, default=9)



def get_env_dims(env):
    """从环境中获取维度信息"""
    try:
        global_obs_dim = get_dim_from_space(env.real_env.share_observation_space)
        local_obs_dim = get_dim_from_space(env.real_env.observation_space)
        action_dim = get_dim_from_space(env.real_env.action_space)
        
        print(f"从环境获取维度: global={global_obs_dim}, local={local_obs_dim}, action={action_dim}")
        return global_obs_dim, local_obs_dim, action_dim
    except Exception as e:
        print(f"获取环境维度失败: {e}")
        return None
def setup_device(args):
    """设置训练设备"""
    if not args.use_cpu and torch.cuda.is_available():
        # 设置可见的CUDA设备
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
        device = torch.device("cuda")
        print(f"使用GPU: {args.cuda_id}")
        print(f"当前GPU型号: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("使用CPU训练")
    return device



# args = parser.parse_args(args, parser)
args = parser.parse_args()
set_seed(args.seed)
torch.set_num_threads(8)
cur_time = datetime.now() + timedelta(hours=0)
args.log_dir += cur_time.strftime("[%m-%d]%H.%M.%S")

writter = SummaryWriter(args.log_dir+args.exp_name) if args.save_log else None

eval_env = Env(args.eval_episodes)
online_train_env = Env(args.online_buffer_size)


# 替换原有的环境创建代码

    

# global_obs_dim = 267
# local_obs_dim = 252
# action_dim = 15
# global_obs_dim = 318
# local_obs_dim = 318
# action_dim = 15

dims = get_env_dims(online_train_env)

if dims is None:
    print("使用默认维度")
    global_obs_dim = 299
    local_obs_dim = 252
    action_dim = 15
else:
    global_obs_dim, local_obs_dim, action_dim = dims

# global_obs_dim = get_dim_from_space(online_train_env.real_env.share_observation_space)
# local_obs_dim = get_dim_from_space(online_train_env.real_env.observation_space)
# action_dim = get_dim_from_space(online_train_env.real_env.action_space)


block_size = args.context_length * 3

print("global_obs_dim: ", global_obs_dim)
print("local_obs_dim: ", local_obs_dim)
print("action_dim: ", action_dim)

##############替换claude模型框架
# mconf_actor = GPT1Config(local_obs_dim, action_dim, block_size,
#                         window_size=3, model_type=args.model_type,n_embd=256,n_layer=6,max_timestep=args.max_timestep)#2 2 32
# model = GPT1(mconf_actor, model_type='actor')

# mconf_critic = GPT1Config(local_obs_dim, action_dim, block_size,
#                         window_size=3, model_type=args.model_type,n_embd=256,n_layer=6,max_timestep=args.max_timestep)
# critic_model = GPT1(mconf_critic, model_type='critic')

mconf_actor = GPTConfig(local_obs_dim, action_dim, block_size,
                        n_layer=2, n_head=2, n_embd=32, model_type=args.model_type, max_timestep=args.max_timestep)#2 2 32
model = GPT(mconf_actor, model_type='actor')

mconf_critic = GPTConfig(global_obs_dim, action_dim, block_size,
                         n_layer=2, n_head=2, n_embd=32, model_type=args.model_type, max_timestep=args.max_timestep)#2 2 32
critic_model = GPT(mconf_critic, model_type='critic')

mconf_frozen_actor = GPTConfig(local_obs_dim, action_dim, block_size,
                              n_layer=2, n_head=2, n_embd=32, model_type=args.model_type, max_timestep=args.max_timestep)
frozen_actor_model = GPT(mconf_frozen_actor, model_type='actor')

# device = setup_device(args)
# model = model.to(device)
# critic_model = critic_model.to(device)

if torch.cuda.is_available():  #torch.cuda.is_available():
    device = torch.cuda.current_device()
    model = torch.nn.DataParallel(model).to(device)
    critic_model = torch.nn.DataParallel(critic_model).to(device)
    frozen_actor_model = torch.nn.DataParallel(frozen_actor_model).to(device)
buffer = ReplayBuffer(block_size, global_obs_dim, local_obs_dim, action_dim)
#buffer.set_counterfactual_mode(use_cf=True)
rollout_worker = RolloutWorker(model, critic_model,frozen_actor_model, buffer, global_obs_dim, local_obs_dim, action_dim)
rollout_worker.frozen_actor_model = frozen_actor_model
used_data_dir = []
for map_name in args.offline_map_lists:
    source_dir = args.offline_data_dir + map_name
    for quality in args.offline_data_quality:
        used_data_dir.append(f"{source_dir}/{quality}/")

buffer.load_offline_data(used_data_dir, args.offline_episode_num, max_epi_length=eval_env.max_timestep)
offline_dataset = buffer.sample()
offline_dataset.stats()

offline_tconf = TrainerConfig(max_epochs=1, batch_size=args.offline_mini_batch_size, learning_rate=args.offline_lr,
                              num_workers=0, mode="offline")
offline_trainer = Trainer(model, critic_model, offline_tconf)

# target_rtgs = offline_dataset.max_rtgs
target_rtgs = 20.
print("offline target_rtgs: ", target_rtgs)
# for i in range(args.offline_epochs):
#     offline_actor_loss, offline_critic_loss, _, __, ___ = offline_trainer.train(offline_dataset,
#                                                                                 args.offline_train_critic)
#     if args.save_log:
#         writter.add_scalar('offline/{args.map_name}/offline_actor_loss', offline_actor_loss, i)
#         writter.add_scalar('offline/{args.map_name}/offline_critic_loss', offline_critic_loss, i)
#     if i % args.offline_eval_interval == 0:
#         aver_return, aver_win_rate, _ = rollout_worker.rollout(eval_env, target_rtgs, train=False)
#         print("offline epoch: %s, return: %s, eval_win_rate: %s" % (i, aver_return, aver_win_rate))
#         if args.save_log:
#             writter.add_scalar('offline/{args.map_name}/aver_return', aver_return.item(), i)
#             writter.add_scalar('offline/{args.map_name}/aver_win_rate', aver_win_rate, i)
#     if args.offline_model_save and i==args.offline_epochs-1:
#         actor_path = args.pre_train_model_path + args.exp_name + '/actor'
#         if not os.path.exists(actor_path):
#             os.makedirs(actor_path)
#         critic_path = args.pre_train_model_path + args.exp_name + '/critic'
#         if not os.path.exists(critic_path):
#             os.makedirs(critic_path)
#         torch.save(model.state_dict(), actor_path + os.sep + str(i) + '.pkl')
#         torch.save(critic_model.state_dict(), critic_path + os.sep + str(i) + '.pkl')


if args.online_epochs > 0 and args.online_pre_train_model_load:
    actor_path = args.pre_train_model_path + args.exp_name + '/actor/' + str(args.online_pre_train_model_id) + '.pkl'
    critic_path = args.pre_train_model_path + args.exp_name + '/critic/' + str(args.online_pre_train_model_id) + '.pkl'
    model.load_state_dict(torch.load(actor_path), strict=False)
    critic_model.load_state_dict(torch.load(critic_path), strict=False)
    # model.load_state_dict(torch.load(actor_path))
    # critic_model.load_state_dict(torch.load(critic_path))
    #冻结参数模型
    rollout_worker.frozen_actor_model.load_state_dict(torch.load(actor_path), strict=False)
    for param in rollout_worker.frozen_actor_model.parameters():
        param.requires_grad = False
    rollout_worker.frozen_actor_model.eval()

online_tconf = TrainerConfig(max_epochs=args.online_ppo_epochs, batch_size=0,
                             learning_rate=args.online_lr, num_workers=0, mode="online",use_lr_scheduler = True)
online_trainer = Trainer(model, critic_model, online_tconf)
online_trainer.set_frozen_model(frozen_actor_model)
buffer.reset(num_keep=0, buffer_size=args.online_buffer_size)

total_steps = 0
rollout_worker.trainer = online_trainer
for i in range(args.online_epochs):
    
    #sample_return, win_rate, steps = rollout_worker.rollout(online_train_env, target_rtgs, train=True) #  win_rate target_rtgs
    sample_return, win_rate, steps = rollout_worker.rollout(online_train_env, target_rtgs, train=True)
   
    


    #new_rtg = online_trainer.update_rtg(win_rate, sample_return)
    
    total_steps += steps
    online_dataset = buffer.sample()
    online_actor_loss, online_critic_loss, entropy, ratio, confidence = online_trainer.train(online_dataset,
                                                                                             args.online_train_critic)
    if args.save_log:
        writter.add_scalar('online/{args.map_name}/online_actor_loss', online_actor_loss, total_steps)
        writter.add_scalar('online/{args.map_name}/online_critic_loss', online_critic_loss, total_steps)
        writter.add_scalar('online/{args.map_name}/entropy', entropy, total_steps)
        writter.add_scalar('online/{args.map_name}/ratio', ratio, total_steps)
        writter.add_scalar('online/{args.map_name}/confidence', confidence, total_steps)
        writter.add_scalar('online/{args.map_name}/sample_return', sample_return, total_steps)

    # if online_dataset.max_rtgs > target_rtgs:
    #     target_rtgs = online_dataset.max_rtgs
    #print("sample return: %s, online target_rtgs: %s" % (sample_return, target_rtgs))
    print("sample return: %s, online target_rtgs: %s" % (sample_return, target_rtgs))
    # if i % args.online_eval_interval == 0:
    #     print(f"Episode {i}: Win Rate={win_rate:.2f}, Return={sample_return:.2f}, RTG={new_rtg:.2f}")
    #     if writter is not None:
    #         writter.add_scalar("train/rtg", new_rtg, i)
    #         writter.add_scalar("train/win_rate", win_rate, i)
    #         writter.add_scalar("train/return", sample_return, i)
    
    if i % args.online_eval_interval == 0:
        #aver_return, aver_win_rate, _ = rollout_worker.rollout(eval_env, target_rtgs, train=False)
        aver_return, aver_win_rate, _ = rollout_worker.rollout_with_imitate(eval_env, target_rtgs, train=False)

         #加加加
    
        if online_trainer.config.use_lr_scheduler:
            # 判断是否需要更新学习率
            should_update = False
            
            # 性能检查条件
            if aver_win_rate > online_trainer.config.win_rate_threshold:
                # 胜率高，可能需要减小学习率
                should_update = True
            elif aver_return > target_rtgs * 0.8:  # 回报达到目标的80%
                # 回报较好，可能需要微调学习率
                should_update = True
            
            # 更新学习率
            if should_update:
                online_trainer.update_scheduler(aver_win_rate, aver_return)
        print("online steps: %s, return: %s, eval_win_rate: %s" % (total_steps, aver_return, aver_win_rate))
        if args.save_log:
            writter.add_scalar('online/{args.map_name}/aver_return', aver_return.item(), total_steps)
            writter.add_scalar('online/{args.map_name}/aver_win_rate', aver_win_rate, total_steps)

online_train_env.real_env.close()
eval_env.real_env.close()
