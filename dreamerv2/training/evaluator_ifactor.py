import numpy as np
from typing import *
import torch 
import os
from dreamerv2.models.actor import DiscreteActionModel
from dreamerv2.models.drssm_ifactor import DRSSM
from dreamerv2.models.dense import DenseModel
from dreamerv2.models.pixel_robodesk import ObsDecoder, ObsEncoder
# TODO
from dreamerv2.utils.visualize import Visualizer
from dreamerv2.utils.identifiability.metrics import compute_r2, test_independence

class Evaluator(object):
    '''
    used this only for minigrid envs
    '''
    def __init__(
        self,
        config,
        device,
    ):
        self.device = device
        self.config = config
        self.action_size = config.action_size
        self.visualizer = Visualizer(config)

    def load_model(self, config, model_path):
        saved_dict = torch.load(model_path)
        obs_shape = config.obs_shape
        action_size = config.action_size
        deter_size = config.rssm_info['deter_size_s1'] + config.rssm_info['deter_size_s2'] + config.rssm_info['deter_size_s3'] + config.rssm_info['deter_size_s4']
        if config.rssm_type == 'continuous':
            stoch_size = config.rssm_info['stoch_size_s1'] + config.rssm_info['stoch_size_s2'] + config.rssm_info['stoch_size_s3'] + config.rssm_info['stoch_size_s4']
        elif config.rssm_type == 'discrete':
            category_size = config.rssm_info['category_size']
            class_size = config.rssm_info['class_size']
            stoch_size = category_size*class_size

        embedding_size = config.embedding_size
        rssm_node_size = config.rssm_node_size
        modelstate_size = stoch_size + deter_size
        asrdeter_size = config.rssm_info['deter_size_s1'] + config.rssm_info['deter_size_s2'] 
        asrstoch_size = config.rssm_info['stoch_size_s1'] + config.rssm_info['stoch_size_s2']

        if config.pixel:
                self.ObsEncoder = ObsEncoder(obs_shape, embedding_size, config.obs_encoder).to(self.device).eval()
                self.ObsDecoder = ObsDecoder(obs_shape, modelstate_size, config.obs_decoder).to(self.device).eval()
        else:
            self.ObsEncoder = DenseModel((embedding_size,), int(np.prod(obs_shape)), config.obs_encoder).to(self.device).eval()
            self.ObsDecoder = DenseModel(obs_shape, modelstate_size, config.obs_decoder).to(self.device).eval()

        self.ActionModel = DiscreteActionModel(action_size, asrdeter_size, asrstoch_size, embedding_size, config.actor, config.expl).to(self.device).eval()
        self.RSSM = DRSSM(action_size, rssm_node_size, embedding_size, self.device, config.rssm_type, config.rssm_info).to(self.device).eval()

        self.RSSM.load_state_dict(saved_dict["RSSM"])
        self.ObsEncoder.load_state_dict(saved_dict["ObsEncoder"])
        self.ObsDecoder.load_state_dict(saved_dict["ObsDecoder"])
        self.ActionModel.load_state_dict(saved_dict["ActionModel"])        

    def eval_saved_agent(self, env, model_path):
        self.load_model(self.config, model_path)
        train_step = model_path.split('/')[-1].split('.')[-2]
        print('train_step', train_step)
        return self.eval_agent(env, self.RSSM, self.ObsEncoder, self.ObsDecoder, self.ActionModel, train_step)

    def eval_visualize(self, env, model_path, interval=40, frame_save=6, visualize_episode=5, random=False):
        self.load_model(self.config, model_path)
        print("eval agent")
        for e in range(visualize_episode):
            obs, score = env.reset(), 0
            done = False
            with torch.no_grad():
                prev_rssmstate = self.RSSM._init_rssm_state(1)
                prev_action = torch.zeros(1, self.action_size).to(self.device)
            video_frames_dict = {"obs":[], "rssm_state_1234":[], "rssm_state_1": [], "rssm_state_2": [], "rssm_state_3": [], "rssm_state_4": [], "rssm_state_12": [], "rssm_state_34": []}
            first_state_flag = True
            iter_cnt = 0
            frame_cnt = 0
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32)
                    if obs.dtype == np.uint8:
                        obs_tensor = obs_tensor.div(255).sub_(0.5)
                    embed = self.ObsEncoder(obs_tensor.unsqueeze(0).to(self.device))
                    _, posterior_rssm_state = self.RSSM.rssm_observe(embed, prev_action, not done, prev_rssmstate)
                    asr_state = self.RSSM.get_asr_state(posterior_rssm_state)
                    if first_state_flag:
                        first_state = posterior_rssm_state
                        first_state_flag = False
                    if e < visualize_episode and iter_cnt % interval == 0:
                        self.visualizer.collect_frames_cartpole(obs_tensor, posterior_rssm_state, first_state, self.RSSM, self.ObsDecoder, video_frames_dict)
                        frame_cnt += 1
                    action = self.ActionModel.optimal_action(asr_state)
                    prev_rssmstate = posterior_rssm_state
                    prev_action = action
                    action = action.squeeze(0).cpu().numpy()
                    if random:
                        action = env.action_space.sample()
                next_obs, rew, done, _ = env.step(action)
                if self.config.eval_render:
                    env.render()
                score += rew
                obs = next_obs
                iter_cnt += 1
                if frame_cnt >= frame_save:
                    break
            # print(f'episode {e}: {score}')
            first_state_flag = True
            save_dir = os.path.join(os.path.split(model_path)[0], 'eval', 'visualize')
            self.visualizer.output_picture(save_dir, e, video_frames_dict)
        
    def eval_agent(self, env, RSSM, ObsEncoder, ObsDecoder, ActionModel, train_step):
        eval_scores = []
        eval_episode = self.config.eval_episode
        visualize_episode = self.config.visualize_episode
        print("eval agent")
        for e in range(eval_episode):
            obs, score = env.reset(), 0
            done = False
            with torch.no_grad():
                prev_rssmstate = RSSM._init_rssm_state(1)
                prev_action = torch.zeros(1, self.action_size).to(self.device)
            video_frames_dict = {"rssm_state_1234":[], "rssm_state_1": [], "rssm_state_2": [], "rssm_state_3": [], "rssm_state_4": [], "rssm_state_12": []}
            first_state_flag = True
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32)
                    if obs.dtype == np.uint8:
                        obs_tensor = obs_tensor.div(255).sub_(0.5)
                    embed = ObsEncoder(obs_tensor.unsqueeze(0).to(self.device))
                    _, posterior_rssm_state = RSSM.rssm_observe(embed, prev_action, not done, prev_rssmstate)
                    asr_state = RSSM.get_asr_state(posterior_rssm_state)
                    if first_state_flag:
                        first_state = posterior_rssm_state
                        first_state_flag = False
                    if e < visualize_episode:
                        self.visualizer.collect_frames(obs_tensor, posterior_rssm_state, first_state, RSSM, ObsDecoder, video_frames_dict)
                    action = ActionModel.optimal_action(asr_state)
                    prev_rssmstate = posterior_rssm_state
                    prev_action = action
                next_obs, rew, done, _ = env.step(action.squeeze(0).cpu().numpy())
                if self.config.eval_render:
                    env.render()
                score += rew
                obs = next_obs
            first_state_flag = True
            eval_scores.append(score)
            if e < visualize_episode:
                self.visualizer.output_video(train_step, e, video_frames_dict)
        print('average evaluation score = ' + str(np.mean(eval_scores)))
        return np.mean(eval_scores)
    
    def eval_score(self, env, RSSM, ObsEncoder, ActionModel, eval_num=5):
        eval_scores = []
        print("eval agent")
        for e in range(eval_num):
            obs, score = env.reset(), 0
            done = False
            with torch.no_grad():
                prev_rssmstate = RSSM._init_rssm_state(1)
                prev_action = torch.zeros(1, self.action_size).to(self.device)
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32)
                    if obs.dtype == np.uint8:
                        obs_tensor = obs_tensor.div(255).sub_(0.5)
                    embed = ObsEncoder(obs_tensor.unsqueeze(0).to(self.device))
                    _, posterior_rssm_state = RSSM.rssm_observe(embed, prev_action, not done, prev_rssmstate)
                    asr_state = RSSM.get_asr_state(posterior_rssm_state)
                    action = ActionModel.optimal_action(asr_state)
                    prev_rssmstate = posterior_rssm_state
                    prev_action = action
                next_obs, rew, done, _ = env.step(action.squeeze(0).cpu().numpy())
                if self.config.eval_render:
                    env.render()
                score += rew
                obs = next_obs
            first_state_flag = True
            eval_scores.append(score)
            print('Evaluation score = ' + str((score)))
        return np.mean(eval_scores), np.std(eval_scores)

    def collect_data(self, env, model_path=None, data_num_dict: Dict = {'train': 10000, 'val': 2500, 'test': 2500}, data_size_dict: Dict = {'s1': 2, 's2': 2, 's3': 1, 's4': 4}, overwrite=False, random=False):
        if model_path is not None:
            self.load_model(self.config, model_path)
        assert(env.full_state == True)
        if model_path is not None:
            new_dir = model_path.split('.')[0]
            os.makedirs(new_dir, exist_ok=True)
            output_file = os.path.join(new_dir + '_' + f"{data_num_dict['train']}_data.npy")
            if os.path.exists(output_file) and not overwrite:
                data = np.load(output_file, allow_pickle=True)
                data = data.item()
                print('loading data succeeds')
                print('data file name: ', output_file)
                return np.asarray(data)
        # key: "train", "val", "test"
        # init data
        stoch_size_s1, stoch_size_s2, stoch_size_s3, stoch_size_s4 = max(self.RSSM.stoch_size_s1, 1), max(self.RSSM.stoch_size_s2, 1), max(self.RSSM.stoch_size_s3, 1), max(self.RSSM.stoch_size_s4, 1)
        dataset_dict = {}
        for key in data_num_dict.keys():
            dataset_dict[key] = {'hs1': np.zeros((data_num_dict[key], stoch_size_s1)), 'hs2': np.zeros((data_num_dict[key], stoch_size_s2)), 'hs3': np.zeros((data_num_dict[key], stoch_size_s3)), 'hs4': np.zeros((data_num_dict[key], stoch_size_s4)), 's1': np.zeros((data_num_dict[key], max(1, data_size_dict['s1']))), 's2': np.zeros((data_num_dict[key], max(1, data_size_dict['s2']))), 's3': np.zeros((data_num_dict[key], max(1, data_size_dict['s3']))), 's4': np.zeros((data_num_dict[key], max(1, data_size_dict['s4']))), 'action': np.zeros((data_num_dict[key], self.action_size))}
        for key, value in data_num_dict.items():
            count = 0
            print(f'collecting data: {key}')
            while count < value:
                obs_dict, score = env.reset(), 0
                done = False
                prev_rssmstate = self.RSSM._init_rssm_state(1)
                prev_action = torch.zeros(1, self.action_size).to(self.device)
                while not done:
                    with torch.no_grad():
                        dataset_dict[key]['s1'][count], dataset_dict[key]['s2'][count], dataset_dict[key]['s3'][count], dataset_dict[key]['s4'][count] = obs_dict['s1'], obs_dict['s2'], obs_dict['s3'], obs_dict['s4']
                        obs = obs_dict['image']
                        obs_tensor = torch.tensor(obs, dtype=torch.float32)
                        if obs.dtype == np.uint8:
                            obs_tensor = obs_tensor.div(255).sub_(0.5)
                        embed = self.ObsEncoder(obs_tensor.unsqueeze(0).to(self.device))
                        _, posterior_rssm_state = self.RSSM.rssm_observe(embed, prev_action, not done, prev_rssmstate)
                        state_dict = self.RSSM.get_mean_state_dict(posterior_rssm_state)
                        for k, v in state_dict.items():
                            if v.shape[-1] == 0: 
                                state_dict[k] = torch.zeros(*v.shape[:-1], 1) 
                        # a = 's4' in state_dict.keys()
                        dataset_dict[key]['hs1'][count] = state_dict['s1'].cpu().squeeze().numpy()
                        dataset_dict[key]['hs2'][count] = state_dict['s2'].cpu().squeeze().numpy()
                        dataset_dict[key]['hs3'][count] = state_dict['s3'].cpu().squeeze().numpy()
                        dataset_dict[key]['hs4'][count] = state_dict['s4'].cpu().squeeze().numpy()
                    if not random:
                        with torch.no_grad():
                            asr_state = self.RSSM.get_asr_state(posterior_rssm_state)
                            action, _ = self.ActionModel(asr_state)
                            prev_rssmstate = posterior_rssm_state
                            prev_action = action
                            action_numpy = action.squeeze(0).cpu().numpy()
                    else:
                        action_numpy = env.action_space.sample()
                        prev_action = torch.tensor(action_numpy).unsqueeze(0).to(self.device)
                    next_obs_dict, rew, done, _ = env.step(action_numpy)
                    dataset_dict[key]['action'][count] = action_numpy
                    count += 1
                    if count >= value:
                        break
                    score += rew
                    obs_dict = next_obs_dict
                    
        if model_path is not None:
            np.save(output_file, dataset_dict, allow_pickle=True)
        return np.array(dataset_dict)
    
    def eval_block_wise(self, env, RSSM, ObsEncoder, ActionModel, random=False, data_size_dict = {'s1': 2, 's2': 2, 's3': 1, 's4': 4}):
        self.RSSM = RSSM
        self.ObsEncoder = ObsEncoder
        data_num_dict = {'train': 5001, 'val': 10, 'test': 5001}
        h_data_size_dict = {'s1': self.RSSM.stoch_size_s1, 's2': self.RSSM.stoch_size_s2, 's3': self.RSSM.stoch_size_s3, 's4': self.RSSM.stoch_size_s4}
        data = self.collect_data(env, data_num_dict=data_num_dict, data_size_dict=data_size_dict, random=random)
        data = data.item()
        # s12, s13, s14, s21, s23, s24, s31, s32, s34, s41, s42, s43 = test_independence(data['train'], data['test'], h_data_size_dict)
        # s predicts hs 13, 24
        s132hs13_r2 = compute_r2(np.concatenate((data['train']['hs1'], data['train']['hs3']), axis=-1), np.concatenate((data['train']['s1'], data['train']['s3']), axis=-1), np.concatenate((data['test']['hs1'], data['test']['hs3']), axis=-1), np.concatenate((data['test']['s1'], data['test']['s3']), axis=-1))
        print('s132hs13_r2: {}'.format(s132hs13_r2))
        s242hs24_r2 = compute_r2(np.concatenate((data['train']['hs2'], data['train']['hs4']), axis=-1), np.concatenate((data['train']['s2'], data['train']['s4']), axis=-1), np.concatenate((data['test']['hs2'], data['test']['hs4']), axis=-1), np.concatenate((data['test']['s2'], data['test']['s4']), axis=-1))
        print('s242hs24_r2: {}'.format(s242hs24_r2))
        # hs predicts s
        hs12s1_r2 = compute_r2(data['train']['s1'], data['train']['hs1'], data['test']['s1'], data['test']['hs1'])
        print('hs12s1_r2: {}'.format(hs12s1_r2))
        hs22s2_r2 = compute_r2(data['train']['s2'], data['train']['hs2'], data['test']['s2'], data['test']['hs2'])
        print('hs22s2_r2: {}'.format(hs22s2_r2))
        hs32s3_r2 = compute_r2(data['train']['s3'], data['train']['hs3'], data['test']['s3'], data['test']['hs3'])
        print('hs32s3_r2: {}'.format(hs32s3_r2))
        hs42s4_r2 = compute_r2(data['train']['s4'], data['train']['hs4'], data['test']['s4'], data['test']['hs4'])
        print('hs42s4_r2: {}'.format(hs42s4_r2))
        # s predicts hs
        s12hs1_r2 = compute_r2(data['train']['hs1'], data['train']['s1'], data['test']['hs1'], data['test']['s1'])
        print('s12hs1_r2: {}'.format(s12hs1_r2))
        s22hs2_r2 = compute_r2(data['train']['hs2'], data['train']['s2'], data['test']['hs2'], data['test']['s2'])
        print('s22hs2_r2: {}'.format(s22hs2_r2))
        s32hs3_r2 = compute_r2(data['train']['hs3'], data['train']['s3'], data['test']['hs3'], data['test']['s3'])
        print('s32hs3_r2: {}'.format(s32hs3_r2))
        s42hs4_r2 = compute_r2(data['train']['hs4'], data['train']['s4'], data['test']['hs4'], data['test']['s4'])
        print('s42hs4_r2: {}'.format(s42hs4_r2))
        
        return np.array([s12hs1_r2, s22hs2_r2, s32hs3_r2, s42hs4_r2, s132hs13_r2, s242hs24_r2, hs12s1_r2, hs22s2_r2, hs32s3_r2, hs42s4_r2])