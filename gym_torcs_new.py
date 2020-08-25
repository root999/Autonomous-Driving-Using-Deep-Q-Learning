import gym
from gym import spaces
import numpy as np
# from os import path
import snakeoil3_gym as snakeoil3
import numpy as np
import copy
import collections as col
import os
import time


class TorcsEnv:
    terminal_judge_start = 500  # Speed limit is applied after this step
    termination_limit_progress = 5  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 50

    initial_reset = True


    def __init__(self, vision=False, throttle=False, gear_change=False):
       #print("Init")
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change
        
        self.initial_run = True

        ##print("launch torcs")
        os.system('pkill torcs')
        time.sleep(0.5)
        if self.vision is True:
            os.system('torcs -nofuel -nodamage -nolaptime  -vision &')
        else:
            os.system('torcs  -nofuel -nodamage -nolaptime &')
        time.sleep(0.5)
        os.system('sh CG_Track2.sh')
        time.sleep(0.5)

        """
        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        """
        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        if vision is False:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf])
            self.observation_space = spaces.Box(low=low, high=high)
        else:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf, 255])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf, 0])
            self.observation_space = spaces.Box(low=low, high=high)

    def reshape_image(self,vision):
        img = np.ndarray((64,64,3))
        for i in range(3):
            img[:, :, i] = 255 - vision[:, i].reshape((64, 64))
        return img
    def step(self, u):
       #print("Step")
        # convert thisAction to the actual torcs actionstr
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]
        action_torcs['accel'] = this_action['accel']
        #  Simple Autnmatic Throttle Control by Snakeoil
        # if self.throttle is False:
        #     target_speed = self.default_speed
        #     if client.S.d['speedX'] < target_speed - (client.R.d['steer']*50):
        #         client.R.d['accel'] += .01
        #     else:
        #         client.R.d['accel'] -= .01

        #     if client.R.d['accel'] > 0.2:
        #         client.R.d['accel'] = 0.2

        #     if client.S.d['speedX'] < 10:
        #         client.R.d['accel'] += 1/(client.S.d['speedX']+.1)

        #     # Traction Control System
        #     if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
        #        (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
        #         action_torcs['accel'] -= .2
        # else:
        #

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            #  Automatic Gear Change by Snakeoil is possible
            action_torcs['gear'] = 1
            if client.S.d['speedX'] > 50:
                action_torcs['gear'] = 2
            if client.S.d['speedX'] > 80:
                action_torcs['gear'] = 3
            if client.S.d['speedX'] > 110:
                action_torcs['gear'] = 4
            if client.S.d['speedX'] > 140:
                action_torcs['gear'] = 5
            if client.S.d['speedX'] > 170:
                action_torcs['gear'] = 6

        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)

        # Reward setting Here #######################################
        # direction-dependent positive reward
        track = np.array(obs['track'])
        sp = np.array(obs['speedX'])
        sp_pre = np.array(obs_pre['speedX'])
        #print(f"speed: {sp} diger: {np.cos(obs['angle'])}")
        progress = sp * np.cos(obs['angle']) - np.abs(sp * np.sin(obs['angle']))- sp * np.abs(obs['trackPos'])
        # progress = sp*np.cos(obs['angle'])
        reward = progress
        episode_terminate = False
        # collision detection
        if obs['damage'] - obs_pre['damage'] > 0:
            reward = -200
            episode_terminate = True
            client.R.d['meta'] = True
        # Termination judgement #########################
        # if sp > 100:
        #     print(f"speedddddd : {sp}")
        #     reward = 0.01
        # elif sp > 50:
        #     print(f"speedddddd : {sp}")
        #     reward = 0.005
        # elif sp > 20:
        #     reward = 0.001
        # if sp >= sp_pre:
        #     reward = 1
        if track.min() < 0:  # Episode is terminated if the car is out of track
            reward = -200
            episode_terminate = True
            client.R.d['meta'] = True

        if self.terminal_judge_start < self.time_step: # Episode terminates if the progress of agent is small
            if progress < self.termination_limit_progress:
                episode_terminate = True
                client.R.d['meta'] = True

        if np.cos(obs['angle']) < 0: # Episode is terminated if the agent runs backward
            episode_terminate = True
            client.R.d['meta'] = True


        if client.R.d['meta'] is True: # Send a reset signal
            episode_terminate = True
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        #return self.get_obs(), reward, client.R.d['meta'], {}
        return self.get_obs(), reward, episode_terminate, {}

    def reset(self, relaunch=False):
        #print("Reset")

        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False
        return self.get_obs()

    def end(self):
        os.system('pkill torcs')

    def get_obs(self):
        return self.observation

    def reset_torcs(self):
       #print("relaunch torcs")
        os.system('pkill torcs')
        time.sleep(0.5)
        if self.vision is True:
            os.system('torcs -nofuel -nodamage -nolaptime -vision &')
        else:
            os.system('torcs -nofuel -nodamage -nolaptime &')
        time.sleep(0.5)
        os.system('sh CG_Track2.sh')
        time.sleep(0.5)

    def agent_to_torcs(self, action):
        if action == 0:
            torcs_action = {'steer': 0.0}
            torcs_action.update({'accel': 0.0})
            #print(f"action {action} active")
        elif action == 1:
            torcs_action = {'steer': 0.-5}
            torcs_action.update({'accel': 0.0})
            #print(f"action {action} active")
        elif action == 2:
            torcs_action = {'steer': 0.5}
            torcs_action.update({'accel': 0.0})
            #print(f"action {action} active")
        elif action == 3:
            torcs_action = {'steer': 0.0}
            torcs_action.update({'accel': 1.0})
            #print(f"action {action} active")
        elif action == 4:
            torcs_action = {'steer': 0.0}
            torcs_action.update({'accel': -0.5})
            #print(f"action {action} active")
        elif action == 5:
            torcs_action = {'steer': -0.5}
            torcs_action.update({'accel': 1.0})
            #print(f"action {action} active")
        elif action == 6:
            torcs_action = {'steer': 0.5}
            torcs_action.update({'accel': 1.0})
            #print(f"action {action} active")
        elif action == 7:
            torcs_action = {'steer': -0.5}
            torcs_action.update({'accel': -0.5})
            #print(f"action {action} active")
        elif action == 8:
            torcs_action = {'steer': 0.5}
            torcs_action.update({'accel': -0.5})
            #print(f"action {action} active")
        elif action == 9:
            torcs_action = {'steer': 0.0}
            torcs_action.update({'accel': 0.5})
            #print(f"action {action} active")
        elif action == 10:
            torcs_action = {'steer': -0.5}
            torcs_action.update({'accel': -1.0})
            #print(f"action {action} active")
        elif action == 11:
            torcs_action = {'steer': 0.5}
            torcs_action.update({'accel': -1.0})
            #print(f"action active {action}")

        # if self.gear_change is True: # gear change action is enabled
        #     torcs_action.update({'gear': u[2]})

        return torcs_action


    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec
        rgb = []
        temp = []
        # convert size 64x64x3 = 12288 to 64x64=4096 2-D list
        # with rgb values grouped together.
        # Format similar to the observation in openai gym
        for i in range(0,12286,3):
            temp.append(image_vec[i])
            temp.append(image_vec[i+1])
            temp.append(image_vec[i+2])
            rgb.append(temp)
            temp = []
        return np.array(rgb, dtype=np.uint8)

    def make_observaton(self, raw_obs):
        if self.vision is False:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ',
                     'opponents',
                     'rpm',
                     'track',
                     'wheelSpinVel']
            Observation = col.namedtuple('Observaion', names)
            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32))
        else:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ',
                     'opponents',
                     'rpm',
                     'track',
                     'wheelSpinVel',
                     'img']
            Observation = col.namedtuple('Observaion', names)

            # Get RGB from observation
            image_rgb = self.obs_vision_to_image_rgb(raw_obs[names[8]])

            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               img=image_rgb)
