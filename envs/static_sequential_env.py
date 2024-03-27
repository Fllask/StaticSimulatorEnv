import copy
import pygame
from typing import Literal
import gymnasium as gym
from gymnasium import spaces
from discrete_blocks_norot import DiscreteBlockNorot as Block
from discrete_simulator_noblocks import DiscreteSimulatorNoBlocks
from physics_scipy import StabilitySolverDiscrete as Ph
import numpy as np
import render_pygame as render
class sequential_discrete_env(gym.Env):
    metadata = {"render_modes": ["human", "agent_input"], "render_fps": 1}

    def __init__(self, render_mode=None,
                 gridsize=[20,20],
                 n_robots=2,
                 n_reg = 2,
                 max_blocks = 50,
                 max_interfaces = 150,
                 block_list: list[Block]|Literal['hex', 'trap','hexlink']='hex',
                 friction_coef: float=0.5,
                 mask_generator=None):
        self.gridsize = gridsize
        self.friction_coef = friction_coef 
        self.n_robots = n_robots 
        self.mask_generator = mask_generator
        self.scale = 10  
        
        

        if block_list == 'hexlink':
            hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=friction_coef)
            linkr = Block([[0,0,0],[0,1,1],[1,0,0],[1,0,1],[1,1,1],[0,1,0]],muc=friction_coef) 
            linkl = Block([[0,0,0],[0,1,1],[1,0,0],[0,1,0],[0,0,1],[-1,1,1]],muc=friction_coef) 
            linkh = Block([[0,0,0],[0,1,1],[1,0,0],[-1,2,1],[0,1,0],[0,2,1]],muc=friction_coef)
            self.block_type = [hexagon,linkh,linkl,linkr]
        elif block_list == 'hex':
            hexagon = Block([[1,0,0],[1,1,1],[1,1,0],[0,2,1],[0,1,0],[0,1,1]],muc=friction_coef)
            self.block_type = [hexagon]
        elif block_list == 'trap':
            trap1 = Block([[1,0,0],[1,1,1],[1,1,0]],muc=friction_coef)
            trap2 = Block([[1,1,1],[1,1,0],[0,2,1]],muc=friction_coef)
            trap3 = Block([[1,1,0],[0,2,1],[0,1,0]],muc=friction_coef)
            trap4 = Block([[0,2,1],[0,1,0],[0,1,1]],muc=friction_coef)
            trap5 = Block([[1,0,0],[0,1,0],[0,1,1]],muc=friction_coef)
            trap6 = Block([[1,0,0],[1,1,1],[0,1,1]],muc=friction_coef)
            self.block_type = [trap1,trap2,trap3,trap4,trap5,trap6]
        elif type(block_list) ==str:
            assert False, f"unkwown preset {block_list}"
        else:
            self.block_type = block_list
        
        self.observation_space = spaces.Dict({"occ": spaces.MultiBinary((self.gridsize[0],self.gridsize[1],2)),
                                                  "obstacles":spaces.MultiBinary((self.gridsize[0],self.gridsize[1],2)),
                                                  "objectives":spaces.MultiBinary((self.gridsize[0],self.gridsize[1],2)),
                                                  "sides":spaces.MultiBinary((self.gridsize[0],self.gridsize[1],2*3)),
                                                  "grounds":spaces.MultiBinary((self.gridsize[0],self.gridsize[1],2)),
                                                  "hold":spaces.MultiBinary((self.gridsize[0],self.gridsize[1],2*self.n_robots)),
                                                  "turn":spaces.Discrete(self.n_robots),
                                                  "mask":spaces.MultiBinary((gridsize[0],gridsize[1],len(self.block_type)))
                                                  })
        
        self.action_space = spaces.Box(low=0,high=np.array([gridsize[0]-1,gridsize[1]-1,len(self.block_type)-1]),dtype=int)

        self.setup = DiscreteSimulatorNoBlocks(maxs=self.gridsize,
                                             n_robots=self.n_robots,
                                             block_choices=None,
                                             n_reg=n_reg,
                                             maxblocks=max_blocks,
                                             maxinterface=max_interfaces,
                                             robot_torque=True
                                             )
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
    def _get_observation(self):
        state = {'occ': self.sim.grid.occ[:-1,:-1]>=0,
                 "grounds":self.sim.grid.occ[:-1,:-1]== 0,
                 'obstacles': self.sim.grid.occ[:-1,:-1]== -2,
                 "objectives":self.sim.grid.occ[:-1,:-1]== -3,
                 "sides":(self.sim.grid.neighbours[:-1,:-1,:,:,0]>-1).reshape((self.gridsize[0],self.gridsize[1],-1)),
                 "hold":np.concatenate([self.sim.grid.hold[:-1,:-1]==(self.sim.turn-rid)%self.n_robots for rid in range(self.n_robots)]),
                 "mask":self.mask_generator(self.sim.grid,self.block_type,['Ph'],False,False,True),
                 "turn":self.sim.turn}
        return state
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.sim = copy.deepcopy(self.setup)

        if options['target_type']=='random_gap':
            gap = self.np_random.integers(options['gap_range'][0],options['gap_range'][1])
            self.sim.add_ground(Block([[i,0,1] for i in range(self.gridsize[0]-1-gap)],muc=self.friction_coef),[0,0])
            self.sim.add_ground(Block([[0,0,1]],muc=self.friction_coef),[self.gridsize[0]-1,0])
        if options.get('obstacle_gen') is not None:
            assert False,"Not implemented"

        observation = self._get_observation()
        info = {'gap':gap}

        if self.render_mode == "human":
            self._render_frame()
    
        return observation, info
    def step(self,action):
        terminated = False
        
        info={}
        valid,closer,blocktype,interfaces,n_obstacles = self.sim.interprete_act('Ph',blocktype=self.block_type[action[2]],x=action[0],y=action[1])
        if valid: #and (self.obstacles_type != 'hard' or tot_obstacles[-1]==0):
            reward = 0
            if np.all(self.sim.grid.min_dist < 1e-5):
                bids = []
                for r in range(self.n_robots):
                    bids.append(self.sim.leave(r))
                if self.sim.check():
                    terminated = True
                    reward += 1
                else:
                    for r,bid in enumerate(bids):
                        self.sim.hold(r,bid)
        else:
            terminated = True
            reward = -1
            #mark the state as terminal
        if self.render_mode == "human":
            self._render_frame()
        observation = self._get_observation()
        return observation, reward, terminated, False, info
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            self.window,canvas = render.open_window(self.gridsize,scale=self.scale)
        else:
            canvas= render.new_caneva(self.gridsize,self.scale)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        render.draw_struct(surface=canvas,grid=self.sim.grid,scale=self.scale)
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
def generate_mask_parametrized(grid,block_choices,action_choice,allow_float,allow_col,last_only):
        n_channels_r = (len(block_choices)+int('S' in action_choice))
        mask = np.zeros((n_channels_r, grid.shape[0],grid.shape[1]),dtype=bool)
        if 'S' in action_choice:
            mask[-1,:,:]=False
            mask[-1,grid.shape[0]//2,grid.shape[1]//2]=True
        if not allow_float:
            for bid,block in enumerate(block_choices):
                non_floating_pos = grid.touch_side(block,last_only)
                mask[bid*np.ones(non_floating_pos.shape[0],dtype=int),non_floating_pos[:,0],non_floating_pos[:,1]]=True
        else:
            mask[:len(block_choices),:,:]=True
        if not allow_col:
            pos = np.array(np.nonzero(mask)).T
            for bid,x,y in pos:
                if bid >= len(block_choices):
                    break
                mask[bid,x,y],*_ = grid.put(block_choices[bid],[x,y],-1,test_col=True)
        return mask 
if __name__ == "__main__":
    print("Start test")
    tmp_env = sequential_discrete_env(mask_generator=generate_mask_parametrized)
    tmp_env.render_mode='rgb_array'
    # wrap the env in the record video
    env = gym.wrappers.RenderCollection(tmp_env,False)
    # env reset for a fresh start
    gym_option = {'target_type':'random_gap',
                  'gap_range':[1,10]}
    obs,info = env.reset(options=gym_option)
    # Start the recorder
    env.render()
    
    for _ in range(1):
        env.step([19,0,0])
        env.render()
        env.step([18,2,0])
        env.render()
        env.step([17,4,0])  # agent policy that uses the observation and info
        env.render()
        
        observation, info = env.reset(options=gym_option)

    # Close the environment
    env.close()
    
    print("End test")