import isaacgym

import os
import hydra
from termcolor import cprint
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn

from dexscrew.algo.ppo.padapt import ProprioAdapt
from dexscrew.tasks import isaacgym_task_map
from dexscrew.utils.reformat import omegaconf_to_dict, print_dict
from dexscrew.utils.misc import set_seed

OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)

class PolicyWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("running_mean", torch.zeros(1))
        self.register_buffer("running_var", torch.ones(1))
        self.register_buffer("running_count", torch.zeros(1))
        self.register_buffer("sa_mean", torch.zeros(1))
        self.register_buffer("sa_var", torch.ones(1))
        self.register_buffer("sa_count", torch.zeros(1))
        self.register_buffer("running_sa", torch.zeros(1))
        self.register_buffer("pc_mean", torch.zeros(1))
        self.register_buffer("pc_var", torch.ones(1))
        self.register_buffer("pc_count", torch.zeros(1))

    
    def load_stats(self, mean, var, count, sa_mean=None, sa_var=None, sa_count=None, pc_mean=None, pc_var=None, pc_count=None):
        self.running_mean = mean
        self.running_var = var
        self.running_count = count
        if sa_mean is not None:
            self.sa_mean = sa_mean
            self.sa_var = sa_var
            self.sa_count = sa_count
        if pc_mean is not None:
            self.pc_mean = pc_mean
            self.pc_var = pc_var
            self.pc_count = pc_count
        
    
    def forward(self, input_dict):
        device = next(self.model.parameters()).device
        input_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in input_dict.items()}
        mu, sigma, value, extrin, extrin_gt = self.model._actor_critic(input_dict)
        return mu, extrin, extrin_gt

def export_jit_model(agent, output_dir: str, obs_dict=None, jit_output_name: str = "student_policy.pt"):
    """
    Export the agent's policy network to a TorchScript file using tracing.
    """
    agent.model = agent.model.cpu()
    agent.model.eval()
    
    if hasattr(agent, 'running_mean_std'):
        agent.running_mean_std = agent.running_mean_std.cpu()
    if hasattr(agent, 'priv_mean_std'):
        agent.priv_mean_std = agent.priv_mean_std.cpu()
    if hasattr(agent, 'sa_mean_std'):
        agent.sa_mean_std = agent.sa_mean_std.cpu()
    if hasattr(agent, 'point_cloud_mean_std'):
        agent.point_cloud_mean_std = agent.point_cloud_mean_std.cpu()

    if isinstance(agent, ProprioAdapt):
        student_obs = obs_dict['obs']
        input_dict = {
            'obs': agent.running_mean_std(student_obs).cpu(),
            'proprio_hist': agent.sa_mean_std(obs_dict['proprio_hist']).cpu(),
            'point_cloud_info': agent.point_cloud_mean_std(obs_dict['point_cloud_info']).cpu()
            if agent.normalize_point_cloud
            else obs_dict['point_cloud_info'].cpu(),
        }
    else:
        raise ValueError(f"Unsupported agent type for JIT export: {type(agent)}")
    
    wrapper = PolicyWrapper(agent.model)
    if hasattr(agent, 'sa_mean_std'):
        wrapper.load_stats(agent.running_mean_std.running_mean, agent.running_mean_std.running_var, agent.running_mean_std.count,
                        sa_mean=agent.sa_mean_std.running_mean, sa_var=agent.sa_mean_std.running_var, sa_count=agent.sa_mean_std.count,
                        pc_mean=agent.point_cloud_mean_std.running_mean, pc_var=agent.point_cloud_mean_std.running_var, pc_count=agent.point_cloud_mean_std.count)
    else:
        wrapper.load_stats(agent.running_mean_std.running_mean, agent.running_mean_std.running_var, agent.running_mean_std.count)
    wrapper.eval()

    with torch.no_grad():
        ts_model = torch.jit.trace(wrapper, input_dict)
    
    os.makedirs(output_dir, exist_ok=True)
    jit_path = os.path.join(output_dir, jit_output_name)
    ts_model.save(jit_path)
    cprint(f"JIT model saved to {jit_path}", 'green')

@hydra.main(config_name='config', config_path='configs')
def main(config: DictConfig):

    cfg_dict = omegaconf_to_dict(config)
    print_dict(cfg_dict)

    config.seed = set_seed(config.seed)

    env = isaacgym_task_map[config.task_name](
        config=omegaconf_to_dict(config.task),
        sim_device=config.sim_device,
        graphics_device_id=config.graphics_device_id,
        headless=config.headless,
    )

    output_dif = os.path.join('outputs', config.train.ppo.output_name)
    os.makedirs(output_dif, exist_ok=True)
    agent = eval(config.train.algo)(env, output_dif, full_config=config, student_dim=24)
    agent.restore_test(config.train.load_path)

    
    # Run episodes
    for episode in range(1):
        agent.set_eval()
        obs_dict = env.reset()
        
        frame_idx = 0
        step_c = 0
        
        while step_c < 3:
            if isinstance(agent, ProprioAdapt):
                student_obs = obs_dict['obs']
                input_dict = {
                    'obs': agent.running_mean_std(student_obs),
                    'proprio_hist': agent.sa_mean_std(obs_dict['proprio_hist']),
                    'point_cloud_info': agent.point_cloud_mean_std(obs_dict['point_cloud_info'])
                    if agent.normalize_point_cloud
                    else obs_dict['point_cloud_info'],
                }
            else:
                raise ValueError(f"Unsupported agent type during evaluation: {type(agent)}")

            mu_gt, extrin, extrin_gt = agent.model.act_inference(input_dict)
            mu = torch.clamp(mu_gt, -1.0, 1.0)

            obs_dict, r, done, info = env.step(mu, extrin_record=None)
            step_c += 1
            print(f"Step {step_c}")

        print(f"Episode {episode + 1} finished after {step_c} steps")

    # Always export a JIT model (filename can be overridden via env var or Hydra config)
    default_jit_name = "student_policy.pt"
    env_jit_name = os.environ.get("JIT_OUTPUT_NAME")
    if env_jit_name:
        jit_output_name = env_jit_name
    else:
        jit_output_name = getattr(config, "jit_output_name", default_jit_name)
    obs_dict_cpu = {}
    for key, value in obs_dict.items():
        if isinstance(value, torch.Tensor):
            obs_dict_cpu[key] = value.cpu()
        else:
            obs_dict_cpu[key] = value
    export_jit_model(agent, 'outputs', obs_dict_cpu, jit_output_name=jit_output_name)

    print("Done")


if __name__ == "__main__":
    main()