import wandb
import torch
import os
import shutil
import time
import math
import io
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.io as spio
import copy
import torch.optim as optim

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from collections import OrderedDict
from datetime import datetime
from sklearn import svm 
from utils import diff_operators
from utils.error_evaluators import scenario_optimization, ValueThresholdValidator, MultiValidator, MLPConditionedValidator, target_fraction, MLP, MLPValidator, SliceSampleGenerator
import scipy.io as spio
from scipy.stats import beta as beta__dist
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

import hj_reachability as hj
import jax.numpy as jnp



class Experiment(ABC):
    def __init__(self, model, dataset, experiment_dir, use_wandb):
        self.model = model
        self.dataset = dataset
        self.experiment_dir = experiment_dir
        self.use_wandb = use_wandb

    @abstractmethod
    def init_special(self):
        raise NotImplementedError

    def _load_checkpoint(self, epoch):
        if epoch == -1:
            model_path = os.path.join(self.experiment_dir, 'training', 'checkpoints', 'model_final.pth')
            self.model.load_state_dict(torch.load(model_path))
        else:
            model_path = os.path.join(self.experiment_dir, 'training', 'checkpoints', 'model_epoch_%04d.pth' % epoch)
            self.model.load_state_dict(torch.load(model_path)['model'])

    def validate(self, device, epoch, save_path, x_resolution, y_resolution, z_resolution, time_resolution):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)
        v_val = 0.75  # --velocity
        w_val = 3.0   # --omega_max
        r_val = 0.25  # --collisionR
        plot_config = self.dataset.dynamics.plot_config()

        state_test_range = self.dataset.dynamics.state_test_range()
        x_min, x_max = state_test_range[plot_config['x_axis_idx']]
        y_min, y_max = state_test_range[plot_config['y_axis_idx']]
        z_min, z_max = state_test_range[plot_config['z_axis_idx']]

        hj_dynamics = hj.systems.Air3d(
            evader_speed=v_val, 
            pursuer_speed=v_val, 
            evader_max_turn_rate=w_val, 
            pursuer_max_turn_rate=w_val
        )
        hj_grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            hj.sets.Box(np.array([-1., -1., -np.pi]), 
                        np.array([1., 1., np.pi])),
            (51, 40, 50), 
            periodic_dims=2
        )
        initial_values = jnp.linalg.norm(hj_grid.states[..., :2], axis=-1) - r_val
        
        # Solver 설정 (Very High Accuracy)
        solver_settings = hj.SolverSettings.with_accuracy(
            "very_high", hamiltonian_postprocessor=hj.solver.backwards_reachable_tube
        )
        times = torch.linspace(0, self.dataset.tMax, time_resolution)
        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        zs = torch.linspace(z_min, z_max, z_resolution)
        xys = torch.cartesian_prod(xs, ys)
        n_cols = 4 
        n_plots = len(times) * len(zs)
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig = plt.figure(figsize=(6 * n_cols, 6 * n_rows))
        plot_idx = 1
        fig = plt.figure(figsize=(5*len(times), 5*len(zs)))
        for i in range(len(times)):
            current_t_scalar = float(times[i].cpu().numpy())
            
            if current_t_scalar == 0:
                hj_values = initial_values
            else:
                # 0초부터 -t초까지 역방향 계산
                hj_values = hj.step(solver_settings, hj_dynamics, hj_grid, 0., initial_values, -current_t_scalar)
            for j in range(len(zs)):
                coords = torch.zeros(x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
                coords[:, 0] = times[i]
                coords[:, 1:] = torch.tensor(plot_config['state_slices'])
                coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
                coords[:, 1 + plot_config['z_axis_idx']] = zs[j]

                with torch.no_grad():
                    model_results = self.model({'coords': self.dataset.dynamics.coord_to_input(coords.to(device))})
                    values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach())
                
                nn_grid_data = 1 * (values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T <= 0)
    
                # ----------------------------------------------------------
                # (B) Plotting
                # ----------------------------------------------------------
                ax = fig.add_subplot(n_rows,n_cols,plot_idx)
                plot_idx+=1
                ax.set_title('t = %0.2f, %s = %0.2f' % (times[i], plot_config['state_labels'][plot_config['z_axis_idx']], zs[j]),fontsize=12,fontweight='bold')
    
                # 1. Neural Net 결과 (Filled Area, 반투명)
                # extent를 정확히 맞춰야 HJ 결과와 겹쳐짐
                s = ax.imshow(nn_grid_data, cmap='bwr', origin='lower', 
                              extent=(x_min, x_max, y_min, y_max), alpha=0.6)
                
                # 2. HJ Solver 결과 (Contour Line)
                # 현재 z값(zs[j])과 가장 가까운 HJ Grid의 z 인덱스 찾기
                target_z = float(zs[j].cpu().numpy())
                # Air3D의 z축(theta)은 2번 인덱스라고 가정 (x, y, theta)
                z_idx_hj = np.abs(hj_grid.coordinate_vectors[2] - target_z).argmin()
                
                # 해당 z-slice 추출 및 Transpose (.T)
                # imshow의 x-y축과 contour의 x-y축 방향을 맞추기 위해 .T가 필요할 수 있음
                hj_slice = hj_values[:, :, z_idx_hj].T 
                
                # 검은색 실선으로 0-level set 그리기
                ax.contour(hj_grid.coordinate_vectors[0], 
                           hj_grid.coordinate_vectors[1], 
                           hj_slice, 
                           levels=[0], colors='black', linewidths=2.0)
        fig.savefig(save_path)
        if self.use_wandb:
            wandb.log({
                'step': epoch,
                'val_plot': wandb.Image(fig),
            })
        plt.close()

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)
            
    def validate1(self, device, epoch, save_path, x_resolution, y_resolution, z_resolution, time_resolution, time_step=None):
        was_training = self.target_model2.training
        self.target_model2.eval()
        self.target_model2.requires_grad_(False)
        v_val = 0.75  # --velocity
        w_val = 3.0   # --omega_max
        r_val = 0.25  # --collisionR
        plot_config = self.dataset.dynamics.plot_config()
    
        state_test_range = self.dataset.dynamics.state_test_range()
        x_min, x_max = state_test_range[plot_config['x_axis_idx']]
        y_min, y_max = state_test_range[plot_config['y_axis_idx']]
        z_min, z_max = state_test_range[plot_config['z_axis_idx']]
    
        # ------------------------------------------------------------------
        # 1. HJ Reachability (Ground Truth) 초기 설정
        # ------------------------------------------------------------------
        # Air3D Dynamics 및 Grid 설정
        hj_dynamics = hj.systems.Air3d(
            evader_speed=v_val, 
            pursuer_speed=v_val, 
            evader_max_turn_rate=w_val, 
            pursuer_max_turn_rate=w_val
        )
        hj_grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            hj.sets.Box(np.array([-1., -1., -np.pi]), 
                        np.array([1., 1., np.pi])),
            (51, 40, 50), 
            periodic_dims=2
        )
        # 초기 Value Function (Target Set: Cylinder r=5)
        initial_values = jnp.linalg.norm(hj_grid.states[..., :2], axis=-1) - r_val
        
        # Solver 설정 (Very High Accuracy)
        solver_settings = hj.SolverSettings.with_accuracy(
            "very_high", hamiltonian_postprocessor=hj.solver.backwards_reachable_tube
        )
    
        # ------------------------------------------------------------------
        # 2. 시각화 루프 준비
        # ------------------------------------------------------------------
        times = torch.linspace(epoch * time_step, (epoch + 1) * (time_step), time_resolution)
        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        zs = torch.linspace(z_min, z_max, z_resolution)
        xys = torch.cartesian_prod(xs, ys)
        n_cols = 4 
        n_plots = len(times) * len(zs)
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig = plt.figure(figsize=(6 * n_cols, 6 * n_rows))
        plot_idx = 1
        for i in range(len(times)):
            # [HJ Solver] 현재 시간(times[i])까지의 해 계산
            # Backward Reachability이므로 target_time은 음수 (-t)로 설정
            current_t_scalar = float(times[i].cpu().numpy())
            
            if current_t_scalar == 0:
                hj_values = initial_values
            else:
                # 0초부터 -t초까지 역방향 계산
                hj_values = hj.step(solver_settings, hj_dynamics, hj_grid, 0., initial_values, -current_t_scalar)
    
            for j in range(len(zs)):
                # ----------------------------------------------------------
                # (A) Neural Network Prediction (Grid Query)
                # ----------------------------------------------------------
                coords = torch.zeros(x_resolution * y_resolution, self.dataset.dynamics.state_dim + 1)
                coords[:, 0] = times[i]
                coords[:, 1:] = torch.tensor(plot_config['state_slices'])
                coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
                coords[:, 1 + plot_config['z_axis_idx']] = zs[j]
    
                with torch.no_grad():
                    model_results = self.target_model2({'coords': self.dataset.dynamics.coord_to_input(coords.to(device))})
                    values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach())
                
                # Data Shaping for Plot
                nn_grid_data = 1 * (values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T <= 0)
    
                # ----------------------------------------------------------
                # (B) Plotting
                # ----------------------------------------------------------
                ax = fig.add_subplot(n_rows,n_cols,plot_idx)
                plot_idx+=1
                ax.set_title('t = %0.2f, %s = %0.2f' % (times[i], plot_config['state_labels'][plot_config['z_axis_idx']], zs[j]),fontsize=12,fontweight='bold')
    
                # 1. Neural Net 결과 (Filled Area, 반투명)
                # extent를 정확히 맞춰야 HJ 결과와 겹쳐짐
                s = ax.imshow(nn_grid_data, cmap='bwr', origin='lower', 
                              extent=(x_min, x_max, y_min, y_max), alpha=0.6)
                
                # 2. HJ Solver 결과 (Contour Line)
                # 현재 z값(zs[j])과 가장 가까운 HJ Grid의 z 인덱스 찾기
                target_z = float(zs[j].cpu().numpy())
                # Air3D의 z축(theta)은 2번 인덱스라고 가정 (x, y, theta)
                z_idx_hj = np.abs(hj_grid.coordinate_vectors[2] - target_z).argmin()
                
                # 해당 z-slice 추출 및 Transpose (.T)
                # imshow의 x-y축과 contour의 x-y축 방향을 맞추기 위해 .T가 필요할 수 있음
                hj_slice = hj_values[:, :, z_idx_hj].T 
                
                # 검은색 실선으로 0-level set 그리기
                ax.contour(hj_grid.coordinate_vectors[0], 
                           hj_grid.coordinate_vectors[1], 
                           hj_slice, 
                           levels=[0], colors='black', linewidths=2.0)
    
                # (옵션) 범례 추가 시 주석 해제
                # if i == 0 and j == 0:
                #     fig.colorbar(s, ax=ax) 
    
        # 저장 및 로깅
        fig.tight_layout()
        fig.savefig(save_path)
        
        if self.use_wandb:
            wandb.log({
                'step': epoch,
                'val_plot': wandb.Image(fig),
            })
        plt.close()
    
        if was_training:
            self.target_model2.train()
            self.target_model2.requires_grad_(True)
            
    
    def train(
            self, device, batch_size, epochs, lr, 
            steps_til_summary, epochs_til_checkpoint, 
            loss_fn, clip_grad, use_lbfgs, adjust_relative_grads, 
            val_x_resolution, val_y_resolution, val_z_resolution, val_time_resolution,
            use_CSL, CSL_lr, CSL_dt, epochs_til_CSL, num_CSL_samples, CSL_loss_frac_cutoff, max_CSL_epochs, CSL_loss_weight, CSL_batch_size,
        ):
        was_eval = not self.model.training
        self.model.train()
        self.model.requires_grad_(True)
        self.device=device

        train_dataloader = DataLoader(self.dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=0)

        optim = torch.optim.Adam(lr=lr, params=self.model.parameters())

        # copy settings from Raissi et al. (2019) and here 
        # https://github.com/maziarraissi/PINNs
        if use_lbfgs:
            optim = torch.optim.LBFGS(lr=lr, params=self.model.parameters(), max_iter=50000, max_eval=50000,
                                    history_size=50, line_search_fn='strong_wolfe')

        training_dir = os.path.join(self.experiment_dir, 'training')
        
        summaries_dir = os.path.join(training_dir, 'summaries')
        if not os.path.exists(summaries_dir):
            os.makedirs(summaries_dir)

        checkpoints_dir = os.path.join(training_dir, 'checkpoints')
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        writer = SummaryWriter(summaries_dir)

        total_steps = 0

        if adjust_relative_grads:
            new_weight = 1

        with tqdm(total=len(train_dataloader) * epochs) as pbar:
            train_losses = []
            last_CSL_epoch = -1
            for epoch in range(0, epochs):
                if self.dataset.pretrain: # skip CSL
                    last_CSL_epoch = epoch
                time_interval_length = (self.dataset.counter/self.dataset.counter_end)*(self.dataset.tMax-self.dataset.tMin)
                CSL_tMax = self.dataset.tMin + int(time_interval_length/CSL_dt)*CSL_dt
                
                # self-supervised learning
                for step, (model_input, gt) in enumerate(train_dataloader):
                    start_time = time.time()
                
                    model_input = {key: value.to(device) for key, value in model_input.items()}
                    gt = {key: value.to(device) for key, value in gt.items()}
                    if 'model_coords' in model_input:
                        model_input['model_coords'].requires_grad_(True)
                    elif 'coords' in model_input:
                        model_input['coords'].requires_grad_(True)
                    model_results = self.model({'coords': model_input['model_coords']})

                    states = self.dataset.dynamics.input_to_coord(model_results['model_in'].detach())[..., 1:]
                    values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1))
                    dvs = self.dataset.dynamics.io_to_dv(model_results['model_in'], model_results['model_out'].squeeze(dim=-1))
                    boundary_values = gt['boundary_values']
                    if self.dataset.dynamics.loss_type == 'brat_hjivi':
                        reach_values = gt['reach_values']
                        avoid_values = gt['avoid_values']
                    dirichlet_masks = gt['dirichlet_masks']

                    if self.dataset.dynamics.loss_type == 'brt_hjivi':
                        losses = loss_fn(states, values, dvs[..., 0], dvs[..., 1:], boundary_values, dirichlet_masks, model_results['model_out'])
                    elif self.dataset.dynamics.loss_type == 'brat_hjivi':
                        losses = loss_fn(states, values, dvs[..., 0], dvs[..., 1:], boundary_values, reach_values, avoid_values, dirichlet_masks, model_results['model_out'])
                    else:
                        raise NotImplementedError
                    
                    if use_lbfgs:
                        def closure():
                            optim.zero_grad()
                            train_loss = 0.
                            for loss_name, loss in losses.items():
                                train_loss += loss.mean() 
                            train_loss.backward()
                            return train_loss
                        optim.step(closure)

                    # Adjust the relative magnitude of the losses if required
                    if self.dataset.dynamics.deepreach_model in ['vanilla', 'diff'] and adjust_relative_grads:
                        if losses['diff_constraint_hom'] > 0.01:
                            params = OrderedDict(self.model.named_parameters())
                            # Gradients with respect to the PDE loss
                            optim.zero_grad()
                            losses['diff_constraint_hom'].backward(retain_graph=True)
                            grads_PDE = []
                            for key, param in params.items():
                                grads_PDE.append(param.grad.view(-1))
                            grads_PDE = torch.cat(grads_PDE)

                            # Gradients with respect to the boundary loss
                            optim.zero_grad()
                            losses['dirichlet'].backward(retain_graph=True)
                            grads_dirichlet = []
                            for key, param in params.items():
                                grads_dirichlet.append(param.grad.view(-1))
                            grads_dirichlet = torch.cat(grads_dirichlet)

                            # # Plot the gradients
                            # import seaborn as sns
                            # import matplotlib.pyplot as plt
                            # fig = plt.figure(figsize=(5, 5))
                            # ax = fig.add_subplot(1, 1, 1)
                            # ax.set_yscale('symlog')
                            # sns.distplot(grads_PDE.cpu().numpy(), hist=False, kde_kws={"shade": False}, norm_hist=True)
                            # sns.distplot(grads_dirichlet.cpu().numpy(), hist=False, kde_kws={"shade": False}, norm_hist=True)
                            # fig.savefig('gradient_visualization.png')

                            # fig = plt.figure(figsize=(5, 5))
                            # ax = fig.add_subplot(1, 1, 1)
                            # ax.set_yscale('symlog')
                            # grads_dirichlet_normalized = grads_dirichlet * torch.mean(torch.abs(grads_PDE))/torch.mean(torch.abs(grads_dirichlet))
                            # sns.distplot(grads_PDE.cpu().numpy(), hist=False, kde_kws={"shade": False}, norm_hist=True)
                            # sns.distplot(grads_dirichlet_normalized.cpu().numpy(), hist=False, kde_kws={"shade": False}, norm_hist=True)
                            # ax.set_xlim([-1000.0, 1000.0])
                            # fig.savefig('gradient_visualization_normalized.png')

                            # Set the new weight according to the paper
                            # num = torch.max(torch.abs(grads_PDE))
                            num = torch.mean(torch.abs(grads_PDE))
                            den = torch.mean(torch.abs(grads_dirichlet))
                            new_weight = 0.9*new_weight + 0.1*num/den
                            losses['dirichlet'] = new_weight*losses['dirichlet']
                        writer.add_scalar('weight_scaling', new_weight, total_steps)

                    # import ipdb; ipdb.set_trace()

                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()

                        if loss_name == 'dirichlet':
                            writer.add_scalar(loss_name, single_loss/new_weight, total_steps)
                        else:
                            writer.add_scalar(loss_name, single_loss, total_steps)
                        train_loss += single_loss

                    train_losses.append(train_loss.item())
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                    if not total_steps % steps_til_summary:
                        torch.save(self.model.state_dict(),
                                os.path.join(checkpoints_dir, 'model_current.pth'))
                        # summary_fn(model, model_input, gt, model_output, writer, total_steps)

                    if not use_lbfgs:
                        optim.zero_grad()
                        train_loss.backward()

                        if clip_grad:
                            if isinstance(clip_grad, bool):
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.)
                            else:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad)

                        optim.step()

                    pbar.update(1)

                    if not total_steps % steps_til_summary:
                        tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
                        if self.use_wandb:
                            wandb.log({
                                'step': epoch,
                                'train_loss': train_loss,
                                'pde_loss': losses['diff_constraint_hom'],
                            })

                    total_steps += 1
                if not (epoch+1) % epochs_til_checkpoint:
                    # Saving the optimizer state is important to produce consistent results
                    checkpoint = { 
                        'epoch': epoch+1,
                        'model': self.model.state_dict(),
                        'optimizer': optim.state_dict()}
                    torch.save(checkpoint,
                        os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % (epoch+1)))
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % (epoch+1)),
                        np.array(train_losses))
                    self.validate(
                        device=device, epoch=epoch+1, save_path=os.path.join(checkpoints_dir, 'BRS_validation_plot_epoch_%04d.png' % (epoch+1)),
                        x_resolution = val_x_resolution, y_resolution = val_y_resolution, z_resolution=val_z_resolution, time_resolution=val_time_resolution)    

                    


       # Boundary Condition Helper
        def h_func(states):
            return self.dataset.dynamics.boundary_fn(states)

        # --------------------------------------------------------------------------
        # 1. 초기 설정
        # --------------------------------------------------------------------------
        refine_segments = 50
        refine_epochs_per_segment = 300
        VI_steps_refine = 10
        
        # [수정] 한 세그먼트 당 총 업데이트 횟수 계산 (Loop Flattening)
        # 데이터를 1번 뽑고 20번 학습하는게 아니라, 20배 더 많이 데이터를 뽑으며 1번씩 학습합니다.
        total_steps_per_segment = refine_epochs_per_segment * VI_steps_refine 
        
        segment_dt = self.dataset.tMax / refine_segments
        refine_losses = []

        # Target Network 1: Temporal Neighbor (V(t-dt)) 담당 (완전 고정)
        self.target_model1 = copy.deepcopy(self.model)
        for param in self.target_model1.parameters():
            param.requires_grad = False

        print("\n" + "="*80)
        print(">>> Starting Stage 2: Backward Refinement (Sharpening the Boundary) <<<")
        print(f"Total Segments: {refine_segments}, Steps per Segment: {total_steps_per_segment}")
        print("="*80)
        
        # --------------------------------------------------------------------------
        # 2. 세그먼트 루프 (Time-to-Go: 0 -> T 방향으로 진행)
        # --------------------------------------------------------------------------
        for seg_idx in range(refine_segments):
            if seg_idx == 0:
                continue
            
            t_start = ((seg_idx) * segment_dt)
            t_end = min(self.dataset.tMax, t_start + segment_dt)
            
            print(f"--- Refinement Segment {seg_idx}/{refine_segments}: [{t_start:.2f}, {t_end:.2f}] ---")

            # Main Model (Learner) 설정
            self.target_model2 = copy.deepcopy(self.model)
            self.target_model2.train() 
            for param in self.target_model2.parameters():
                param.requires_grad = True
            
            self.optim_refine = torch.optim.Adam(self.target_model2.parameters(), lr=1e-5)

            # [핵심] Spatial Target Net 설정 (V(x_next) 담당, Soft Update 대상)
            self.spatial_target_net = copy.deepcopy(self.target_model2)
            self.spatial_target_net.eval()
            for param in self.spatial_target_net.parameters():
                param.requires_grad = False
            loss_fn = torch.nn.HuberLoss(delta=0.1,reduction='none')
            # ----------------------------------------------------------------------
            # 3. 학습 루프 (Resampling Loop)
            # ----------------------------------------------------------------------
            with tqdm(total=total_steps_per_segment, desc=f"Refining [{t_start:.2f}~{t_end:.2f}]") as pbar:
                for step in range(total_steps_per_segment):
                    
                    # (A) 매 스텝마다 데이터 새로 샘플링 (발산 방지 핵심!)
                    batch_size = int(8192 * 4)
                    n_steps_trajectory = 5
                    n_uniform = int(batch_size * 0.4)
                    n_seeds = int((batch_size - n_uniform) / (1 + n_steps_trajectory))

                    states_uniform = self.dataset.sample_states(n_uniform).to(self.device)
                    temp_states = self.dataset.sample_states(int(n_seeds * 2)).to(self.device)
                    temp_t = torch.ones((temp_states.shape[0], 1), device=self.device) * t_start
                    temp_coords = torch.cat([temp_t, temp_states], dim=-1)

                    # 시드 선별 (경계면 근처)
                    with torch.no_grad():
                        temp_in = self.dataset.dynamics.coord_to_input(temp_coords)
                        # 여기선 target_model1(안정된 모델) 기준으로 샘플링
                        temp_out = self.spatial_target_net({'coords': temp_in})['model_out']
                        temp_val = self.dataset.dynamics.io_to_value(temp_in, temp_out.squeeze(-1))
                        _, indices = torch.sort(torch.abs(temp_val), descending=False)
                        seed_states = temp_states[indices[:n_seeds]]

                    # Trajectory 생성
                    traj_states_list = [seed_states] 
                    curr_stream = seed_states.clone()
                    
                    with torch.enable_grad(): # Gradient for optimal control
                        for _ in range(n_steps_trajectory):
                            curr_stream.requires_grad_(True)
                            t_input = torch.ones((curr_stream.shape[0], 1), device=self.device) * t_start
                            stream_coords = torch.cat([t_input, curr_stream], dim=-1)
                            u_dim = self.dataset.dynamics.control_dim
                            v_dim = self.dataset.dynamics.disturbance_dim
                            u_rand = torch.rand((curr_stream.shape[0], u_dim), device=self.device) * 6.0 - 3.0
                            stream_in = self.dataset.dynamics.coord_to_input(stream_coords)
                            # Control 계산 시에는 target_model1 사용 (안정성)
                            stream_out = self.spatial_target_net({'coords': stream_in})['model_out']
                            val_stream = self.dataset.dynamics.io_to_value(stream_in, stream_out.squeeze(-1))
                            
                            grads = torch.autograd.grad(outputs=val_stream, inputs=curr_stream,
                                                        grad_outputs=torch.ones_like(val_stream),
                                                        create_graph=False)[0]
                            dv_dx_stream = grads.detach()
                            
                            u_opt = self.dataset.dynamics.optimal_control(curr_stream.detach(), dv_dx_stream)
                            v_rand = torch.rand((curr_stream.shape[0], v_dim), device=self.device) * 6.0 - 3.0
                            epsilon = 0.5
                            mask = (torch.rand((curr_stream.shape[0], 1), device=self.device) < epsilon).float()
                            v_opt = self.dataset.dynamics.optimal_disturbance(curr_stream.detach(), dv_dx_stream)
                            v_mix = mask * v_rand + (1.0 - mask) * v_opt
                            u_mix = mask * u_rand + (1.0 - mask) * u_opt
                            sim_dt = 0.08
                            k1 = self.dataset.dynamics.dsdt(curr_stream.detach(), u_mix, v_opt)
                        
                            # k2: 중간점 1 (0.5 dt 이동)
                            x_k2 = curr_stream.detach() + 0.5 * sim_dt * k1
                            x_k2[..., 2] = (x_k2[..., 2] + math.pi) % (2 * math.pi) - math.pi # 각도 정규화
                            k2 = self.dataset.dynamics.dsdt(x_k2, u_mix, v_opt)
                        
                            # k3: 중간점 2 (0.5 dt 이동, k2 기울기 사용)
                            x_k3 = curr_stream.detach() + 0.5 * sim_dt * k2
                            x_k3[..., 2] = (x_k3[..., 2] + math.pi) % (2 * math.pi) - math.pi
                            k3 = self.dataset.dynamics.dsdt(x_k3, u_mix, v_opt)
                        
                            # k4: 끝점 (dt 이동, k3 기울기 사용)
                            x_k4 = curr_stream.detach() + sim_dt * k3
                            x_k4[..., 2] = (x_k4[..., 2] + math.pi) % (2 * math.pi) - math.pi
                            k4 = self.dataset.dynamics.dsdt(x_k4, u_mix, v_opt)
                        
                            # 최종 RK4 업데이트 (가중 평균)
                            # 오차 O(dt^5)로 줄어들어 원 궤도를 정확히 따라감
                            dx_rk4 = (sim_dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                            next_x_space = curr_stream.detach() + dx_rk4
                        
                            # 최종 위치 각도 정규화
                            next_x_space[..., 2] = (next_x_space[..., 2] + math.pi) % (2 * math.pi) - math.pi
                            # ds = self.dataset.dynamics.dsdt(curr_stream.detach(), u_mix, v_opt)
                            
                            # # Trajectory step (여기 dt는 물리 시뮬용)
                            # next_stream = curr_stream.detach() + ds * 0.06
                            # next_stream[..., 2] = (next_stream[..., 2] + math.pi) % (2 * math.pi) - math.pi
                            traj_states_list.append(next_x_space)
                            curr_stream = next_x_space

                    states_trajectory = torch.cat(traj_states_list, dim=0).detach()
                    aug_N = int(batch_size * 0.01)  # 전체 배치의 5%
                    aug_states = self.dataset.sample_states(aug_N).to(self.device)

                    sides = torch.randint(0, 2, (aug_N,), device=self.device).float() 
                    aug_angle = (1 - sides) * (-math.pi) + sides * (math.pi)
                    aug_states[:, 2] = (aug_angle + math.pi) % (2 * math.pi) - math.pi
                    
                    # 상태 벡터의 각도 차원(index 2)에 덮어쓰기
                    aug_states[:, 2] = aug_angle
                    temp_states = torch.cat([states_uniform, states_trajectory], dim=0)

                    
                    cutoff_idx = batch_size - aug_N
                    if temp_states.shape[0] > cutoff_idx:
                        temp_states = temp_states[:cutoff_idx]
                    
                    rand_states = torch.cat([temp_states, aug_states], dim=0)

                    # (혹시라도 모자랄 경우를 대비한 안전장치)
                    if rand_states.shape[0] > batch_size:
                        rand_states = rand_states[:batch_size]

                    rand_times = torch.ones((rand_states.shape[0], 1), device=self.device) * t_start
                    batch_coords = torch.cat([rand_times, rand_states], dim=-1).detach()
                    
                    # --------------------------------------------------------------
                    # (B) Target Calculation (안정화된 로직)
                    # --------------------------------------------------------------
                    batch_coords.requires_grad_(True)
                    
                    # 1. Optimal Control u, v 계산
                    model_input = self.dataset.dynamics.coord_to_input(batch_coords)
                    model_input = model_input.detach().requires_grad_(True)
                    
                    # Control을 구할 때는 현재 학습중인 모델(target_model2)의 Gradient를 참조해도 됨 (혹은 spatial_target_net)
                    output_raw = self.spatial_target_net({'coords': model_input})['model_out']
                    grads = self.dataset.dynamics.io_to_dv(model_input, output_raw.squeeze(-1))
                    dv_dx = grads[..., 1:]
                    
                    batch_states = batch_coords[..., 1:]
                    with torch.no_grad():
                        u_fixed = self.dataset.dynamics.optimal_control(batch_states, dv_dx)
                        v_fixed = self.dataset.dynamics.optimal_disturbance(batch_states, dv_dx)

                    batch_coords.requires_grad_(False)

                    with torch.no_grad():
                        curr_t = batch_coords[..., 0:1]
                        curr_x = batch_coords[..., 1:]

                        # [Temporal Target] t - dt (Time-to-Go가 줄어드는 방향 = Future/Target)
                        # 반드시 Frozen Model (target_model1) 사용
                        prev_t_time = torch.clamp(curr_t - segment_dt, min=self.dataset.tMin)
                        coords_T = torch.cat([prev_t_time, curr_x], dim=-1)
                        in_T = self.dataset.dynamics.coord_to_input(coords_T)
                        out_T_raw = self.target_model1({'coords': in_T})['model_out']
                        val_T = self.dataset.dynamics.io_to_value(in_T, out_T_raw.squeeze(-1)).unsqueeze(-1)

                        # [Spatial Target] x_next (Equation 4 RHS)
                        # 반드시 Spatial Snapshot (spatial_target_net) 사용
                        k1 = self.dataset.dynamics.dsdt(curr_x, u_fixed, v_fixed)
                        
                        # k2: 중간점 1 (0.5 dt 이동)
                        x_k2 = curr_x + 0.5 * segment_dt * k1
                        x_k2[..., 2] = (x_k2[..., 2] + math.pi) % (2 * math.pi) - math.pi # 각도 정규화
                        k2 = self.dataset.dynamics.dsdt(x_k2, u_fixed, v_fixed)
                        
                        # k3: 중간점 2 (0.5 dt 이동, k2 기울기 사용)
                        x_k3 = curr_x + 0.5 * segment_dt * k2
                        x_k3[..., 2] = (x_k3[..., 2] + math.pi) % (2 * math.pi) - math.pi
                        k3 = self.dataset.dynamics.dsdt(x_k3, u_fixed, v_fixed)
                        
                        # k4: 끝점 (dt 이동, k3 기울기 사용)
                        x_k4 = curr_x + segment_dt * k3
                        x_k4[..., 2] = (x_k4[..., 2] + math.pi) % (2 * math.pi) - math.pi
                        k4 = self.dataset.dynamics.dsdt(x_k4, u_fixed, v_fixed)
                        
                        # 최종 RK4 업데이트 (가중 평균)
                        # 오차 O(dt^5)로 줄어들어 원 궤도를 정확히 따라감
                        dx_rk4 = (segment_dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                        next_x_space = curr_x + dx_rk4
                        
                        # 최종 위치 각도 정규화
                        next_x_space[..., 2] = (next_x_space[..., 2] + math.pi) % (2 * math.pi) - math.pi
                        coords_S = torch.cat([curr_t, next_x_space], dim=-1)
                        in_S = self.dataset.dynamics.coord_to_input(coords_S)
                        
                        # [핵심 변경] target_model2 대신 spatial_target_net 사용
                        out_S_raw = self.spatial_target_net({'coords': in_S})['model_out']
                        val_S = self.dataset.dynamics.io_to_value(in_S, out_S_raw.squeeze(-1)).unsqueeze(-1)

                        # Mixing Logic (Alpha)
                        diff = torch.abs(val_S.detach() - val_T)
                        alpha = torch.tanh(diff * 10.0)
                        mixed_target = 0.505 * val_S.detach() + 0.505 * val_T
                        bellman_target = mixed_target # 필요시 alpha 적용: alpha * mixed + (1-alpha) * val_S
                        
                        # Boundary Condition (Min-Max)
                        h_val = h_func(curr_x).view(-1, 1)
                        correction_bias = 0.005
                        target_V = torch.min(h_val, bellman_target).detach()

                    # --------------------------------------------------------------
                    # (C) Loss & Update
                    # --------------------------------------------------------------
                    in_batch = self.dataset.dynamics.coord_to_input(batch_coords)
                    pred_raw = self.target_model2({'coords': in_batch})['model_out']
                    pred_V_phys = self.dataset.dynamics.io_to_value(in_batch, pred_raw.squeeze(-1)).unsqueeze(-1)
                    with torch.no_grad():
                        old_raw = self.spatial_target_net({'coords': in_batch})['model_out']
                        pred_V_old = self.dataset.dynamics.io_to_value(in_batch, old_raw.squeeze(-1)).unsqueeze(-1)
                    ppo_eps = 0.1
                    pred_V_clipped = pred_V_old + torch.clamp(pred_V_phys - pred_V_old, -ppo_eps, ppo_eps)
                    loss1=loss_fn(pred_V_clipped, target_V)
                    loss2=loss_fn(pred_V_phys,target_V)
                    loss=loss2
                    grad_norm = torch.norm(dv_dx, p=2, dim=-1)
                    min_slope = 0.001
                    slope_ratio = torch.clamp(grad_norm / min_slope, max=1.0)
                    
                    loss_barrier = torch.relu(min_slope-grad_norm)
                    # Boundary Pixel Weighting
                    band_width = 0.02
                    is_boundary = (torch.abs(target_V) < band_width).float()
                    pixel_weights = 1.0 + 1.0 * is_boundary # 필요 시 사용, 지금은 MSE 단순화
                    mask_narrow_band = (torch.abs(target_V) < band_width).float()
                    num_boundary_pixels = torch.sum(mask_narrow_band)
                    if num_boundary_pixels > 0:
                        loss_slope = torch.sum(loss_barrier * mask_narrow_band) / num_boundary_pixels
                    else:
                        loss_slope = torch.tensor(0.0, device=self.device)
                    MSE = (pred_V_phys - target_V) ** 2
                    k_lse = 50.0 
                    errors = torch.abs(pred_V_phys - target_V)
                    lse_loss = torch.logsumexp(k_lse * errors, dim=0) / k_lse
                    # loss = torch.mean(loss)+1.0*loss_slope 

                    # normal_grad = self.dataset.dynamics._last_normal_grad

                    # 전체 배치 중 마지막 5%(Boundary 조건)만 추출
                    # boundary_size = int(normal_grad.shape[0] * 0.05)
                    # normal_grad_bound = normal_grad[-boundary_size:]
                    
                    # # 페널티 Loss 생성
                    # lambda_normal = 1.0 # 학습이 불안정하면 0.1로, 발산하면 10.0으로 조절
                    # loss_normal = torch.mean(normal_grad_bound ** 2)
                    
                    # 5. 페널티 Loss 생성 (MSE)
                    loss=lse_loss+loss_slope
                    self.optim_refine.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.target_model2.parameters(), max_norm=1.0)
                    self.optim_refine.step()

                    # --------------------------------------------------------------
                    # (D) Soft Update
                    # --------------------------------------------------------------
                    # with torch.no_grad():
                    #     tau = 0.005 # 매 스텝 0.5%씩 반영 (Resampling 하므로 충분함)
                    #     for target_param, main_param in zip(self.spatial_target_net.parameters(), self.target_model2.parameters()):
                    #         target_param.data.copy_(
                    #             (1.0 - tau) * target_param.data + tau * main_param.data
                    #         )
                    target_update_freq = 10
                    if step % target_update_freq == 0:
                        self.spatial_target_net.load_state_dict(self.target_model2.state_dict())
                    refine_losses.append(loss.item())
                    
                    if step % 100 == 0:
                        pbar.set_postfix({'Loss': f"{loss.item():.2e}"})
                        pbar.update(100)

            # ----------------------------------------------------------------------
            # 4. 세그먼트 종료 후 처리
            # ----------------------------------------------------------------------
            checkpoint = {
                'segment_idx': seg_idx,
                'time_range': (t_start, t_end),
                'model': self.target_model2.state_dict(),
                'optimizer': self.optim_refine.state_dict()
            }
            save_name = f'refine_seg_{seg_idx:02d}_T_{t_start:.2f}.pth'
            torch.save(checkpoint, os.path.join(checkpoints_dir, save_name))
            np.savetxt(os.path.join(checkpoints_dir, 'refine_losses.txt'), np.array(refine_losses))

            # Validation Plot
            plot_name = f'Refine_Plot_Seg_{seg_idx:02d}_T_{t_start:.2f}.png'
            self.validate1(
                device=self.device, 
                epoch=seg_idx,
                save_path=os.path.join(checkpoints_dir, plot_name),
                x_resolution=val_x_resolution, y_resolution=val_y_resolution, 
                z_resolution=val_z_resolution, time_resolution=val_time_resolution,
                time_step=segment_dt
            )
            
            print(f"[Segment {seg_idx}] Saved: {save_name} | Plot: {plot_name} | Final Loss: {loss.item():.5f}")
            
            # 다음 세그먼트(Time-to-Go가 더 큰 시간)를 위해 현재 모델을 Freeze하여 target_model1으로 승격
            self.target_model1 = copy.deepcopy(self.target_model2)
            for param in self.target_model1.parameters():
                param.requires_grad = False

        print("\n" + "="*80)
        print("Pure Backward Refinement Finished Successfully.")
        print("="*80)      
        
        with tqdm(total=len(train_dataloader) * epochs) as pbar:

            last_CSL_epoch = -1
            for epoch in range(0, epochs):
                if self.dataset.pretrain: # skip CSL
                    last_CSL_epoch = epoch
                time_interval_length = (self.dataset.counter/self.dataset.counter_end)*(self.dataset.tMax-self.dataset.tMin)
                CSL_tMax = self.dataset.tMin + int(time_interval_length/CSL_dt)*CSL_dt    
                # cost-supervised learning (CSL) phase
                if use_CSL and not self.dataset.pretrain and (epoch-last_CSL_epoch) >= epochs_til_CSL:
                    last_CSL_epoch = epoch
                    
                    # generate CSL datasets
                    self.model.eval()

                    CSL_dataset = scenario_optimization(
                        device=device, model=self.model, policy=self.model, dynamics=self.dataset.dynamics,
                        tMin=self.dataset.tMin, tMax=CSL_tMax, dt=CSL_dt,
                        set_type="BRT", control_type="value", # TODO: implement option for BRS too
                        scenario_batch_size=min(num_CSL_samples, 100000), sample_batch_size=min(10*num_CSL_samples, 1000000),
                        sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=0.0),
                        max_scenarios=num_CSL_samples, tStart_generator=lambda n : torch.zeros(n).uniform_(self.dataset.tMin, CSL_tMax)
                    )
                    CSL_coords = torch.cat((CSL_dataset['times'].unsqueeze(-1), CSL_dataset['states']), dim=-1)
                    CSL_costs = CSL_dataset['costs']

                    num_CSL_val_samples = int(0.1*num_CSL_samples)
                    CSL_val_dataset = scenario_optimization(
                        model=self.model, policy=self.model, dynamics=self.dataset.dynamics,
                        tMin=self.dataset.tMin, tMax=CSL_tMax, dt=CSL_dt,
                        set_type="BRT", control_type="value", # TODO: implement option for BRS too
                        scenario_batch_size=min(num_CSL_val_samples, 100000), sample_batch_size=min(10*num_CSL_val_samples, 1000000),
                        sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=0.0),
                        max_scenarios=num_CSL_val_samples, tStart_generator=lambda n : torch.zeros(n).uniform_(self.dataset.tMin, CSL_tMax)
                    )
                    CSL_val_coords = torch.cat((CSL_val_dataset['times'].unsqueeze(-1), CSL_val_dataset['states']), dim=-1)
                    CSL_val_costs = CSL_val_dataset['costs']

                    CSL_val_tMax_dataset = scenario_optimization(
                        model=self.model, policy=self.model, dynamics=self.dataset.dynamics,
                        tMin=self.dataset.tMin, tMax=self.dataset.tMax, dt=CSL_dt,
                        set_type="BRT", control_type="value", # TODO: implement option for BRS too
                        scenario_batch_size=min(num_CSL_val_samples, 100000), sample_batch_size=min(10*num_CSL_val_samples, 1000000),
                        sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(v_min=float('-inf'), v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=0.0),
                        max_scenarios=num_CSL_val_samples # no tStart_generator, since I want all tMax times
                    )

                    
                    CSL_val_tMax_coords = torch.cat((CSL_val_tMax_dataset['times'].unsqueeze(-1), CSL_val_tMax_dataset['states']), dim=-1)
                    CSL_val_tMax_costs = CSL_val_tMax_dataset['costs']
                    
                    self.model.train()

                    # CSL optimizer
                    CSL_optim = torch.optim.Adam(lr=CSL_lr, params=self.model.parameters())

                    # initial CSL val loss
                    CSL_val_results = self.model({'coords': self.dataset.dynamics.coord_to_input(CSL_val_coords.to(device))})
                    CSL_val_preds = self.dataset.dynamics.io_to_value(CSL_val_results['model_in'], CSL_val_results['model_out'].squeeze(dim=-1))
                    CSL_val_errors = CSL_val_preds - CSL_val_costs.to(device)
                    CSL_val_loss = torch.mean(torch.pow(CSL_val_errors, 2))
                    CSL_initial_val_loss = CSL_val_loss
                    if self.use_wandb:
                        wandb.log({
                            "step": epoch,
                            "CSL_val_loss": CSL_val_loss.item()
                        })

                    # initial self-supervised learning (SSL) val loss
                    # right now, just took code from dataio.py and the SSL training loop above; TODO: refactor all this for cleaner modular code
                    CSL_val_states = CSL_val_coords[..., 1:].to(device)
                    CSL_val_dvs = self.dataset.dynamics.io_to_dv(CSL_val_results['model_in'], CSL_val_results['model_out'].squeeze(dim=-1))
                    CSL_val_boundary_values = self.dataset.dynamics.boundary_fn(CSL_val_states)
                    if self.dataset.dynamics.loss_type == 'brat_hjivi':
                        CSL_val_reach_values = self.dataset.dynamics.reach_fn(CSL_val_states)
                        CSL_val_avoid_values = self.dataset.dynamics.avoid_fn(CSL_val_states)
                    CSL_val_dirichlet_masks = CSL_val_coords[:, 0].to(device) == self.dataset.tMin # assumes time unit in dataset (model) is same as real time units
                    if self.dataset.dynamics.loss_type == 'brt_hjivi':
                        SSL_val_losses = loss_fn(CSL_val_states, CSL_val_preds, CSL_val_dvs[..., 0], CSL_val_dvs[..., 1:], CSL_val_boundary_values, CSL_val_dirichlet_masks)
                    elif self.dataset.dynamics.loss_type == 'brat_hjivi':
                        SSL_val_losses = loss_fn(CSL_val_states, CSL_val_preds, CSL_val_dvs[..., 0], CSL_val_dvs[..., 1:], CSL_val_boundary_values, CSL_val_reach_values, CSL_val_avoid_values, CSL_val_dirichlet_masks)
                    else:
                        NotImplementedError
                    SSL_val_loss = SSL_val_losses['diff_constraint_hom'].mean() # I assume there is no dirichlet (boundary) loss here, because I do not ever explicitly generate source samples at tMin (i.e. torch.all(CSL_val_dirichlet_masks == False))
                    if self.use_wandb:
                        wandb.log({
                            "step": epoch,
                            "SSL_val_loss": SSL_val_loss.item()
                        })

                    # CSL training loop
                    for CSL_epoch in tqdm(range(max_CSL_epochs)):
                        CSL_idxs = torch.randperm(num_CSL_samples)
                        for CSL_batch in range(math.ceil(num_CSL_samples/CSL_batch_size)):
                            CSL_batch_idxs = CSL_idxs[CSL_batch*CSL_batch_size:(CSL_batch+1)*CSL_batch_size]
                            CSL_batch_coords = CSL_coords[CSL_batch_idxs]

                            CSL_batch_results = self.model({'coords': self.dataset.dynamics.coord_to_input(CSL_batch_coords.to(device))})
                            CSL_batch_preds = self.dataset.dynamics.io_to_value(CSL_batch_results['model_in'], CSL_batch_results['model_out'].squeeze(dim=-1))
                            CSL_batch_costs = CSL_costs[CSL_batch_idxs].to(device)
                            CSL_batch_errors = CSL_batch_preds - CSL_batch_costs
                            CSL_batch_loss = CSL_loss_weight*torch.mean(torch.pow(CSL_batch_errors, 2))

                            CSL_batch_states = CSL_batch_coords[..., 1:].to(device)
                            CSL_batch_dvs = self.dataset.dynamics.io_to_dv(CSL_batch_results['model_in'], CSL_batch_results['model_out'].squeeze(dim=-1))
                            CSL_batch_boundary_values = self.dataset.dynamics.boundary_fn(CSL_batch_states)
                            if self.dataset.dynamics.loss_type == 'brat_hjivi':
                                CSL_batch_reach_values = self.dataset.dynamics.reach_fn(CSL_batch_states)
                                CSL_batch_avoid_values = self.dataset.dynamics.avoid_fn(CSL_batch_states)
                            CSL_batch_dirichlet_masks = CSL_batch_coords[:, 0].to(device) == self.dataset.tMin # assumes time unit in dataset (model) is same as real time units
                            if self.dataset.dynamics.loss_type == 'brt_hjivi':
                                SSL_batch_losses = loss_fn(CSL_batch_states, CSL_batch_preds, CSL_batch_dvs[..., 0], CSL_batch_dvs[..., 1:], CSL_batch_boundary_values, CSL_batch_dirichlet_masks)
                            elif self.dataset.dynamics.loss_type == 'brat_hjivi':
                                SSL_batch_losses = loss_fn(CSL_batch_states, CSL_batch_preds, CSL_batch_dvs[..., 0], CSL_batch_dvs[..., 1:], CSL_batch_boundary_values, CSL_batch_reach_values, CSL_batch_avoid_values, CSL_batch_dirichlet_masks)
                            else:
                                NotImplementedError
                            SSL_batch_loss = SSL_batch_losses['diff_constraint_hom'].mean() # I assume there is no dirichlet (boundary) loss here, because I do not ever explicitly generate source samples at tMin (i.e. torch.all(CSL_batch_dirichlet_masks == False))
                            
                            CSL_optim.zero_grad()
                            SSL_batch_loss.backward(retain_graph=True)
                            if (not use_lbfgs) and clip_grad: # no adjust_relative_grads, because I assume even with adjustment, the diff_constraint_hom remains unaffected and the only other loss (dirichlet) is zero
                                if isinstance(clip_grad, bool):
                                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.)
                                else:
                                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad)
                            CSL_batch_loss.backward()
                            CSL_optim.step()
                        
                        # evaluate on CSL_val_dataset
                        CSL_val_results = self.model({'coords': self.dataset.dynamics.coord_to_input(CSL_val_coords.to(device))})
                        CSL_val_preds = self.dataset.dynamics.io_to_value(CSL_val_results['model_in'], CSL_val_results['model_out'].squeeze(dim=-1))
                        CSL_val_errors = CSL_val_preds - CSL_val_costs.to(device)
                        CSL_val_loss = torch.mean(torch.pow(CSL_val_errors, 2))
                    
                        CSL_val_states = CSL_val_coords[..., 1:].to(device)
                        CSL_val_dvs = self.dataset.dynamics.io_to_dv(CSL_val_results['model_in'], CSL_val_results['model_out'].squeeze(dim=-1))
                        CSL_val_boundary_values = self.dataset.dynamics.boundary_fn(CSL_val_states)
                        if self.dataset.dynamics.loss_type == 'brat_hjivi':
                            CSL_val_reach_values = self.dataset.dynamics.reach_fn(CSL_val_states)
                            CSL_val_avoid_values = self.dataset.dynamics.avoid_fn(CSL_val_states)
                        CSL_val_dirichlet_masks = CSL_val_coords[:, 0].to(device) == self.dataset.tMin # assumes time unit in dataset (model) is same as real time units
                        if self.dataset.dynamics.loss_type == 'brt_hjivi':
                            SSL_val_losses = loss_fn(CSL_val_states, CSL_val_preds, CSL_val_dvs[..., 0], CSL_val_dvs[..., 1:], CSL_val_boundary_values, CSL_val_dirichlet_masks)
                        elif self.dataset.dynamics.loss_type == 'brat_hjivi':
                            SSL_val_losses = loss_fn(CSL_val_states, CSL_val_preds, CSL_val_dvs[..., 0], CSL_val_dvs[..., 1:], CSL_val_boundary_values, CSL_val_reach_values, CSL_val_avoid_values, CSL_val_dirichlet_masks)
                        else:
                            raise NotImplementedError
                        SSL_val_loss = SSL_val_losses['diff_constraint_hom'].mean() # I assume there is no dirichlet (boundary) loss here, because I do not ever explicitly generate source samples at tMin (i.e. torch.all(CSL_val_dirichlet_masks == False))
                    
                        CSL_val_tMax_results = self.model({'coords': self.dataset.dynamics.coord_to_input(CSL_val_tMax_coords.to(device))})
                        CSL_val_tMax_preds = self.dataset.dynamics.io_to_value(CSL_val_tMax_results['model_in'], CSL_val_tMax_results['model_out'].squeeze(dim=-1))
                        CSL_val_tMax_errors = CSL_val_tMax_preds - CSL_val_tMax_costs.to(device)
                        CSL_val_tMax_loss = torch.mean(torch.pow(CSL_val_tMax_errors, 2))
                        
                        # log CSL losses, recovered_safe_set_fracs
                        if self.dataset.dynamics.set_mode == 'reach':
                            CSL_train_batch_theoretically_recoverable_safe_set_frac = torch.sum(CSL_batch_costs.to(device) < 0) / len(CSL_batch_preds)
                            CSL_train_batch_recovered_safe_set_frac = torch.sum(CSL_batch_preds < torch.min(CSL_batch_preds[CSL_batch_costs.to(device) > 0])) / len(CSL_batch_preds)
                            CSL_val_theoretically_recoverable_safe_set_frac = torch.sum(CSL_val_costs.to(device) < 0) / len(CSL_val_preds)
                            CSL_val_recovered_safe_set_frac = torch.sum(CSL_val_preds < torch.min(CSL_val_preds[CSL_val_costs.to(device) > 0])) / len(CSL_val_preds)
                            CSL_val_tMax_theoretically_recoverable_safe_set_frac = torch.sum(CSL_val_tMax_costs.to(device) < 0) / len(CSL_val_tMax_preds)
                            CSL_val_tMax_recovered_safe_set_frac = torch.sum(CSL_val_tMax_preds < torch.min(CSL_val_tMax_preds[CSL_val_tMax_costs.to(device) > 0])) / len(CSL_val_tMax_preds)
                        elif self.dataset.dynamics.set_mode == 'avoid':
                            CSL_train_batch_theoretically_recoverable_safe_set_frac = torch.sum(CSL_batch_costs.to(device) > 0) / len(CSL_batch_preds)
                            CSL_train_batch_recovered_safe_set_frac = torch.sum(CSL_batch_preds > torch.max(CSL_batch_preds[CSL_batch_costs.to(device) < 0])) / len(CSL_batch_preds)
                            CSL_val_theoretically_recoverable_safe_set_frac = torch.sum(CSL_val_costs.to(device) > 0) / len(CSL_val_preds)
                            CSL_val_recovered_safe_set_frac = torch.sum(CSL_val_preds > torch.max(CSL_val_preds[CSL_val_costs.to(device) < 0])) / len(CSL_val_preds)
                            CSL_val_tMax_theoretically_recoverable_safe_set_frac = torch.sum(CSL_val_tMax_costs.to(device) > 0) / len(CSL_val_tMax_preds)
                            CSL_val_tMax_recovered_safe_set_frac = torch.sum(CSL_val_tMax_preds > torch.max(CSL_val_tMax_preds[CSL_val_tMax_costs.to(device) < 0])) / len(CSL_val_tMax_preds)
                        else:
                            raise NotImplementedError
                        if self.use_wandb:
                            wandb.log({
                                "step": epoch+(CSL_epoch+1)*int(0.5*epochs_til_CSL/max_CSL_epochs),
                                "CSL_train_batch_loss": CSL_batch_loss.item(),
                                "SSL_train_batch_loss": SSL_batch_loss.item(),
                                "CSL_val_loss": CSL_val_loss.item(),
                                "SSL_val_loss": SSL_val_loss.item(),
                                "CSL_val_tMax_loss": CSL_val_tMax_loss.item(),
                                "CSL_train_batch_theoretically_recoverable_safe_set_frac": CSL_train_batch_theoretically_recoverable_safe_set_frac.item(),
                                "CSL_val_theoretically_recoverable_safe_set_frac": CSL_val_theoretically_recoverable_safe_set_frac.item(),
                                "CSL_val_tMax_theoretically_recoverable_safe_set_frac": CSL_val_tMax_theoretically_recoverable_safe_set_frac.item(),
                                "CSL_train_batch_recovered_safe_set_frac": CSL_train_batch_recovered_safe_set_frac.item(),
                                "CSL_val_recovered_safe_set_frac": CSL_val_recovered_safe_set_frac.item(),
                                "CSL_val_tMax_recovered_safe_set_frac": CSL_val_tMax_recovered_safe_set_frac.item(),
                            })

                        if CSL_val_loss < CSL_loss_frac_cutoff*CSL_initial_val_loss:
                            break

                if not (epoch+1) % epochs_til_checkpoint:
                    # Saving the optimizer state is important to produce consistent results
                    checkpoint = { 
                        'epoch': epoch+1,
                        'model': self.model.state_dict(),
                        'optimizer': optim.state_dict()}
                    torch.save(checkpoint,
                        os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % (epoch+1)))
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % (epoch+1)),
                        np.array(train_losses))
                    self.validate(
                        device=device, epoch=epoch+1, save_path=os.path.join(checkpoints_dir, 'BRS_validation_plot_epoch_%04d.png' % (epoch+1)),
                        x_resolution = val_x_resolution, y_resolution = val_y_resolution, z_resolution=val_z_resolution, time_resolution=val_time_resolution)

        if was_eval:
            self.model.eval()
            self.model.requires_grad_(False)    
            
    def save_visualization(self, epoch, save_dir):
        try:
            resolution = 200
            x = np.linspace(-1, 1, resolution)
            y = np.linspace(-1, 1, resolution)
            X, Y = np.meshgrid(x, y)
            Theta = np.zeros_like(X) # 0도
            Time = np.ones_like(X) * self.dataset.tMax 

            coords = np.stack([Time, X, Y, Theta], axis=-1)
            coords_flat = coords.reshape(-1, 4)
            coords_tensor = torch.FloatTensor(coords_flat).to(self.device)
            
            # Dynamics 변환
            model_input = self.dataset.dynamics.coord_to_input(coords_tensor)

            with torch.no_grad():
                # 현재 학습 중인 model1 사용
                model_out = self.model1({'coords': model_input})['model_out']
                values = model_out.cpu().numpy().reshape(resolution, resolution)

            plt.figure(figsize=(6, 6))
            plt.title(f"Dubins Reachability (Theta=0)\nEpoch: {epoch}")
            cp = plt.contourf(X, Y, values, levels=20, cmap='RdBu_r')
            plt.colorbar(cp)
            plt.contour(X, Y, values, levels=[0], colors='black', linewidths=2)
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.grid(True)
            plt.arrow(0, 0, 0.2, 0, head_width=0.05, head_length=0.1, fc='k', ec='k')
            
            # 파일로 저장 (plt.show 대신)
            save_path = os.path.join(save_dir, f'vis_epoch_{epoch}.png')
            plt.savefig(save_path)
            plt.close() # 메모리 해제
            # tqdm.write(f">>> Image Saved: {save_path}")
            
        except Exception as e:
            tqdm.write(f"Warning: Visualization failed: {e}")

    def _load_checkpoint(self, epoch):
        if epoch == -1:
            model_path = os.path.join(self.experiment_dir, 'training', 'checkpoints', 'model_final.pth')
            self.model.load_state_dict(torch.load(model_path))
        else:
            model_path = os.path.join(self.experiment_dir, 'training', 'checkpoints', 'model_epoch_%d.pth' % epoch)
            self.model.load_state_dict(torch.load(model_path))

    def test(self, device, current_time, last_checkpoint, checkpoint_dt, dt, num_scenarios, num_violations, set_type, control_type, data_step, checkpoint_toload=None,
             gt_data_path=None):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)
        if data_step in ["plot_basic_recovery", 'run_basic_recovery', 'plot_ND', 'run_robust_recovery', 'plot_robust_recovery','eval_w_gt']:
            testing_dir = self.experiment_dir
        else:
            testing_dir = os.path.join(
                self.experiment_dir, 'testing_%s' % current_time.strftime('%m_%d_%Y_%H_%M'))
            if os.path.exists(testing_dir):
                overwrite = input(
                    "The testing directory %s already exists. Overwrite? (y/n)" % testing_dir)
                if not (overwrite == 'y'):
                    print('Exiting.')
                    quit()
                shutil.rmtree(testing_dir)
            os.makedirs(testing_dir)

        if checkpoint_toload is None:
            print('running cross-checkpoint testing')

            # checkpoint x simulation_time square matrices
            sidelen = 10
            assert (last_checkpoint /
                    checkpoint_dt) % sidelen == 0, 'checkpoints cannot be even divided by sidelen'
            BRT_volumes_matrix = np.zeros((sidelen, sidelen))
            BRT_errors_matrix = np.zeros((sidelen, sidelen))
            BRT_error_rates_matrix = np.zeros((sidelen, sidelen))
            BRT_error_region_fracs_matrix = np.zeros((sidelen, sidelen))

            exBRT_volumes_matrix = np.zeros((sidelen, sidelen))
            exBRT_errors_matrix = np.zeros((sidelen, sidelen))
            exBRT_error_rates_matrix = np.zeros((sidelen, sidelen))
            exBRT_error_region_fracs_matrix = np.zeros((sidelen, sidelen))

            checkpoints = np.linspace(0, last_checkpoint, num=sidelen+1)[1:]
            checkpoints[-1] = -1
            times = np.linspace(self.dataset.tMin,
                                self.dataset.tMax, num=sidelen+1)[1:]
            print('constructing matrices for')
            print('checkpoints:', checkpoints)
            print('times:', times)
            for i in tqdm(range(sidelen), desc='Checkpoint'):
                self._load_checkpoint(epoch=checkpoints[i])
                for j in tqdm(range(sidelen), desc='Simulation Time', leave=False):
                    # get BRT volume, error, error rate, error region fraction
                    results = scenario_optimization(
                        device=device, model=self.model,policy=self.model,dynamics=self.dataset.dynamics, tMin=self.dataset.tMin, tMax=times[
                            j], dt=dt,
                        set_type=set_type, control_type=control_type,
                        scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000),
                        sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(
                            v_min=0.0, v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(
                            v_min=float('-inf'), v_max=0.0),
                        max_scenarios=num_scenarios, max_samples=1000*num_scenarios)
                    BRT_volumes_matrix[i, j] = results['valid_sample_fraction']
                    if results['maxed_scenarios']:
                        BRT_errors_matrix[i,
                                          j] = results['max_violation_error']
                        BRT_error_rates_matrix[i,
                                               j] = results['violation_rate']
                        BRT_error_region_fracs_matrix[i, j] = target_fraction(device=device,
                            model=self.model, dynamics=self.dataset.dynamics, t=times[j],
                            sample_validator=ValueThresholdValidator(
                                v_min=float('-inf'), v_max=0.0),
                            target_validator=ValueThresholdValidator(
                                v_min=-results['max_violation_error'], v_max=0.0),
                            num_samples=num_scenarios, batch_size=min(10*num_scenarios, 1000000))
                    else:
                        BRT_errors_matrix[i, j] = np.NaN
                        BRT_error_rates_matrix[i, j] = np.NaN
                        BRT_error_region_fracs_matrix[i, j] = np.NaN

                    # get exBRT error, error rate, error region fraction
                    results = scenario_optimization(
                        device=device, model=self.model,policy=self.model,dynamics=self.dataset.dynamics, tMin=self.dataset.tMin, tMax=times[
                            j], dt=dt,
                        set_type=set_type, control_type=control_type,
                        scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000),
                        sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[None]*self.dataset.dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(
                            v_min=0.0, v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(
                            v_min=float('-inf'), v_max=0.0),
                        max_scenarios=num_scenarios, max_samples=1000*num_scenarios)
                    exBRT_volumes_matrix[i,
                                         j] = results['valid_sample_fraction']
                    if results['maxed_scenarios']:
                        exBRT_errors_matrix[i,
                                            j] = results['max_violation_error']
                        exBRT_error_rates_matrix[i,
                                                 j] = results['violation_rate']
                        exBRT_error_region_fracs_matrix[i, j] = target_fraction(
                            device=device,model=self.model, dynamics=self.dataset.dynamics, t=times[j],
                            sample_validator=ValueThresholdValidator(
                                v_min=0.0, v_max=float('inf')),
                            target_validator=ValueThresholdValidator(
                                v_min=0.0, v_max=results['max_violation_error']),
                            num_samples=num_scenarios, batch_size=min(10*num_scenarios, 1000000))
                    else:
                        exBRT_errors_matrix[i, j] = np.NaN
                        exBRT_error_rates_matrix[i, j] = np.NaN
                        exBRT_error_region_fracs_matrix[i, j] = np.NaN

            # save the matrices
            matrices = {
                'BRT_volumes_matrix': BRT_volumes_matrix,
                'BRT_errors_matrix': BRT_errors_matrix,
                'BRT_error_rates_matrix': BRT_error_rates_matrix,
                'BRT_error_region_fracs_matrix': BRT_error_region_fracs_matrix,
                'exBRT_volumes_matrix': exBRT_volumes_matrix,
                'exBRT_errors_matrix': exBRT_errors_matrix,
                'exBRT_error_rates_matrix': exBRT_error_rates_matrix,
                'exBRT_error_region_fracs_matrix': exBRT_error_region_fracs_matrix,
            }
            for name, arr in matrices.items():
                with open(os.path.join(testing_dir, f'{name}.npy'), 'wb') as f:
                    np.save(f, arr)

            # plot the matrices
            matrices = {
                'BRT_volumes_matrix': [
                    BRT_volumes_matrix, 'BRT Fractions of Test State Space'
                ],
                'BRT_errors_matrix': [
                    BRT_errors_matrix, 'BRT Errors'
                ],
                'BRT_error_rates_matrix': [
                    BRT_error_rates_matrix, 'BRT Error Rates'
                ],
                'BRT_error_region_fracs_matrix': [
                    BRT_error_region_fracs_matrix, 'BRT Error Region Fractions'
                ],
                'exBRT_volumes_matrix': [
                    exBRT_volumes_matrix, 'exBRT Fractions of Test State Space'
                ],
                'exBRT_errors_matrix': [
                    exBRT_errors_matrix, 'exBRT Errors'
                ],
                'exBRT_error_rates_matrix': [
                    exBRT_error_rates_matrix, 'exBRT Error Rates'
                ],
                'exBRT_error_region_fracs_matrix': [
                    exBRT_error_region_fracs_matrix, 'exBRT Error Region Fractions'
                ],
            }
            for name, data in matrices.items():
                cmap = matplotlib.cm.get_cmap('Reds')
                cmap.set_bad(color='blue')
                fig, ax = plt.subplots(1, 1)
                ax.set_xticks(range(sidelen))
                ax.set_yticks(range(sidelen))
                ax.set_xticklabels(np.round_(times, decimals=2))
                ax.set_yticklabels(np.linspace(
                    0, last_checkpoint, num=sidelen+1)[1:])
                plt.xlabel('Simulation Time')
                plt.ylabel('Checkpoint')
                ax.imshow(data[0], cmap=cmap)
                plt.title(data[1])
                for (y, x), label in np.ndenumerate(data[0]):
                    plt.text(x, y, '%.7f' %
                             label, ha='center', va='center', fontsize=4)
                plt.savefig(os.path.join(testing_dir, name + '.png'), dpi=600)
                plt.clf()
                # log version
                cmap = matplotlib.cm.get_cmap('Reds')
                cmap.set_bad(color='blue')
                fig, ax = plt.subplots(1, 1)
                ax.set_xticks(range(sidelen))
                ax.set_yticks(range(sidelen))
                ax.set_xticklabels(np.round_(times, decimals=2))
                ax.set_yticklabels(np.linspace(
                    0, last_checkpoint, num=sidelen+1)[1:])
                plt.xlabel('Simulation Time')
                plt.ylabel('Checkpoint')
                new_matrix = np.log(data[0])
                ax.imshow(new_matrix, cmap=cmap)
                plt.title('(Log) ' + data[1])
                for (y, x), label in np.ndenumerate(new_matrix):
                    plt.text(x, y, '%.7f' %
                             label, ha='center', va='center', fontsize=4)
                plt.savefig(os.path.join(
                    testing_dir, name + '_log' + '.png'), dpi=600)
                plt.clf()

        else:
            print('running specific-checkpoint testing')
            self._load_checkpoint(checkpoint_toload)

            model = self.model
            dataset = self.dataset
            dynamics = dataset.dynamics

            
            if data_step == "eval_w_gt":
                coords=torch.load(os.path.join(gt_data_path,"coords.pt")).to(device)
                gt_values=torch.load(os.path.join(gt_data_path,"gt_values.pt")).to(device)
                with torch.no_grad():
                    results = model(
                        {'coords': self.dataset.dynamics.coord_to_input(coords)})
                    pred_values = self.dataset.dynamics.io_to_value(results['model_in'].detach(
                        ), results['model_out'].squeeze(dim=-1).detach())
                mse = torch.pow(pred_values-gt_values,2).mean()


                # print(results['batch_state_trajs'].shape)
                gt_values=gt_values.cpu().numpy()
                pred_values=pred_values.cpu().numpy()
                fp=np.argwhere(np.logical_and(gt_values < 0, pred_values >= 0)).shape[0]/pred_values.shape[0]
                fn=np.argwhere(np.logical_and(gt_values >= 0, pred_values < 0)).shape[0]/pred_values.shape[0]
                np.save(os.path.join(
                    testing_dir, f"mse.npy"),mse.cpu().numpy())
                np.save(os.path.join(
                    testing_dir, f"fp.npy"),torch.tensor([fp]))
                np.save(os.path.join(
                    testing_dir, f"fn.npy"),torch.tensor([fn]))
                print("False positive: %0.4f, False negative: %0.4f"%(fp,
                        fn))
                print("MSE: ", mse)

            if data_step == 'plot_robust_recovery':
                epsilons=-np.load(os.path.join(testing_dir, f'epsilons.npy'))+1
                deltas=np.load(os.path.join(testing_dir, f'deltas.npy'))
                target_eps=0.01
                delta_level=deltas[np.argmin(np.abs(epsilons-target_eps))]
                fig,values_slices = self.plot_recovery_fig(
                    dataset, dynamics, model, delta_level)
                fig.savefig(os.path.join(
                    testing_dir, f'robust_BRTs_1e-2.png'), dpi=800)
                np.save(os.path.join(testing_dir, f'values_slices'),values_slices)

            if data_step == 'run_robust_recovery':
                logs = {}
                # rollout samples all over the state space
                beta_ = 1e-10
                N = 300000
                logs['beta_'] = beta_
                logs['N'] = N
                delta_level = float(
                    'inf') if dynamics.set_mode in ['reach','reach_avoid'] else float('-inf')

                results = scenario_optimization(device=device,
                    model=model, policy=model,dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=min(N, 100000), sample_batch_size=10*min(N, 10000),
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(v_min=float(
                        '-inf'), v_max=delta_level) if dynamics.set_mode in ['reach','reach_avoid'] else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode in ['reach','reach_avoid'] else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=N, max_samples=1000*min(N, 10000))

                sns.set_style('whitegrid')
                costs_ = results['costs'].cpu().numpy()
                values_ = results['values'].cpu().numpy()
                unsafe_cost_safe_value_indeces = np.argwhere(
                    np.logical_and(costs_ < 0, values_ >= 0))

                print("k max: ", unsafe_cost_safe_value_indeces.shape[0])

                # determine delta_level_max
                delta_level_max = np.max(
                    values_[unsafe_cost_safe_value_indeces])
                print("delta_level_max: ", delta_level_max)

                # for each delta level, determine (1) the corresponding volume;
                # (2) k and and corresponding epsilon
                ks = []
                epsilons = []
                volumes = []

                for delta_level_ in np.arange(0, delta_level_max, delta_level_max/100):
                    k = int(np.argwhere(np.logical_and(
                        costs_ < 0, values_ >= delta_level_)).shape[0])
                    eps = beta__dist.ppf(beta_,  N-k, k+1)
                    volume = values_[values_ >= delta_level_].shape[0]/values_.shape[0]
                    
                    ks.append(k)
                    epsilons.append(eps)
                    volumes.append(volume)

                # plot epsilon volume graph
                fig1, ax1 = plt.subplots()
                color = 'tab:red'
                ax1.set_xlabel('volumes')
                ax1.set_ylabel('epsilons', color=color)
                ax1.plot(volumes, epsilons, color=color)
                ax1.tick_params(axis='y', labelcolor=color)

                ax2 = ax1.twinx()

                color = 'tab:blue'
                ax2.set_ylabel('number of outliers', color=color)
                ax2.plot(volumes, ks, color=color)
                ax2.tick_params(axis='y', labelcolor=color)

                plt.title("beta_=1e-10, N =3e6")
                fig1.savefig(os.path.join(
                    testing_dir, f'robust_verification_results.png'), dpi=800)
                plt.close(fig1)
                np.save(os.path.join(testing_dir, f'epsilons'),
                        epsilons)
                np.save(os.path.join(testing_dir, f'volumes'),
                        volumes)
                np.save(os.path.join(testing_dir, f'deltas'),
                        np.arange(0, delta_level_max, delta_level_max/100))
                np.save(os.path.join(testing_dir, f'ks'),
                        ks)
                
            if data_step == 'run_basic_recovery':
                logs = {}

                # 0. explicit statement of probabilistic guarantees, N, \beta, \epsilon
                beta = 1e-16
                epsilon = 1e-3
                N = int(math.ceil((2/epsilon)*(np.log(1/beta)+1)))
                M = 5

                logs['beta'] = beta
                logs['epsilon'] = epsilon
                logs['N'] = N
                logs['M'] = M

                # 1. execute algorithm for tMax
                # record state/learned_value/violation for each while loop iteration
                delta_level = float(
                    'inf') if dynamics.set_mode in ['reach','reach_avoid'] else float('-inf')
                algorithm_iters = []
                for i in range(M):
                    print('algorithm iter', str(i))
                    results = scenario_optimization(device=device,
                        model=model, policy=model,dynamics=dynamics,
                        tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                        set_type=set_type, control_type=control_type,
                        scenario_batch_size=min(N, 100000), sample_batch_size=10*min(N, 10000),
                        sample_generator=SliceSampleGenerator(
                            dynamics=dynamics, slices=[None]*dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(v_min=float(
                            '-inf'), v_max=delta_level) if dynamics.set_mode in ['reach','reach_avoid'] else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                            'inf')) if dynamics.set_mode in ['reach','reach_avoid'] else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                        max_scenarios=N, max_samples=1000*min(N, 10000))
                    if not results['maxed_scenarios']:
                        delta_level = float(
                            '-inf') if dynamics.set_mode in ['reach','reach_avoid'] else float('inf')
                        break
                    algorithm_iters.append(
                        {
                            'states': results['states'],
                            'values': results['values'],
                            'violations': results['violations']
                        }
                    )
                    if results['violation_rate'] == 0:
                        break
                    violation_levels = results['values'][results['violations']]
                    delta_level_arg = np.argmin(
                        violation_levels) if dynamics.set_mode in ['reach','reach_avoid'] else np.argmax(violation_levels)
                    delta_level = violation_levels[delta_level_arg].item()

                    print('violation_rate:', str(results['violation_rate']))
                    print('delta_level:', str(delta_level))
                    print('valid_sample_fraction:', str(
                        results['valid_sample_fraction'].item()))
                    sns.set_style('whitegrid')
                    # density_plot=sns.kdeplot(results['costs'].cpu().numpy(), bw=0.5)
                    # density_plot=sns.displot(results['costs'].cpu().numpy(), x="cost function")
                    # fig1 = density_plot.get_figure()
                    fig1 = plt.figure()
                    plt.hist(results['costs'].cpu().numpy(), bins=200)
                    fig1.savefig(os.path.join(
                        testing_dir, f'cost distribution.png'), dpi=800)
                    plt.close(fig1)
                    fig2 = plt.figure()
                    plt.hist(results['costs'].cpu().numpy() -
                             results['values'].cpu().numpy(), bins=200)
                    fig2.savefig(os.path.join(
                        testing_dir, f'diff distribution.png'), dpi=800)
                    plt.close(fig1)

                logs['algorithm_iters'] = algorithm_iters
                logs['delta_level'] = delta_level

                # 2. record solution volume, recovered volume
                S = 1000000
                logs['S'] = S
                logs['learned_volume'] = target_fraction(device=device,
                    model=model, dynamics=dynamics, t=dataset.tMax,
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    target_validator=ValueThresholdValidator(v_min=float(
                        '-inf'), v_max=0.0) if dynamics.set_mode in ['reach','reach_avoid'] else ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
                    num_samples=S,
                    batch_size=min(S, 1000000),
                ).item()
                logs['recovered_volume'] = target_fraction(device=device,
                    model=model, dynamics=dynamics, t=dataset.tMax,
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    target_validator=ValueThresholdValidator(v_min=float(
                        '-inf'), v_max=delta_level) if dynamics.set_mode in ['reach','reach_avoid'] else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')),
                    num_samples=S,
                    batch_size=min(S, 1000000)
                ).item()

                results = scenario_optimization(device=device,
                    model=model, policy=model,dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=min(S, 100000), sample_batch_size=10*min(S, 10000),
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode in ['reach','reach_avoid'] else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=S, max_samples=1000*min(S, 10000))
                if results['maxed_scenarios']:
                    logs['theoretically_recoverable_volume'] = 1 - \
                        results['violation_rate']
                else:
                    logs['theoretically_recoverable_volume'] = 0

                print('learned_volume', str(logs['learned_volume']))
                print('recovered_volume', str(logs['recovered_volume']))
                print('theoretically_recoverable_volume', str(
                    logs['theoretically_recoverable_volume']))

                # 3. validate theoretical guarantees via mass sampling
                results = scenario_optimization(device=device,
                    model=model, policy=model,dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=min(S, 100000), sample_batch_size=10*min(S, 10000),
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(v_min=float(
                        '-inf'), v_max=delta_level) if dynamics.set_mode in ['reach','reach_avoid'] else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode in ['reach','reach_avoid'] else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=S, max_samples=1000*min(S, 10000))
                if results['maxed_scenarios']:
                    logs['recovered_violation_rate'] = results['violation_rate']
                else:
                    logs['recovered_violation_rate'] = 0
                print('recovered_violation_rate', str(
                    logs['recovered_violation_rate']))

                with open(os.path.join(testing_dir, 'basic_logs.pickle'), 'wb') as f:
                    pickle.dump(logs, f)

            if data_step == 'plot_basic_recovery':
                with open(os.path.join(self.experiment_dir, 'basic_logs.pickle'), 'rb') as f:
                    logs = pickle.load(f)

                # 0.
                print('N:', str(logs['N']))
                print('M:', str(logs['M']))
                print('beta:', str(logs['beta']))
                print('epsilon:', str(logs['epsilon']))
                print('S:', str(logs['S']))
                print('delta level', str(logs['delta_level']))
                delta_level = logs['delta_level']
                print('learned volume', str(logs['learned_volume']))
                print('recovered volume', str(logs['recovered_volume']))
                print('theoretically recoverable volume', str(
                    logs['theoretically_recoverable_volume']))
                print('recovered violation rate', str(
                    logs['recovered_violation_rate']))

                # fig, _ = self.plot_recovery_fig(
                #     dataset, dynamics, model, delta_level)
                plot_config = self.dataset.dynamics.plot_config()

                state_test_range = self.dataset.dynamics.state_test_range()
                
                times = [self.dataset.tMax]
                if plot_config['z_axis_idx'] == -1:
                    fig = self.plotSingleFig(
                        state_test_range, plot_config, 512, 512, times, delta_level)
                else:
                    fig= self.plotMultipleFigs(
                        state_test_range, plot_config, 512, 512, 5, times, delta_level)
                # plt.tight_layout()
                fig.savefig(os.path.join(
                    testing_dir, f'basic_BRTs.png'), dpi=800)
                np.save(os.path.join(
                    testing_dir, f'volumes.npy'), np.array([float(logs['learned_volume']),
                                                            float(logs['recovered_volume']),float(logs['theoretically_recoverable_volume'])]))

            if data_step == 'plot_ND':
                with open(os.path.join(self.experiment_dir, 'basic_logs.pickle'), 'rb') as f:
                    logs = pickle.load(f)

                # 0.
                print('N:', str(logs['N']))
                print('M:', str(logs['M']))
                print('beta:', str(logs['beta']))
                print('epsilon:', str(logs['epsilon']))
                print('S:', str(logs['S']))
                print('delta level', str(logs['delta_level']))
                delta_level = logs['delta_level']
                print('learned volume', str(logs['learned_volume']))
                print('recovered volume', str(logs['recovered_volume']))
                print('theoretically recoverable volume', str(
                    logs['theoretically_recoverable_volume']))
                print('recovered violation rate', str(
                    logs['recovered_violation_rate']))

                
                plot_config = self.dataset.dynamics.plot_config()

                state_test_range = self.dataset.dynamics.state_test_range()
                
                times = [self.dataset.tMax]
                if plot_config['z_axis_idx'] == -1:
                    fig = self.plotSingleFig(
                        state_test_range, plot_config, 512, 512, times, delta_level)
                else:
                    fig= self.plotMultipleFigs(
                        state_test_range, plot_config, 512, 512, 5, times, delta_level)
                    
                x_resolution=512
                y_resolution=512
                x_min, x_max = state_test_range[plot_config['x_axis_idx']]
                y_min, y_max = state_test_range[plot_config['y_axis_idx']]

                xs = torch.linspace(x_min, x_max, x_resolution)
                ys = torch.linspace(y_min, y_max, y_resolution)
                xys = torch.cartesian_prod(xs, ys)
                Xg, Yg = torch.meshgrid(xs, ys)
                
                ## Plot Set and Value Fn
                fig = plt.figure(figsize=(5*len(times), 2*5*1), facecolor='white')
                
                plt.rcParams['text.usetex'] = False

                # for i in range(3*len(times)):
                for i in range(2*len(times)):
                    
                    if i >= len(times):
                        ax = fig.add_subplot(2, len(times), 1+i)
                    else:
                        ax = fig.add_subplot(2, len(times), 1+i, projection='3d')
                    ax.set_title(r"t =" + "%0.2f" % (times[i % len(times)]))

                    ## Define Grid Slice to Plot

                    coords = torch.zeros(x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
                    coords[:, 0] = times[i % len(times)]
                    coords[:, 1:] = torch.tensor(plot_config['state_slices']) # initialized to zero (nothing else to set!)

                    # xN - (xi = xj) plane
                    if i >= len(times):
                        pad_label = 0
                        ax.set_xlabel(r"$x_N$", fontsize=12, labelpad=pad_label); ax.set_ylabel(r"$x_i = x_j$", fontsize=12, labelpad=pad_label)
                        ax.set_xticks([-1, 1])
                        ax.set_xticklabels([r'$-1$', r'$1$'])
                        ax.set_yticks([-1, 1])
                        ax.set_yticklabels([r'$-1$', r'$1$'])

                        ax_pad = 0
                        ax.xaxis.set_tick_params(pad=ax_pad)
                        ax.yaxis.set_tick_params(pad=ax_pad)

                    else:
                        pad_label = 6
                        ax.set_xlabel(r"$x_N$", fontsize=12, labelpad=pad_label); 
                        ax.set_ylabel(r"$x_i = x_j$", fontsize=12, labelpad=pad_label); 
                        ax.set_zlabel(r"$V$", fontsize=12, labelpad=10) #, labelpad=pad_label)
                        ax.set_xticks([-1, 0, 1])
                        ax.set_xticklabels([r'$-1$', r'$0$', r'$1$'])
                        ax.set_yticks([-1, 0, 1])
                        ax.set_yticklabels([r'$-1$', r'$0$', r'$1$'])
                        ax.set_zticks([])
                        ax.zaxis.label.set_position((-0.1, 0.5))
                        
                        ax.xaxis.pane.fill = False
                        ax.yaxis.pane.fill = False
                        ax.zaxis.pane.fill = False
                        
                        ax_pad = 0
                        ax.xaxis.set_tick_params(pad=ax_pad)
                        ax.yaxis.set_tick_params(pad=ax_pad)
                        ax.zaxis.set_tick_params(pad=ax_pad)

                    coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                    coords[:, 2:] = (xys[:, 1] * torch.ones(self.dataset.dynamics.N-1, xys.size()[0])).t()

                    with torch.no_grad():
                        model_results = self.model({'coords': self.dataset.dynamics.coord_to_input(coords.to(device))})
                        values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach())
                    
                    learned_value = values.detach().cpu().numpy().reshape(x_resolution, y_resolution)

                    xig = torch.arange(-0.99, 1.01, 0.02) # 100 x 100
                    X1g, X2g = torch.meshgrid(xig, xig)

                    Vgt=np.load("./dynamics/vgt_40D.npy")
                    ## Make Value-Based Colormap
                    
                    cmap_name = "RdBu"

                    if learned_value.min() > 0:
                        # RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', [(1,1,1), (0.5,0.5,1), (0,0,1), (0,0,1)])
                        scaled_colors = np.vstack((matplotlib.colormaps[cmap_name](np.linspace(0.6, 1., 256))))
                        RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', scaled_colors)

                    elif learned_value.max() < 0:
                        # RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', [(1,0,0), (1,0,0), (1,0.5,0.5), (1,1,1)])
                        scaled_colors = np.vstack((matplotlib.colormaps[cmap_name](np.linspace(0., 0.4, 256))))
                        RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', scaled_colors)

                    else:
                        # n_bins_high = int(256 * (learned_value.max()/(learned_value.max() - learned_value.min())) // 1)
                        n_bins_high = round(256 * learned_value.max()/(learned_value.max() - learned_value.min()))


                        offset = 0
                        scaled_colors = np.vstack((matplotlib.colormaps[cmap_name](np.linspace(0., 0.4, 256-n_bins_high+offset)), matplotlib.colormaps[cmap_name](np.linspace(0.6, 1., n_bins_high-offset))))
                        RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', scaled_colors)
                    
                    if i >= len(times):

                        ## Plot Zero-level Set of Learned Value
                        
                        # s = ax.imshow(1*(learned_value.T <= 0), cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
                        # s = ax.imshow(learned_value.T, cmap='coolwarm_r', origin='lower', extent=(-1., 1., -1., 1.))
                        s = ax.contourf(Xg, Yg, learned_value, cmap=RdWhBl_vscaled, levels=256)
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        cbar = fig.colorbar(s, cax=cax)
                        cbar.set_ticks([learned_value.min(), 0., learned_value.max()])  # Define custom tick locations
                        cbar.set_ticklabels([f'{learned_value.min():1.1f}', '0', f'{learned_value.max():1.1f}'])  # Define custom tick labels

                        ## Plot Ground-Truth Zero-Level Contour

                        ax.contour(X1g, X2g, Vgt, [0.], linewidths=4, alpha=0.7, colors='brown')
                        ax.contour(Xg, Yg, learned_value, [0.], linewidths=1, alpha=1, colors='k',linestyles="--")
                        ax.contour(Xg, Yg, learned_value, [-0.0744], linewidths=1, alpha=1, colors='k',linestyles="-")
                    
                    else:


                        ax.view_init(elev=15, azim=-60)
                        ax.set_facecolor((1, 1, 1, 1))
                        surf = ax.plot_surface(Xg, Yg, learned_value, cmap=RdWhBl_vscaled, alpha=0.8) #cmap='bwr_r')

                        cbar = fig.colorbar(surf, ax=ax, fraction=0.02, pad=0.0)

                        ax.set_zlim(learned_value.min() - (learned_value.max() - learned_value.min())/5)

                        ax.contour(Xg, Yg, learned_value, zdir='z', offset=ax.get_zlim()[0], colors='k', levels=[0.]) #cmap='bwr_r')

                fig.savefig(os.path.join(
                    testing_dir, f'basic_BRTs.png'), dpi=800)
                
                
        if was_training:
            self.model.train()
            self.model.requires_grad_(True)

    def plotSingleFig(self, state_test_range, plot_config, x_resolution, y_resolution, times, delta_level = None):
        x_min, x_max = state_test_range[plot_config['x_axis_idx']]
        y_min, y_max = state_test_range[plot_config['y_axis_idx']]
        
        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        xys = torch.cartesian_prod(xs, ys)
        fig = plt.figure(figsize=(6, 5*len(times)))
        X, Y = np.meshgrid(xs, ys)
        for i in range(len(times)):
            coords = torch.zeros(
                x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
            coords[:, 0] = times[i]
            coords[:, 1:] = torch.tensor(plot_config['state_slices'])
            coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
            coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]

            with torch.no_grad():
                model_results = self.model(
                    {'coords': self.dataset.dynamics.coord_to_input(coords.to(device))})

                values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(
                ), model_results['model_out'].squeeze(dim=-1).detach())

            ax = fig.add_subplot(len(times), 1, 1 + i)
            ax.set_title('t = %0.2f' % (times[i]))
            BRT_img = values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T
            max_value = np.amax(BRT_img)
            min_value = np.amin(BRT_img)
            imshow_kwargs = {
                'vmax': max_value,
                'vmin': min_value,
                'cmap': 'coolwarm_r',
                'extent': (x_min, x_max, y_min, y_max),
                'origin': 'lower',
            }
            ax.imshow(BRT_img, **imshow_kwargs)
            lx=self.dataset.dynamics.boundary_fn(coords.to(device)[...,1:]).detach().cpu().numpy().reshape(x_resolution, y_resolution).T
            zero_contour = ax.contour(X, 
                                Y, 
                                BRT_img, 
                                levels=[0.0],  
                                colors="black",  
                                linewidths=2,    
                                linestyles='--')  
                
            failure_set_contour = ax.contour(X, 
                            Y, 
                            lx, 
                            levels=[0.0],  
                            colors="saddlebrown",  
                            linewidths=2,    
                            linestyles='-')  
            if delta_level is not None:
                delta_contour = ax.contour(X, 
                            Y, 
                            BRT_img, 
                            levels=[delta_level],  
                            colors="black",  
                            linewidths=2,    
                            linestyles='-')  
        return fig

    def plotMultipleFigs(self, device,state_test_range, plot_config, x_resolution, y_resolution, z_resolution, times, delta_level = None):
        x_min, x_max = state_test_range[plot_config['x_axis_idx']]
        y_min, y_max = state_test_range[plot_config['y_axis_idx']]
        z_min, z_max = state_test_range[plot_config['z_axis_idx']]


        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        zs = torch.linspace(z_min, z_max, z_resolution)
        xys = torch.cartesian_prod(xs, ys)

        fig = plt.figure(figsize=(6*len(zs),5*len(times)))
        X, Y = np.meshgrid(xs, ys)
        for i in range(len(times)):
            for j in range(len(zs)):
                coords = torch.zeros(
                    x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
                coords[:, 0] = times[i]
                coords[:, 1:] = torch.tensor(plot_config['state_slices'])
                coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
                coords[:, 1 + plot_config['z_axis_idx']] = zs[j]

                lx=self.dataset.dynamics.boundary_fn(coords.to(device)[...,1:]).detach().cpu().numpy().reshape(x_resolution, y_resolution).T
                with torch.no_grad():
                    model_results = self.model(
                        {'coords': self.dataset.dynamics.coord_to_input(coords.to(device))})
                    values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(
                        ), model_results['model_out'].squeeze(dim=-1).detach())

                ax = fig.add_subplot(len(times), len(zs), (j+1) + i*len(zs))
                ax.set_title('t = %0.2f, %s = %0.2f' % (
                    times[i], plot_config['state_labels'][plot_config['z_axis_idx']], zs[j]))

                BRT_img = values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T

                max_value = np.amax(BRT_img)
                min_value = np.amin(BRT_img)
                imshow_kwargs = {
                    'vmax': max_value,
                    'vmin': min_value,
                    'cmap': 'coolwarm_r',
                    'extent': (x_min, x_max, y_min, y_max),
                    'origin': 'lower',
                }

                s1 = ax.imshow(BRT_img, **imshow_kwargs)
                fig.colorbar(s1)
                zero_contour = ax.contour(X, 
                                Y, 
                                BRT_img, 
                                levels=[0.0],  
                                colors="black",  
                                linewidths=2,    
                                linestyles='--')  
                
                failure_set_contour = ax.contour(X, 
                                Y, 
                                lx, 
                                levels=[0.0],  
                                colors="saddlebrown",  
                                linewidths=2,    
                                linestyles='-')  
                
                if delta_level is not None:
                    delta_contour = ax.contour(X, 
                                Y, 
                                BRT_img, 
                                levels=[delta_level],  
                                colors="black",  
                                linewidths=2,    
                                linestyles='-')  
   
        return fig    
    def plot_recovery_fig(self, device,dataset, dynamics, model, delta_level):
        # 1. for ground truth slices (if available), record (higher-res) grid of learned values
        # plot (with ground truth) learned BRTs, recovered BRTs
        z_res = 5
        plot_config = dataset.dynamics.plot_config()
        if os.path.exists(os.path.join(self.experiment_dir, 'ground_truth.mat')):
            ground_truth = spio.loadmat(os.path.join(
                self.experiment_dir, 'ground_truth.mat'))
            if 'gmat' in ground_truth:
                ground_truth_xs = ground_truth['gmat'][..., 0][:, 0, 0]
                ground_truth_ys = ground_truth['gmat'][..., 1][0, :, 0]
                ground_truth_zs = ground_truth['gmat'][..., 2][0, 0, :]
                ground_truth_values = ground_truth['data']
                ground_truth_ts = np.linspace(
                    0, 1, ground_truth_values.shape[3])

            elif 'g' in ground_truth:
                ground_truth_xs = ground_truth['g']['vs'][0, 0][0][0][:, 0]
                ground_truth_ys = ground_truth['g']['vs'][0, 0][1][0][:, 0]
                ground_truth_zs = ground_truth['g']['vs'][0, 0][2][0][:, 0]
                ground_truth_ts = ground_truth['tau'][0]
                ground_truth_values = ground_truth['data']

            # idxs to plot
            x_idxs = np.linspace(0, len(ground_truth_xs)-1,
                                 len(ground_truth_xs)).astype(dtype=int)
            y_idxs = np.linspace(0, len(ground_truth_ys)-1,
                                 len(ground_truth_ys)).astype(dtype=int)
            z_idxs = np.linspace(0, len(ground_truth_zs) -
                                 1, z_res).astype(dtype=int)
            t_idxs = np.array([len(ground_truth_ts)-1]).astype(dtype=int)

            # indexed ground truth to plot
            ground_truth_xs = ground_truth_xs[x_idxs]
            ground_truth_ys = ground_truth_ys[y_idxs]
            ground_truth_zs = ground_truth_zs[z_idxs]
            ground_truth_ts = ground_truth_ts[t_idxs]
            ground_truth_values = ground_truth_values[
                x_idxs[:, None, None, None],
                y_idxs[None, :, None, None],
                z_idxs[None, None, :, None],
                t_idxs[None, None, None, :]
            ]
            ground_truth_grids = ground_truth_values

            xs = ground_truth_xs
            ys = ground_truth_ys
            zs = ground_truth_zs
        else:
            ground_truth_grids = None
            resolution = 512
            xs = np.linspace(*dynamics.state_test_range()
                             [plot_config['x_axis_idx']], resolution)
            ys = np.linspace(*dynamics.state_test_range()
                             [plot_config['y_axis_idx']], resolution)
            zs = np.linspace(*dynamics.state_test_range()
                             [plot_config['z_axis_idx']], z_res)

        xys = torch.cartesian_prod(torch.tensor(xs), torch.tensor(ys))
        value_grids = np.zeros((len(zs), len(xs), len(ys)))
        for i in range(len(zs)):
            coords = torch.zeros(xys.shape[0], dataset.dynamics.state_dim + 1)
            coords[:, 0] = dataset.tMax
            coords[:, 1:] = torch.tensor(plot_config['state_slices'])
            coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
            coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
            if dataset.dynamics.state_dim > 2:
                coords[:, 1 + plot_config['z_axis_idx']] = zs[i]

            model_results = model(
                {'coords': dataset.dynamics.coord_to_input(coords.to(device))})
            values = dataset.dynamics.io_to_value(model_results['model_in'].detach(
            ), model_results['model_out'].detach().squeeze(dim=-1)).detach().cpu()
            value_grids[i] = values.reshape(len(xs), len(ys))

        fig = plt.figure()
        fig.suptitle(plot_config['state_slices'], fontsize=8)
        x_min, x_max = dataset.dynamics.state_test_range()[
            plot_config['x_axis_idx']]
        y_min, y_max = dataset.dynamics.state_test_range()[
            plot_config['y_axis_idx']]

        for i in range(len(zs)):
            values = value_grids[i]

            # learned BRT and recovered BRT
            ax = fig.add_subplot(1, len(zs), (i+1))
            ax.set_title('%s = %0.2f' % (
                plot_config['state_labels'][plot_config['z_axis_idx']], zs[i]), fontsize=8)

            image = np.full((*values.shape, 3), 255, dtype=int)
            BRT = values < 0
            recovered_BRT = values < delta_level

            if dynamics.set_mode in ['reach','reach_avoid']:
                image[BRT] = np.array([252, 227, 152])
                self.overlay_border(image, BRT, np.array([249, 188, 6]))
                image[recovered_BRT] = np.array([155, 241, 249])
                self.overlay_border(image, recovered_BRT,
                                    np.array([15, 223, 240]))
                if ground_truth_grids is not None:
                    self.overlay_ground_truth(image, i, ground_truth_grids)
            else:
                image[recovered_BRT] = np.array([155, 241, 249])
                image[BRT] = np.array([252, 227, 152])
                self.overlay_border(image, BRT, np.array([249, 188, 6]))
                # overlay recovered border over learned BRT
                self.overlay_border(image, recovered_BRT,
                                    np.array([15, 223, 240]))
                if ground_truth_grids is not None:
                    self.overlay_ground_truth(image, i, ground_truth_grids)

            ax.imshow(image.transpose(1, 0, 2), origin='lower',
                      extent=(x_min, x_max, y_min, y_max))

            ax.set_xlabel(plot_config['state_labels']
                          [plot_config['x_axis_idx']])
            ax.set_ylabel(plot_config['state_labels']
                          [plot_config['y_axis_idx']])
            ax.set_xticks([x_min, x_max])
            ax.set_yticks([y_min, y_max])
            ax.tick_params(labelsize=6)
            if i != 0:
                ax.set_yticks([])
        return fig, value_grids

    def overlay_ground_truth(self, image, z_idx, ground_truth_grids):
        thickness = max(0, image.shape[0] // 120 - 1)
        ground_truth_grid = ground_truth_grids[:, :, z_idx, 0]
        ground_truth_brts = ground_truth_grid < 0
        for x in range(ground_truth_brts.shape[0]):
            for y in range(ground_truth_brts.shape[1]):
                if not ground_truth_brts[x, y]:
                    continue
                neighbors = [
                    (x, y+1),
                    (x, y-1),
                    (x+1, y+1),
                    (x+1, y),
                    (x+1, y-1),
                    (x-1, y+1),
                    (x-1, y),
                    (x-1, y-1),
                ]
                for neighbor in neighbors:
                    if neighbor[0] >= 0 and neighbor[1] >= 0 and neighbor[0] < ground_truth_brts.shape[0] and neighbor[1] < ground_truth_brts.shape[1]:
                        if not ground_truth_brts[neighbor]:
                            image[x-thickness:x+1, y-thickness:y +
                                  1+thickness] = np.array([50, 50, 50])
                            break

    def overlay_border(self, image, set, color):
        thickness = max(0, image.shape[0] // 120 - 1)
        for x in range(set.shape[0]):
            for y in range(set.shape[1]):
                if not set[x, y]:
                    continue
                neighbors = [
                    (x, y+1),
                    (x, y-1),
                    (x+1, y+1),
                    (x+1, y),
                    (x+1, y-1),
                    (x-1, y+1),
                    (x-1, y),
                    (x-1, y-1),
                ]
                for neighbor in neighbors:
                    if neighbor[0] >= 0 and neighbor[1] >= 0 and neighbor[0] < set.shape[0] and neighbor[1] < set.shape[1]:
                        if not set[neighbor]:
                            image[x-thickness:x+1, y -
                                  thickness:y+1+thickness] = color
                            break
                        
class DeepReach(Experiment):
    def init_special(self):
        pass
