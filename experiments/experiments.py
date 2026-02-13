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

        plot_config = self.dataset.dynamics.plot_config()

        state_test_range = self.dataset.dynamics.state_test_range()
        x_min, x_max = state_test_range[plot_config['x_axis_idx']]
        y_min, y_max = state_test_range[plot_config['y_axis_idx']]
        z_min, z_max = state_test_range[plot_config['z_axis_idx']]

        times = torch.linspace(0, self.dataset.tMax, time_resolution)
        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        zs = torch.linspace(z_min, z_max, z_resolution)
        xys = torch.cartesian_prod(xs, ys)
        
        fig = plt.figure(figsize=(5*len(times), 5*len(zs)))
        for i in range(len(times)):
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
                
                ax = fig.add_subplot(len(times), len(zs), (j+1) + i*len(zs))
                ax.set_title('t = %0.2f, %s = %0.2f' % (times[i], plot_config['state_labels'][plot_config['z_axis_idx']], zs[j]))
                s = ax.imshow(1*(values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T <= 0), cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
                fig.colorbar(s) 
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
            
    def validate1(self, device, epoch, save_path, x_resolution, y_resolution, z_resolution, time_resolution):
        was_training = self.model1.training
        self.model1.eval()
        self.model1.requires_grad_(False)

        plot_config = self.dataset.dynamics.plot_config()

        state_test_range = self.dataset.dynamics.state_test_range()
        x_min, x_max = state_test_range[plot_config['x_axis_idx']]
        y_min, y_max = state_test_range[plot_config['y_axis_idx']]
        z_min, z_max = state_test_range[plot_config['z_axis_idx']]

        times = torch.linspace(0, self.dataset.tMax, time_resolution)
        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        zs = torch.linspace(z_min, z_max, z_resolution)
        xys = torch.cartesian_prod(xs, ys)
        
        fig = plt.figure(figsize=(5*len(times), 5*len(zs)))
        for i in range(len(times)):
            for j in range(len(zs)):
                coords = torch.zeros(x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
                coords[:, 0] = times[i]
                coords[:, 1:] = torch.tensor(plot_config['state_slices'])
                coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
                coords[:, 1 + plot_config['z_axis_idx']] = zs[j]

                with torch.no_grad():
                    model_results = self.model1({'coords': self.dataset.dynamics.coord_to_input(coords.to(device))})
                    values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach())
                
                ax = fig.add_subplot(len(times), len(zs), (j+1) + i*len(zs))
                ax.set_title('t = %0.2f, %s = %0.2f' % (times[i], plot_config['state_labels'][plot_config['z_axis_idx']], zs[j]))

                raw_data = values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T 
                # s = ax.imshow(raw_data, cmap='seismic', origin='lower', extent=(-1., 1., -1., 1.))
                s = ax.imshow(1*(values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T <= 0), cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
                fig.colorbar(s) 
        fig.savefig(save_path)
        if self.use_wandb:
            wandb.log({
                'step': epoch,
                'val_plot': wandb.Image(fig),
            })
        plt.close()

        if was_training:
            self.model1.train()
            self.model1.requires_grad_(True)
            
    
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

                    


        def safe_clone(model):
            """
            copy.deepcopy가 실패할 때 사용하는 직렬화 방식 복제
            (계산 그래프를 끊고 순수 파라미터만 복사함)
            """
            buffer = io.BytesIO()
            torch.save(model, buffer) # 메모리에 저장
            buffer.seek(0)
            return torch.load(buffer,weights_only=False) # 메모리에서 불러오기 (새 객체 생성됨)
        # ----------------------------------------------------------------------
        # [초기화] 단일 모델 설정 (Centralized)
        # ----------------------------------------------------------------------
        
        # 1. Main Model 설정
        # self.model을 복제하지 않고 바로 사용합니다.
        self.model.to(self.device)
        self.model.train() 
        for param in self.model.parameters():
            param.requires_grad = True # 확실하게 학습 가능 상태로 설정

        # 2. Target Network (학습 안정성을 위해 복제 필요)
        # Main Model의 현재 상태를 복사해서 만듭니다.
        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()
        self.target_model.to(self.device)
        for p in self.target_model.parameters(): 
            p.requires_grad = False
        
        # 3. Anchor Model (Catastrophic Forgetting 방지용, 복제 필요)
        self.anchor_model = copy.deepcopy(self.model)
        self.anchor_model.eval()
        self.anchor_model.to(self.device)
        for param in self.anchor_model.parameters():
            param.requires_grad = False

        # 4. Optimizer 정의 (Main Model에 대해서만)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=2e-5)
        # Hyperparameters
        alpha = 0.005
        epochs = 2000
        max_MPI_epochs = 10  # 한 에폭당 반복 횟수
        total_iterations = epochs

        # Boundary Condition Helper
        def h_func(states):
            return self.dataset.dynamics.boundary_fn(states)

        # Checkpoint 경로 설정
        checkpoints_dir = os.path.join(self.experiment_dir, 'training', 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)
        total_steps = 0
        train_losses = []

        # Anchor Weight Settings
        initial_anchor_weight = 200.0
        decay_rate = 0.9995
        min_anchor_weight = 1.0

        # ==============================================================================
        # [Main Training Loop - Centralized]
        # ==============================================================================
        with tqdm(total=total_iterations, desc="Training (Centralized)") as pbar:
            for epoch in range(epochs):
                
                # 0. Hyperparameter Update
                current_anchor_weight = max(
                    min_anchor_weight, 
                    initial_anchor_weight * (decay_rate ** epoch)
                )

                # 1. Time Scheduling (Backward Curriculum)
                progress = epoch / epochs
                curriculum_factor = min(1.0, progress * 1.5)
                current_t_max = self.dataset.tMax * curriculum_factor

                # 2. Update Target Network
                # 매 에폭 시작 시 Target을 현재 모델로 갱신 (Soft update 대신 Hard update 방식 사용 중)
                self.target_model.load_state_dict(self.model.state_dict())

                # ----------------------------------------------------------------------
                # Inner Loop (Gradient Updates)
                # ----------------------------------------------------------------------
                # Centralized에서는 VI_steps(교차 업데이트)가 필요 없고, 
                # Target이 고정된 상태에서 배치 업데이트만 수행하면 됩니다

                    # ------------------------------------------------------------------
                    # 5. Bellman Target Construction (Coordinate Splitting)
                    # V(x,t) = max(h(x), 0.5*V(t-alpha, x) + 0.5*V(t, x+f*alpha))
                    # ------------------------------------------------------------------
                VI_steps = 10  # ★ [복구] 한 배치에 대해 반복 학습할 횟수 (고정점 반복)
                for pi_step in range(max_MPI_epochs):
                    
                    # ------------------------------------------------------------------
                    # 3. Sampling (배치 생성)
                    # ------------------------------------------------------------------
                    batch_size = 8192

                    n_frontier = int(batch_size * 0.3)
                    n_uniform = batch_size - n_frontier
                    # Main Batch: Time [0, current_t_max]
                    t_uniform = (current_t_max) * torch.rand(n_uniform, 1, device=self.device)
                    frontier_width = max(0.1, current_t_max * 0.1) 
                    t_frontier = (frontier_width) * torch.rand(n_frontier, 1, device=self.device) + (current_t_max - frontier_width)
                    rand_times = torch.cat([t_uniform, t_frontier], dim=0)
                    rand_states = self.dataset.sample_states(batch_size).to(self.device)
                    batch_coords = torch.cat([rand_times, rand_states], dim=-1)

                    # Anchor Batch: Time [0, tMax] (전체 영역 보존)
                    rand_times2 = (self.dataset.tMax) * torch.rand(batch_size, 1, device=self.device)
                    rand_states2 = self.dataset.sample_states(batch_size).to(self.device)
                    batch_coords2 = torch.cat([rand_times2, rand_states2], dim=-1)

                    # ------------------------------------------------------------------
                    # 4. Optimal Control Calculation (via Analytical Gradient)
                    # ★ 변경 핵심: model1 vs model2 경쟁 없이, 기울기로 u*, v* 즉시 계산
                    # ------------------------------------------------------------------
                    
                    # (1) Gradients 계산을 위해 requires_grad 켬
                    batch_coords.requires_grad_(True)
                    model_input = self.dataset.dynamics.coord_to_input(batch_coords)
                    model_input = model_input.detach().requires_grad_(True)

                    # (2) Target Model을 통해 현재 가치함수의 형상(Shape) 파악
                    output_raw = self.target_model({'coords': model_input})['model_out']
                    
                    # (3) Automatic Differentiation (dv/dt, dv/dx)
                    grads = self.dataset.dynamics.io_to_dv(model_input, output_raw.squeeze(-1))
                    dv_dx = grads[..., 1:] # Spatial derivatives (Optimal Control에 필요)

                    # (4) Analytical Optimal Control
                    # Hamiltonian H(x, p) = max_v min_u [p * f(x, u, v)]
                    # 따라서 u* = argmin, v* = argmax (Reachability 기준)
                    batch_states = batch_coords[..., 1:]
                    
                    # 미분값(dv_dx)을 이용해 최적의 제어(u)와 방해(v)를 바로 구함
                    with torch.no_grad():
                        u_fixed = self.dataset.dynamics.optimal_control(batch_states, dv_dx)
                        v_fixed = self.dataset.dynamics.optimal_disturbance(batch_states, dv_dx)

                    # 미분 계산 끝났으므로 grad 끔 (메모리 절약)
                    batch_coords.requires_grad_(False)

                    for vi_iter in range(VI_steps):
                        
                        # (A) Bellman Target Construction
                        with torch.no_grad():
                            curr_t = batch_coords[..., 0:1]
                            curr_x = batch_coords[..., 1:]

                            # --- Time Step: V(t - alpha, x) ---
                            prev_t_time = torch.clamp(curr_t - alpha, min=self.dataset.tMin)
                            coords_T = torch.cat([prev_t_time, curr_x], dim=-1)
                            
                            in_T = self.dataset.dynamics.coord_to_input(coords_T)
                            out_T_raw = self.target_model({'coords': in_T})['model_out']
                            val_T = self.dataset.dynamics.io_to_value(in_T, out_T_raw.squeeze(-1)).unsqueeze(-1)

                            # --- Space Step: V(t, x + f(u*, v*) * alpha) ---
                            # 위에서 고정한 u_fixed, v_fixed 사용
                            dx = self.dataset.dynamics.dsdt(curr_x, u_fixed, v_fixed)
                            next_x_space = curr_x + dx * alpha
                            coords_S = torch.cat([curr_t, next_x_space], dim=-1)

                            in_S = self.dataset.dynamics.coord_to_input(coords_S)
                            out_S_raw = self.target_model({'coords': in_S})['model_out']
                            val_S = self.dataset.dynamics.io_to_value(in_S, out_S_raw.squeeze(-1)).unsqueeze(-1)

                            # --- Combine ---
                            bellman_target = 0.5 * val_T + 0.5 * val_S
                            bellman_target_clamped = torch.clamp(bellman_target, min=-10.0, max=10.0)
                            
                            boundary_val = h_func(curr_x)
                            target_V = torch.min(boundary_val, bellman_target_clamped)

                            # Anchor Target (한 번 계산하면 되지만, 구조상 루프 안에서 계산해도 무방)
                            in_batch2 = self.dataset.dynamics.coord_to_input(batch_coords2)
                            anchor_out_raw = self.anchor_model({'coords': in_batch2})['model_out']
                            anchor_out_phys = self.dataset.dynamics.io_to_value(in_batch2, anchor_out_raw.squeeze(-1)).unsqueeze(-1)

                        # (B) Model Update (Gradient Descent)
                        # Main Prediction
                        in_batch = self.dataset.dynamics.coord_to_input(batch_coords)
                        pred_raw = self.model({'coords': in_batch})['model_out']
                        pred_V_phys = self.dataset.dynamics.io_to_value(in_batch, pred_raw.squeeze(-1)).unsqueeze(-1)

                        # Anchor Prediction
                        pred_raw2 = self.model({'coords': in_batch2})['model_out']
                        pred_V_phys2 = self.dataset.dynamics.io_to_value(in_batch2, pred_raw2.squeeze(-1)).unsqueeze(-1)

                        # Loss Calculation
                        loss = torch.mean((pred_V_phys - target_V) ** 2) + \
                               current_anchor_weight * torch.mean((pred_V_phys2 - anchor_out_phys) ** 2)

                        self.optim.zero_grad()
                        loss.backward()
                        self.optim.step()

                    # Logging
                    total_steps += 1
                    train_losses.append(loss.item())
                    
                    if pi_step % 5 == 0: # Update progress bar periodically
                        pbar.set_postfix({'Loss': f"{loss.item():.2e}", 'TargetMean': f"{target_V.mean():.2f}"})
                        pbar.update(1)

                # ----------------------------------------------------------------------
                # 7. Checkpointing & Validation
                # ----------------------------------------------------------------------
                if (epoch + 1) % 10 == 0:
                    checkpoint = {
                        'epoch': epoch + 1,
                        'model': self.model.state_dict(),
                        'optimizer': self.optim.state_dict()
                    }
                    # 모델이 하나이므로 이름도 단순화
                    torch.save(checkpoint, os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % (epoch + 1)))
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses.txt'), np.array(train_losses))
                    
                    # Validation
                    # self.validate1 함수가 내부적으로 self.model1을 쓴다면 self.model로 바꿔서 호출하거나
                    # 해당 함수를 수정해야 합니다. 여기서는 self.model을 사용한다고 가정합니다.
                    self.validate(
                        device=self.device, epoch=epoch+1, 
                        save_path=os.path.join(checkpoints_dir, 'BRS_validation_plot1_epoch_%04d.png' % (epoch+1)),
                        x_resolution=val_x_resolution, y_resolution=val_y_resolution, 
                        z_resolution=val_z_resolution, time_resolution=val_time_resolution
                    )
                    
                    print(f"\n[Epoch {epoch+1}] Model Saved. Loss: {loss.item():.5f}")
                    
        self.model.to(self.device)
        self.model.train() 
        for param in self.model.parameters():
            param.requires_grad = True # 확실하게 학습 가능 상태로 설정
            
        # 2. Target Network (학습 안정성을 위해 복제 필요)
        # Main Model의 현재 상태를 복사해서 만듭니다.
        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()
        self.target_model.to(self.device)
        for p in self.target_model.parameters(): 
            p.requires_grad = False
            
        # 3. Anchor Model (Catastrophic Forgetting 방지용, 복제 필요)
        self.anchor_model = copy.deepcopy(self.model)
        self.anchor_model.eval()
        self.anchor_model.to(self.device)
        for param in self.anchor_model.parameters():
            param.requires_grad = False
            
        # Hyperparameters
        alpha = 0.005
        epochs = 2000
        max_MPI_epochs = 10  # 한 에폭당 반복 횟수
        total_iterations = epochs 

        # Boundary Condition Helper
        def h_func(states):
            return self.dataset.dynamics.boundary_fn(states)

        # Checkpoint 경로 설정
        checkpoints_dir = os.path.join(self.experiment_dir, 'training', 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)
        total_steps = 0
        train_losses = []

        # Anchor Weight Settings
        initial_anchor_weight = 200.0
        decay_rate = 0.9995
        min_anchor_weight = 1.0

        # ==============================================================================
        # [Main Training Loop - Centralized]
        # ==============================================================================
        with tqdm(total=total_iterations, desc="Training (Centralized)") as pbar:
            for epoch in range(epochs):
                
                # 0. Hyperparameter Update
                current_anchor_weight = max(
                    min_anchor_weight, 
                    initial_anchor_weight * (decay_rate ** epoch)
                )

                # 1. Time Scheduling (Backward Curriculum)
                progress = epoch / epochs
                curriculum_factor = min(1.0, progress * 1.5)
                current_t_max = self.dataset.tMax * curriculum_factor

                # 2. Update Target Network
                # 매 에폭 시작 시 Target을 현재 모델로 갱신 (Soft update 대신 Hard update 방식 사용 중)
                self.target_model.load_state_dict(self.model.state_dict())

                # ----------------------------------------------------------------------
                # Inner Loop (Gradient Updates)
                # ----------------------------------------------------------------------
                # Centralized에서는 VI_steps(교차 업데이트)가 필요 없고, 
                # Target이 고정된 상태에서 배치 업데이트만 수행하면 됩니다.
                VI_steps = 10  # ★ [복구] 한 배치에 대해 반복 학습할 횟수 (고정점 반복)
                for pi_step in range(max_MPI_epochs):
                    
                    # ------------------------------------------------------------------
                    # 3. Sampling (배치 생성)
                    # ------------------------------------------------------------------
                    batch_size = 8192
                    n_frontier = int(batch_size * 0.3)
                    n_uniform = batch_size - n_frontier
                    # Main Batch: Time [0, current_t_max]
                    t_uniform = (current_t_max) * torch.rand(n_uniform, 1, device=self.device)
                    frontier_width = max(0.1, current_t_max * 0.1) 
                    t_frontier = (frontier_width) * torch.rand(n_frontier, 1, device=self.device) + (current_t_max - frontier_width)
                    rand_times = torch.cat([t_uniform, t_frontier], dim=0)
                    rand_states = self.dataset.sample_states(batch_size).to(self.device)
                    batch_coords = torch.cat([rand_times, rand_states], dim=-1)

                    # Anchor Batch: Time [0, tMax] (전체 영역 보존)
                    rand_times2 = (self.dataset.tMax) * torch.rand(batch_size, 1, device=self.device)
                    rand_states2 = self.dataset.sample_states(batch_size).to(self.device)
                    batch_coords2 = torch.cat([rand_times2, rand_states2], dim=-1)

                    # ------------------------------------------------------------------
                    # 4. Optimal Control Calculation (via Analytical Gradient)
                    # ★ 변경 핵심: model1 vs model2 경쟁 없이, 기울기로 u*, v* 즉시 계산
                    # ------------------------------------------------------------------
                    
                    # (1) Gradients 계산을 위해 requires_grad 켬
                    batch_coords.requires_grad_(True)
                    model_input = self.dataset.dynamics.coord_to_input(batch_coords)
                    model_input = model_input.detach().requires_grad_(True)

                    # (2) Target Model을 통해 현재 가치함수의 형상(Shape) 파악
                    output_raw = self.target_model({'coords': model_input})['model_out']
                    
                    # (3) Automatic Differentiation (dv/dt, dv/dx)
                    grads = self.dataset.dynamics.io_to_dv(model_input, output_raw.squeeze(-1))
                    dv_dx = grads[..., 1:] # Spatial derivatives (Optimal Control에 필요)

                    # (4) Analytical Optimal Control
                    # Hamiltonian H(x, p) = max_v min_u [p * f(x, u, v)]
                    # 따라서 u* = argmin, v* = argmax (Reachability 기준)
                    batch_states = batch_coords[..., 1:]
                    
                    # 미분값(dv_dx)을 이용해 최적의 제어(u)와 방해(v)를 바로 구함
                    with torch.no_grad():
                        u_fixed = self.dataset.dynamics.optimal_control(batch_states, dv_dx)
                        v_fixed = self.dataset.dynamics.optimal_disturbance(batch_states, dv_dx)

                    # 미분 계산 끝났으므로 grad 끔 (메모리 절약)
                    batch_coords.requires_grad_(False)

                    for vi_iter in range(VI_steps):
                        
                        # (A) Bellman Target Construction
                        with torch.no_grad():
                            curr_t = batch_coords[..., 0:1]
                            curr_x = batch_coords[..., 1:]

                            # --- Time Step: V(t - alpha, x) ---
                            prev_t_time = torch.clamp(curr_t - alpha, min=self.dataset.tMin)
                            coords_T = torch.cat([prev_t_time, curr_x], dim=-1)
                            
                            in_T = self.dataset.dynamics.coord_to_input(coords_T)
                            out_T_raw = self.target_model({'coords': in_T})['model_out']
                            val_T = self.dataset.dynamics.io_to_value(in_T, out_T_raw.squeeze(-1)).unsqueeze(-1)

                            # --- Space Step: V(t, x + f(u*, v*) * alpha) ---
                            # 위에서 고정한 u_fixed, v_fixed 사용
                            dx = self.dataset.dynamics.dsdt(curr_x, u_fixed, v_fixed)
                            next_x_space = curr_x + dx * alpha
                            coords_S = torch.cat([curr_t, next_x_space], dim=-1)

                            in_S = self.dataset.dynamics.coord_to_input(coords_S)
                            out_S_raw = self.target_model({'coords': in_S})['model_out']
                            val_S = self.dataset.dynamics.io_to_value(in_S, out_S_raw.squeeze(-1)).unsqueeze(-1)

                            # --- Combine ---
                            bellman_target = 0.5 * val_T + 0.5 * val_S
                            bellman_target_clamped = torch.clamp(bellman_target, min=-10.0, max=10.0)
                            
                            boundary_val = h_func(curr_x)
                            target_V = torch.min(boundary_val, bellman_target_clamped)

                            # Anchor Target (한 번 계산하면 되지만, 구조상 루프 안에서 계산해도 무방)
                            in_batch2 = self.dataset.dynamics.coord_to_input(batch_coords2)
                            anchor_out_raw = self.anchor_model({'coords': in_batch2})['model_out']
                            anchor_out_phys = self.dataset.dynamics.io_to_value(in_batch2, anchor_out_raw.squeeze(-1)).unsqueeze(-1)

                        # (B) Model Update (Gradient Descent)
                        # Main Prediction
                        in_batch = self.dataset.dynamics.coord_to_input(batch_coords)
                        pred_raw = self.model({'coords': in_batch})['model_out']
                        pred_V_phys = self.dataset.dynamics.io_to_value(in_batch, pred_raw.squeeze(-1)).unsqueeze(-1)

                        # Anchor Prediction
                        pred_raw2 = self.model({'coords': in_batch2})['model_out']
                        pred_V_phys2 = self.dataset.dynamics.io_to_value(in_batch2, pred_raw2.squeeze(-1)).unsqueeze(-1)

                        # Loss Calculation
                        loss = torch.mean((pred_V_phys - target_V) ** 2) + \
                               current_anchor_weight * torch.mean((pred_V_phys2 - anchor_out_phys) ** 2)

                        self.optim.zero_grad()
                        loss.backward()
                        self.optim.step()

                    # Logging
                    total_steps += 1
                    train_losses.append(loss.item())
                    
                    if pi_step % 5 == 0: # Update progress bar periodically
                        pbar.set_postfix({'Loss': f"{loss.item():.2e}", 'TargetMean': f"{target_V.mean():.2f}"})
                        pbar.update(1)

                # ----------------------------------------------------------------------
                # 7. Checkpointing & Validation
                # ----------------------------------------------------------------------
                if (epoch + 1) % 10 == 0:
                    checkpoint = {
                        'epoch': epoch + 1,
                        'model': self.model.state_dict(),
                        'optimizer': self.optim.state_dict()
                    }
                    # 모델이 하나이므로 이름도 단순화
                    torch.save(checkpoint, os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % (epoch + 1)))
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses.txt'), np.array(train_losses))
                    
                    # Validation
                    # self.validate1 함수가 내부적으로 self.model1을 쓴다면 self.model로 바꿔서 호출하거나
                    # 해당 함수를 수정해야 합니다. 여기서는 self.model을 사용한다고 가정합니다.
                    self.validate(
                        device=self.device, epoch=epoch+1, 
                        save_path=os.path.join(checkpoints_dir, 'BRS_validation_plot2_epoch_%04d.png' % (epoch+1)),
                        x_resolution=val_x_resolution, y_resolution=val_y_resolution, 
                        z_resolution=val_z_resolution, time_resolution=val_time_resolution
                    )
                    
                    print(f"\n[Epoch {epoch+1}] Model Saved. Loss: {loss.item():.5f}")
                    
        print("Training Finished (Centralized).")
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