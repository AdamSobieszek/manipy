import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup
from torchdiffeq import odeint
from tqdm import tqdm
import numpy as np
import os
import argparse
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Assuming these local modules are in the python path
from full_manipy.models.flow_models import VectorFieldTransformer, VectorFieldModel, RatingODE
from full_manipy.models.rating_models import EnsembleRegressor
from full_manipy.models.layers import AlphaBetaRegressor


# --- Visualization Function ---
def visualize_and_compare_trajectories(teacher_model, student_model, rating_model,
                                         initial_points, val_data_for_pca, writer,
                                         global_step, num_steps=10, device='cpu'):
    """
    Computes, visualizes, and compares the ODE trajectories of the teacher and student models.
    """
    print("Generating trajectory comparison plot...")
    teacher_model.eval()
    student_model.eval()

    # 1. Fit PCA on validation data to create a 2D projection
    pca = PCA(n_components=2)
    pca.fit(val_data_for_pca.cpu().numpy())

    # 2. Setup ODE solvers
    t_span = torch.linspace(0, 10, num_steps, device=device)
    teacher_ode = RatingODE(teacher_model)
    student_ode = RatingODE(student_model)

    # 3. Compute trajectories
    with torch.no_grad():
        traj_teacher = odeint(teacher_ode, initial_points, t_span, method='rk4', options={'step_size': 10.0/num_steps})
        traj_student = odeint(student_ode, initial_points, t_span, method='rk4', options={'step_size': 10.0/num_steps})
        # Shape of traj tensors: (num_steps, num_trajectories, 512)

    # 4. Calculate step-wise MSE divergence
    mse_divergence = torch.mean((traj_teacher - traj_student)**2, dim=(1, 2)).cpu().numpy()

    # 5. Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=120)
    fig.suptitle(f'Trajectory Comparison @ Step {global_step}', fontsize=16)

    # Subplot 1: 2D Trajectory Visualization
    ax1.set_title('Teacher vs. Student Trajectories (PCA Projection)')
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')
    for i in range(initial_points.shape[0]):
        # Project trajectories to 2D
        traj_2d_teacher = pca.transform(traj_teacher[:, i, :].cpu().numpy())
        traj_2d_student = pca.transform(traj_student[:, i, :].cpu().numpy())

        # Plot
        ax1.plot(traj_2d_teacher[:, 0], traj_2d_teacher[:, 1], 'b-', alpha=0.5, label='Teacher' if i == 0 else "")
        ax1.plot(traj_2d_student[:, 0], traj_2d_student[:, 1], 'r--', alpha=0.7, label='Student' if i == 0 else "")
        ax1.scatter(traj_2d_teacher[0, 0], traj_2d_teacher[0, 1], c='g', marker='o', s=30, zorder=5, label='Start' if i == 0 else "")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Subplot 2: Trajectory Divergence (MSE)
    ax2.set_title('Trajectory Divergence over Time')
    ax2.plot(range(num_steps), mse_divergence, 'm-o')
    ax2.set_xlabel('Integration Step')
    ax2.set_ylabel('Mean Squared Error')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_xticks(range(num_steps))

    # 6. Log figure to TensorBoard
    writer.add_figure('Evaluation/Trajectory_Comparison', fig, global_step)
    plt.close(fig) # Avoid displaying the plot in the notebook/console

# --- Dataset and Collate (Unchanged) ---
class LatentVectorDataset(Dataset):
    def __init__(self, generator, num_samples, device, batch_size=128, val=False, noise_level=0.03, rating_model=None):
        self.generator = generator
        self.num_samples = num_samples
        self.device = device
        self.z_dim = self.generator.z_dim
        self.val = val
        self.noise_level = noise_level
        print(f"Pre-generating {num_samples} W vectors...")
        all_w = []
        all_ratings = []
        self.generator.eval()
        with torch.no_grad():
            for _ in tqdm(range(int(np.ceil(num_samples / batch_size))), desc="Generating W vectors"):
                z = torch.randn(batch_size, self.z_dim, device=self.device)
                w = self.generator.mapping(z, None, truncation_psi=max(0.01,np.random.randn()/3+0.99) if not self.val else 1)[:, 0]
                w = w + torch.randn_like(w) * self.noise_level
                all_w.append(w)
                all_ratings.append(rating_model(w))
        self.w_vectors = torch.cat(all_w, dim=0)[:num_samples]
        self.w_vectors = self.w_vectors 
        self.ratings = torch.cat(all_ratings, dim=0)[:num_samples]
        print(f"Dataset ready with {self.w_vectors.shape[0]} vectors on device {self.w_vectors.device}.")

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.w_vectors[idx], self.ratings[idx]


# --- Main Distillation Function ---
def distill_vector_field_model(config):
    # --- 1. Setup Environment ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
    DTYPE = torch.float32
    print(f"Using device: {DEVICE}")
    dim_name = config.dim_name
    print(f"Using dimension: {dim_name}")
    config.log_dir = config.log_dir + f'_{dim_name}'
    writer = SummaryWriter(log_dir=config.log_dir)

    # --- 2. Load Models ---
    sys.path.append('/Users/adamsobieszek/PycharmProjects/psychGAN/content/psychGAN/stylegan3')
    with open('/Users/adamsobieszek/PycharmProjects/psychGAN/stylegan2-ffhq-1024x1024.pkl', 'rb') as fp:
        stylegan_generator = pickle.load(fp)['G_ema'].to(DEVICE)
        stylegan_generator.eval()
    models = [AlphaBetaRegressor(dim=512).to(DEVICE) for _ in range(8)]
    for i, m_state in enumerate([torch.load(f"/Users/adamsobieszek/PycharmProjects/psychGAN/best_models/model_{dim_name}_v{3+i}.pt", map_location=DEVICE) for i in range(8)]):
        models[i].load_state_dict(m_state)
        models[i].eval()
    rating_model = EnsembleRegressor(models, model_kwargs={'output':"logit"}).to(DEVICE)
    rating_model.eval()
    teacher_model = VectorFieldTransformer(rating_model=rating_model, dim=512, depth=8, num_heads=8, dim_head=48, num_registers=32, dropout=0.15, add_rating_gradient=True, use_rotary=True).to(DEVICE)
    checkpoint_path = f'/Users/adamsobieszek/PycharmProjects/psychGAN/{dim_name}_final_final_final.pt'
    teacher_model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    teacher_model.eval()
    print("Teacher model (VectorFieldTransformer) loaded.")
    student_model = VectorFieldModel(input_dim=512, hidden_dims=[1024, 2048, 4096, 2048, 1024], output_dim=512, dropout_rate=0.1, center_w_avg=stylegan_generator.mapping.w_avg.to(DEVICE)).to(DEVICE, DTYPE)
    student_model.train()
    print("Student model (VectorFieldModel) created.")

    # --- 2a. COMPILE MODELS ---
    # Note: torch.compile() is introduced in PyTorch 2.0.
    # The first run of a compiled model will have some overhead for JIT compilation.
    # print("Compiling models with torch.compile()...")
    # rating_model = torch.compile(rating_model)
    # teacher_model = torch.compile(teacher_model)
    # student_model = torch.compile(student_model)
    # print("Models compiled.")


    # --- 3. DataLoaders with Custom Collate ---

    train_dataset = LatentVectorDataset(stylegan_generator, config.num_train_samples, DEVICE, 1024, noise_level=config.noise_level, rating_model=rating_model)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    val_dataset = LatentVectorDataset(stylegan_generator, config.num_val_samples, DEVICE, 1024, val=True, rating_model=rating_model)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # --- 4. Optimizer, Scheduler, and Loss ---
    optimizer = optim.AdamW(student_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = 100
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    mse_loss = nn.MSELoss()
    cosine_loss = lambda v1, v2: (1 - nn.functional.cosine_similarity(v1, v2)).mean()

    # --- 5. Training Loop ---
    global_step = 0
    for epoch in range(config.num_epochs):
        student_model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}", leave=False)
        for w_batch, rating_batch in progress_bar:
            w_batch, rating_batch = w_batch.to(DEVICE), rating_batch.to(DEVICE).view(-1, 1)
            optimizer.zero_grad()
            with torch.no_grad():
                v_teacher = teacher_model(w_batch)
            v_student = student_model(w_batch)
            loss_mse_val = mse_loss(v_student, v_teacher)
            loss_cos_val = cosine_loss(v_student, v_teacher)
            total_loss = loss_mse_val + config.cosine_weight * loss_cos_val
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            writer.add_scalar('Loss/train_total', total_loss.item(), global_step)
            writer.add_scalar('Loss/train_mse', loss_mse_val.item(), global_step)
            writer.add_scalar('Loss/train_cosine', loss_cos_val.item(), global_step)
            writer.add_scalar('Meta/learning_rate', scheduler.get_last_lr()[0], global_step)
            progress_bar.set_postfix(loss=f"{total_loss.item():.4f}")
            global_step += 1

        # --- 6. Validation and Functional Evaluation ---
        if (epoch + 1) % config.eval_every == 0:
            student_model.eval()
            val_loss_mse, val_loss_cos = 0, 0
            with torch.no_grad():
                for w_batch, rating_batch in val_loader:
                    w_batch, rating_batch = w_batch.to(DEVICE), rating_batch.to(DEVICE).view(-1, 1)
                    v_teacher = teacher_model(w_batch)
                    v_student = student_model(w_batch)
                    val_loss_mse += mse_loss(v_student, v_teacher).item()
                    val_loss_cos += cosine_loss(v_student, v_teacher).item()
            avg_val_mse = val_loss_mse / len(val_loader)
            avg_val_cos = val_loss_cos / len(val_loader)
            writer.add_scalar('Loss/val_mse', avg_val_mse, global_step)
            writer.add_scalar('Loss/val_cosine', avg_val_cos, global_step)
            print(f"\nEpoch {epoch+1} | Val MSE: {avg_val_mse:.4f}, Val Cosine Loss: {avg_val_cos:.4f}")

            # *** CALL THE NEW VISUALIZATION FUNCTION ***
            initial_points_for_vis = val_dataset.w_vectors[:config.num_vis_trajectories].to(DEVICE)
            visualize_and_compare_trajectories(
                teacher_model, student_model, rating_model,
                initial_points=initial_points_for_vis,
                val_data_for_pca=val_dataset.w_vectors, # Use all val data for stable PCA
                writer=writer,
                global_step=global_step,
                device=DEVICE
            )

            # Save a checkpoint
            torch.save(student_model.state_dict(), os.path.join(config.log_dir, f'student_epoch_{epoch+1}.pth'))
        if epoch % 1 == 0:
            del train_dataset, train_loader
            if DEVICE == 'mps':
                torch.mps.empty_cache()
            elif DEVICE == 'cuda':
                torch.cuda.empty_cache()
            train_dataset = LatentVectorDataset(stylegan_generator, config.num_train_samples, DEVICE, 1024, noise_level=config.noise_level, rating_model=rating_model)
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    writer.close()
    print("Distillation training finished.")
    return student_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Distill a VectorFieldTransformer to an MLP.")
    parser.add_argument('--dim_name', type=str, default='attractive', help='Dimension to train on.')
    parser.add_argument('--log_dir', type=str, default=f'runs/distillation_{datetime.now().strftime("%Y%m%d_%H%M%S")}', help='Directory for TensorBoard logs.')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='AdamW weight decay.')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size.')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of training epochs.')
    parser.add_argument('--num_train_samples', type=int, default=50_000, help='Number of samples per epoch for training.')
    parser.add_argument('--num_val_samples', type=int, default=5000, help='Number of samples for validation.')
    parser.add_argument('--eval_every', type=int, default=4, help='Run validation and ODE eval every N epochs.')
    parser.add_argument('--noise_level', type=float, default=0.03, help='Std deviation of noise for data augmentation.')
    parser.add_argument('--cosine_weight', type=float, default=0.5, help='Weight for the cosine similarity loss term.')
    # New argument for visualization
    parser.add_argument('--num_vis_trajectories', type=int, default=50, help='Number of trajectories to plot for visualization.')

    args = parser.parse_args()
    distilled_model = distill_vector_field_model(args)