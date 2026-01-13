import torch
import matplotlib.pyplot as plt
from net import DNN
from matplotlib import gridspec
from scipy.interpolate import griddata
import numpy as np
from main3 import PINN
import os
from matplotlib.ticker import MaxNLocator
import pandas as pd
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Define parameters
x_step = 100
x_min = 0
x_max = 10000
t_step = 60
t_min = 0
t_max = 86400

# Load model
model = PINN()
model.net.load_state_dict(torch.load('model_a100.param'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = 'real_data.csv'
df = pd.read_csv(file_path, header=0)
x_real = df.iloc[:, 0].values
t_real = df.iloc[:, 1].values
h_real = df.iloc[:, 2].values
u_real = df.iloc[:, 3].values
z_real = df.iloc[:, 4].values
k = model.k
print(k)

# Create meshgrid
x = np.arange(x_min, x_max + x_step, x_step)
t = np.arange(t_min, t_max + t_step, t_step)
X, T = np.meshgrid(x, t)
xt_points = torch.tensor(np.stack([X.flatten(), T.flatten()], axis=1), dtype=torch.float32).to(device)

# Compute predictions
with torch.no_grad():
    predictions = model.net(xt_points).detach().cpu().numpy()
    h_pred = predictions[:, 0:1].reshape(len(t), len(x))
    u_pred = predictions[:, 1:2].reshape(len(t), len(x))
    z_pred = predictions[:, 2:3].reshape(len(t), len(x))

# 对真实数据进行插值
X_grid, T_grid = np.meshgrid(x, t)
points = np.column_stack((x_real, t_real))
h_real_grid = griddata(points, h_real, (X_grid, T_grid), method='linear')
u_real_grid = griddata(points, u_real, (X_grid, T_grid), method='linear')
z_real_grid = griddata(points, z_real, (X_grid, T_grid), method='linear')

# 计算误差
error_h = np.abs(h_pred - h_real_grid)
error_u = np.abs(u_pred - u_real_grid)
error_z = np.abs(z_pred - z_real_grid)

# 计算相对L2误差
relative_l2_error_h = np.linalg.norm(h_real_grid - h_pred) / np.linalg.norm(h_real_grid)
relative_l2_error_u = np.linalg.norm(u_real_grid - u_pred) / np.linalg.norm(u_real_grid)
relative_l2_error_z = np.linalg.norm(z_real_grid - z_pred) / np.linalg.norm(z_real_grid)

# 打印相对 L2 误差
print(f'Relative L2 Error of h: {relative_l2_error_h:.4f}')
print(f'Relative L2 Error of u: {relative_l2_error_u:.4f}')
print(f'Relative L2 Error of z: {relative_l2_error_z:.4f}')

time_points = [0*3600, 12*3600, 24*3600]  # 0h, 6h, 12h, 18h, 24h
time_labels = ['0',  '12',  '24']


# Plotting
plt.rc('font', family='Times New Roman')

fig2 = plt.figure(figsize=(20, 12), facecolor='white')
gs2 = gridspec.GridSpec(3, 1,
                        height_ratios=[1, 1, 1],
                        top=0.96,
                        bottom=0.06,
                        left=0.1,
                        right=0.95,
                        hspace=0.25)


# 定义颜色和线型
pred_colors = ['red', 'blue', 'orange']  # 预测值的颜色
line_styles = ['-', '--']  # 实线用于真实值，虚线用于预测值
fontsize=15

ax1 = plt.subplot(gs2[0, 0])
for i, tp in enumerate(time_points):
    idx = np.argmin(np.abs(t - tp))
    # 预测值 - 彩色虚线
    ax1.plot(x, h_pred[idx, :],
             color=pred_colors[i],
             linestyle='--',
             label=f'PINN result at t={time_labels[i]}h',
             linewidth=1.5)
    # 真实值 - 黑色实线
    ax1.plot(x, h_real_grid[idx, :],
             color='black',
             linestyle='-',
             label=f'CFD result at t={time_labels[i]}h',
             linewidth=1.5,
             alpha=0.8)

ax1.set_title("Comparison between PINN results and CFD results", fontsize=fontsize, fontweight='bold')
ax1.set_ylabel("Water depth(m)", fontsize=fontsize)
ax1.legend(ncol=1, fontsize=12, loc='best')
ax1.grid(True, alpha=0.5)

ax2 = plt.subplot(gs2[1, 0])
for i, tp in enumerate(time_points):
    idx = np.argmin(np.abs(t - tp))
    ax2.plot(x, u_pred[idx, :],
             color=pred_colors[i],
             linestyle='--',
             label=f'PINN result at t={time_labels[i]}h',
             linewidth=1.5)
    ax2.plot(x, u_real_grid[idx, :],
             color='black',
             linestyle='-',
             label=f'CFD result at t={time_labels[i]}h',
             linewidth=1.5,
             alpha=0.8)
ax2.set_ylabel("Velocity(m/s)", fontsize=fontsize)
ax2.legend(ncol=1, fontsize=12, loc='best')
ax2.grid(True, alpha=0.5)

ax3 = plt.subplot(gs2[2, 0])
for i, tp in enumerate(time_points):
    idx = np.argmin(np.abs(t - tp))
    ax3.plot(x, z_pred[idx, :],
             color=pred_colors[i],
             linestyle='--',
             label=f'PINN result at t={time_labels[i]}h',
             linewidth=1.5)
    ax3.plot(x, z_real_grid[idx, :],
             color='black',
             linestyle='-',
             label=f'CFD result at t={time_labels[i]}h',
             linewidth=1.5,
             alpha=0.8)
ax3.set_ylabel("Bottom elevation(m)", fontsize=fontsize)
ax3.set_xlabel("Distance(m)", fontsize=fontsize)
ax3.legend(ncol=1, fontsize=12, loc='best')
ax3.grid(True, alpha=0.5)

plt.savefig("vector_output2.tiff",  # 或 .svg/.eps
           format="tiff",
           bbox_inches="tight",
           dpi=600)  # 内嵌位图的分辨率（如果有）


plt.tight_layout()
plt.show()