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
t_step2 = 200
# Create meshgrid
x = np.arange(x_min, x_max + x_step, x_step)
t = np.arange(t_min, t_max + t_step, t_step)
t2 = np.arange(t_min, t_max + t_step2, t_step2)
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



select_location = {x_min,2500, 5000, 7500, x_max}
time_points = [0*3600, 12*3600, 24*3600]  # 0h, 6h, 12h, 18h, 24h
time_labels = ['0h',  '12h',  '24h']
Alpha=0.5

# Plotting
plt.rc('font', family='Times New Roman')

# ============================================================================
# 图1: 真实值、预测值和误差的云图
# ============================================================================
fig1 = plt.figure(figsize=(18, 12),
               facecolor='white',
               tight_layout=False)
gs1 = gridspec.GridSpec(3, 3,
                      top=0.985,
                      bottom=0.074,
                      left=0.05,
                      right=0.99,
                      hspace=0.25,
                      wspace=0.15)

# 第一行：真实值
ax1 = plt.subplot(gs1[0, 0])
contour1 = ax1.contourf(X_grid, T_grid, h_real_grid, levels=100, cmap='coolwarm')
ax1.set_ylabel("Time (s)", fontsize=15)
ax1.set_xlabel("x (m)", fontsize=15)
cbar1 = plt.colorbar(contour1, ax=ax1)
for loc in select_location:
    ax1.scatter([loc] * len(t2), t2, color='black', s=5, alpha=Alpha)

ax2 = plt.subplot(gs1[1, 0])
contour2 = ax2.contourf(X_grid, T_grid, u_real_grid, levels=100, cmap='coolwarm')
ax2.set_ylabel("Time (s)", fontsize=15)
ax2.set_xlabel("x (m)", fontsize=15)
cbar2 = plt.colorbar(contour2, ax=ax2)
for loc in select_location:
    ax2.scatter([loc] * len(t2), t2, color='black', s=5, alpha=Alpha)

ax3 = plt.subplot(gs1[2, 0])
contour3 = ax3.contourf(X_grid, T_grid, z_real_grid, levels=100, cmap='coolwarm')
ax3.set_ylabel("Time (s)", fontsize=15)
ax3.set_xlabel("x (m)", fontsize=15)
cbar3 = plt.colorbar(contour3, ax=ax3)

# 第二行：预测值
ax4 = plt.subplot(gs1[0, 1])
contour4 = ax4.contourf(X_grid, T_grid, h_pred, levels=100, cmap='coolwarm')
ax4.set_ylabel("Time (s)", fontsize=15)
ax4.set_xlabel("x (m)", fontsize=15)
cbar4 = plt.colorbar(contour4, ax=ax4)

ax5 = plt.subplot(gs1[1, 1])
contour5 = ax5.contourf(X_grid, T_grid, u_pred, levels=100, cmap='coolwarm')
ax5.set_ylabel("Time (s)", fontsize=15)
ax5.set_xlabel("x (m)", fontsize=15)
cbar5 = plt.colorbar(contour5, ax=ax5)

ax6 = plt.subplot(gs1[2, 1])
contour6 = ax6.contourf(X_grid, T_grid, z_pred, levels=100, cmap='coolwarm')
ax6.set_ylabel("Time (s)", fontsize=15)
ax6.set_xlabel("x (m)", fontsize=15)
cbar6 = plt.colorbar(contour6, ax=ax6)

# 第三行：误差
ax7 = plt.subplot(gs1[0, 2])
contour7 = ax7.contourf(X_grid, T_grid, error_h, levels=100, cmap='coolwarm')
ax7.set_ylabel("Time (s)", fontsize=15)
ax7.set_xlabel("x (m)", fontsize=15)
cbar7 = plt.colorbar(contour7, ax=ax7)

ax8 = plt.subplot(gs1[1, 2])
contour8 = ax8.contourf(X_grid, T_grid, error_u, levels=100, cmap='coolwarm')
ax8.set_ylabel("Time (s)", fontsize=15)
ax8.set_xlabel("x (m)", fontsize=15)
cbar8 = plt.colorbar(contour8, ax=ax8)

ax9 = plt.subplot(gs1[2, 2])
contour9 = ax9.contourf(X_grid, T_grid, error_z, levels=100, cmap='coolwarm')
ax9.set_ylabel("Time (s)", fontsize=15)
ax9.set_xlabel("x (m)", fontsize=15)
cbar9 = plt.colorbar(contour9, ax=ax9)
plt.figtext(0.055, 0.96, 'h_real',fontweight='bold',
           bbox={'facecolor':'lightgray', 'alpha':0.0, 'pad':10,},
           fontsize=14)
plt.figtext(0.055, 0.64, 'u_real',fontweight='bold',
           bbox={'facecolor':'lightgray', 'alpha':0.0, 'pad':10},
           fontsize=14)
plt.figtext(0.055, 0.31, 'z_real',fontweight='bold',
           bbox={'facecolor':'lightgray', 'alpha':0.0, 'pad':10},
           fontsize=14)
#pred
plt.figtext(0.38, 0.96, 'h_pred',fontweight='bold',
           bbox={'facecolor':'lightgray', 'alpha':0.0, 'pad':10},
           fontsize=14)
plt.figtext(0.38, 0.64, 'u_pred',fontweight='bold',
           bbox={'facecolor':'lightgray', 'alpha':0.0, 'pad':10},
           fontsize=14)
plt.figtext(0.38, 0.31, 'z_pred',fontweight='bold',
           bbox={'facecolor':'lightgray', 'alpha':0.0, 'pad':10},
           fontsize=14)
#error
plt.figtext(0.71, 0.96, '|h_real-h_pred|',fontweight='bold',
           bbox={'facecolor':'lightgray', 'alpha':0.0, 'pad':10},
           fontsize=14)
plt.figtext(0.71, 0.64, '|u_real-u_pred|',fontweight='bold',
           bbox={'facecolor':'lightgray', 'alpha':0.0, 'pad':10},
           fontsize=14)
plt.figtext(0.71, 0.31, '|z_real-z_pred|',fontweight='bold',
           bbox={'facecolor':'lightgray', 'alpha':0.0, 'pad':10},
           fontsize=14)


# plt.savefig("vector_output.tiff",  # 或 .svg/.eps
#            format="tiff",
#            bbox_inches="tight",
#            dpi=600)  # 内嵌位图的分辨率（如果有）


plt.tight_layout()
plt.show()