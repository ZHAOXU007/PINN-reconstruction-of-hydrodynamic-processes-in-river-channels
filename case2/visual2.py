import torch
import matplotlib.pyplot as plt
from net import DNN
from matplotlib import gridspec
import pandas as pd
from scipy.interpolate import griddata
import numpy as np
from main2 import PINN
import os
from matplotlib.ticker import MaxNLocator
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Define parameters
x_step = 15
x_min=0
x_max=1500
t_step = 15
t_min=0
t_max=10800

# Load model
model = PINN()
model.net.load_state_dict(torch.load('discontinuous_ss72.param'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

k = model.k
print(k)


def h_acu( x, t):
    h = 16 - z_acu(x) + 4 - 4 * torch.sin(torch.pi * (4 * t / 86400 + 0.5))
    return h


def u_acu( x, t):
    u = torch.pi * (x - x_max) / 5400 / h_acu(x, t) * torch.cos(torch.pi * (4 * t / 86400 + 0.5))
    return u


def z_acu( x):
    distance = torch.abs(x - 1500 / 2)
    # 使用逐元素的条件赋值
    z = torch.where(distance <= 1500 / 8,
                    torch.tensor(8.0, device=device),
                    torch.tensor(0.0, device=device))
    return z

x = np.arange(x_min, x_max + x_step, x_step)
t = np.arange(t_min, t_max + t_step, t_step)
X, T = np.meshgrid(x, t)
xt_points = torch.tensor(np.stack([X.flatten(), T.flatten()], axis=1), dtype=torch.float32).to(device)

t0=10800
garph_x = np.arange(x_min, x_max + x_step, x_step).reshape(-1, 1)
garph_t = t0*np.ones_like(garph_x)
garph_xt = torch.tensor(np.concatenate([garph_x, garph_t], axis=1), dtype=torch.float32).to(device)
garph_h = h_acu(garph_xt[:, 0:1], garph_xt[:, 1:2]).to(device)
garph_u = u_acu(garph_xt[:, 0:1], garph_xt[:, 1:2]).to(device)
garph_z = z_acu(garph_xt[:, 0:1]).to(device)
t1=3600
garph_t2 = t1*np.ones_like(garph_x)
garph_xt2 = torch.tensor(np.concatenate([garph_x, garph_t2], axis=1), dtype=torch.float32).to(device)
garph_h2 = h_acu(garph_xt2[:, 0:1], garph_xt2[:, 1:2]).to(device)
garph_u2 = u_acu(garph_xt2[:, 0:1], garph_xt2[:, 1:2]).to(device)
garph_z2 = z_acu(garph_xt2[:, 0:1]).to(device)



with torch.no_grad():
    h_pred = model.net(garph_xt)[:, 0:1].detach().cpu().numpy()
    u_pred = model.net(garph_xt)[:, 1:2].detach().cpu().numpy()/k
    z_pred = model.net(garph_xt)[:, 2:3].detach().cpu().numpy()

    h_pred2 = model.net(garph_xt2)[:, 0:1].detach().cpu().numpy()
    u_pred2 = model.net(garph_xt2)[:, 1:2].detach().cpu().numpy()/k
    z_pred2 = model.net(garph_xt2)[:, 2:3].detach().cpu().numpy()
plt.rc('font', family='Times New Roman')  # 设置字体为 serif，大小为 12
plt.figure(figsize=(12, 8))
mark1=[0, 25, 50,75,100]
mark2=[0, 17, 33,50,67,83,100]
mark3=[0, 12, 25,37,50,63,75,87,100]
mark4=[0, 17, 37,38,50,62,63,83,100]
# 水深图
plt.subplot(3, 1, 1)
plt.plot(garph_xt[:, 0:1].detach().cpu().numpy(), garph_h[:, 0:1].detach().cpu().numpy(),
         label=f'Analytical Solution at t={t0}s', color='black',linestyle='-',marker='o', markevery=mark4, markersize=5)
plt.plot(garph_xt2[:, 0:1].detach().cpu().numpy(), garph_h2[:, 0:1].detach().cpu().numpy(),
         label=f'Analytical Solution at t={t1}s', color='black',linestyle='-',marker='o', markevery=mark4, markersize=5)
plt.plot(garph_xt[:, 0:1].detach().cpu().numpy(), h_pred[:, 0:1], label=f'PINN Prediction at t={t0}s', color='red',linestyle='--')
plt.plot(garph_xt2[:, 0:1].detach().cpu().numpy(), h_pred2[:, 0:1], label=f'PINN Prediction at t={t1}s', color='blue',linestyle='--')

plt.title('Difference between PINN and Analytical solution')
plt.ylabel('Water Depth(m)')
plt.grid()
plt.legend()

# # 流速图
plt.subplot(3, 1, 2)
plt.plot(garph_xt[:, 0:1].detach().cpu().numpy(), garph_u[:, 0:1].detach().cpu().numpy(),
         label=f'Analytical Solution at t={t0}s', color='black',linestyle='-',marker='o', markevery=mark4, markersize=5)
plt.plot(garph_xt2[:, 0:1].detach().cpu().numpy(), garph_u2[:, 0:1].detach().cpu().numpy(),
         label=f'Analytical Solution at t={t1}s', color='black',linestyle='-',marker='o', markevery=mark4, markersize=5)
plt.plot(garph_xt[:, 0:1].detach().cpu().numpy(), u_pred[:, 0:1], label=f'PINN Prediction at t={t0}s', color='red',linestyle='--')
plt.plot(garph_xt2[:, 0:1].detach().cpu().numpy(), u_pred2[:, 0:1], label=f'PINN Prediction at t={t1}s', color='blue',linestyle='--')

# plt.title('Velocity (u)')
plt.ylabel('Velocity(m/s)')
plt.grid()
plt.legend()

# 地形图
plt.subplot(3, 1, 3)
plt.plot(garph_xt[:, 0:1].detach().cpu().numpy(), garph_z[:, 0:1].detach().cpu().numpy(),
           label='Analytical Solution', color='black',linestyle='-')
plt.plot(garph_xt[:, 0:1].detach().cpu().numpy(), z_pred[:, 0:1], label=f'PINN at t={t0}s', color='red',linestyle='--')
plt.plot(garph_xt2[:, 0:1].detach().cpu().numpy(), z_pred2[:, 0:1], label=f'PINN at t={t1}s', color='blue',linestyle='--')

# plt.title('Terrain (z)')
plt.ylabel('Bottom Elevation(m)')
plt.xlabel('Distance(m)')
plt.grid()
plt.legend()

plt.figtext(0.02, 0.98, '(d)', fontsize=14, fontweight='bold', va='top')
plt.tight_layout()
plt.savefig("s72.tiff",  # 或 .svg/.eps
           format="tiff",
           bbox_inches="tight",
           dpi=600)  # 内嵌位图的分辨率（如果有）
plt.show()
