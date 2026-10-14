import torch
import matplotlib.pyplot as plt
from net import DNN
from matplotlib import gridspec
import numpy as np
from main2 import PINN
import os
from matplotlib.ticker import MaxNLocator

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Define parameters
x_step = 10
x_min=0
x_max=1500
t_step = 60
t_min=0
t_max=10800

# Load model
model = PINN()
model.net.load_state_dict(torch.load('discontinuous_ss72.param'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

k = model.k

inlet_point = 0
x_point1 = 250
x_point2 = 555
x_point3 = 570
x_point4 = 930
x_point5 = 945
x_point6 = 1250
x_point7 = 750
select_location = {x_min,250, 555, 570, 930, 945, 1250, 750,x_max}
t_step2 = 200
print(k)
# Define functions
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


# Create meshgrid
x = np.arange(x_min, x_max + x_step, x_step)
t = np.arange(t_min, t_max + t_step, t_step)
t2 = np.arange(t_min, t_max + t_step2, t_step2)
X, T = np.meshgrid(x, t)
xt_points = torch.tensor(np.stack([X.flatten(), T.flatten()], axis=1), dtype=torch.float32).to(device)

# Compute real values
h_real = h_acu(xt_points[:, 0:1], xt_points[:, 1:2]).detach().cpu().numpy().reshape(len(t), len(x))
u_real = u_acu(xt_points[:, 0:1], xt_points[:, 1:2]).detach().cpu().numpy().reshape(len(t), len(x))
z_real = z_acu(xt_points[:, 0:1]).detach().cpu().numpy().reshape(len(t), len(x))

# Compute predictions
with torch.no_grad():
    h_pred = model.net(xt_points).detach().cpu().numpy()[:, 0:1].reshape(len(t), len(x))
    u_pred = model.net(xt_points).detach().cpu().numpy()[:, 1:2].reshape(len(t), len(x)) / k
    z_pred = model.net(xt_points).detach().cpu().numpy()[:, 2:3].reshape(len(t), len(x))

# Print shapes for debugging
print(f"Shapes - h_real: {h_real.shape}, h_pred: {h_pred.shape}")

# Compute errors
# error_h = np.log10(np.abs(h_pred - h_real))
# error_u = np.log10(np.abs(u_pred - u_real))
# error_z = np.log10(np.abs(z_pred - z_real))
error_h = np.abs(h_pred - h_real)
error_u = np.abs(u_pred - u_real)
error_z = np.abs(z_pred - z_real)
# Check the size of error arrays
print(f"Error sizes - error_h: {error_h.size}, error_u: {error_u.size}, error_z: {error_z.size}")

# Relative L2 errors
relative_l2_h_error = np.linalg.norm(h_real - h_pred) / np.linalg.norm(h_real)
relative_l2_u_error = np.linalg.norm(u_real - u_pred) / np.linalg.norm(u_real)
relative_l2_z_error = np.linalg.norm(z_real - z_pred) / np.linalg.norm(z_real)
print(f'relative_l2_h_error: {relative_l2_h_error:.4f}')
print(f'relative_l2_u_error: {relative_l2_u_error:.4f}')
print(f'relative_l2_z_error: {relative_l2_z_error:.4f}')

# Plotting
plt.rc('font', family='Times New Roman')
plt.figure(figsize=(18, 12))

fig = plt.figure(figsize=(18, 12),
               facecolor='white',
               tight_layout=False)  # 必须禁用tight_layout

# 使用GridSpec精确控制（适合复杂布局）
gs = gridspec.GridSpec(3, 3,
                      top=0.985,
                      bottom=0.074,
                      left=0.05,
                      right=0.99,
                      hspace=0.25,
                      wspace=0.15)


Size=1
Alpha=0.5
# h_pred plot
ax_1 = plt.subplot(gs[0, 1])
contour1 = ax_1.contourf(x, t, h_pred, levels=100, cmap='coolwarm')
#ax_1.set_title(r"h_pred", fontsize=15)
ax_1.set_xlabel("x (m)", fontsize=15)
ax_1.set_ylabel("time (s)", fontsize=15)
cbar1 = plt.colorbar(contour1, ax=ax_1)
cbar1.locator = MaxNLocator(integer=True)
cbar1.update_ticks()


# u_pred plot
ax2 = plt.subplot(gs[1, 1])
contour2 = ax2.contourf(x, t, u_pred, levels=100, cmap='coolwarm')
#ax2.set_title("u_pred", fontsize=15)
ax2.set_xlabel("x (m)", fontsize=15)
ax2.set_ylabel("time (s)", fontsize=15)
cbar2 = plt.colorbar(contour2, ax=ax2)
cbar2.locator = MaxNLocator(integer=True)
cbar2.update_ticks()


# z_pred plot
ax3 = plt.subplot(gs[2, 1])
contour3 = ax3.contourf(x, t, z_pred, levels=100, cmap='coolwarm')
#ax3.set_title("z_pred", fontsize=15)
ax3.set_xlabel("x (m)", fontsize=15)
ax3.set_ylabel("time (s)", fontsize=15)
cbar3 = plt.colorbar(contour3, ax=ax3)
cbar3.locator = MaxNLocator(integer=True)
cbar3.update_ticks()


# h_real plot
ax_11 = plt.subplot(gs[0, 0])
contour11 = ax_11.contourf(x, t, h_real, levels=100, cmap='coolwarm')
#ax_11.set_title(r"h_real", fontsize=15)
ax_11.set_xlabel("x (m)", fontsize=15)
ax_11.set_ylabel("time (s)", fontsize=15)
cbar11 = plt.colorbar(contour11, ax=ax_11)
cbar11.locator = MaxNLocator(integer=True)
cbar11.update_ticks()
# for loc in select_location:
#     ax_11.scatter([loc]*len(t2), t2, color='black',s=Size, alpha=Alpha)
for loc in select_location:
    ax_11.scatter([loc] * len(t2), t2, color='black', s=5, alpha=Alpha)

# u_real plot
ax22 = plt.subplot(gs[1, 0])
contour22 = ax22.contourf(x, t, u_real, levels=100, cmap='coolwarm')
#ax22.set_title("u_real", fontsize=15)
ax22.set_xlabel("x (m)", fontsize=15)
ax22.set_ylabel("time (s)", fontsize=15)
cbar22 = plt.colorbar(contour22, ax=ax22)
cbar22.locator = MaxNLocator(integer=True)
cbar22.update_ticks()
for loc in select_location:
    ax22.scatter([loc]*len(t2), t2, color='black', s=5, alpha=Alpha)
# z_real plot
ax33 = plt.subplot(gs[2, 0])
contour33 = ax33.contourf(x, t, z_real, levels=100, cmap='coolwarm')
#ax33.set_title("z_real", fontsize=15)
ax33.set_xlabel("x (m)", fontsize=15)
ax33.set_ylabel("time (s)", fontsize=15)
cbar33 = plt.colorbar(contour33, ax=ax33)
cbar33.locator = MaxNLocator(integer=True)
cbar33.update_ticks()


# Error plots (新增的三个误差图)
# h error plot
ax_err_h = plt.subplot(gs[0, 2])
contour_err_h = ax_err_h.contourf(x, t, error_h, levels=100, cmap='coolwarm')
#ax_err_h.set_title(r"h Error (Pred - Real)", fontsize=15)
ax_err_h.set_xlabel("x (m)", fontsize=15)
ax_err_h.set_ylabel("time (s)", fontsize=15)
cbar_err_h = plt.colorbar(contour_err_h, ax=ax_err_h)


# u error plot
ax_err_u = plt.subplot(gs[1, 2])
contour_err_u = ax_err_u.contourf(x, t, error_u, levels=100, cmap='coolwarm')
#ax_err_u.set_title("u Error (Pred - Real)", fontsize=15)
ax_err_u.set_xlabel("x (m)", fontsize=15)
ax_err_u.set_ylabel("time (s)", fontsize=15)
cbar_err_u = plt.colorbar(contour_err_u, ax=ax_err_u)


# z error plot
ax_err_z = plt.subplot(gs[2, 2])
contour_err_z = ax_err_z.contourf(x, t, error_z, levels=100, cmap='coolwarm')
#ax_err_z.set_title("z Error (Pred - Real)", fontsize=15)
ax_err_z.set_xlabel("x (m)", fontsize=15)
ax_err_z.set_ylabel("time (s)", fontsize=15)
cbar_err_z = plt.colorbar(contour_err_z, ax=ax_err_z)
#real
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
plt.tight_layout()
def onclick(event):
    if event.inaxes:
        print(f'点击位置: 数据坐标=({event.xdata:.2f},{event.ydata:.2f}), '
              f'相对坐标=({event.x/fig.get_figwidth():.2f},'
              f'{event.y/fig.get_figheight():.2f})')

fig.canvas.mpl_connect('button_press_event', onclick)
plt.savefig("vector_output.tiff",  # 或 .svg/.eps
           format="tiff",
           bbox_inches="tight",
           dpi=600)  # 内嵌位图的分辨率（如果有）
plt.show()





