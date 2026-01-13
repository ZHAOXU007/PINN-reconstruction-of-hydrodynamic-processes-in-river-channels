import torch
import matplotlib.pyplot as plt
from net import DNN
from matplotlib import gridspec
import pandas as pd
from scipy.interpolate import griddata
import numpy as np
from main1 import PINN
import os
from matplotlib.ticker import MaxNLocator
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
x_step = 100
x_min=0
x_max=14000
t_step = 60
t_min=0
t_max=14400



model = PINN()
model.net.load_state_dict(torch.load(f'model_k1000.param'))

k = model.k
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def h_acu( x, t):
    h = 54.5 - 40 * x / 14000 - 10 * torch.sin(torch.pi * (4 * x / 14000 - 0.5)) - 4 * torch.sin(
        torch.pi * (4 * t / 86400 + 0.5))
    return h


def u_acu( x, t):
    u = torch.pi * (x - 14000) / 5400 / h_acu(x, t) * torch.cos(torch.pi * (4 * t / 86400 + 0.5))
    return u


def z_acu( x):
    z = 10 + 40 * x / 14000 + 10 * torch.sin(torch.pi * (4 * x / 14000 - 0.5))
    return z



x = np.arange(x_min, x_max + x_step, x_step)
t = np.arange(t_min, t_max + t_step, t_step)
X, T = np.meshgrid(x, t)
xt_points = torch.tensor(np.stack([X.flatten(), T.flatten()], axis=1), dtype=torch.float32).to(device)

t0=3600
garph_x = np.arange(x_min, x_max + x_step, x_step).reshape(-1, 1)
garph_t = t0*np.ones_like(garph_x)
garph_xt = torch.tensor(np.concatenate([garph_x, garph_t], axis=1), dtype=torch.float32).to(device)
garph_h = h_acu(garph_xt[:, 0:1], garph_xt[:, 1:2]).to(device)
garph_u = u_acu(garph_xt[:, 0:1], garph_xt[:, 1:2]).to(device)
garph_z = z_acu(garph_xt[:, 0:1]).to(device)
t1=14400
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

# 水深图
plt.subplot(3, 1, 1)
plt.plot(garph_xt[:, 0:1].detach().cpu().numpy(), garph_h[:, 0:1].detach().cpu().numpy(),
         label=f'Analytical Solution at t={t0}s', color='black',linestyle='-')
plt.plot(garph_xt2[:, 0:1].detach().cpu().numpy(), garph_h2[:, 0:1].detach().cpu().numpy(),
         label=f'Analytical Solution at t={t1}s', color='black',linestyle='-')
plt.plot(garph_xt[:, 0:1].detach().cpu().numpy(), h_pred[:, 0:1], label=f'PINN Prediction at t={t0}s', color='red',linestyle='--')
plt.plot(garph_xt2[:, 0:1].detach().cpu().numpy(), h_pred2[:, 0:1], label=f'PINN Prediction at t={t1}s', color='blue',linestyle='--')

plt.title('Difference between PINN and Analytical solution')
plt.ylabel('Water Depth(m)')
plt.grid()
plt.legend()

# # 流速图
plt.subplot(3, 1, 2)
plt.plot(garph_xt[:, 0:1].detach().cpu().numpy(), garph_u[:, 0:1].detach().cpu().numpy(),
         label=f'Analytical Solution at t={t0}s', color='black',linestyle='-')
plt.plot(garph_xt2[:, 0:1].detach().cpu().numpy(), garph_u2[:, 0:1].detach().cpu().numpy(),
         label=f'Analytical Solution at t={t1}s', color='black',linestyle='-')
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

plt.tight_layout()
plt.savefig("CROSS_C.tiff",  # 或 .svg/.eps
           format="tiff",
           bbox_inches="tight",
           dpi=600)  # 内嵌位图的分辨率（如果有）
plt.show()


def export_comparison_data(output_file=f'{k}_results.xlsx'):
    # 准备数据容器
    data_dict = {
        'x': garph_xt[:, 0].cpu().numpy(),
        't0': t0,
        't1': t1,

        # 水深数据
        'h_true_t0': garph_h[:, 0].cpu().numpy(),
        'h_pred_t0': h_pred[:, 0],
        'h_true_t1': garph_h2[:, 0].cpu().numpy(),
        'h_pred_t1': h_pred2[:, 0],

        # 流速数据
        'u_true_t0': garph_u[:, 0].cpu().numpy(),
        'u_pred_t0': u_pred[:, 0],
        'u_true_t1': garph_u2[:, 0].cpu().numpy(),
        'u_pred_t1': u_pred2[:, 0],

        # 地形数据
        'z_true': garph_z[:, 0].cpu().numpy(),
        'z_pred_t0': z_pred[:, 0],
        'z_pred_t1': z_pred2[:, 0]
    }

    # 创建DataFrame
    df = pd.DataFrame(data_dict)

    # 计算误差
    df['h_error_t0'] = df['h_true_t0'] - df['h_pred_t0']
    df['h_error_t1'] = df['h_true_t1'] - df['h_pred_t1']
    df['u_error_t0'] = df['u_true_t0'] - df['u_pred_t0']
    df['u_error_t1'] = df['u_true_t1'] - df['u_pred_t1']
    df['z_error_t0'] = df['z_true'] - df['z_pred_t0']
    df['z_error_t1'] = df['z_true'] - df['z_pred_t1']

    # 创建误差统计表
    error_stats = pd.DataFrame({
        'Variable': ['Water Depth (t0)', 'Water Depth (t1)',
                     'Velocity (t0)', 'Velocity (t1)',
                     'Terrain (t0)', 'Terrain (t1)'],
        'MAE': [
            df['h_error_t0'].abs().mean(),
            df['h_error_t1'].abs().mean(),
            df['u_error_t0'].abs().mean(),
            df['u_error_t1'].abs().mean(),
            df['z_error_t0'].abs().mean(),
            df['z_error_t1'].abs().mean()
        ],
        'RMSE': [
            np.sqrt((df['h_error_t0'] ** 2).mean()),
            np.sqrt((df['h_error_t1'] ** 2).mean()),
            np.sqrt((df['u_error_t0'] ** 2).mean()),
            np.sqrt((df['u_error_t1'] ** 2).mean()),
            np.sqrt((df['z_error_t0'] ** 2).mean()),
            np.sqrt((df['z_error_t1'] ** 2).mean())
        ],
        'Max Error': [
            df['h_error_t0'].abs().max(),
            df['h_error_t1'].abs().max(),
            df['u_error_t0'].abs().max(),
            df['u_error_t1'].abs().max(),
            df['z_error_t0'].abs().max(),
            df['z_error_t1'].abs().max()
        ]
    })

    # 保存到Excel
    with pd.ExcelWriter(output_file) as writer:
        df.to_excel(writer, sheet_name='Raw Data', index=False)
        error_stats.to_excel(writer, sheet_name='Error Statistics', index=False)

    print(f'Data saved to {output_file}')
    return df, error_stats


# 使用示例
df, stats = export_comparison_data()
print("\nError Statistics:")
print(stats)
