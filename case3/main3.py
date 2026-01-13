import sys

sys.path.append(".")
import numpy as np
import torch
from torch.autograd import grad
from net import DNN
from scipy.io import loadmat
import pandas as pd
import os
import matplotlib.pyplot as plt
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

x_step = 100
x_min=0
x_max=10000
t_step = 60
t_min=0
t_max=86400

inlet_point1 = 0
x_point1 = 2500
x_point2 = 5000
x_point3 = 7500
outlet_point = 10000



# 定义上界和下界
ub = np.array([x_max])
lb = np.array([x_min])

class PINN:
    def __init__(self):
        self.k = 1
        self.gravity = 9.81
        self.A = 10000.0  # 最大冲刷深度 [m]
        self.c = 10000.0 / 86400  # 冲刷波速度 ≈ 0.463 [m/s]
        self.lam = 1e-4 / 86400  # 衰减系数 ≈ 1.157e-10 [1/s]
        self.gamma = 2e-5  # 距离衰减系数 [1/m]
        self.K = 1.0 * 86400  # 阶跃陡度系数 ≈ 1.728e6 [1/s]
        self.S0 = 0.001  # 初始坡度
        self.L = 10000  # 河道长度 [m]
        self.recorded_data = []  # 用于存储记录的数据
        self.net = DNN(dim_in=2, dim_out=3, n_layer=6, n_node=20, ub=ub, lb=lb).to(
            device
        )

        self.adam = torch.optim.Adam(self.net.parameters())
        self.lbfgs = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )
        self.iter = 0
        self.get_training_data()


    def get_training_data(self):
        """生成各类训练数据"""
        # 初始条件数据 (t=0)
        # 观测数据
        data1 = pd.read_csv("./0km.csv")
        self.inlet_xt = torch.tensor(np.column_stack([data1["xi"], data1["ti"]]), dtype=torch.float32).to(device)
        self.inlet_hi = torch.tensor(data1["hi"].values.reshape(-1, 1), dtype=torch.float32).to(device)
        self.inlet_ui = torch.tensor(data1["ui"].values.reshape(-1, 1), dtype=torch.float32).to(device)
        self.inlet_zi = torch.tensor(data1["zi"].values.reshape(-1, 1), dtype=torch.float32).to(device)
        data2 = pd.read_csv("./2.5km.csv")
        self.Data2_xt = torch.tensor(np.column_stack([data2["xi"], data2["ti"]]), dtype=torch.float32).to(device)
        self.Data2_hi = torch.tensor(data2["hi"].values.reshape(-1, 1), dtype=torch.float32).to(device)
        self.Data2_ui = torch.tensor(data2["ui"].values.reshape(-1, 1), dtype=torch.float32).to(device)
        data3 = pd.read_csv("./5.0km.csv")
        self.Data3_xt = torch.tensor(np.column_stack([data3["xi"], data3["ti"]]), dtype=torch.float32).to(device)
        self.Data3_hi = torch.tensor(data3["hi"].values.reshape(-1, 1), dtype=torch.float32).to(device)
        self.Data3_ui = torch.tensor(data3["ui"].values.reshape(-1, 1), dtype=torch.float32).to(device)
        data4 = pd.read_csv("./7.5km.csv")
        self.Data4_xt = torch.tensor(np.column_stack([data4["xi"], data4["ti"]]), dtype=torch.float32).to(device)
        self.Data4_hi = torch.tensor(data4["hi"].values.reshape(-1, 1), dtype=torch.float32).to(device)
        self.Data4_ui = torch.tensor(data4["ui"].values.reshape(-1, 1), dtype=torch.float32).to(device)
        data5 = pd.read_csv("./10.0km.csv")
        self.outlet_xt = torch.tensor(np.column_stack([data5["xi"], data5["ti"]]), dtype=torch.float32).to(device)
        self.outlet_hi = torch.tensor(data5["hi"].values.reshape(-1, 1), dtype=torch.float32).to(device)
        self.outlet_ui = torch.tensor(data5["ui"].values.reshape(-1, 1), dtype=torch.float32).to(device)
        self.outlet_zi = torch.tensor(data5["zi"].values.reshape(-1, 1), dtype=torch.float32).to(device)
        # PDE配点
        x = np.arange(x_min, x_max + x_step, x_step)
        t = np.arange(t_min, t_max + t_step, t_step)
        X, T = np.meshgrid(x, t)
        self.pde_points = torch.tensor(np.stack([X.flatten(), T.flatten()], axis=1), dtype=torch.float32).to(device)





    def gradients(self, outputs, inputs):
        return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)


    def loss_PDE(self):
        """PDE约束损失"""
        self.pde_points.requires_grad = True
        h_pred=self.net(self.pde_points)[:,0:1]
        u_pred = self.net(self.pde_points)[:, 1:2]
        zb = self.net(self.pde_points)[:, 2:3]

        h2 = h_pred*u_pred*u_pred
        hu = u_pred*h_pred
        # 计算各阶导数
        dh_dt = self.gradients(h_pred, self.pde_points)[0][:,1:2]
        dhu_dt = self.gradients(hu, self.pde_points)[0][:, 1:2]
        dh_dx=self.gradients(h_pred, self.pde_points)[0][:,0:1]
        dh2_dx = self.gradients(h2, self.pde_points)[0][:,0:1]
        dhu_dx = self.gradients(hu, self.pde_points)[0][:,0:1]
        dz_dx = self.gradients(zb, self.pde_points)[0][:,0:1]

        e1 = dh_dt + 1/self.k*dhu_dx

        e2 = 1/self.k*dhu_dt + 1/self.k*1/self.k*dh2_dx +self.gravity* h_pred*dh_dx+ self.gravity * h_pred * dz_dx

        return torch.mean(torch.square(e1)) + torch.mean(torch.square(e2))

    def loss_obs(self):
        """观测数据损失"""
        h_pred1 = self.net(self.inlet_xt)[:, 0:1]
        h_pred2 = self.net(self.Data2_xt)[:, 0:1]
        h_pred3 = self.net(self.Data3_xt)[:, 0:1]
        h_pred4 = self.net(self.Data4_xt)[:, 0:1]
        h_pred5 = self.net(self.outlet_xt)[:, 0:1]

        u_pred1 = self.net(self.inlet_xt)[:, 1:2]
        u_pred2 = self.net(self.Data2_xt)[:, 1:2]
        u_pred3 = self.net(self.Data3_xt)[:, 1:2]
        u_pred4 = self.net(self.Data4_xt)[:, 1:2]
        u_pred5 = self.net(self.outlet_xt)[:, 1:2]



        return torch.mean(torch.square(h_pred1 - self.inlet_hi)) + torch.mean(torch.square(u_pred1 - self.inlet_ui)) \
               + torch.mean(torch.square(h_pred2 - self.Data2_hi)) + torch.mean(torch.square(u_pred2 - self.Data2_ui)) \
               + torch.mean(torch.square(h_pred3 - self.Data3_hi)) + torch.mean(torch.square(u_pred3 - self.Data3_ui)) \
               + torch.mean(torch.square(h_pred4 - self.Data4_hi)) + torch.mean(torch.square(u_pred4 - self.Data4_ui)) \
               + torch.mean(torch.square(h_pred5 - self.outlet_hi)) + torch.mean(torch.square(u_pred5 - self.outlet_ui))


    def loss_obs_z(self):
        z_pred1 = self.net(self.inlet_xt)[:, 2:3]
        z_pred2 = self.net(self.outlet_xt)[:, 2:3]

        return torch.mean(torch.square(z_pred1 - self.inlet_zi)) + torch.mean(torch.square(z_pred2 - self.outlet_zi))

    def closure(self):
        self.adam.zero_grad()
        self.lbfgs.zero_grad()

        loss_pde = self.loss_PDE()
        loss_obs = self.loss_obs()
        loss_obs_z =self.loss_obs_z()
        loss = 100*loss_pde + loss_obs + loss_obs_z

        loss.backward()

        self.iter += 1
        print(
            f"\r{self.iter} loss : {loss.item():.3e},pde : {loss_pde.item():.5f},obs:{loss_obs.item():.5f}",
            end="",
        )
        if self.iter % 500 == 0:
            print("")

        # 记录 mse_data, mse_pde 和参数
        self.recorded_data.append({
            'Iteration': self.iter,
            'MSE_PDE': loss_pde.item(),
            'MSE_OBS': loss_obs.item(),



        })
        return loss


if __name__ == "__main__":
    pinn = PINN()
    for i in range(10000):
        pinn.closure()
        pinn.adam.step()
        # 每隔一定的迭代次数绘制一次状态
        # if i % 500 == 0:  # 每500次迭代绘制一次
        #     pinn.plot_current_state()
    pinn.lbfgs.step(pinn.closure)
    #pinn.plot_current_state()
    #torch.save(pinn.net.state_dict(), ".weight.pt")
    torch.save(pinn.net.state_dict(), f'model_a100.param')
    # 将记录的数据保存到 Excel
    df = pd.DataFrame(pinn.recorded_data)
    df.to_excel("training_log_a100.xlsx", index=False)