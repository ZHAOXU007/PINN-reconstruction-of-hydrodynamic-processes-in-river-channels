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

x_step = 15
x_min=0
x_max=1500
t_step = 15
t_min=0
t_max=10800

inlet_point = 0
x_point1 = 250
x_point2 = 555
x_point3 = 570
x_point4 = 930
x_point5 = 945
x_point6 = 1250
x_point7 = 750
outlet_point = x_max
# 定义上界和下界
ub = np.array([x_max])
lb = np.array([x_min])

class PINN:
    def __init__(self):
        self.k = 500
        self.gravity = 9.81
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


    def h_acu(self, x,t):
        h = 16-self.z_acu(x)+4-4*torch.sin(torch.pi*(4*t/86400+0.5))
        return h

    def u_acu(self, x,t):
        u = torch.pi*(x-x_max)/5400/self.h_acu(x,t)*torch.cos(torch.pi*(4*t/86400+0.5))
        return self.k*u

    def z_acu(self, x):
        distance = torch.abs(x - 1500 / 2)
        # 使用逐元素的条件赋值
        z = torch.where(distance <= 1500 / 8,
                        torch.tensor(8.0, device=device),
                        torch.tensor(0.0, device=device))
        return z

    def get_training_data(self):
        """生成各类训练数据"""
        # 初始条件数据 (t=0)
        init_x = np.arange(x_min, x_max + x_step, x_step).reshape(-1, 1)
        init_t = np.zeros_like(init_x)
        self.init_xt = torch.tensor(np.concatenate([init_x, init_t], axis=1), dtype=torch.float32).to(device)
        self.init_h = self.h_acu(self.init_xt[:,0:1],self.init_xt[:,1:2]).to(device)
        self.init_u = self.u_acu(self.init_xt[:, 0:1], self.init_xt[:, 1:2]).to(device)
        #self.pde_points = torch.tensor(np.arange(x_min, x_max + x_step, x_step).reshape(-1, 1), dtype=torch.float32).to(device)

        # obs point
        bound1_t = np.arange(t_min, t_max + t_step, t_step).reshape(-1, 1)
        bound1_x = x_point1 * np.ones_like(bound1_t)
        self.bound1_xt = torch.tensor(np.concatenate([bound1_x, bound1_t], axis=1), dtype=torch.float32).to(device)
        self.bound1_h = self.h_acu(self.bound1_xt[:, 0:1], self.bound1_xt[:, 1:2]).to(device)
        self.bound1_u = self.u_acu(self.bound1_xt[:, 0:1], self.bound1_xt[:, 1:2]).to(device)
        self.bound1_z = self.z_acu(self.bound1_xt[:, 0:1]).to(device)
        # 边界条件数据 (x=l)
        bound2_t = np.arange(t_min, t_max + t_step, t_step).reshape(-1, 1)
        bound2_x = x_point2 * np.ones_like(bound2_t)
        self.bound2_xt = torch.tensor(np.concatenate([bound2_x, bound2_t], axis=1), dtype=torch.float32).to(device)
        self.bound2_h = self.h_acu(self.bound2_xt[:, 0:1], self.bound2_xt[:, 1:2]).to(device)
        self.bound2_u = self.u_acu(self.bound2_xt[:, 0:1], self.bound2_xt[:, 1:2]).to(device)
        self.bound2_z = self.z_acu(self.bound2_xt[:, 0:1]).to(device)

        bound3_t = np.arange(t_min, t_max + t_step, t_step).reshape(-1, 1)
        bound3_x = x_point3 * np.ones_like(bound3_t)
        self.bound3_xt = torch.tensor(np.concatenate([bound3_x, bound3_t], axis=1), dtype=torch.float32).to(device)
        self.bound3_h = self.h_acu(self.bound3_xt[:, 0:1], self.bound3_xt[:, 1:2]).to(device)
        self.bound3_u = self.u_acu(self.bound3_xt[:, 0:1], self.bound3_xt[:, 1:2]).to(device)
        self.bound3_z = self.z_acu(self.bound3_xt[:, 0:1]).to(device)

        bound4_t = np.arange(t_min, t_max + t_step, t_step).reshape(-1, 1)
        bound4_x = x_point4 * np.ones_like(bound4_t)
        self.bound4_xt = torch.tensor(np.concatenate([bound4_x, bound4_t], axis=1), dtype=torch.float32).to(device)
        self.bound4_h = self.h_acu(self.bound4_xt[:, 0:1], self.bound4_xt[:, 1:2]).to(device)
        self.bound4_u = self.u_acu(self.bound4_xt[:, 0:1], self.bound4_xt[:, 1:2]).to(device)
        self.bound4_z = self.z_acu(self.bound4_xt[:, 0:1]).to(device)
        # 边界条件数据 (x=l)
        bound5_t = np.arange(t_min, t_max + t_step, t_step).reshape(-1, 1)
        bound5_x = x_point5 * np.ones_like(bound5_t)
        self.bound5_xt = torch.tensor(np.concatenate([bound5_x, bound5_t], axis=1), dtype=torch.float32).to(device)
        self.bound5_h = self.h_acu(self.bound5_xt[:, 0:1], self.bound5_xt[:, 1:2]).to(device)
        self.bound5_u = self.u_acu(self.bound5_xt[:, 0:1], self.bound5_xt[:, 1:2]).to(device)
        self.bound5_z = self.z_acu(self.bound5_xt[:, 0:1]).to(device)

        bound6_t = np.arange(t_min, t_max + t_step, t_step).reshape(-1, 1)
        bound6_x = x_point6 * np.ones_like(bound6_t)
        self.bound6_xt = torch.tensor(np.concatenate([bound6_x, bound6_t], axis=1), dtype=torch.float32).to(device)
        self.bound6_h = self.h_acu(self.bound6_xt[:, 0:1], self.bound6_xt[:, 1:2]).to(device)
        self.bound6_u = self.u_acu(self.bound6_xt[:, 0:1], self.bound6_xt[:, 1:2]).to(device)
        self.bound6_z = self.z_acu(self.bound6_xt[:, 0:1]).to(device)

        bound7_t = np.arange(t_min, t_max + t_step, t_step).reshape(-1, 1)
        bound7_x = x_point7 * np.ones_like(bound7_t)
        self.bound7_xt = torch.tensor(np.concatenate([bound7_x, bound7_t], axis=1), dtype=torch.float32).to(device)
        self.bound7_h = self.h_acu(self.bound7_xt[:, 0:1], self.bound7_xt[:, 1:2]).to(device)
        self.bound7_u = self.u_acu(self.bound7_xt[:, 0:1], self.bound7_xt[:, 1:2]).to(device)
        self.bound7_z = self.z_acu(self.bound7_xt[:, 0:1]).to(device)
        #
        # bound8_t = np.arange(t_min, t_max + t_step, t_step).reshape(-1, 1)
        # bound8_x = x_point8 * np.ones_like(bound8_t)
        # self.bound8_xt = torch.tensor(np.concatenate([bound8_x, bound8_t], axis=1), dtype=torch.float32).to(device)
        # self.bound8_h = self.h_acu(self.bound8_xt[:, 0:1], self.bound8_xt[:, 1:2]).to(device)
        # self.bound8_u = self.u_acu(self.bound8_xt[:, 0:1], self.bound8_xt[:, 1:2]).to(device)
        # self.bound8_z = self.z_acu(self.bound8_xt[:, 0:1]).to(device)
        #
        # bound9_t = np.arange(t_min, t_max + t_step, t_step).reshape(-1, 1)
        # bound9_x = x_point9 * np.ones_like(bound9_t)
        # self.bound9_xt = torch.tensor(np.concatenate([bound9_x, bound9_t], axis=1), dtype=torch.float32).to(device)
        # self.bound9_h = self.h_acu(self.bound9_xt[:, 0:1], self.bound9_xt[:, 1:2]).to(device)
        # self.bound9_u = self.u_acu(self.bound9_xt[:, 0:1], self.bound9_xt[:, 1:2]).to(device)
        # self.bound9_z = self.z_acu(self.bound9_xt[:, 0:1]).to(device)


        # # 定义边界点列表 (根据实际情况调整x_points的顺序)
        # x_points = [x_point1, x_point2, x_point3, x_point4, x_point5, x_point6]
        #
        # for i in range(6):
        #     # 生成时间序列
        #     bound_t = np.arange(t_min, t_max + t_step, t_step).reshape(-1, 1)
        #     # 生成空间坐标（固定x值）
        #     bound_x = x_points[i] * np.ones_like(bound_t)
        #     # 组合成坐标矩阵
        #     bound_xt = torch.tensor(np.concatenate([bound_x, bound_t], axis=1),
        #                             dtype=torch.float32).to(device)
        #
        #     # 计算各项参数
        #     h = self.h_acu(bound_xt[:, 0:1], bound_xt[:, 1:2]).to(device)
        #     u = self.u_acu(bound_xt[:, 0:1], bound_xt[:, 1:2]).to(device)
        #     z = self.z_acu(bound_xt[:, 0:1]).to(device)
        #
        #     # 动态设置属性
        #     setattr(self, f"bound{i + 1}_xt", bound_xt)
        #     setattr(self, f"bound{i + 1}_h", h)
        #     setattr(self, f"bound{i + 1}_u", u)
        #     setattr(self, f"bound{i + 1}_z", z)


        # PDE配点
        # x1 = np.arange(x_min, 500 + 50, 50)
        # x2 = np.arange(500, 1000 + 5, 5)
        # x3 = np.arange(1000, 1500 + 50, 50)
        # x = np.concatenate([x1, x2,x3], axis=0)
        x = np.arange(x_min, x_max + x_step, x_step)
        t = np.arange(t_min, t_max + t_step, t_step)
        X, T = np.meshgrid(x, t)
        self.pde_points = torch.tensor(np.stack([X.flatten(), T.flatten()], axis=1), dtype=torch.float32).to(device)

        # boundary
        inlet_t = np.arange(t_min, t_max + t_step, t_step).reshape(-1, 1)
        inlet_x = inlet_point * np.ones_like(inlet_t)
        self.inlet_xt = torch.tensor(np.concatenate([inlet_x, inlet_t], axis=1), dtype=torch.float32).to(device)
        self.inlet_h = self.h_acu(self.inlet_xt[:, 0:1], self.inlet_xt[:, 1:2]).to(device)
        self.inlet_u = self.u_acu(self.inlet_xt[:, 0:1], self.inlet_xt[:, 1:2]).to(device)
        self.inlet_z = self.z_acu(self.inlet_xt[:, 0:1]).to(device)
        # 边界条件数据 (x=l)
        outlet_t = np.arange(t_min, t_max + t_step, t_step).reshape(-1, 1)
        outlet_x = outlet_point * np.ones_like(outlet_t)
        self.outlet_xt = torch.tensor(np.concatenate([outlet_x, outlet_t], axis=1), dtype=torch.float32).to(device)
        self.outlet_h = self.h_acu(self.outlet_xt[:, 0:1], self.outlet_xt[:, 1:2]).to(device)
        self.outlet_u = self.u_acu(self.outlet_xt[:, 0:1], self.outlet_xt[:, 1:2]).to(device)
        self.outlet_z = self.z_acu(self.outlet_xt[:, 0:1]).to(device)


        garph_x = np.arange(x_min, x_max + x_step, x_step).reshape(-1, 1)
        garph_t = 3600*np.ones_like(garph_x)
        self.garph_xt = torch.tensor(np.concatenate([garph_x, garph_t], axis=1), dtype=torch.float32).to(device)
        self.garph_h = self.h_acu(self.garph_xt[:, 0:1], self.garph_xt[:, 1:2]).to(device)
        self.garph_u = self.u_acu(self.garph_xt[:, 0:1], self.garph_xt[:, 1:2]).to(device)/self.k
        self.garph_z = self.z_acu(self.garph_xt[:, 0:1]).to(device)

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
        dh_dx =self.gradients(h_pred, self.pde_points)[0][:,0:1]

        dh2_dx = self.gradients(h2, self.pde_points)[0][:,0:1]
        dhu_dx = self.gradients(hu, self.pde_points)[0][:,0:1]
        dz_dx = self.gradients(zb, self.pde_points)[0][:,0:1]
        dz_dt = self.gradients(zb, self.pde_points)[0][:, 1:2]

        du_dx = self.gradients(u_pred, self.pde_points)[0][:, 0:1]

        lambda_val = 0.1  # 需调试优化
        eta_v = lambda_val * (torch.abs(du_dx) - du_dx) + 1.0

        e1 = dh_dt + 1/self.k*dhu_dx

        e2 = 1/self.k*dhu_dt + 1/self.k*1/self.k*dh2_dx +self.gravity* h_pred*dh_dx+ self.gravity * h_pred * dz_dx
        # return torch.mean(torch.square(e1)) + torch.mean(torch.square(e2))
        # 加权后的残差损失（MSE_F）
        e1_weighted = e1 / eta_v
        e2_weighted = e2 / eta_v
        mse_f = torch.mean(torch.square(e1_weighted)) + torch.mean(torch.square(e2_weighted))

        return  mse_f


    def loss_obs(self):
        """观测数据损失"""
        h_pred1 = self.net(self.bound1_xt)[:, 0:1]
        h_pred2 = self.net(self.bound2_xt)[:, 0:1]
        h_pred3 = self.net(self.bound3_xt)[:, 0:1]
        h_pred4 = self.net(self.bound4_xt)[:, 0:1]
        h_pred5 = self.net(self.bound5_xt)[:, 0:1]
        h_pred6 = self.net(self.bound6_xt)[:, 0:1]
        h_pred7 = self.net(self.bound7_xt)[:, 0:1]
        # h_pred8 = self.net(self.bound8_xt)[:, 0:1]
        # h_pred9 = self.net(self.bound9_xt)[:, 0:1]

        u_pred1 = self.net(self.bound1_xt)[:, 1:2]
        u_pred2 = self.net(self.bound2_xt)[:, 1:2]
        u_pred3 = self.net(self.bound3_xt)[:, 1:2]
        u_pred4 = self.net(self.bound4_xt)[:, 1:2]
        u_pred5 = self.net(self.bound5_xt)[:, 1:2]
        u_pred6 = self.net(self.bound6_xt)[:, 1:2]
        u_pred7 = self.net(self.bound7_xt)[:, 1:2]
        # u_pred8 = self.net(self.bound8_xt)[:, 1:2]
        # u_pred9 = self.net(self.bound9_xt)[:, 1:2]

        return torch.mean(torch.square(h_pred1 - self.bound1_h)) + torch.mean(torch.square(u_pred1 - self.bound1_u)) \
               + torch.mean(torch.square(h_pred2 - self.bound2_h)) + torch.mean(torch.square(u_pred2 - self.bound2_u)) \
               + torch.mean(torch.square(h_pred3 - self.bound3_h)) + torch.mean(torch.square(u_pred3 - self.bound3_u)) \
               + torch.mean(torch.square(h_pred4 - self.bound4_h)) + torch.mean(torch.square(u_pred4 - self.bound4_u)) \
               + torch.mean(torch.square(h_pred5 - self.bound5_h)) + torch.mean(torch.square(u_pred5 - self.bound5_u)) \
               + torch.mean(torch.square(h_pred6 - self.bound6_h)) + torch.mean(torch.square(u_pred6 - self.bound6_u))\
                + torch.mean(torch.square(h_pred7 - self.bound7_h)) + torch.mean(torch.square(u_pred7 - self.bound7_u))


    def loss_B1(self):
        """观测数据损失"""
        h_pred = self.net(self.inlet_xt)[:, 0:1]
        u_pred = self.net(self.inlet_xt)[:, 1:2]
        z_pred = self.net(self.inlet_xt)[:, 2:3]
        return torch.mean(torch.square(h_pred - self.inlet_h)) + torch.mean(torch.square(u_pred - self.inlet_u))\
               + torch.mean(torch.square(z_pred-self.inlet_z))

    def loss_B2(self):
        """观测数据损失"""
        h_pred = self.net(self.outlet_xt)[:, 0:1]
        u_pred = self.net(self.outlet_xt)[:, 1:2]
        z_pred = self.net(self.outlet_xt)[:, 2:3]
        return torch.mean(torch.square(h_pred - self.outlet_h)) + torch.mean(torch.square(u_pred - self.outlet_u))\
               + torch.mean(torch.square(z_pred - self.outlet_z))


    def loss_init(self):
        """观测数据损失"""
        h_pred = self.net(self.init_xt)[:, 0:1]
        u_pred = self.net(self.init_xt)[:, 1:2]

        return torch.mean(torch.square(h_pred - self.init_h)) + torch.mean(torch.square(u_pred - self.init_u))



    def closure(self):
        self.adam.zero_grad()
        self.lbfgs.zero_grad()


        loss_pde = self.loss_PDE()
        loss_obs = self.loss_obs()

        loss_boundary = self.loss_B1() + self.loss_B2()
        loss = 1000000*loss_pde + loss_obs  + 10*loss_boundary

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
            'MSE_Data': loss_obs.item(),
            'MSE_PDE': loss_pde.item(),


        })
        return loss

    def plot_current_state(self):
        """实时观测 h, u, z 变化的函数"""
        with torch.no_grad():
            h = self.net(self.garph_xt)[:, 0:1].detach().cpu().numpy()  # 使用 .detach()
            u = self.net(self.garph_xt)[:, 1:2].detach().cpu().numpy()/self.k  # 使用 .detach()
            z = self.net(self.garph_xt)[:, 2:3].detach().cpu().numpy()  # 使用 .detach()

        plt.figure(figsize=(12, 8))

        # 水深图
        plt.subplot(3, 1, 1)
        plt.plot(self.garph_xt[:, 0:1].detach().cpu().numpy(), h[:, 0:1], label='Predicted Water Depth (h)', color='red',linestyle='--')
        plt.plot(self.garph_xt[:, 0:1].detach().cpu().numpy(), self.garph_h[:, 0:1].detach().cpu().numpy(), label='Real Water Depth (h)', color='blue',linestyle='-')
        plt.title('Water Depth (h)')
        plt.ylabel('h Value')
        plt.grid()
        plt.legend()

        # 流速图
        plt.subplot(3, 1, 2)
        plt.plot(self.garph_xt[:, 0:1].detach().cpu().numpy(), u[:, 0:1], label='Predicted Velocity (u)', color='red',linestyle='--')
        plt.plot(self.garph_xt[:, 0:1].detach().cpu().numpy(), self.garph_u[:, 0:1].detach().cpu().numpy(),
                 label='Real Velocity (u)', color='blue',linestyle='-')
        plt.title('Velocity (u)')
        plt.ylabel('u Value')
        plt.grid()
        plt.legend()

        # 地形图
        plt.subplot(3, 1, 3)
        plt.plot(self.garph_xt[:, 0:1].detach().cpu().numpy(), z[:, 0:1], label='Predicted Terrain (z)', color='red',linestyle='--')
        plt.plot(self.garph_xt[:, 0:1].detach().cpu().numpy(), self.garph_z[:, 0:1].detach().cpu().numpy(),
                   label='Real Terrain (z)', color='blue',linestyle='-')
        plt.title('Terrain (z)')
        plt.ylabel('z Value')
        plt.xlabel('x Value')
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.show()

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
    torch.save(pinn.net.state_dict(), f'discontinuous_ss72.param')
    # 将记录的数据保存到 Excel
    df = pd.DataFrame(pinn.recorded_data)
    df.to_excel("training_logas72.xlsx", index=False)