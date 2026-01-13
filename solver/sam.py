import torch

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls, rho=0.05, **kwargs):
        """
        base_optimizer_cls: 传入类，如 torch.optim.AdamW, torch.optim.SGD
        rho: 邻域半径
        **kwargs: 传给 base_optimizer 的所有参数 (lr, weight_decay, etc.)
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        # 1. 准备 defaults
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        # 2. 初始化真正的优化器
        # 使用 self.param_groups 确保两者引用的是同一组参数内存和配置
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        
        # 3. 非常重要：让 SAM 的 param_groups 指向 base_optimizer 的
        # 这样外部修改 optimizer.param_groups (如 Scheduler) 时，base_optimizer 会同步生效
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None: continue
                # 计算扰动 e_w
                e_w = p.grad * scale.to(p)
                # 应用扰动：w = w + e_w
                p.add_(e_w)
                # 记忆扰动，用于第二步还原
                self.state[p]['e_w'] = e_w
                
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                # 还原参数：w = w - e_w
                p.sub_(self.state[p]['e_w'])
        
        # 此时 p.grad 已经是第二次 backward 算出的梯度了（在 w+e_w 处算的）
        # 执行 base_optimizer 的真正的更新步骤
        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def _grad_norm(self):
        # 跨设备和参数组计算全局梯度范数
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm