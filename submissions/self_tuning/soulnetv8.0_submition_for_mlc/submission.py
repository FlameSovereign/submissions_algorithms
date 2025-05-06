
import torch

class SoulnetFOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super(SoulnetFOptimizer, self).__init__(params, defaults)

        self.task_embed = torch.nn.ParameterDict({
            'default':    torch.nn.Parameter(torch.tensor(1.0)),
            'resnet':     torch.nn.Parameter(torch.tensor(1.0)),
            'deepspeech': torch.nn.Parameter(torch.tensor(1.0)),
            'wmt':        torch.nn.Parameter(torch.tensor(1.0))
        })
        self.current_task = 'default'

    def set_task(self, task_name):
        self.current_task = task_name if task_name in self.task_embed else 'default'

    def step(self, closure=None):
        """Performs a single optimization step.

        closure: A 0-arg function that recomputes the loss and
                 performs loss.backward().
        """
        loss = None
        # —— 1. 先在允许计算梯度的上下文里调用 closure(),
        #     closure() 应该完成 forward + backward()
        if closure is not None:
            loss = closure()

        # —— 2. 然后再在 no_grad 环境下更新参数
        with torch.no_grad():
            for group in self.param_groups:
                lr = group['lr']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad  # 这里 grad 仍然有 grad_fn，但不影响我们读取 .data

                    state = self.state[p]
                    # 初始化状态
                    if 'GH_trace' not in state:
                        state['GH_trace'] = torch.zeros_like(p.data)
                    if 'loss_ma' not in state:
                        state['loss_ma'] = torch.tensor(0.0, device=p.device)
                    if 'trace_history' not in state:
                        state['trace_history'] = []

                    # 更新 GH_trace
                    state['GH_trace'].mul_(0.95).add_(0.05 * grad)

                    # 计算 loss moving average
                    if loss is not None:
                        state['loss_ma'] = 0.9 * state['loss_ma'] + 0.1 * loss.item()
                    stability_factor = torch.exp(-state['loss_ma'])

                    # 频谱过滤示例
                    state['trace_history'].append(grad.clone())
                    if len(state['trace_history']) > 5:
                        state['trace_history'].pop(0)
                        fft_stack = torch.stack([
                            torch.fft.fft(h.flatten())
                            for h in state['trace_history']
                        ])
                        spectrum_mean = torch.mean(torch.abs(fft_stack), dim=0)
                        if torch.max(spectrum_mean) > 10:  # placeholder threshold
                            grad = grad * 0.9

                    # 任务嵌入因子
                    embed_factor = self.task_embed[self.current_task].item()

                    # 最终更新
                    update = (grad - state['GH_trace']) * embed_factor * stability_factor
                    p.data.add_(-lr * update)

        return loss
