import math
import random
import os
import torch  # 导入 torch

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from torch.multiprocessing import Value 

class LazyScheduledMLMProbProvider:
    """
    支持延迟初始化的MLM概率提供者。
    total_steps 在训练开始时由Callback设置。
    """
    def __init__(
        self, 
        shared_step: Value,  # 接收一个共享的Value对象
        start_prob: float = 0.25, 
        end_prob: float = 0.15,
        schedule_type: str = 'linear',
    ):
        self.start_prob = start_prob
        self.end_prob = end_prob
        self.schedule_type = schedule_type
        
        self.total_steps = -1 
        self._shared_step = shared_step  # 保存共享对象
        self._initialized = False

    def initialize(self, total_steps: int):
        """由Callback调用，用于完成初始化"""
        if not self._initialized:
            self.total_steps = total_steps
            self._initialized = True
            
            # 在初始化时打印 rank 信息
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                print(f"[Rank {rank}] MLM Prob Provider Initialized with total_steps = {self.total_steps}")
            else:
                print(f"[Single Process] MLM Prob Provider Initialized with total_steps = {self.total_steps}")

    def get_prob(self) -> float:
        if not self._initialized or self.total_steps <= 0:

            print(f"Warning: MLM Provider not initialized. Returning start_prob={self.start_prob}")
            return self.start_prob

        # 从共享内存中读取当前的step
        current_step = self._shared_step.value
        
        progress = min(1.0, current_step / self.total_steps)
        
        if self.schedule_type == 'linear':
            prob = self.start_prob + (self.end_prob - self.start_prob) * progress
        elif self.schedule_type == 'cosine':
            cosine_progress = 0.5 * (1 + math.cos(math.pi * progress))
            prob = self.end_prob + (self.start_prob - self.end_prob) * cosine_progress
        elif self.schedule_type == 'random':
            prob = random.uniform(min(self.start_prob, self.end_prob), max(self.start_prob, self.end_prob))
            
        else:
            assert False, f"VALID schedule_type missing, get {self.schedule_type}"
 
        prob =  (1 - 1e-3) * prob + 1e-3
        return prob

    def __call__(self) -> float:
        return self.get_prob()




from torch.multiprocessing import Value # 导入Value

class LazyMLMProbSchedulerCallback(TrainerCallback):
    """
    一个回调，用于初始化Provider并更新共享的step。
    """
    def __init__(self, prob_provider: LazyScheduledMLMProbProvider, shared_step: Value):
        self.prob_provider = prob_provider
        self.shared_step = shared_step # 保存共享对象

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.prob_provider.initialize(state.max_steps)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

        self.shared_step.value = state.global_step + 1