from transformers import ModernBertConfig





class NiuConfig(ModernBertConfig):
    model_type = "niu"
    def __init__(
        self,
        # ... other parameters
        use_token_change_task: bool = False,  # <-- 新增的开关
        token_change_loss_weight: float = 1.0, # <-- 相关的权重也放在这里
        **kwargs
    ):
        # ...
        self.use_token_change_task = use_token_change_task
        self.token_change_loss_weight = token_change_loss_weight
        self.mask_token_id = 50284
        super().__init__(**kwargs)