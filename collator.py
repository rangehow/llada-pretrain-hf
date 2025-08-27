import random
import torch
from typing import Dict, List, Any, Callable, Union
from transformers import PreTrainedTokenizer



class NTPCollator:
    """
    用于NTP（Next Token Prediction）预训练任务的数据整理器
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        text_key: str = 'text',
    ):
        """
        Args:
            tokenizer: 预训练的分词器
            max_length: 最大序列长度
            text_key: 如果指定，会从examples中提取该key对应的文本并转换为input_ids
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_key = text_key
        
        # 获取特殊token的id
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id  # 添加BOS token
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        处理一个batch的数据
        
        Args:
            examples: 包含'input_ids'键或指定text_key的字典列表
            
        Returns:
            包含input_ids, attention_mask, labels的字典
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for example in examples:
            # 获取input_ids：优先使用现有的input_ids，否则从文本转换
            if 'input_ids' in example:
                input_ids = example['input_ids']
            elif self.text_key and self.text_key in example:
                # 从文本转换为input_ids
                text = example[self.text_key]
                encoded = self.tokenizer.encode(
                    text,
                    add_special_tokens=False,  # 手动添加BOS和EOS
                    truncation=True,
                    max_length=self.max_length - 2  # 留出BOS和EOS位置
                )
                input_ids = encoded
            else:
                raise ValueError(f"Example must contain either 'input_ids' or '{self.text_key}' key")
            
            # 确保序列以BOS开头，EOS结尾
            if input_ids[0] != self.bos_token_id:
                input_ids = [self.bos_token_id] + input_ids
            if input_ids[-1] != self.eos_token_id:
                input_ids = input_ids + [self.eos_token_id]
            
            # 确保序列长度不超过max_length
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
            
            # 创建NTP标签：直接使用input_ids，让模型前向处理移位
            labels = input_ids.copy()  # 直接复制，不做移位操作
            
            # 创建attention mask
            attention_mask = [1] * len(input_ids)
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
        
        # Padding处理
        batch_input_ids = self._pad_sequences(batch_input_ids, self.pad_token_id)
        batch_attention_mask = self._pad_sequences(batch_attention_mask, 0)
        batch_labels = self._pad_sequences(batch_labels, -100)
        
        return {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long),
            'labels': torch.tensor(batch_labels, dtype=torch.long),
            'return_dict': True,
        }
    
    def _pad_sequences(self, sequences: List[List[int]], pad_value: int) -> List[List[int]]:
        """
        对序列进行padding
        """
        max_len = max(len(seq) for seq in sequences)
        padded_sequences = []
        
        for seq in sequences:
            padded_seq = seq + [pad_value] * (max_len - len(seq))
            padded_sequences.append(padded_seq)
        
        return padded_sequences
    






class LLaDACollator:
    """
    用于BERT预训练MLM任务的数据整理器
    """
    
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        text_key: str = 'text',
    ):
        
        self.tokenizer = tokenizer
        
        self.max_length = max_length
        self.text_key = text_key  # 保存text_key
        # 获取特殊token的id
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.eos_token_id = tokenizer.eos_token_id  # 添加EOS token
        self.bos_token_id = tokenizer.bos_token_id  # 添加BOS token
        
        # 可用于随机替换的token范围
        self.vocab_size = tokenizer.vocab_size
        
        
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        处理一个batch的数据
        
        Args:
            examples: 包含'input_ids'键或指定text_key的字典列表
            
        Returns:
            包含input_ids, attention_mask, labels, token_change_labels的字典
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        batch_mlm_probs = []  # 添加用于存储每个样本MLM概率的列表
        
        for example in examples:
            # 获取input_ids：优先使用现有的input_ids，否则从文本转换
            if 'input_ids' in example:
                input_ids = example['input_ids']
            elif self.text_key and self.text_key in example:
                # 从文本转换为input_ids
                text = example[self.text_key]
                encoded = self.tokenizer.encode(
                    text, 
                    add_special_tokens=False,  # 我们手动添加BOS和结尾token
                    truncation=True,
                    max_length=self.max_length - 2  # 留出BOS和结尾token位置
                )
                input_ids = encoded
            else:
                raise ValueError(f"Example must contain either 'input_ids' or '{self.text_key}' key")
            

            if not input_ids or input_ids[0] != self.bos_token_id:
                input_ids = [self.bos_token_id] + input_ids

            input_ids.append(self.eos_token_id)


            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]




            current_mlm_prob = self._get_mlm_probability()  # 为每个样本获取MLM概率
            batch_mlm_probs.append(current_mlm_prob)  # 添加到列表中
            
            # 创建MLM mask和labels
            masked_input_ids, labels = self._mask_tokens(input_ids, current_mlm_prob)
         
            # 创建attention mask
            attention_mask = [1] * len(masked_input_ids)
            
            batch_input_ids.append(masked_input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        
        # Padding处理
        batch_input_ids = self._pad_sequences(batch_input_ids, self.pad_token_id)
        batch_attention_mask = self._pad_sequences(batch_attention_mask, 0)
        batch_labels = self._pad_sequences(batch_labels, -100)

  
 
        return {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long),
            'labels': torch.tensor(batch_labels, dtype=torch.long),
            'current_mlm_prob': torch.tensor(batch_mlm_probs, dtype=torch.float),  # 使用列表转换
            'return_dict': True,
        }
    
    def _get_mlm_probability(self, eps: float = 1e-3) -> float:
        t = random.uniform(0, 1)
        # 这是一个简单的线性噪声调度
        p_mask = (1 - eps) * t + eps
        return p_mask

    def _mask_tokens(self, input_ids: List[int], current_mlm_prob ) -> tuple:
        """
        对输入序列进行MLM masking
        
        Args:
            input_ids: 输入token序列
            
        Returns:
            (masked_input_ids, labels, token_change_labels): mask后的序列、MLM标签和token变化标签
        """
        
        labels = [-100] * len(input_ids)  
  
        maskable_positions = []

        for i, token_id in enumerate(input_ids):
            if token_id not in [self.cls_token_id, self.pad_token_id, self.bos_token_id, self.sep_token_id]:  # BOS也不能被mask
                maskable_positions.append(i)
                
        
        # 强制处理结尾token（EOS或SEP）
        # if end_token_position is not None:
        #     labels[end_token_position] = input_ids[end_token_position]  # 保存原始结尾token作为标签
            
        #     input_ids[end_token_position] = self.mask_token_id
                
        #     if end_token_position in maskable_positions:
        #         maskable_positions.remove(end_token_position)
        
        # 随机选择其他需要mask的位置
        num_to_mask = max(1, int(len(maskable_positions) * current_mlm_prob))
        masked_positions = random.sample(maskable_positions, 
                                       min(num_to_mask, len(maskable_positions)))
        
        for pos in masked_positions:
            labels[pos] = input_ids[pos]  # 保存原始token作为标签
            input_ids[pos] = self.mask_token_id
               

        return input_ids, labels
    
    def _pad_sequences(self, sequences: List[List[int]], pad_value: int) -> List[List[int]]:
        """
        对序列进行padding
        """
        max_len = max(len(seq) for seq in sequences)
        padded_sequences = []
        
        for seq in sequences:
            padded_seq = seq + [pad_value] * (max_len - len(seq))
            padded_sequences.append(padded_seq)
        
        return padded_sequences
