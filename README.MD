
# 在开始前需要了解的

## 关于词表
由于预训练，模型和词表都是从头开始，我们这里没有提供从给定数据集学习词表的方法。目前项目里直接使用了modernbert的词表，主要适用于英文。如果你需要多语言词表，可以考虑使用qwen2以上系列的词表。

## 本地资源文件配置
这里主要是配置数据集，参考dataset_config.json，注意这里的数据集名字是项目用于加载训练所需要的，所以不要起重复了。

## 扩展数据集
参考 utils/load_dataset.py
可以直接在huggingface上找到预训练数据集，然后告诉GPT类助手你需要加入的数据集ID。需要注意，这个函数的名字需要和上面dataset_config.json的一致，如果你不需要从本地导入而是直接从huggingface导入，那就无所谓了。
这里只有一点需要注意，文本域的column name需要是'text'，如果不是，可以参考我在ultra_fineweb的写法(其文本域是content)，在最后加上一个
```python
    dataset = dataset.rename_column('content','text')
```

## 打印更多想要观察的东西
huggingface的trainer不支持打印预设的kv对，我们的项目通过修改trainer原生的支持了这一点。需要额外打印的东西需要在模型的forward返回，并且在main.py里的keys_you_want_to_log加上。
```python
trainer = MultipleLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, # <-- 变量名更新
        eval_dataset=eval_dataset,   # <-- 新增，传递验证集
        data_collator=collator,
        # callbacks=None if args.mode == 'llama' or args.mode == 'llada' else [lazy_prob_scheduler_callback],
        keys_you_want_to_log = ['lm_loss','current_mlm_prob','masked_lm_loss']
    )
```

## 恢复训练
本项目支持自动重训，无需手动操作。

## 扩展性
项目通过accelerate和deepspeed支持多卡/多节点多卡的ddp/tp/pp训练，具体参见accelerate文档和github示例。

## 一些快于transformers的bug fix
我们的trainer目前有两个feature早于latest的huggingface trainer，都是旨在保证中断训练后恢复的一致性。解决的问题具体参考
https://github.com/huggingface/transformers/pull/40347
https://github.com/huggingface/transformers/issues/39755 （随机状态加载时间这个问题还需要进一步fix）


## 课程学习
适配加噪率课程学习，可以不以随机从[0,1]采样加噪率，而是根据训练步数决定加噪率。组件已经全部完成了，只是不知道这个做法有没有意义，所以不是项目的默认行为。可以通过callbacks打开。这个写起来有点麻烦，看看后面有人要用再说吧。

# 开始训练
bash main.sh就行，修改里面对应的参数为你所需要的即可，参数可以参考main.py里argparse的参数列表。

# 评测
logprob_eval下面也有同样的main.sh, 不过他还没有到达我的预期状态，后面会继续优化。目前logprob eval已经支持一定数量的benchmark，而且在推断速度上可以自动以数据并行形式利用多卡同时推断（这比对模型进行tp/pp高效的多）

## TODO
1. 对LLaDA的采样改为FP64的gumbel softmax，以解决MDM精度问题影响生成路径的多样性（参考nvidia的time agnostic那篇论文）
2. 加入类似dream的基于pretrained ARM的CPT过程。
3. ADD SFT

