# Chinese-poetry-CCL25-Eval-task51th
尝试
## 1. 项目背景与目标

### 1.1 项目背景
本项目针对 CCL 2025 古诗词理解与推理评测任务展开，该任务要求模型在不使用 RAG 技术的前提下，对古诗词内容进行深度理解和情感推理。古诗词因其高度凝练性和丰富的文化底蕴，对模型的语言理解能力和知识储备提出了极高要求。

### 1.2 任务分解
- **TaskA（古诗词理解）**：包括词语释义和句子翻译两个子任务
- **TaskB（情感推理）**：识别诗词表达的情感，从多个选项中选择最符合的一项

### 1.3 项目目标
1. 实现三种大型语言模型（Qwen 1.5-7 B、Qwen 2.5-7 B、Qwen 3-8 B）的 LoRA 微调
2. 在有限的训练数据（仅约 164 首诗）条件下优化模型性能
3. 达成总分≥0.8 的竞赛指标（Baseline 为 0.667）
4. 确保训练过程的稳定性和可复现性

## 2. 技术架构与选型

### 2.1 基础架构
Poetry_pretraining_package/
├── configs/                    # 配置文件目录
│   └── qwen 25_config. Yaml     # Qwen 2.5 模型配置
├── competition_data/           # 竞赛数据
├── models/                     # 模型存储目录
├── logs/                       # 日志目录
├── src/                        # 源代码
└── scripts/                    # 运行脚本
```
poetry_pretraining_package/
├── configs/                    # 配置文件目录
│   └── qwen25_config.yaml     # Qwen2.5模型配置
├── competition_data/           # 竞赛数据
├── models/                     # 模型存储目录
├── logs/                       # 日志目录
├── src/                        # 源代码
└── scripts/                    # 运行脚本
```

### 2.2 模型选型与对比

| 模型 | 参数规模 | 上下文窗口 | 预训练语料 | 特点 |
|------|--------|------------|-----------|------|
| Qwen 1.5-7 B | 70 亿 | 32 K | 通用中文+多语言 | 训练稳定，部署便捷 |
| Qwen 2.5-7 B | 70 亿 | 128 K | 更新语料库 | 长文本处理强，上下文理解深 |
| Qwen 3-8 B | 80 亿 | 128 K+ | 最新语料 | 知识丰富，推理能力强 |

### 2.3 技术栈选择

| 技术组件 | 选型 | 作用 |
|---------|------|------|
| 基础框架 | PyTorch | 深度学习基础支持 |
| 模型加载 | Transformers | 预训练模型加载与接口 |
| 参数高效微调 | PEFT (LoRA) | 减少训练参数，提高效率 |
| 训练优化 | deepspeed/8 bit 优化 | 内存占用优化 |
| 评估工具 | BLEU, BertScore | 生成质量评估 |

### 2.4 LoRA 配置设计

```python
lora_config = LoraConfig(
    r=64,                      # LoRA秩
    lora_alpha=128,            # 缩放参数
    lora_dropout=0.1,          # Dropout率
    bias="none",               # 偏置项处理方式
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj" 
    ]                          # 目标模块
)
```

## 3. 预训练实施过程

### 3.1 数据处理流程

1. **数据加载与清洗**：加载竞赛提供的诗词数据，处理特殊符号
2. **提示工程**：设计结构化提示模板，引导模型理解任务
3. **标签处理**：统一标签格式，确保模型输出一致性

```python
def build_prompt(item):
    """构建提示模板"""
    return f"""
[任务] 古诗词赏析与理解
[诗词标题] {item.get('title', '无题')}
[作者] {item.get('author', '佚名')}
[诗词内容] {item.get('content', '')}

请解释以下词语的含义：
{', '.join(item.get('qa_words', []))}

请翻译以下诗句：
{', '.join(item.get('qa_sents', []))}

请从以下选项中选出最符合这首诗词表达的情感：
{item.get('choose', '')}
"""
```

### 3.2 训练策略设计

我们采用了针对不同模型的差异化训练策略：

| 模型 | 批次大小 | 梯度累积 | 学习率 | 其他优化 |
|------|---------|----------|--------|----------|
| Qwen 1.5-7 B | 4 | 4 | 2 e-5 | 标准 LoRA |
| Qwen 2.5-7 B | 1 | 16 | 2 e-5 | 内存限制为 42 GB，手动训练循环 |
| Qwen 3-8 B | 1 | 16 | 2 e-5 | LoRA+8 bit 量化 |

### 3.3 训练监控与验证

- **训练损失监控**：实时监控损失下降情况，异常值报警
- **梯度检查**：定期检查梯度是否为 NaN 或过大/过小
- **中间检查点**：每轮结束保存检查点，支持断点续训
- **验证评估**：使用与最终评测相同指标进行中间验证

## 4. 技术难点与解决方案

### 4.1 内存优化挑战

**问题**：Qwen 2.5-7 B 在 L 20 GPU (48 GB 显存) 上训练出现 OOM 错误
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 74.00 MiB 
(GPU 0; 47.50 GiB total capacity; 46.99 GiB already allocated; 38.56 MiB free; 
47.12 GiB reserved in total by PyTorch)
```

**解决方案**：
1. 减小批次大小（从 4 降至 1）
2. 增加梯度累积步数（从 4 增至 16）
3. 设置显式内存限制（42 GB）
4. 优化设备映射格式，使用整数索引而非字符串

```python
# 修改前
max_memory[f"cuda:{i}"] = f"{args.max_memory_MB}MB"

# 修改后
max_memory[i] = f"{args.max_memory_MB}MB"
```

### 4.2 梯度计算问题

**问题**：训练出现零损失和 NaN 梯度
```
{'loss': 0.0, 'grad_norm': nan, 'learning_rate': 0.0, 'epoch': 0.06}
```

**根本原因**：
1. LoRA 层未正确激活，导致无梯度流
2. 优化器状态初始化错误
3. 学习率传递异常

**解决方案**：开发自定义训练循环，绕过 Transformers Trainer API
```python
def train_with_custom_loop(model, tokenizer, dataloader, args):
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
```

关键改进：
1. 显式设置 `model.train()` 确保训练模式
2. 手动激活 LoRA 权重与 GPU 设备绑定
3. 添加梯度检查断言确保梯度传播

### 4.3 模型兼容性问题

各模型架构存在差异，需要专门优化：

| 模型 | 架构差异 | 适配解决方案 |
|------|---------|-------------|
| Qwen 1.5-7 B | 标准架构 | 直接使用标准 LoRA |
| Qwen 2.5-7 B | 注意力机制更新 | 动态识别目标模块而非硬编码 |
| Qwen 3-8 B | 结构复杂度提升 | 模块级分析+精细 LoRA 配置 |

关键代码：
```python
# 动态识别LoRA目标模块
qwen_target_modules = []
for name, _ in model.named_modules():
    if any(substr in name for substr in ["q_proj", "k_proj", "v_proj", "o_proj", 
                                        "gate_proj", "up_proj", "down_proj"]):
        if name not in qwen_target_modules:
            qwen_target_modules.append(name)
```

## 5. 实验结果与对比分析

### 5.1 实验指标概览

| 模型 | TaskA 得分 | TaskB 得分 | 总分 | 相对提升 |
|------|----------|----------|------|----------|
| Baseline (Qwen 2.5-Zero) | 0.564 | 0.771 | 0.667 | - |
| Qwen 1.5-7 B (LoRA) | 0.576 | 0.780 | 0.678 | +1.6% |
| Qwen 2.5-7 B (LoRA) | 0.581 | 0.790 | 0.686 | +2.8% |
| Qwen 3-8 B (LoRA) | 0.599 | 0.820 | 0.710 | +6.4% |

### 5.2 任务细分指标

| 模型 | 词语 BLEU | 词语 BertScore | 句子 BLEU | 句子 BertScore | 情感准确率 |
|------|----------|--------------|----------|--------------|------------|
| Baseline | 0.230 | 0.873 | 0.241 | 0.911 | 0.771 |
| Qwen 1.5-7 B | 0.250 | 0.880 | 0.260 | 0.915 | 0.780 |
| Qwen 2.5-7 B | 0.253 | 0.885 | 0.265 | 0.920 | 0.790 |
| Qwen 3-8 B | 0.275 | 0.900 | 0.285 | 0.935 | 0.820 |

### 5.3 定性分析

以下是对三个模型在几个具体案例上的输出质量对比：

#### 词语释义示例：对"霜"的解释
- **Qwen 1.5-7 B**：基础解释，未涉及文学象征
- **Qwen 2.5-7 B**：增加了象征意义，更全面
- **Qwen 3-8 B**：多层次解读，文学性强

#### 情感推理能力：王维《鹿柴》
- **Qwen 1.5-7 B**："宁静"（基本正确但笼统）
- **Qwen 2.5-7 B**："闲适"（更符合诗境）
- **Qwen 3-8 B**："超然物外的闲适与宁静"（捕捉禅意与超脱）

### 5.4 训练效率分析

| 模型 | GPU 内存需求 | 每轮训练时间 | 优化器步数 | 收敛轮次 |
|------|------------|------------|----------|----------|
| Qwen 1.5-7 B | 约 24 GB | 约 40 分钟 | 936 | 2 轮 |
| Qwen 2.5-7 B | 约 46 GB | 约 65 分钟 | 936 | 2 轮 |
| Qwen 3-8 B | 约 47 GB | 约 80 分钟 | 936 | 2 轮 |

## 6. 经验总结与反思

### 6.1 技术成功经验

1. **参数高效微调**：LoRA 技术有效减少训练参数，使特大模型微调成为可能
2. **内存优化策略**：通过梯度累积和批次调整解决 OOM 问题
3. **手动训练循环**：绕过 API 限制，实现更精细的训练控制
4. **动态架构适配**：针对不同模型架构动态识别目标模块

### 6.2 项目不足与反思

1. **评估指标优化不足**：未充分针对古汉语特点优化 BLEU 和 BertScore
2. **提示工程单一**：未针对不同诗体设计差异化提示模板
3. **数据增强欠缺**：未实施有效的数据增强策略来应对小数据集挑战
4. **耗时调试**：训练脚本设计不够健壮，导致多次失败尝试

### 6.3 知识转化与复用

| 技术经验 | 应用场景 | 通用价值 |
|---------|----------|----------|
| 手动训练循环 | 小数据集微调 | 提供更精细的训练控制 |
| 内存优化策略 | 大模型训练 | 允许在有限资源下训练更大模型 |
| 动态模块识别 | 模型架构变更 | 提高代码对模型更新的适应性 |
| 梯度检查机制 | 异常检测 | 提前发现训练中的数值问题 |

## 7. 未来优化方向

### 7.1 短期优化方向

1. **评估指标优化**：
   ```python
   def weighted_bertscore(candidates, references):
       """加权BERTScore，降低古汉语虚词权重"""
       function_words = ["之", "乎", "者", "也", "兮", "其", "斯", "焉"]
       # 降低虚词对得分的影响
       weighted_score = original_score * (1 + 0.1 * (1 - function_word_ratio))
       return weighted_score
   ```

2. **数据增强策略**：
   - 平仄替换与扰动
   - 词序调整保持语义
   - 古文风格迁移技术

3. **结构化提示改进**：
   ```python
   def enhanced_prompt(item):
       """增强版提示模板"""
       # 检测诗歌格式
       poetry_form = ""
       if len(content.split('，')[0]) == 5:
           poetry_form = "[五言]"
       elif len(content.split('，')[0]) == 7:
           poetry_form = "[七言]"
       
       return f"""
       [任务] 古诗词赏析与理解
       [诗词标题] {item.get('title', '无题')}
       [诗体形式] {poetry_form}
       [诗词内容] {item.get('content', '')}
       """
   ```

### 7.2 中长期研究方向

1. **古诗专用分词器**：开发针对古汉语的分词和词元化策略
2. **韵律感知注意力机制**：增强模型对诗词韵律结构的理解
3. **知识蒸馏技术**：将 Qwen 3-8 B 的优势知识蒸馏到更小模型
4. **平仄结构编码器**：开发专门编码诗词平仄结构的神经网络层

## 附录：关键代码片段

### LoRA 配置示例
```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    lora_alpha=128,
    lora_dropout=0.1,
    target_modules=qwen_target_modules,
    inference_mode=False,
    bias="none",
)
```

### 梯度检查机制
```python
# 检验梯度是否能够正确计算
try:
    dummy_loss = sum([p.mean() for p in grad_params])
    dummy_loss.backward()
    logger.info("梯度检查通过: 可正确计算梯度")
    # 清理梯度，准备实际训练
    model.zero_grad()
except Exception as e:
    logger.error(f"梯度检查失败: {e}")
    raise
```

### 温度退火采样
```python
class TemperatureScheduler:
    def __init__(self, initial_temp=0.7, min_temp=0.2, steps=3):
        self.temps = np.linspace(initial_temp, min_temp, steps)
        
    def get_temperatures(self):
        return self.temps
```

---

*报告编制：Jam Sylph*  
*日期：2025 年 5 月*
