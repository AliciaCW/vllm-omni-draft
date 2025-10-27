# AR+DiT 分离性能测试

## 背景

### 任务介绍

当前vLLM只支持多模态输入，文字输出的推理任务，我们vLLM-Omni后续准备将项目拓展到支持多模态输出的任务，具体的方案是通过将AR和生成模型（如DiT)分离后通过不同的Engine Core来处理同一请求中的不同模型，因此目前需要一个分离测试方案来验证性能提升效果。

需要考虑的场景：离线/在线推理，图像生成/编辑任务，再扩展到音频以及输出数据类型为多模态混合的任务。

初期考虑离线的推理。

除性能测试外，还需要考虑核心功能验证测试，以确保系统的基本功能正常。

**分离式架构性能测试：**
验证我们的分离式架构对比原始模型在延迟，吞吐量以及资源使用上都有所提升。

**分离式架构流程测试：**

重点验证请求在Proxy API server、vLLM prefill和vLLM decode三个组件间的流转是否正确。例如，确认vLLM prefill阶段产生的KV缓存能否被vLLM decode阶段正确接收并用于后续生成。

**生成效果验证(以Qwen Image为例)**

基础图像生成：测试不同参数（如提示词、尺寸width/height、推理步数num_inference_steps）下的图像生成效果。

文本渲染：测试模型生成包含准确中英文文本（如海报、标志）的图像能力。

图像编辑：使用Qwen-Image-Edit模型，测试其根据文本指令对现有图像进行编辑（如修改内容、风格迁移）的功能。



### 性能测试主要指标

关键性能指标：

1. 延迟：
   首Token延迟 (Time to First Token, TTFT) 和每个输出Token的时间 (Time Per Output Token, TPOT)。

2. 吞吐量：每秒处理token数量
   请求吞吐量 (requests/s) 和Token吞吐量 (tokens/s)

3. 资源使用：
    GPU显存占用、利用率和CPU使用率。

压力与稳定性场景：

1. 并发测试：在不同并发度 (--max-concurrency) 下测试性能表现，找到系统瓶颈。

2. 请求速率测试：使用不同的请求速率 (--request-rate)，观察系统在恒定压力下的表现。

3. 长文本与长时测试：使用不同输入长度 (--random-input-len) 的提示词，测试性能变化。并进行长时间高负载测试，检查是否存在内存泄漏或性能下降。

（包括但不限于上面的指标）


### 理论估算的数据

| 类型       | 可计算性 | 方法             |
| ---------- | -------- | ---------------- |
| 性能       | ✅        | 分阶段延迟分解   |
| 显存需求   | ✅        | 通过参数规模计算 |
| 批处理增益 |          |                  |
| 瓶颈识别   |          |                  |


- 性能： 由于是分离架构，需要分别计算AR部分和DiT部分的性能，并考虑两者之间的数据传输（如果分布在不同的设备上）。AR部分主要是自回归生成，DiT部分则是扩散模型的多步去噪。每个步骤的计算量可以通过模型参数量和输入输出大小来估算。


```python
# 总延迟 = Prefill延迟 + Decode延迟
Total_Latency = T_prefill + N_tokens × T_per_token

# Prefill阶段：受计算瓶颈限制
T_prefill ≈ (模型FLOPs × 序列长度) / GPU峰值FLOPS

# Decode阶段：受内存带宽限制  
T_per_token ≈ (KV缓存大小 + 权重大小) / GPU内存带宽
```

- 显存需求：
    分别计算AR模型和DiT模型的参数显存。模型的参数所占显存可以通过参数数量乘以每个参数所占字节数来计算。对于推理，AR模型主要考虑参数和KV缓存；对于DiT模型，显存需求还与采样步数有关。实际推理测试中，Qwen Image Edit加载后使用的显存可以达到70G。

```python

# 总显存 = 模型权重 + KV缓存 + 激活值 + 框架开销
Total_VRAM = Model_Memory + KV_Cache + Activation + Overhead
# 模型权重显存
Model_Memory = 参数数量 × 每参数字节数（FP8=1字节）

# KV缓存显存
KV_Cache = 2 × batch_size × seq_len × layers × hidden_dim × 数据类型大小




# DiT 显存
Total_VRAM = Model_Memory + Activation_Memory + KV_Cache + Diffusion_States + Overhead

Activation_Memory = batch_size × seq_len × hidden_dim × 激活值系数 × 数据类型大小
KV_Cache = 2 × batch_size × seq_len × num_layers × hidden_dim × 数据类型大小
# 对于图像patch序列
seq_len = (image_size / patch_size)²
Diffusion_States = batch_size × num_timesteps_kept × latent_dim × 数据类型大小

```

- 批处理增益：
    在vLLM中，批处理增益主要通过PagedAttention和动态批处理来实现。理论上，随着批处理大小增加，吞吐量会提升，但延迟可能会增加。对于DiT部分，批处理增益同样存在，但要注意扩散模型每一步的计算量较大，批处理大小可能受显存限制。理论上讲，小batch_size：接近线性增长； 中等batch_size：增长斜率下降；大batch_size：趋于饱和。 实际上，在前期测试时，Qwen Image Edit能接受的最大bs为4，（输入图片尺寸1024*1024）。

- 瓶颈识别：    
    根据初期的分离测试，DiT是性能瓶颈所在 （AR耗时<1s, DiT耗时为几十秒不等）。

### 注意的点：

分离式架构：单独监控 prefill 和 decode 阶段的资源消耗与耗时，以便精准定位性能瓶颈。

AR模型误差累积：对于自回归（AR）模型，测试长序列生成时可能出现的误差累积问题。

合成负载需要贴近真实业务场景: 尽量在基准测试使用的数据（如输入/输出Token长度分布）贴近真实业务场景。

关注P99/P95延迟：平均延迟可能会掩盖一些极端情况，高百分位延迟（如P99）更值得考虑。


## 测试实验设计：

### 测试环境：

至少2张 80GB Graph Card

### 使用到的模型/数据集

模型： 

    实验组 - vLLM-Omni version Qwen Image
    对照组 - 原生Qwen Image: Qwen/Qwen-Image-Edit


![images/arch.png](images/arch.png)

数据集：

    单图像数据集：Image edit数据集，包含单轮对话和多轮对话图像编辑任务
    多图像数据集： tbd.






### 实验参数

#### 控制变量
dtype: bf16

#### 自变量

- batch size:[1, 2, 4]
- prompt length: [512, 2048, 2560]
- input image size: [1024*1024, ..]
- output image size: [1376*768, ..]

### 性能数据记录

    batch_size		
    平均TTFT (s)	
    平均TPOT (s)	
    吞吐量 (req/s)	
    吞吐量 (tokens/s)	
    GPU显存占用 (GB)

（在线时需要加上并发数和请求总量）

可以考虑画出折线图：

    性能曲线图: 吞吐量 vs. 并发数、 延迟 vs. 并发数、吞吐量 vs. batch_size。
    资源使用图: 显存占用随batch_size变化图。
