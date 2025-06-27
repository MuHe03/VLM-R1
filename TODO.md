## TODO List

### 1. 模型搭建 -- Mu He
- [ ] 搭建 Qwen + Mask2Former 组合模型
  - 参考 LISA 范式设计整体架构
  - 不使用 LISA 的 Visual Encoder，改为采用 Qwen 内置的 Visual Model
  - 保证 Visual Encoder 和 Qwen 模块的接口兼容

### 2. 数据集与 Prompt 调整 -- Dingyi Zhou
- [ ] 参考 LISA 的训练数据集
  - 结合 LISA 输出的 Text Token、Mask 结果
  - 整合 LISA Prompt 结构与 Qwen Prompt 格式
  - 调整数据集适配 Qwen 的 Prompt 接口规范

### 3. Stage 1：SFT 微调
- [ ] 参考 VLM-R1 方案，完成以下任务：
  - 将 LLAMA-factor 训练的 Qwen 模型替换为本项目自定义模型
  - 完成接口适配，确保数据流畅通
  - **注意：需冻结 Visual Model，避免其参数更新**

### 4. Stage 2：GRPO 微调
- [ ] 进行 GRPO 微调阶段
  - 保持 Mask2Former 模型参数冻结状态
