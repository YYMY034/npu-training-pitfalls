# npu-training-pitfalls

按**错误签名**索引的国产算力（昇腾 NPU 等）训练踩坑数据库。你在 CANN / torch_npu / HCCL / DeepSpeed / OpenI 平台上撞到的每一个错误串，都应该能在这里搜到根因、修复、版本边界和复现证据。

**为什么存在**：国产栈最贵的时间税是重复调试已知问题——同一个版本矩阵、同一个静默数值错、同一个缺内核，被每个团队重新踩一遍，因为答案没有被索引到故障现场。本仓库把知识放在**你搜索的那串错误本身**。

## 怎么用

1. 把你看到的**错误字符串**（如 `ERR02200`、`Can not support soc_version: 104`、`bfloat16 of tbe.dsl.transdata is not supported`）在本页 Ctrl+F，或在 GitHub 全站搜 `<错误串> npu-training-pitfalls`
2. 命中后看：症状 → 根因 → 修复 → 版本边界 → 证据链接
3. 机器可读版：[`index.json`](index.json)（`sig` 字段即签名，可接进你自己的诊断脚本）

## 索引（按层分组，v0.1 共 53 条）

### 环境安装（env-install, 10 条）
`Failed to load the backend extension: torch_npu`（PyPI torch aarch64 与 torch_npu ABI 互斥）· venv 克隆 shebang 陷阱 · vllm/transformers/torch 版本墙三连 · triton-ascend 重装互踩 · fla-npu 缺失 · 默认内核空壳 ……

### CANN / 硬件（cann-hardware, 9 条）
`ERR00100` aclnnInplaceCopy（910C 需 A3 变体）· `ACL 507034` 坏卡调度 · `500002` / `Can not support soc_version: 104` · `aicore Illegal instruction in NLLLoss` · vllm-ascend 0.30 内核库 vs CANN 9.1 硬失败 …

### 数值正确性（numerics, 5 条）⚠️ 最危险：不报错、结果是错的
`signbit(-0.0) = False`（910B2 三模式一致，CUDA 正常）· cross_entropy **静默接受 int32 targets**（CUDA 拒绝——"NPU 能跑、CUDA 才崩"）· GroupNorm backward 静默零梯度（310P3 eager）· CANN 版本改变训练数值（7.0.RC1 梯度爆炸）· GradScaler 拒收 fp16 梯度（双栈一致，需 fp32 master）
→ **开训前先跑 [train_audit.py](https://openi.pcl.ac.cn/yymy050/sd-crosshw/src/branch/master/conformance-kit/train_audit.py)（~10 秒，16 探针 + fuzz）**

### 算子覆盖（kernels-ops, 3 条）
`E90003 bfloat16 of tbe.dsl.transdata is not supported`（旧栈）· `FlexAttention is only supported on CUDA, CPU or HPU` · `Conv3DBackpropFilter` 缺失（VLM 微调断点）

### 分布式（distributed, 1 条 + 关联）
`ERR02200 DIST call hccl api failed`（双机 allreduce/save 失败簇）· DDP bucket stride 警告（挂 3 年 "Analysing"）

### 框架适配（framework, 10 条）
ZeRO-3 `no_sync` 断言 + `EE1001` 流同步 · Megatron 并行配置等价性缺口 · bitsandbytes 仅 CUDA（QLoRA 不可用）· vllm-ascend 内核库/CANN 版本硬失败 …

### 平台（platform, 6 条）
调试任务**换记录 ID** → 停止指向旧 ID 空转计费 · 分中心排队 · NFS 陈旧 `__pycache__` · contents PUT 200 但静默不落盘 · /tmp/code 30 天回收 …

## 证据纪律

每条 entry 带证据（我们的实测 JSON 或上游 issue 链接）；未验证的条目在 `note` 里标注。验证协议见 [DECISION_CARD R0](https://openi.pcl.ac.cn/yymy050/sd-crosshw/raw/branch/master/DECISION_CARD.md)。

## 贡献

踩到新坑：开 issue 附**完整错误串** + CANN/torch_npu 版本 + 复现步骤（可复现者优先收编）。修好的条目划线保留（后来者仍会搜到）。

## 数据来源与状态

种子语料：[sd-crosshw](https://openi.pcl.ac.cn/yymy050/sd-crosshw) 两周 910B2 实测事故库（OWN-001..026，全部至少复现一次）+ 全网 issue 挖掘（SCAN-*，带上游链接）。v0.1 手工整理，2026-11-23。
