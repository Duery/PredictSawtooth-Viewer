# PredictSawtooth Viewer

面向 HL-3 锯齿波形预测的本地模型结果查看器。选择模型与炮号，在真实 **IP + SXR**
波形上定位 160 ms 历史窗口，然后运行训练完成模型的只读推理并查看预测结果。

```text
选择模型与炮号  →  定位 IP + SXR 时间窗  →  预测 / 真实未来 / Persistence 对照
```

## 主要能力

- 自动扫描标准化 `model_registry/`，展示模型、配置、炮号和就绪状态。
- 支持 TimeXer、WPMixer、TimesFM 2.5 LoRA 和 TimesFM 3.0。
- TimesFM 3.0 提供单通道 Zero-shot 与 13 通道 Native multivariate 两个版本。
- 保持各模型训练时的通道顺序、采样率、窗口和预处理契约。
- 展示 MAE、NRMSE、相关系数、R²、Persistence 和推理耗时。
- 深色/浅色主题；英文使用 Inter，中文使用加粗得意黑。

> 页面执行的是训练完成模型的**推理**，不会训练、更新或覆盖模型权重。IP + SXR
> 双轨波形只用于定位时间段，不会改变模型的真实输入通道。

## 快速启动

环境要求：Windows 或 Linux、Python 3.10+。

```powershell
git clone https://github.com/Duery/PredictSawtooth-Viewer.git
cd PredictSawtooth-Viewer

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item config.example.toml config.toml
.\.venv\Scripts\python.exe -B app.py --no-browser
```

访问 `http://127.0.0.1:7860/`。

Linux/macOS 将虚拟环境命令替换为：

```bash
python -m venv .venv
./.venv/bin/python -m pip install -e '.[dev]'
cp config.example.toml config.toml
./.venv/bin/python -B app.py --no-browser
```

## 数据和模型

公开仓库只包含源码、模型包元数据和第三方许可证，不包含：

- HL-3 原始波形、处理缓存、人工标签或保存结果；
- TimeXer、WPMixer、LoRA adapter 或 TimesFM checkpoint；
- Hugging Face token、本机路径和运行时输出。

将数据缓存放在 `Data/cache/`，模型按
[模型包规范](model_registry/MODEL_PACKAGE_SPEC.md) 放入 `model_registry/`。也可以在本机
`config.toml` 中设置外部只读路径：

```toml
[results]
root = "D:/path/to/outputs"

[inference]
model_registry = "D:/path/to/model_registry"
data_cache = "D:/path/to/data_cache"
code_root = "D:/path/to/WPMixer"
device = "cpu"
```

TimesFM 3.0 权重来自
[`google/timesfm-3.0-pytorch`](https://huggingface.co/google/timesfm-3.0-pytorch)，
适用 TimesFM Non-Commercial License v1.0。请在使用前阅读模型许可证。

## 项目结构

```text
frontend/          页面、主题、字体与交互
src/               结果扫描、模型注册、窗口选择、推理与指标
model_registry/    模型包规范、manifest 和本地权重位置
tests/             扫描、前端契约和真实推理回归测试
vendor/            TimesFM 3.0 推理所需的固定源码与许可证
app.py             本地 HTTP 服务入口
```

完整设计见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

没有外部数据或模型权重时，选择页仍可启动；依赖真实工件的推理测试会明确跳过。

## 使用边界

- 当前结果属于回顾性窗口推理，不等同于实时控制验证。
- 波形拟合指标不等同于锯齿崩塌事件级预警指标。
- 炮号和模型配置可以公开，但原始实验数据与权重不随仓库发布。

本仓库目前未授予统一的项目级开源许可证。仓库内第三方代码、字体和模型分别遵循其随附
许可证。
