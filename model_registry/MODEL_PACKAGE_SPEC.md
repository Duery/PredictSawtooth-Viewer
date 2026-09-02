# PredictSawtooth 模型包规范 v1

把一个完整模型包放入 `model_registry/<model_package_id>/`，重新启动软件后会自动扫描。
模型目录只读；实时输出不会写回模型包。

## 必需文件

```text
<model_package_id>/
├── manifest.json
├── checkpoint.pt 或 model.safetensors
├── run_config.json
├── scaler.json 或 preprocessing.json
└── completed.json
```

字段定义见同目录的 `manifest.schema.json`。可复制当前任一同族模型包的
`manifest.json`，再替换权重、配置、窗口参数、结果关联和炮号。

## 自动接入范围

- 已实现 Runner：`wpmixer.timexer.v1`、`wpmixer.legacy.v1`、
  `timesfm25.lora.v1`、`timesfm3.zero_shot.v1`、`timesfm3.multivariate.v1`。
- 同结构 TimeXer checkpoint：模型包完整并且参数、输入通道、时间窗一致时可直接接入。
- 当前 legacy WPMixer、TimesFM 2.5 LoRA 和 TimesFM 3.0 Zero-shot 可按现有 manifest
  及 TimesFM 3.0 Native multivariate 可按现有 manifest 接入；其他结构需先实现并注册
  Runner。
- `manifest.json` 不能指定任意 Python 脚本；Runner 必须由软件代码显式注册。

TimesFM 2.5 LoRA 模型包还需在 `artifacts.adapter_config` 登记 adapter 配置，并在
`base_model_id` 指定已经存在于本机 Hugging Face 缓存的基座。TimesFM 3.0 模型包把
官方 `config.json`、`model.safetensors` 和许可证放在 `base_model/`，运行时仅从该本地
目录加载，后端不会联网下载。

## 结果集关联

`compatible_result.relative_dir` 是相对于 `config.toml` 中 `[results].root` 的结果目录，
`compatible_result.variant` 对应结果扫描器中的模型变体。两者完全匹配后，选择页才显示
“可实时推理”。

没有预先保存结果的 Zero-shot 模型可使用 `catalog_entry` 提供选择页元数据；只有模型包
完整并通过校验时才会出现，不创建虚假的结果目录或指标。

## 数据要求

浏览器只发送模型包、炮号和历史终点。后端从 `[inference].data_cache` 读取真实缓存，
并按照 manifest 中的通道顺序、scaler、采样率和窗口参数构造输入。IP/SXR 导航数据不
替代模型训练时的完整输入。
