# 物业工程名称智能分类 Demo

这个 Demo 包含两部分：

- `frontend/`：静态前端页面，已拆分为 `index.html + styles.css + app.js`
- `backend/`：FastAPI 后端，已拆分为 `api / core / data / models / services`

## 1. 安装依赖

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
```

## 2. 确认 Ollama 已运行

```bash
ollama list
ollama run qwen3:8b
```

只要本地 `http://127.0.0.1:11434` 可访问即可。

## 2.1. 可选：使用 LM Studio 测试本地 35B 模型

Windows + WSL 示例环境变量：

```bash
export LLM_PROVIDER=lmstudio
export LMSTUDIO_BASE_URL=http://172.18.0.1:1234/v1
export LMSTUDIO_MODEL=qwen/qwen3.6-35b-a3b
export LMSTUDIO_API_KEY=lm-studio
export LLM_TIMEOUT_SECONDS=60
```

LM Studio 需要开启 `Serve on local network`。

手动运行示例：

```bash
python scripts/batch_classify_excel.py excel_inputs/d1_error_sample.xlsx -o excel_outputs_lmstudio --overwrite --mode llm
```

注意：这条命令用于本地手动测试，不应由 Codex 执行。

## 3. 启动后端

```bash
cd backend
source .venv/bin/activate
uvicorn app:app --reload
```

默认地址：`http://127.0.0.1:8000`

## 4. 打开前端

直接双击 `frontend/index.html`，或在 VS Code 里用 Live Server 打开。

如果需要改前端请求地址，可在 `frontend/index.html` 中调整：

```html
<script>
  window.API_BASE = "http://127.0.0.1:8000";
</script>
```

## 5. Excel 批量处理约定

- 第一列：工程名称
- 结果追加四列：一级分类、二级分类、分类方式、分类依据

## 6. 展示亮点

- 输入一句工程名称，立即返回细分类结果
- 支持 Excel 批量分类，更像企业可用工具
- 规则优先，减少常见分类抖动
- LLM 辅助分类，兼顾覆盖率与展示效果
- 后端已模块化，便于继续扩展规则、配置和测试

## 7. 运行测试

```bash
cd backend
source .venv/bin/activate
python -m unittest discover -s tests -p "test_*.py"
```

## 8. 本地批量跑 Excel 回归

不想通过前端页面逐个上传时，可以直接批量处理一个目录下的 Excel：

```bash
cd /home/jerrylmr/githubRepository/engcost-classifier-demo
source backend/.venv/bin/activate
python scripts/batch_classify_excel.py /path/to/excel_dir --overwrite
```

默认会把结果输出到 `/path/to/excel_dir/classified_results/`。

常用参数：

```bash
python scripts/batch_classify_excel.py /path/to/excel_dir --overwrite
python scripts/batch_classify_excel.py /path/to/excel_dir -o /path/to/output_dir --overwrite
python scripts/batch_classify_excel.py /path/to/input.xlsx -o /path/to/output.xlsx --overwrite
python scripts/batch_classify_excel.py /path/to/input.xlsx -o /path/to/output.xlsx --overwrite --mode llm
```

`--mode` 支持 `auto`、`llm`、`rule`。默认 `auto` 保持规则优先并用 LLM 兜底；`llm` 会直接用完整目录交给 LLM；`rule` 只跑规则，规则无结果时默认兜底。

脚本默认会跳过已经带 `_分类结果` 或 `_classified` 后缀的文件。

## 9. 固定三级目录

当前分类只使用固定目录文件：

```text
backend/config/catalog.json
```

分类结果直接返回一级、二级、三级目录，不再支持 `RULE_SOURCE` 双轨切换。

## 10. 分析分类结果

可以直接对整个结果目录做汇总分析，并导出一份 Excel 报表：

```bash
cd /home/jerrylmr/githubRepository/engcost-classifier-demo
source backend/.venv/bin/activate
python scripts/analyze_excel_outputs.py excel_outputs
```

默认输出：

```text
excel_outputs/分析汇总.xlsx
```

汇总文件包含 4 个 sheet：

- `总览`
- `一级分类统计`
- `二级分类统计`
- `重点样本`

也可以自定义输出路径：

```bash
python scripts/analyze_excel_outputs.py excel_outputs -o reports/分析汇总.xlsx
```

## 11. 前端分析结果文件

前端页面现在支持上传**已分类结果 Excel** 并直接展示：

- 总览摘要
- 匹配类型统计
- 一级分类统计
- 二级分类统计
- 重点样本

前提是文件中已经包含这些结果列：

- `一级分类`
- `二级分类`
- `三级分类`
- `分类方式`
- `置信度`
- `匹配类型`
- `是否建议复核`
- `候选目录ID`
- `候选目录`
- `分类依据`

## 12. 可继续扩展

- 记录分类日志和命中率
- 增加“人工修正后回写训练集”功能
- 接入你现有的 demo 首页或系统菜单
