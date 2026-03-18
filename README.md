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
- LLM 兜底，满足“有 AI”的展示需求
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
python scripts/batch_classify_excel.py /path/to/excel_dir --rule-source python
```

默认会把结果输出到 `/path/to/excel_dir/classified_results/`。

常用参数：

```bash
python scripts/batch_classify_excel.py /path/to/excel_dir --overwrite --rule-source python
python scripts/batch_classify_excel.py /path/to/excel_dir -o /path/to/output_dir --rule-source json
```

脚本默认会跳过已经带 `_分类结果` 或 `_classified` 后缀的文件。

## 9. JSON 配置化与对比

当前规则已经支持双轨运行：

- 默认 `RULE_SOURCE=json`：使用 `backend/config/*.json` 中的配置
- `RULE_SOURCE=python`：仍可切回 Python 基线规则做对比

导出当前 Python 基线到 JSON：

```bash
python scripts/export_rules_to_json.py
```

对同一批输入分别跑 Python 版和 JSON 版：

```bash
source backend/.venv/bin/activate
python scripts/batch_classify_excel.py excel_inputs -o excel_outputs_python --overwrite --rule-source python
python scripts/batch_classify_excel.py excel_inputs -o excel_outputs_json --overwrite --rule-source json
```

对比两套结果：

```bash
source backend/.venv/bin/activate
python scripts/compare_excel_outputs.py excel_outputs_python excel_outputs_json --csv compare_reports/python_vs_json_diff.csv
```

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
- 结构统计
- 一级分类统计
- 二级分类统计
- 重点样本

前提是文件中已经包含这些结果列：

- `一级分类`
- `二级分类`
- `分类方式`
- `分类依据`
- `是否复合工程`
- `是否建议复核`
- `结构类型`

## 12. 可继续扩展

- 把分类体系从代码中迁移到 JSON 配置文件
- 记录分类日志和命中率
- 增加“人工修正后回写训练集”功能
- 接入你现有的 demo 首页或系统菜单
