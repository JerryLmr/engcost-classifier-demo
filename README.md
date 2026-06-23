# 物业工程名称智能分类 Demo

这个 Demo 包含两部分：

- `frontend/`：静态前端页面，已拆分为 `index.html + styles.css + app.js`
- `backend/`：FastAPI 后端，统一走 CP/CF 标准目录分类链路

## 1. 安装依赖

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
```

## 2. 确认 LLM 服务已运行

默认使用 Ollama：

```bash
ollama list
ollama run qwen3:8b
```

只要本地 `http://127.0.0.1:11434` 可访问即可。

如需使用 LM Studio，可以手动设置环境变量：

```bash
export LLM_PROVIDER=lmstudio
export LMSTUDIO_BASE_URL=http://172.18.0.1:1234/v1
export LMSTUDIO_MODEL=qwen/qwen3.6-35b-a3b
export LMSTUDIO_API_KEY=lm-studio
export LLM_TIMEOUT_SECONDS=60
```

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

## 5. 分类链路

当前只保留一条主链路：

```text
normalize_project_text
→ match_aliases
→ llm_select_catalog_item_from_full_catalog
→ llm_select_repair_status
→ decide_review
```

- CP/CF 标准目录来自 `backend/config/standard_catalog.json`
- alias 规则来自 `backend/config/text_aliases.json`
- normalizer、alias、动作词和复核提示只作为上下文；catalog_id 由完整 compact 标准目录 LLM 选择并经过标准目录校验
- 最终 `OUT_OF_SCOPE`、复合工程和复核建议由标准目录后处理统一决定

## 6. Excel 批量处理

第一列为工程名称。输出列统一为：

```text
工程名称
catalog_id
一级分类
二级分类
维修状态
标准对象
是否复合工程
复合目录
是否紧急维修
是否白蚁相关
是否建议复核
分类依据
```

通过前端上传 Excel 和通过脚本批量处理，都会使用同一条标准目录链路。

## 7. 本地批量跑 Excel

```bash
cd /home/jerrylmr/githubRepository/engcost-classifier-demo
source backend/.venv/bin/activate
python scripts/batch_classify_excel.py /path/to/excel_dir --overwrite
```

默认输出到 `/path/to/excel_dir/classified_results/`。

常用参数：

```bash
python scripts/batch_classify_excel.py /path/to/excel_dir --overwrite
python scripts/batch_classify_excel.py /path/to/excel_dir -o /path/to/output_dir --overwrite
python scripts/batch_classify_excel.py /path/to/input.xlsx -o /path/to/output.xlsx --overwrite
```

脚本默认会跳过已经带 `_分类结果` 或 `_classified` 后缀的文件。

## 8. 分析分类结果

可以对标准目录分类结果做汇总分析，并导出一份 Excel 报表：

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

## 9. 运行测试

```bash
cd backend
source .venv/bin/activate
python -m unittest discover -s tests -p "test_*.py"
```

## 10. 可继续扩展

- 增加“人工修正后回写训练集”功能
