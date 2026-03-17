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

## 8. 可继续扩展

- 把分类体系从代码中迁移到 JSON 配置文件
- 记录分类日志和命中率
- 增加“人工修正后回写训练集”功能
- 接入你现有的 demo 首页或系统菜单
