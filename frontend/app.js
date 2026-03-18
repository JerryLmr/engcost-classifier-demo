const API_BASE = window.API_BASE || "http://127.0.0.1:8000";

const projectText = document.getElementById("projectText");
const resultBox = document.getElementById("resultBox");
const singleStatus = document.getElementById("singleStatus");
const excelStatus = document.getElementById("excelStatus");
const excelFile = document.getElementById("excelFile");
const analysisFile = document.getElementById("analysisFile");
const analysisStatus = document.getElementById("analysisStatus");
const analysisResults = document.getElementById("analysisResults");
const summaryMetrics = document.getElementById("summaryMetrics");
const structureMetrics = document.getElementById("structureMetrics");
const level1Table = document.getElementById("level1Table");
const level2Table = document.getElementById("level2Table");
const focusTable = document.getElementById("focusTable");
const rStructureType = document.getElementById("rStructureType");
const rCompositeReasonWrap = document.getElementById("rCompositeReasonWrap");
const rCompositeReason = document.getElementById("rCompositeReason");
const rCandidatesWrap = document.getElementById("rCandidatesWrap");
const rCandidates = document.getElementById("rCandidates");

function getMethodLabel(method) {
  if (method === "LLM 兜底") {
    return "LLM 辅助分类";
  }
  if (method === "降级兜底") {
    return "体系外默认分类";
  }
  return method;
}

function getStructureTypeLabel(structureType) {
  if (structureType === "single_project") {
    return "单一工程";
  }
  if (structureType === "multi_system_same_domain") {
    return "同域多系统";
  }
  if (structureType === "composite_project") {
    return "复合工程";
  }
  return structureType || "";
}

function buildFocusTitle(row) {
  const details = [];
  if (row.reason) {
    details.push(`分类依据：${row.reason}`);
  }
  if (row.composite_reason) {
    details.push(`复合原因：${row.composite_reason}`);
  }
  if (Array.isArray(row.secondary_candidates) && row.secondary_candidates.length) {
    details.push(`候选分类：${row.secondary_candidates.join("、")}`);
  }
  return details.join("\n");
}

function setSingleStatus(message) {
  singleStatus.innerHTML = message;
}

function resetSingleResult() {
  projectText.value = "";
  resultBox.hidden = true;
  rStructureType.textContent = "";
  rCompositeReasonWrap.hidden = true;
  rCompositeReason.textContent = "";
  rCandidatesWrap.hidden = true;
  rCandidates.textContent = "";
  setSingleStatus(`后端地址：<code>${API_BASE}</code>`);
}

function renderMetrics(container, items) {
  container.innerHTML = items
    .map(
      (item) => `
        <div class="metric">
          <span class="label">${item.label}</span>
          <span class="value">${item.value}</span>
        </div>
      `,
    )
    .join("");
}

function renderCountTable(container, rows, emptyLabel) {
  if (!rows.length) {
    container.innerHTML = `<tr><td colspan="2">${emptyLabel}</td></tr>`;
    return;
  }
  container.innerHTML = rows
    .map((row) => `<tr><td>${row.name}</td><td>${row.count}</td></tr>`)
    .join("");
}

function renderFocusSamples(rows) {
  if (!rows.length) {
    focusTable.innerHTML = '<tr><td colspan="9">当前没有重点样本</td></tr>';
    return;
  }
  focusTable.innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td title="${buildFocusTitle(row)}">${row.project_name}</td>
          <td>${row.level1}</td>
          <td>${row.level2}</td>
          <td>${getMethodLabel(row.method)}</td>
          <td>${row.is_composite ? "是" : "否"}</td>
          <td>${row.needs_review ? "是" : "否"}</td>
          <td title="${buildFocusTitle(row)}">${getStructureTypeLabel(row.structure_type)}</td>
          <td>${row.composite_reason || "-"}</td>
          <td>${Array.isArray(row.secondary_candidates) && row.secondary_candidates.length ? row.secondary_candidates.join("、") : "-"}</td>
        </tr>
      `,
    )
    .join("");
}

async function handleSingleClassify() {
  const text = projectText.value.trim();
  if (!text) {
    setSingleStatus("请先输入工程名称");
    return;
  }

  setSingleStatus("正在分类...");
  try {
    const response = await fetch(`${API_BASE}/api/classify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "分类失败");
    }

    resultBox.hidden = false;
    document.getElementById("rProject").textContent = data.project_name;
    document.getElementById("rLevel1").textContent = data.level1;
    document.getElementById("rLevel2").textContent = data.level2;
    document.getElementById("rMethod").textContent = getMethodLabel(data.method);
    rStructureType.textContent = getStructureTypeLabel(data.structure_type);
    if (data.composite_reason) {
      rCompositeReasonWrap.hidden = false;
      rCompositeReason.textContent = data.composite_reason;
    } else {
      rCompositeReasonWrap.hidden = true;
      rCompositeReason.textContent = "";
    }
    if (Array.isArray(data.secondary_candidates) && data.secondary_candidates.length) {
      rCandidatesWrap.hidden = false;
      rCandidates.textContent = data.secondary_candidates.join("、");
    } else {
      rCandidatesWrap.hidden = true;
      rCandidates.textContent = "";
    }
    document.getElementById("rReason").textContent = data.reason;
    setSingleStatus("分类完成");
  } catch (error) {
    setSingleStatus(`错误：${error.message}`);
  }
}

async function handleExcelClassify() {
  if (!excelFile.files.length) {
    excelStatus.textContent = "请先选择 Excel 文件";
    return;
  }

  excelStatus.textContent = "正在处理 Excel，请稍候...";
  try {
    const formData = new FormData();
    formData.append("file", excelFile.files[0]);

    const response = await fetch(`${API_BASE}/api/classify-excel`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Excel 处理失败");
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = excelFile.files[0].name.replace(/\.xlsx?$|\.xlsm$/i, "") + "_分类结果.xlsx";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    excelStatus.textContent = "Excel 处理完成，结果已开始下载";
  } catch (error) {
    excelStatus.textContent = `错误：${error.message}`;
  }
}

async function handleExcelAnalyze() {
  if (!analysisFile.files.length) {
    analysisStatus.textContent = "请先选择已分类结果 Excel 文件";
    return;
  }

  analysisStatus.textContent = "正在分析，请稍候...";
  analysisResults.hidden = true;

  try {
    const formData = new FormData();
    formData.append("file", analysisFile.files[0]);

    const response = await fetch(`${API_BASE}/api/analyze-excel`, {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "分析失败");
    }

    renderMetrics(summaryMetrics, [
      { label: "总记录数", value: data.summary.total_records },
      { label: "规则优先", value: data.summary.rule_method_count },
      { label: "LLM 辅助分类", value: data.summary.llm_method_count },
      { label: "体系外默认分类", value: data.summary.fallback_method_count },
      { label: "复合工程", value: data.summary.composite_count },
      { label: "建议复核", value: data.summary.review_count },
    ]);

    renderMetrics(structureMetrics, [
      { label: "单一工程", value: data.structure_counts.single_project },
      { label: "同域多系统", value: data.structure_counts.multi_system_same_domain },
      { label: "复合工程", value: data.structure_counts.composite_project },
    ]);

    renderCountTable(level1Table, data.level1_top, "暂无一级分类统计");
    renderCountTable(level2Table, data.level2_top, "暂无二级分类统计");
    renderFocusSamples(data.focus_samples);

    analysisResults.hidden = false;
    analysisStatus.textContent = "分析完成";
  } catch (error) {
    analysisStatus.textContent = `错误：${error.message}`;
  }
}

document.querySelectorAll(".example").forEach((element) => {
  element.addEventListener("click", () => {
    projectText.value = element.textContent.trim();
  });
});

document.getElementById("classifyBtn").addEventListener("click", handleSingleClassify);
document.getElementById("clearBtn").addEventListener("click", resetSingleResult);
document.getElementById("excelBtn").addEventListener("click", handleExcelClassify);
document.getElementById("analyzeBtn").addEventListener("click", handleExcelAnalyze);

resetSingleResult();
