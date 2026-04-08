const API_BASE = window.API_BASE || "http://127.0.0.1:8000";

const projectText = document.getElementById("projectText");
const auditProjectText = document.getElementById("auditProjectText");
const singleModeBtn = document.getElementById("singleModeBtn");
const excelModeBtn = document.getElementById("excelModeBtn");
const singleFeaturePanel = document.getElementById("singleFeaturePanel");
const excelFeaturePanel = document.getElementById("excelFeaturePanel");
const resultBox = document.getElementById("resultBox");
const auditResultBox = document.getElementById("auditResultBox");
const singleStatus = document.getElementById("singleStatus");
const auditStatus = document.getElementById("auditStatus");
const excelStatus = document.getElementById("excelStatus");
const excelFile = document.getElementById("excelFile");
const excelFileName = document.getElementById("excelFileName");
const excelProgressWrap = document.getElementById("excelProgressWrap");
const excelProgressFill = document.getElementById("excelProgressFill");
const excelProgressLabel = document.getElementById("excelProgressLabel");
const excelProgressValue = document.getElementById("excelProgressValue");
const analysisFile = document.getElementById("analysisFile");
const analysisStatus = document.getElementById("analysisStatus");
const analysisResults = document.getElementById("analysisResults");
const analysisFileName = document.getElementById("analysisFileName");
const summaryMetrics = document.getElementById("summaryMetrics");
const sourceLegend = document.getElementById("sourceLegend");
const sourceDonut = document.getElementById("sourceDonut");
const structureBars = document.getElementById("structureBars");
const riskMetrics = document.getElementById("riskMetrics");
const riskChip = document.getElementById("riskChip");
const level1Table = document.getElementById("level1Table");
const level2Table = document.getElementById("level2Table");
const focusTable = document.getElementById("focusTable");
const focusSearch = document.getElementById("focusSearch");
const focusLevel1 = document.getElementById("focusLevel1");
const focusMethod = document.getElementById("focusMethod");
const focusStructure = document.getElementById("focusStructure");
const focusCompositeOnly = document.getElementById("focusCompositeOnly");
const focusReviewOnly = document.getElementById("focusReviewOnly");
const clearFocusFilters = document.getElementById("clearFocusFilters");
const focusSummary = document.getElementById("focusSummary");
const rStructureType = document.getElementById("rStructureType");
const rCompositeReasonWrap = document.getElementById("rCompositeReasonWrap");
const rCompositeReason = document.getElementById("rCompositeReason");
const rCandidatesWrap = document.getElementById("rCandidatesWrap");
const rCandidates = document.getElementById("rCandidates");
const auditBtn = document.getElementById("auditBtn");
const auditResetBtn = document.getElementById("auditResetBtn");
const auditDisplayResult = document.getElementById("auditDisplayResult");
const auditManualReview = document.getElementById("auditManualReview");
const auditProjectName = document.getElementById("auditProjectName");
const auditMappedObjects = document.getElementById("auditMappedObjects");
const auditTags = document.getElementById("auditTags");
const auditReasons = document.getElementById("auditReasons");
const auditReasonCodes = document.getElementById("auditReasonCodes");
const auditBasisDocuments = document.getElementById("auditBasisDocuments");
const auditPath = document.getElementById("auditPath");

let focusSamplesRaw = [];
const focusSortState = {
  key: null,
  direction: "asc",
};
let excelProcessingTimer = null;
let isAuditLoading = false;
let auditResult = null;
let auditError = "";

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function getMethodLabel(method) {
  if (method === "LLM 辅助分类" || method === "体系外默认分类") {
    return method;
  }
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

function getMethodRank(method) {
  if (method === "体系外默认分类" || method === "降级兜底") {
    return 0;
  }
  if (method === "LLM 辅助分类" || method === "LLM 兜底") {
    return 1;
  }
  return 2;
}

function getStructureRank(structureType) {
  if (structureType === "composite_project") {
    return 0;
  }
  if (structureType === "multi_system_same_domain") {
    return 1;
  }
  return 2;
}

function getBooleanRank(value) {
  return value ? 0 : 1;
}

function isTruthyFlag(value) {
  return value === true || value === "是";
}

function normalizeFocusSample(row) {
  return {
    ...row,
    is_composite: isTruthyFlag(row.is_composite),
    needs_review: isTruthyFlag(row.needs_review),
    secondary_candidates: Array.isArray(row.secondary_candidates) ? row.secondary_candidates : [],
    composite_reason: row.composite_reason || "",
  };
}

function compareText(a, b) {
  return String(a || "").localeCompare(String(b || ""), "zh-CN");
}

function updateFocusSortIndicators() {
  document.querySelectorAll(".sort-trigger").forEach((button) => {
    const baseLabel = button.dataset.baseLabel || button.textContent;
    button.dataset.baseLabel = baseLabel;
    if (button.dataset.sortKey === focusSortState.key) {
      const arrow = focusSortState.direction === "asc" ? " ↑" : " ↓";
      button.textContent = `${baseLabel}${arrow}`;
      button.classList.add("active");
    } else {
      button.textContent = baseLabel;
      button.classList.remove("active");
    }
  });
}

function populateFocusFilterOptions(rows) {
  const uniqueValues = (key) =>
    [...new Set(rows.map((row) => row[key]).filter((value) => value && String(value).trim() !== ""))].sort((a, b) =>
      compareText(a, b),
    );

  focusLevel1.innerHTML =
    '<option value="">全部一级分类</option>' +
    uniqueValues("level1").map((value) => `<option value="${value}">${value}</option>`).join("");

  focusMethod.innerHTML =
    '<option value="">全部分类方式</option>' +
    uniqueValues("method")
      .map((value) => `<option value="${value}">${getMethodLabel(value)}</option>`)
      .join("");

  focusStructure.innerHTML =
    '<option value="">全部结构类型</option>' +
    uniqueValues("structure_type")
      .map((value) => `<option value="${value}">${getStructureTypeLabel(value)}</option>`)
      .join("");
}

function getFilteredAndSortedFocusSamples() {
  const keyword = focusSearch.value.trim().toLowerCase();
  let rows = focusSamplesRaw.filter((row) => {
    if (keyword && !String(row.project_name || "").toLowerCase().includes(keyword)) {
      return false;
    }
    if (focusLevel1.value && row.level1 !== focusLevel1.value) {
      return false;
    }
    if (focusMethod.value && row.method !== focusMethod.value) {
      return false;
    }
    if (focusStructure.value && row.structure_type !== focusStructure.value) {
      return false;
    }
    if (focusCompositeOnly.checked && !isTruthyFlag(row.is_composite)) {
      return false;
    }
    if (focusReviewOnly.checked && !isTruthyFlag(row.needs_review)) {
      return false;
    }
    return true;
  });

  if (!focusSortState.key) {
    return rows;
  }

  rows = [...rows].sort((left, right) => {
    let result = 0;
    switch (focusSortState.key) {
      case "project_name":
      case "level1":
      case "level2":
        result = compareText(left[focusSortState.key], right[focusSortState.key]);
        break;
      case "method":
        result = getMethodRank(left.method) - getMethodRank(right.method);
        break;
      case "is_composite":
      case "needs_review":
        result = getBooleanRank(left[focusSortState.key]) - getBooleanRank(right[focusSortState.key]);
        break;
      case "structure_type":
        result = getStructureRank(left.structure_type) - getStructureRank(right.structure_type);
        break;
      default:
        result = 0;
    }
    if (result === 0) {
      result = compareText(left.project_name, right.project_name);
    }
    return focusSortState.direction === "asc" ? result : -result;
  });

  return rows;
}

function setSingleStatus(message) {
  singleStatus.innerHTML = message;
}

function setAuditStatus(message) {
  auditStatus.textContent = message;
}

function switchFeatureMode(mode) {
  const singleMode = mode === "single";
  singleFeaturePanel.hidden = !singleMode;
  excelFeaturePanel.hidden = singleMode;
  singleFeaturePanel.classList.toggle("active", singleMode);
  excelFeaturePanel.classList.toggle("active", !singleMode);
  singleModeBtn.classList.toggle("active", singleMode);
  excelModeBtn.classList.toggle("active", !singleMode);
}

function clearExcelProcessingTimer() {
  if (excelProcessingTimer) {
    clearInterval(excelProcessingTimer);
    excelProcessingTimer = null;
  }
}

function setExcelProgress(percent, label) {
  const safePercent = Math.max(0, Math.min(100, Math.round(percent)));
  excelProgressWrap.hidden = false;
  excelProgressFill.style.width = `${safePercent}%`;
  excelProgressValue.textContent = `${safePercent}%`;
  if (label) {
    excelProgressLabel.textContent = label;
  }
}

function resetExcelProgress() {
  clearExcelProcessingTimer();
  excelProgressWrap.hidden = true;
  excelProgressFill.style.width = "0%";
  excelProgressValue.textContent = "0%";
  excelProgressLabel.textContent = "等待开始";
}

function startExcelProcessingProgress() {
  clearExcelProcessingTimer();
  let current = 72;
  setExcelProgress(current, "正在处理 Excel，请稍候...");
  excelProcessingTimer = setInterval(() => {
    if (current >= 92) {
      clearExcelProcessingTimer();
      return;
    }
    current += current < 85 ? 3 : 1;
    setExcelProgress(current, "正在处理 Excel，请稍候...");
  }, 600);
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

function setAuditLoadingState(loading) {
  isAuditLoading = loading;
  auditBtn.disabled = loading;
  auditResetBtn.disabled = loading;
}

function formatMatchScore(score) {
  return typeof score === "number" ? score.toFixed(2) : "-";
}

function renderAuditPillList(container, values, emptyLabel) {
  if (!Array.isArray(values) || !values.length) {
    container.innerHTML = `<span class="audit-empty">${escapeHtml(emptyLabel)}</span>`;
    return;
  }
  container.innerHTML = values.map((value) => `<span class="pill">${escapeHtml(value)}</span>`).join("");
}

function renderAuditList(container, items, emptyLabel) {
  if (!items.length) {
    container.innerHTML = `<div class="audit-empty">${escapeHtml(emptyLabel)}</div>`;
    return;
  }
  container.innerHTML = items.join("");
}

function renderMappedObjects(mappedObjects) {
  const items = (Array.isArray(mappedObjects) ? mappedObjects : []).map(
    (item) => `
      <div class="audit-list-item">
        <strong>${escapeHtml(item.full_path || "")}</strong>
        <div class="audit-doc-meta">匹配分 ${escapeHtml(formatMatchScore(item.match_score))}</div>
      </div>
    `,
  );
  renderAuditList(auditMappedObjects, items, "未命中明确对象目录");
}

function renderReasons(reasons) {
  const items = (Array.isArray(reasons) ? reasons : []).map(
    (reason) => `<div class="audit-list-item">${escapeHtml(reason)}</div>`,
  );
  renderAuditList(auditReasons, items, "暂无明确原因说明");
}

function renderBasisDocuments(documents) {
  const items = (Array.isArray(documents) ? documents : []).map((document) => {
    const primary = document.display_name || document.title || "未命名依据";
    const secondary = [document.title, document.article, document.section]
      .filter((value) => value && value !== primary)
      .join(" / ");
    return `
      <div class="audit-list-item">
        <strong>${escapeHtml(primary)}</strong>
        <div class="audit-doc-meta">${escapeHtml(secondary || "暂无条款定位信息")}</div>
      </div>
    `;
  });
  renderAuditList(auditBasisDocuments, items, "暂无明确法规依据展示");
}

function renderAuditResult(data) {
  auditResult = data;
  auditError = "";
  auditResultBox.hidden = false;
  auditDisplayResult.textContent = data.display_result || "-";
  auditManualReview.hidden = data.manual_review_required !== true;
  auditProjectName.textContent = data.project_name || "";
  renderMappedObjects(data.mapped_objects);
  renderAuditPillList(auditTags, data.normalized_tags, "暂无标签");
  renderReasons(data.reasons);
  renderAuditPillList(auditReasonCodes, data.reason_codes, "暂无原因码");
  renderBasisDocuments(data.basis_documents);
  renderAuditPillList(auditPath, data.audit_path, "暂无审计路径");
}

function resetAuditResult() {
  auditResult = null;
  auditError = "";
  auditProjectText.value = "";
  auditResultBox.hidden = true;
  auditDisplayResult.textContent = "";
  auditManualReview.hidden = true;
  auditProjectName.textContent = "";
  auditMappedObjects.innerHTML = "";
  auditTags.innerHTML = "";
  auditReasons.innerHTML = "";
  auditReasonCodes.innerHTML = "";
  auditBasisDocuments.innerHTML = "";
  auditPath.innerHTML = "";
  setAuditLoadingState(false);
  setAuditStatus("输入工程描述后开始审计");
}

function renderMetrics(container, items) {
  container.innerHTML = items
    .map(
      (item) => `
        <div class="dashboard-metric ${item.tone || ""}">
          <div>
            <span class="label">${item.label}</span>
            <span class="value">${item.value}</span>
          </div>
          <span class="metric-icon" aria-hidden="true">${item.icon || ""}</span>
        </div>
      `,
    )
    .join("");
}

function renderSourceAnalysis(summary) {
  const items = [
    {
      label: "规则优先",
      value: summary.rule_method_count,
      color: "#3b82f6",
    },
    {
      label: "LLM辅助",
      value: summary.llm_method_count,
      color: "#8b5cf6",
    },
    {
      label: "体系外默认",
      value: summary.fallback_method_count,
      color: "#f59e0b",
    },
  ];
  const total = Math.max(
    1,
    items.reduce((sum, item) => sum + item.value, 0),
  );
  let offset = 0;
  const radius = 42;
  const circumference = 2 * Math.PI * radius;

  sourceDonut.innerHTML = `
    <circle class="donut-track" cx="60" cy="60" r="${radius}"></circle>
    ${items
      .map((item) => {
        const length = (item.value / total) * circumference;
        const circle = `
          <circle
            class="donut-segment"
            cx="60"
            cy="60"
            r="${radius}"
            stroke="${item.color}"
            stroke-dasharray="${length} ${circumference - length}"
            stroke-dashoffset="${-offset}"
          ></circle>
        `;
        offset += length;
        return circle;
      })
      .join("")}
    <circle class="donut-hole" cx="60" cy="60" r="28"></circle>
  `;

  sourceLegend.innerHTML = items
    .map((item) => {
      const percent = total ? ((item.value / total) * 100).toFixed(1) : "0.0";
      return `
        <div class="legend-item">
          <div class="legend-item-main">
            <span class="legend-dot" style="background:${item.color}"></span>
            <span class="legend-label">${item.label}</span>
          </div>
          <div class="legend-item-meta">
            <strong>${item.value}</strong>
            <span>(${percent}%)</span>
          </div>
        </div>
      `;
    })
    .join("");
}

function renderStructureBars(counts) {
  const items = [
    {
      label: "单一工程",
      value: counts.single_project,
      color: "#22c55e",
    },
    {
      label: "复合工程",
      value: counts.composite_project,
      color: "#f43f5e",
    },
    {
      label: "同域多系统",
      value: counts.multi_system_same_domain,
      color: "#fb923c",
    },
  ];
  const total = Math.max(
    1,
    items.reduce((sum, item) => sum + item.value, 0),
  );
  structureBars.innerHTML = items
    .map((item) => {
      const percent = (item.value / total) * 100;
      return `
        <div class="structure-row">
          <div class="structure-row-head">
            <span>${item.label}</span>
            <strong>${item.value}</strong>
          </div>
          <div class="structure-track">
            <div class="structure-fill" style="width:${percent}%;background:${item.color}"></div>
          </div>
        </div>
      `;
    })
    .join("");
}

function renderRiskMetrics(summary, counts) {
  const items = [
    {
      label: "需复核数据",
      value: summary.review_count,
      desc: "包含所有异常分类项",
      tone: "pink",
    },
    {
      label: "复合工程",
      value: summary.composite_count,
      desc: "结构复杂，需人工确认",
      tone: "yellow",
    },
    {
      label: "同域多系统",
      value: counts.multi_system_same_domain,
      desc: "跨系统关联风险",
      tone: "orange",
    },
    {
      label: "默认分类",
      value: summary.fallback_method_count,
      desc: "规则未覆盖项",
      tone: "slate",
    },
  ];
  riskChip.textContent = `共 ${summary.review_count} 项待处理`;
  riskMetrics.innerHTML = items
    .map(
      (item) => `
        <div class="risk-item ${item.tone}">
          <div class="risk-item-label">${item.label}</div>
          <div class="risk-item-value">${item.value}</div>
          <div class="risk-item-desc">${item.desc}</div>
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

function refreshFocusSamplesView() {
  const rows = getFilteredAndSortedFocusSamples();
  renderFocusSamples(rows);
  focusSummary.textContent = `当前显示 ${rows.length} / ${focusSamplesRaw.length} 条`;
  updateFocusSortIndicators();
}

function resetFocusFilters() {
  focusSearch.value = "";
  focusLevel1.value = "";
  focusMethod.value = "";
  focusStructure.value = "";
  focusCompositeOnly.checked = false;
  focusReviewOnly.checked = false;
  focusSortState.key = null;
  focusSortState.direction = "asc";
  refreshFocusSamplesView();
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

async function handleAuditSubmit() {
  const projectName = auditProjectText.value.trim();
  if (!projectName) {
    auditError = "请先输入工程描述";
    setAuditStatus(auditError);
    return;
  }
  if (isAuditLoading) {
    return;
  }

  setAuditLoadingState(true);
  auditError = "";
  auditResultBox.hidden = true;
  setAuditStatus("正在审计...");

  try {
    const response = await fetch(`${API_BASE}/api/audit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_name: projectName }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || "审计失败");
    }

    renderAuditResult(data);
    setAuditStatus("审计完成");
  } catch (error) {
    auditResult = null;
    auditError = error.message || "审计失败";
    auditResultBox.hidden = true;
    setAuditStatus(`错误：${auditError}`);
  } finally {
    setAuditLoadingState(false);
  }
}

async function handleExcelClassify() {
  if (!excelFile.files.length) {
    excelStatus.textContent = "请先选择 Excel 文件";
    return;
  }

  clearExcelProcessingTimer();
  excelStatus.textContent = "正在处理 Excel，请稍候...";
  setExcelProgress(8, "准备上传文件...");

  try {
    const formData = new FormData();
    formData.append("file", excelFile.files[0]);

    await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_BASE}/api/classify-excel`);
      xhr.responseType = "blob";

      xhr.upload.onprogress = (event) => {
        if (!event.lengthComputable) {
          setExcelProgress(35, "正在上传 Excel...");
          return;
        }
        const uploadPercent = 10 + (event.loaded / event.total) * 50;
        setExcelProgress(uploadPercent, "正在上传 Excel...");
      };

      xhr.upload.onload = () => {
        startExcelProcessingProgress();
      };

      xhr.onload = async () => {
        clearExcelProcessingTimer();
        if (xhr.status < 200 || xhr.status >= 300) {
          try {
            const errorText = await xhr.response.text();
            const error = JSON.parse(errorText);
            reject(new Error(error.detail || "Excel 处理失败"));
          } catch (_error) {
            reject(new Error("Excel 处理失败"));
          }
          return;
        }

        setExcelProgress(100, "处理完成，准备下载结果...");
        const blob = xhr.response;
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = excelFile.files[0].name.replace(/\.xlsx?$|\.xlsm$/i, "") + "_分类结果.xlsx";
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
        excelStatus.textContent = "Excel 处理完成，结果已开始下载";
        resolve();
      };

      xhr.onerror = () => {
        clearExcelProcessingTimer();
        reject(new Error("Excel 上传或处理失败"));
      };

      xhr.send(formData);
    });
  } catch (error) {
    setExcelProgress(0, "处理失败");
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
      { label: "总数据量", value: data.summary.total_records, icon: "▦", tone: "blue" },
      { label: "自动分类（规则）", value: data.summary.rule_method_count, icon: "✓", tone: "green" },
      { label: "LLM 辅助分类", value: data.summary.llm_method_count, icon: "≋", tone: "violet" },
      { label: "建议复核", value: data.summary.review_count, icon: "!", tone: "red" },
    ]);
    renderSourceAnalysis(data.summary);
    renderStructureBars(data.structure_counts);
    renderRiskMetrics(data.summary, data.structure_counts);

    renderCountTable(level1Table, data.level1_top, "暂无一级分类统计");
    renderCountTable(level2Table, data.level2_top, "暂无二级分类统计");
    focusSamplesRaw = (data.focus_samples || []).map(normalizeFocusSample);
    populateFocusFilterOptions(focusSamplesRaw);
    resetFocusFilters();

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
auditBtn.addEventListener("click", handleAuditSubmit);
auditResetBtn.addEventListener("click", resetAuditResult);
document.getElementById("excelBtn").addEventListener("click", handleExcelClassify);
document.getElementById("analyzeBtn").addEventListener("click", handleExcelAnalyze);
singleModeBtn.addEventListener("click", () => switchFeatureMode("single"));
excelModeBtn.addEventListener("click", () => switchFeatureMode("excel"));
projectText.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    handleSingleClassify();
  }
});
auditProjectText.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    handleAuditSubmit();
  }
});
excelFile.addEventListener("change", () => {
  if (excelFileName) {
    excelFileName.textContent = excelFile.files.length ? excelFile.files[0].name : "未选择任何文件";
  }
});
analysisFile.addEventListener("change", () => {
  analysisFileName.textContent = analysisFile.files.length ? analysisFile.files[0].name : "未选择文件";
});
focusSearch.addEventListener("input", refreshFocusSamplesView);
focusLevel1.addEventListener("change", refreshFocusSamplesView);
focusMethod.addEventListener("change", refreshFocusSamplesView);
focusStructure.addEventListener("change", refreshFocusSamplesView);
focusCompositeOnly.addEventListener("change", refreshFocusSamplesView);
focusReviewOnly.addEventListener("change", refreshFocusSamplesView);
focusCompositeOnly.addEventListener("input", refreshFocusSamplesView);
focusReviewOnly.addEventListener("input", refreshFocusSamplesView);
clearFocusFilters.addEventListener("click", resetFocusFilters);
document.querySelectorAll(".sort-trigger").forEach((button) => {
  button.addEventListener("click", () => {
    const key = button.dataset.sortKey;
    if (focusSortState.key === key) {
      focusSortState.direction = focusSortState.direction === "asc" ? "desc" : "asc";
    } else {
      focusSortState.key = key;
      focusSortState.direction = "asc";
    }
    refreshFocusSamplesView();
  });
});
updateFocusSortIndicators();
focusSummary.textContent = "当前显示 0 / 0 条";
resetExcelProgress();

switchFeatureMode("single");
resetSingleResult();
resetAuditResult();
