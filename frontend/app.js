const API_BASE = window.API_BASE || "http://127.0.0.1:8000";

const projectText = document.getElementById("projectText");
const resultBox = document.getElementById("resultBox");
const singleStatus = document.getElementById("singleStatus");
const excelStatus = document.getElementById("excelStatus");
const excelFile = document.getElementById("excelFile");

function setSingleStatus(message) {
  singleStatus.innerHTML = message;
}

function resetSingleResult() {
  projectText.value = "";
  resultBox.hidden = true;
  setSingleStatus(`后端地址：<code>${API_BASE}</code>`);
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
    document.getElementById("rMethod").textContent = data.method;
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

document.querySelectorAll(".example").forEach((element) => {
  element.addEventListener("click", () => {
    projectText.value = element.textContent.trim();
  });
});

document.getElementById("classifyBtn").addEventListener("click", handleSingleClassify);
document.getElementById("clearBtn").addEventListener("click", resetSingleResult);
document.getElementById("excelBtn").addEventListener("click", handleExcelClassify);

resetSingleResult();
