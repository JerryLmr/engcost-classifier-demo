# 维修资金审计系统设计说明

## 1. 项目目标

在现有 `engcost-classifier-demo` 项目基础上，新增“维修资金审计链路”。

系统目标不是做一个纯 LLM 问答系统，而是构建一个：

- 规则引擎主导
- RAG 知识检索辅助
- 可持续维护
- 可解释
- 可扩展

的维修资金审计系统。

---

## 2. 总体设计原则

### 2.1 一期原则

一期先实现：

- 对象识别
- 标签映射
- 规则裁决
- 审计结果输出

一期不实现：

- 完整登录权限系统
- 完整知识库管理后台
- 复杂前端改造
- LLM 自动裁决

---

### 2.2 二期预留

二期预留：

- 法规/流程文件上传
- 文档切片
- PostgreSQL / pgvector 存储
- RAG 检索依据
- 用户权限与发布机制

---

### 2.3 核心原则

- 规则引擎决定怎么判
- RAG 负责检索依据和承接新增法规
- LLM 仅用于抽取与解释，不负责最终裁决

---

## 3. 总体架构

输入层  
→ 解析层  
→ 对象识别层  
→ 标签映射层  
→ 规则引擎层  
→ 结果输出层  
→ 展示与解释层  

并行知识链：

法规/流程文件上传  
→ 文档解析  
→ 切片  
→ 向量化  
→ PostgreSQL / pgvector  
→ 审计时检索相关依据  

---

## 4. 分层说明

---

### 4.1 输入层

支持输入：

- 一句话项目描述
- 后续扩展：PDF / Word / Excel / 图片

示例：

- 电梯曳引机更换
- 外墙渗漏维修
- 树木修剪
- 楼道窗户玻璃维修
- 防盗门油漆、人行道闸、绿化补种、电子门禁

---

### 4.2 解析层

职责：

- 文本清洗
- 拆分复合工程
- 提取结构化字段

说明：

- 一期以规则为主
- LLM 如使用，仅用于抽取，不参与裁决

---

### 4.3 对象识别层

对应：

- `backend/config/object_catalog.json`
- `backend/services/mapping_service.py`

职责：

- 将输入文本映射为标准维修对象
- 输出对象列表

#### object_catalog 结构

顶层：

- version
- catalog_name
- catalog_stage
- updated_at
- total_records
- items

每条对象：

- id
- level_1
- level_2
- level_3
- full_path
- status

说明：

- 不包含 source_file / version_note / sheet / seq / menu / allocation_scope / remark

---

### 4.4 标签映射层

对应：

- `backend/config/rule_mapping.json`

职责：

- 将对象映射为审计标签

标签体系：

- repairable_object
- shared_part
- shared_facility
- daily_service_exclusion
- daily_greening
- cleaning_sanitation
- inspection_testing
- new_construction
- emergency_scope
- gray_case
- multi_project
- unknown

---

### 4.5 规则引擎层

对应：

- `backend/config/rule_engine.json`
- `backend/services/audit_service.py`

职责：

- 基于标签裁决结果

内部结果：

- compliant
- non_compliant
- need_supplement
- manual_review

对外展示：

- compliant → 初步符合
- non_compliant → 疑似违规
- need_supplement → 需补充材料
- manual_review → 建议人工复核

流程：

- input_check_flow
- exclusion_flow
- manual_review_flow
- gray_case_review_flow
- emergency_flow
- normal_flow

---

### 4.6 灰区机制

gray_case 是标签，不是结果。

#### 强灰区

→ manual_review_flow  
→ manual_review  

#### 弱灰区

→ gray_case_review_flow  

结果：

- need_supplement
- manual_review
- 或转 normal_flow

---

### 4.7 复合工程处理

输入如：

防盗门油漆、人行道闸、绿化补种、电子门禁

处理：

- 必须拆分
- 多对象 → MULTI_PROJECT / CROSS_DOMAIN_PROJECT
- 禁止只取一个关键词判断

---

### 4.8 输出层

对应：

- `backend/config/output_schema.json`

输出字段：

- project_name
- matched_object_ids
- normalized_tags
- overall_result
- display_result
- reason_codes
- reasons
- basis_documents
- missing_items
- audit_path
- manual_review_required

---

## 5. RAG 与 PostgreSQL 设计（预留）

### 5.1 目标

支持：

- 新法规上传
- 自动切片
- 向量存储
- 审计时检索依据

---

### 5.2 定位

RAG 用于：

- 展示依据
- 检索条文
- 支撑维护

不用于：

- 直接裁决

---

### 5.3 数据结构

#### 文档表

- doc_id
- 标题
- 来源
- 文号
- 发布时间
- 地域范围
- 状态

#### 切片表

- chunk_id
- doc_id
- chunk_text
- chapter
- article
- tags

#### 向量

- embedding（pgvector）

---

### 5.4 更新机制

流程：

上传  
→ 切片  
→ 待审核  
→ 发布  
→ 生效  

---

## 6. 权限设计（预留）

### 角色

#### admin

- 上传法规
- 发布规则
- 管理系统

#### rule_editor

- 编辑规则草稿
- 维护目录

#### auditor

- 使用审计
- 查看结果

---

### 原则

- 审计人员不能改规则
- 上传法规不能直接生效
- 必须有发布机制

---

## 7. 项目迁移原则

### 保留

- FastAPI
- 前端
- classify 接口

### 降级

- taxonomy.json → legacy

### 新标准

- object_catalog
- rule_mapping
- rule_engine
- output_schema

---

## 8. 一期开发范围

实现：

- mapping_service
- audit_service
- audit_routes
- test_audit_engine

打通：

输入 → 对象 → 标签 → 审计 → 输出

不实现：

- LLM裁决
- 知识库系统
- 权限系统
- 前端重构

---

## 9. 系统定位

这是一个：

- 规则驱动
- RAG辅助
- 可解释
- 可扩展

的审计系统。