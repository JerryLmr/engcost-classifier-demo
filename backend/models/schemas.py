from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    project_name: str
    level1: str
    level2: str
    level3: str
    method: str
    confidence: str
    match_type: str
    needs_review: bool
    candidate_ids: List[str]
    candidate_labels: List[str] = Field(default_factory=list)
    reason: str


class CountItem(BaseModel):
    name: str
    count: int


class AnalyzeSummary(BaseModel):
    total_records: int
    rule_method_count: int
    llm_method_count: int
    fallback_method_count: int
    review_count: int


class MatchTypeCounts(BaseModel):
    single: int
    cross_domain: int
    same_domain_multi_item: int
    low_confidence: int
    llm_fallback: int
    fallback: int


class FocusSample(BaseModel):
    source_file: str
    row_num: int
    project_name: str
    level1: str
    level2: str
    level3: str
    method: str
    confidence: str
    match_type: str
    reason: str
    needs_review: bool
    candidate_ids: List[str] = Field(default_factory=list)
    candidate_labels: List[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    summary: AnalyzeSummary
    match_type_counts: MatchTypeCounts
    level1_top: List[CountItem]
    level2_top: List[CountItem]
    focus_samples: List[FocusSample]


class ScopeFacts(BaseModel):
    is_common_part: Optional[bool] = None
    is_common_facility: Optional[bool] = None
    is_private_part: Optional[bool] = None
    is_property_service_scope: Optional[bool] = None
    is_public_window: Optional[bool] = None
    is_original_standard_restoration: Optional[bool] = None
    is_function_restoration: Optional[bool] = None
    is_upgrade: Optional[bool] = None
    is_relocation: Optional[bool] = None
    is_function_improvement: Optional[bool] = None
    is_capacity_expansion: Optional[bool] = None


class ProcessFacts(BaseModel):
    has_vote: Optional[bool] = None
    vote_passed: Optional[bool] = None
    has_announcement: Optional[bool] = None
    announcement_completed: Optional[bool] = None
    has_budget_review: Optional[bool] = None
    budget_review_required: Optional[bool] = None
    has_contract: Optional[bool] = None
    procurement_method: Optional[str] = None
    selected_vendor_count: Optional[int] = None


class DocumentFacts(BaseModel):
    has_site_photos: Optional[bool] = None
    has_rectification_notice: Optional[bool] = None
    has_damage_assessment: Optional[bool] = None
    has_construction_plan: Optional[bool] = None
    has_completion_report: Optional[bool] = None
    has_acceptance_record: Optional[bool] = None
    has_settlement_report: Optional[bool] = None
    has_invoice: Optional[bool] = None
    has_payment_proof: Optional[bool] = None
    has_owner_signature_sheet: Optional[bool] = None
    has_emergency_proof: Optional[bool] = None


class TimelineFacts(BaseModel):
    application_date: Optional[str] = None
    vote_date: Optional[str] = None
    announcement_date: Optional[str] = None
    budget_review_date: Optional[str] = None
    contract_sign_date: Optional[str] = None
    construction_start_date: Optional[str] = None
    construction_end_date: Optional[str] = None
    acceptance_date: Optional[str] = None
    invoice_date: Optional[str] = None
    payment_date: Optional[str] = None
    emergency_report_date: Optional[str] = None


class AmountFacts(BaseModel):
    amount: Optional[float] = None
    budget_amount: Optional[float] = None
    approved_amount: Optional[float] = None
    settlement_amount: Optional[float] = None
    invoice_amount: Optional[float] = None
    quoted_vendor_count: Optional[int] = None
    unit_price_reference_available: Optional[bool] = None
    amount_deviation_ratio: Optional[float] = None


class EmergencyFacts(BaseModel):
    is_emergency: Optional[bool] = None
    emergency_reason: Optional[str] = None
    emergency_hazard_type: Optional[str] = None
    temporary_measure_taken: Optional[bool] = None
    post_emergency_vote_required: Optional[bool] = None
    post_emergency_vote_completed: Optional[bool] = None


class GrayCaseFacts(BaseModel):
    repair_scope_description: Optional[str] = None
    damage_description: Optional[str] = None
    repair_extent: Optional[str] = None
    location_detail: Optional[str] = None
    damage_count: Optional[int] = None
    gray_case_evidence_complete: Optional[bool] = None


class SourceDocument(BaseModel):
    document_type: str
    file_name: Optional[str] = None
    document_id: Optional[str] = None
    page_hint: Optional[str] = None


class DocumentParseContext(BaseModel):
    source_documents: List[SourceDocument] = Field(default_factory=list)
    extracted_document_fields: Dict[str, Any] = Field(default_factory=dict)
    field_confidence_map: Dict[str, float] = Field(default_factory=dict)


class AuditRequest(BaseModel):
    project_name: str
    project_desc: Optional[str] = None
    facts: Optional[Dict[str, Any]] = None
    is_common_part: Optional[bool] = None
    is_common_facility: Optional[bool] = None
    is_emergency: Optional[bool] = None
    is_out_of_warranty: Optional[bool] = None
    amount: Optional[float] = None
    has_vote: Optional[bool] = None
    has_announcement: Optional[bool] = None
    has_budget_review: Optional[bool] = None
    has_site_photos: Optional[bool] = None
    has_rectification_notice: Optional[bool] = None
    has_construction_plan: Optional[bool] = None
    has_completion_report: Optional[bool] = None
    has_invoice: Optional[bool] = None
    is_public_window: Optional[bool] = None
    location_detail: Optional[str] = None
    damage_description: Optional[str] = None
    repair_scope_description: Optional[str] = None
    repair_extent: Optional[str] = None
    is_property_service_scope: Optional[bool] = None
    gray_case_evidence_complete: Optional[bool] = None
    is_function_restoration: Optional[bool] = None
    is_upgrade: Optional[bool] = None
    is_relocation: Optional[bool] = None
    is_private_part: Optional[bool] = None
    scope_facts: Optional[ScopeFacts] = None
    process_facts: Optional[ProcessFacts] = None
    document_facts: Optional[DocumentFacts] = None
    timeline_facts: Optional[TimelineFacts] = None
    amount_facts: Optional[AmountFacts] = None
    emergency_facts: Optional[EmergencyFacts] = None
    gray_case_facts: Optional[GrayCaseFacts] = None
    document_parse_context: Optional[DocumentParseContext] = None


class MappedObject(BaseModel):
    id: int
    full_path: str
    match_score: float
    match_method: str


class BasisDocument(BaseModel):
    display_name: Optional[str] = None
    source_type: Optional[str] = None
    title: Optional[str] = None
    issuer: Optional[str] = None
    document_no: Optional[str] = None
    article: Optional[str] = None
    section: Optional[str] = None


class AuditSubResult(BaseModel):
    applicable: bool
    result: Optional[str] = None
    display_result: Optional[str] = None
    reason_codes: List[str] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)
    missing_items: List[str] = Field(default_factory=list)
    basis_documents: List[BasisDocument] = Field(default_factory=list)
    audit_path: List[str] = Field(default_factory=list)
    facts_used: List[str] = Field(default_factory=list)


class AuditSubAudits(BaseModel):
    scope_audit: AuditSubResult
    process_audit: AuditSubResult
    document_completeness_audit: AuditSubResult
    timeline_audit: AuditSubResult
    amount_audit: AuditSubResult
    emergency_audit: AuditSubResult


class SummaryConclusion(BaseModel):
    type: str = "needs_more_info"
    scope_prelim_pass: bool = False
    conflict_detected: bool = False
    gap_categories: List[str] = Field(default_factory=list)
    primary_message: str = ""
    display_summary: str = ""


class DebugGapItem(BaseModel):
    category: str
    rule_id: Optional[str] = None
    stage: Optional[str] = None
    result: Optional[str] = None
    reason_codes: List[str] = Field(default_factory=list)


class DebugDecision(BaseModel):
    blocking_gaps: List[DebugGapItem] = Field(default_factory=list)
    advisory_gaps: List[DebugGapItem] = Field(default_factory=list)
    selected_overall_signal: Optional[str] = None


class AuditResponse(BaseModel):
    project_name: str
    mapped_objects: List[MappedObject] = Field(default_factory=list)
    matched_object_ids: List[int] = Field(default_factory=list)
    normalized_tags: List[str] = Field(default_factory=list)
    overall_result: str
    display_result: str
    reason_codes: List[str] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)
    basis_documents: List[BasisDocument] = Field(default_factory=list)
    missing_items: List[str] = Field(default_factory=list)
    audit_path: List[str] = Field(default_factory=list)
    manual_review_required: bool
    sub_audits: AuditSubAudits
    document_extraction_targets: Dict[str, List[str]] = Field(default_factory=dict)
    summary_conclusion: SummaryConclusion = Field(default_factory=SummaryConclusion)
    display_summary: str = ""
    debug_decision: Optional[DebugDecision] = None
