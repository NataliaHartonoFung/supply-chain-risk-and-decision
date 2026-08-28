import yaml
import time
import json
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", 
    datefmt="%Y-%m-%d %H:%M:%S"
    )
logger = logging.getLogger(__name__)

load_dotenv()

with open("config.yaml", "r") as file:
    CONFIG = yaml.safe_load(file)
    
@dataclass
class LLMExecutionStats:
    latency_sec: float = 0.0
    tokens_per_sec: float = 0.0
    json_parsable: bool = False
    vram_used_gb: Optional[float] = None
    input_tokens: int = 0   # ADD
    output_tokens: int = 0  # ADD

@dataclass
class AgentResult:
    score: float
    reasons: List[str]
    summary: str
    stats: LLMExecutionStats

@dataclass
class SupervisorDecision:
    risk_level: str
    global_score: float
    recommendations: List[str]
    negative_constraint_followed: bool
    stats: LLMExecutionStats

@dataclass
class FinalReport:
    row_index: int
    agent_outputs: Dict[str, AgentResult]
    supervisor_decision: SupervisorDecision

@dataclass
class EvaluationMetrics:
    strict_json_adherence_rate: float
    negative_constraint_adherence_rate: float
    avg_inference_latency_tps: float
    peak_vram_usage_gb: float
    total_requests: int
    total_worker_in_tokens: int
    total_worker_out_tokens: int
    total_sup_in_tokens: int
    total_sup_out_tokens: int

# Worker Agent
def _strip_fences(text: str) -> str:
    if text.startswith("```json"):
        return text[7:-3].strip()
    if text.startswith("```"):
        return text[3:-3].strip()
    return text

def _get_ollama_vram_gb() -> Optional[float]:
    """Reads VRAM usage from nvidia-smi. Returns None if unavailable."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            return round(int(result.stdout.strip().split("\n")[0]) / 1024, 3)
    except Exception:
        pass
    return None

class WorkerAgent:
    def __init__(self, agent_name: str, config: Dict[str, Any], llm: Any):
        self.name: str = agent_name
        self.features: List[str] = config['features']
        self.rules: List[Dict[str, Any]] = config['rules']
        self.scoring_config: List[Dict[str, Any]] = config['scoring']
        self.llm: Any = llm
        self.prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_template(config['prompt'])

    def _apply_rules(self, data_input: Dict[str, float]) -> List[str]:
        """Applies deterministic thresholds to detect anomalies and generate reasons."""
        reasons: List[str] = []
        for rule in self.rules:
            feature: str = rule['feature']
            value: float = data_input.get(feature)
            if value is None:
                continue

            threshold = rule['threshold']
            condition: str = rule['condition']
            triggered: bool = False

            if condition == 'greater_than' and value > threshold:
                triggered = True
            elif condition == 'less_than' and value < threshold:
                triggered = True
            elif condition == 'outside_range' and (value < threshold[0] or value > threshold[1]):
                triggered = True
            elif condition == 'equals' and value == threshold:
                triggered = True

            if triggered:
                reasons.append(rule['reason'].format(value=value, threshold=threshold))
                
        return reasons

    def _calculate_score(self, data_input: Dict[str, float]) -> float:
        """Calculates a deterministic 0.0 to 1.0 risk score based on the config."""  
        def _score_for(config_item: Dict[str, Any]) -> float:
            feature = config_item['feature']
            value = data_input.get(feature)
            if value is None:
                return 0.0
            calc_type = config_item['type']
            if calc_type == 'direct':
                return float(value)
            if calc_type == 'categorical':
                return float(config_item['mapping'].get(value, 0.0))
            if calc_type in ('normalize', 'inverse_normalize'):
                min_val, max_val = config_item['range']
                if max_val <= min_val:
                    return 0.0
                clamped = max(min_val, min(value, max_val))
                normalized = (clamped - min_val) / (max_val - min_val)
                return 1.0 - normalized if calc_type == 'inverse_normalize' else normalized
            if calc_type == 'deviation_normalize':
                safe_min, safe_max = config_item['safe_range']
                bound_min, bound_max = config_item['bounds']
                if value < safe_min:
                    return min(1.0, (safe_min - value) / (safe_min - bound_min))
                if value > safe_max:
                    return min(1.0, (value - safe_max) / (bound_max - safe_max))
            return 0.0
        return max((_score_for(c) for c in self.scoring_config), default=0.0)    

    def analyze(self, data_row: Dict[str, Any]) -> AgentResult:
        data_input: Dict[str, float] = {k: float(data_row[k]) for k in self.features if k in data_row}

        # Deterministic Processing
        reasons: List[str] = self._apply_rules(data_input)
        score: float = self._calculate_score(data_input)
        
        summary: str = "No anomalies detected."
        stats: LLMExecutionStats = LLMExecutionStats()

        # LLM Invocation & Benchmarking (Only if anomalies exist)
        if reasons:
            prompt = self.prompt_template.invoke({"reasons_list": "\n- ".join(reasons)})
            
            start_time: float = time.perf_counter()
            response: AIMessage = self.llm.invoke(prompt)
            stats.latency_sec = time.perf_counter() - start_time
            stats.vram_used_gb = _get_ollama_vram_gb()

            # Attempt to extract output tokens for TPS calculation
            try:
                in_tokens = response.response_metadata.get('prompt_eval_count', 0) or \
                    (response.usage_metadata or {}).get('input_tokens', 0)
                out_tokens = response.response_metadata.get('eval_count', 0) or \
                     (response.usage_metadata or {}).get('output_tokens', 0)
                stats.input_tokens = in_tokens
                stats.output_tokens = out_tokens
                if out_tokens and stats.latency_sec > 0:
                    stats.tokens_per_sec = out_tokens / stats.latency_sec
            except AttributeError:
                pass

            # Benchmark Reliability Pillar: Strict JSON Adherence
            raw_text: str = _strip_fences(response.content.strip())
            try:
                parsed_json = json.loads(raw_text)
                summary = parsed_json.get("summary", "JSON parsed but 'summary' key missing.")
                stats.json_parsable = True
            except json.JSONDecodeError:
                summary = f"[BENCHMARK FAILURE] LLM failed to output valid JSON. Raw output: {raw_text[:50]}..."
                stats.json_parsable = False

        return AgentResult(
            score=score,
            reasons=reasons,
            summary=summary,
            stats=stats
        )

# Supervisor Agent Definitions
def run_supervisor(
    agent_outputs: Dict[str, AgentResult],
    supervisor_llm: Any,
    config: Dict[str, Any]
) -> SupervisorDecision:
    """
    Aggregates worker risks, dynamically generates a global risk score, and 
    benchmarks the supervisor LLM's synthesis and constraint adherence.
    """
    supervisor_config: Dict[str, Any] = config['supervisor']
    
    # Calculate the weighted global score
    global_score: float = 0.0
    for agent_name, result in agent_outputs.items():
        weight: float = supervisor_config['agent_weights'].get(agent_name, 0.0)
        global_score += result.score * weight
        
    # Determine the risk level based on thresholds
    thresholds: Dict[str, float] = supervisor_config['risk_thresholds']
    risk_level: str = "LOW"
    if global_score >= thresholds['HIGH']:
        risk_level = "HIGH"
    elif global_score >= thresholds['MODERATE']:
        risk_level = "MODERATE"

    # Prepare the synthesized summary for the LLM
    report_summary: str = "\n".join(
        f"Agent '{name}' detected: {result.summary} (Score: {result.score:.2f})"
        for name, result in agent_outputs.items() if result.reasons
    ) or "No anomalies were detected by any agent."

    # Prompt Preparation & LLM Invocation
    prompt = supervisor_prompt_template.invoke({
        "risk_level": risk_level,
        "global_score": global_score,
        "report_summary": report_summary.strip()
    })

    stats: LLMExecutionStats = LLMExecutionStats()
    start_time: float = time.perf_counter()
    
    response: AIMessage = supervisor_llm.invoke(prompt)
    stats.latency_sec = time.perf_counter() - start_time

    # Hardware Metrics (TPS extraction)
    try:
        in_tokens = response.response_metadata.get('prompt_eval_count', 0) or \
                     (response.usage_metadata or {}).get('input_tokens', 0)
        out_tokens = response.response_metadata.get('eval_count', 0) or \
                     (response.usage_metadata or {}).get('output_tokens', 0)
        stats.input_tokens = in_tokens
        stats.output_tokens = out_tokens
        if out_tokens and stats.latency_sec > 0:
            stats.tokens_per_sec = out_tokens / stats.latency_sec
    except AttributeError:
        pass

    # Extract JSON and Evaluate Constraints
    raw_text: str = response.content.strip()
    recommendations: List[str] = []
    negative_constraint_followed: bool = raw_text.startswith("{") or raw_text.startswith("```json")
    clean_text: str = _strip_fences(raw_text)

    # Benchmark Reliability Pillar: Strict JSON Adherence
    try:
        parsed_json = json.loads(clean_text)
        recommendations = parsed_json.get("recommendations", [])
        if not isinstance(recommendations, list):
            recommendations = [str(recommendations)]
        stats.json_parsable = True
        
        # Evaluate "Prefix critical... with IMMEDIATE:" constraint
        if risk_level == "HIGH" and recommendations:
            has_immediate_prefix = any(rec.strip().startswith("IMMEDIATE:") for rec in recommendations)
            if not has_immediate_prefix:
                negative_constraint_followed = False
                
    except json.JSONDecodeError:
        recommendations = [f"[BENCHMARK FAILURE] Failed to parse Supervisor JSON. Raw output: {raw_text[:50]}..."]
        stats.json_parsable = False
        negative_constraint_followed = False

    return SupervisorDecision(
        risk_level=risk_level,
        global_score=global_score,
        recommendations=recommendations,
        negative_constraint_followed=negative_constraint_followed,
        stats=stats
    )

# Initialize LLM-SLM
logger.info(f"Initializing Worker Model (Local Edge): {CONFIG['worker_model']}")
llm_worker = ChatOpenAI(
    model=CONFIG["worker_model"],
    temperature=0.0,
    timeout=15.0,
    max_retries=3
)

logger.info(f"Initializing Supervisor Model (Cloud Baseline): {CONFIG['supervisor_model']}")
llm_supervisor = ChatOpenAI(
    model=CONFIG["supervisor_model"],
    temperature=0.0,
    timeout=15.0,
    max_retries=3
)

supervisor_prompt_template = ChatPromptTemplate.from_template(CONFIG['supervisor']['prompt'])

# Initialize Worker Agents
agents: Dict[str, WorkerAgent] = {}
for agent_name, agent_config in CONFIG.items():
    if agent_name.startswith("agent_"):
        agents[agent_name] = WorkerAgent(
            agent_name=agent_name,
            config=agent_config,
            llm=llm_worker
        )

logger.info(f"Successfully initialized {len(agents)} worker agents.")

# Workflow & Evaluation Logic
def process_row(
    row_index: int,
    data_row: Dict[str, Any],
    agents_dict: Dict[str, WorkerAgent],
    supervisor: Any,
    config: Dict[str, Any]
) -> FinalReport:
    """
    Orchestrates the processing of a single data row through all agents and the supervisor.
    """
    agent_outputs: Dict[str, AgentResult] = {}
    
    # Run all worker agents serially
    for agent_name, agent in agents_dict.items():
        agent_outputs[agent_name] = agent.analyze(data_row)

    # Supervisor aggregate results and make a final decision
    supervisor_decision: SupervisorDecision = run_supervisor(
        agent_outputs=agent_outputs, 
        supervisor_llm=supervisor, 
        config=config
    )

    # Create final structured report
    return FinalReport(
        row_index=row_index,
        agent_outputs=agent_outputs,
        supervisor_decision=supervisor_decision
    )

def evaluate_single_row_metrics(final_report: FinalReport) -> Dict[str, Any]:
    row_metrics: Dict[str, Any] = {
        "json_successes": 0,
        "total_llm_calls": 0,
        "negative_constraint_followed": int(final_report.supervisor_decision.negative_constraint_followed),
        "tps_measurements": [],
        "worker_in": 0,
        "worker_out": 0,
        "sup_in": 0,
        "sup_out": 0
    }
    sup_stats = final_report.supervisor_decision.stats
    row_metrics["total_llm_calls"] += 1
    row_metrics["sup_in"] += sup_stats.input_tokens
    row_metrics["sup_out"] += sup_stats.output_tokens
    if sup_stats.json_parsable:
        row_metrics["json_successes"] += 1
    if sup_stats.tokens_per_sec > 0:
        row_metrics["tps_measurements"].append(sup_stats.tokens_per_sec)

    for agent_name, pred_result in final_report.agent_outputs.items():
        if pred_result.reasons:
            worker_stats = pred_result.stats
            row_metrics["total_llm_calls"] += 1
            row_metrics["worker_in"] += worker_stats.input_tokens
            row_metrics["worker_out"] += worker_stats.output_tokens
            if worker_stats.json_parsable:
                row_metrics["json_successes"] += 1
            if worker_stats.tokens_per_sec > 0:
                row_metrics["tps_measurements"].append(worker_stats.tokens_per_sec)
    return row_metrics

def run_benchmark(start_idx: int = 0, end_idx: Optional[int] = None) -> Tuple[List[FinalReport], Optional[EvaluationMetrics]]:
    """
    Executes the Edge SLM benchmarking pipeline across a sliced batch of the dataset.
    """
    logger.info("Starting SLM benchmarking process...")
    
    data_path: str = CONFIG.get("benchmark_data_path", "data/phase3.csv")
    
    data_df: pd.DataFrame = pd.read_csv(data_path)
    logger.info(f"Loaded {len(data_df)} total rows from {data_path}.")

    # Slice dataframe
    sliced_df = data_df.iloc[start_idx:end_idx]
    logger.info(f"Processing batch slice: Rows {start_idx} to {end_idx if end_idx is not None else 'end'} ({len(sliced_df)} rows).")

    all_reports: List[FinalReport] = []
    all_raw_metrics: List[Dict[str, Any]] = []
    
    print("Start benchmark loop...")
    start_time = time.time()
    
    # Processing & Evaluation Loop
    for index, row in sliced_df.iterrows():
        # Process pipeline
        final_report: FinalReport = process_row(
            row_index=index,
            data_row=row.to_dict(),
            agents_dict=agents,
            supervisor=llm_supervisor,
            config=CONFIG
        )
        all_reports.append(final_report)
        
        row_metrics: Dict[str, Any] = evaluate_single_row_metrics(final_report)
        all_raw_metrics.append(row_metrics)
        
    # Aggregate Metrics into the 3 Pillars
    if not all_raw_metrics:
        logger.warning("No rows were benchmarked. Check your dataset slicing.")
        return all_reports, None

    total_llm_calls: int = sum(m["total_llm_calls"] for m in all_raw_metrics)
    total_json_success: int = sum(m["json_successes"] for m in all_raw_metrics)
    total_neg_constraints: int = sum(m["negative_constraint_followed"] for m in all_raw_metrics)
    all_tps: List[float] = [tps for m in all_raw_metrics for tps in m["tps_measurements"]]

    total_worker_in: int = sum(m["worker_in"] for m in all_raw_metrics)
    total_worker_out: int = sum(m["worker_out"] for m in all_raw_metrics)
    total_sup_in: int = sum(m["sup_in"] for m in all_raw_metrics)
    total_sup_out: int = sum(m["sup_out"] for m in all_raw_metrics)
    
    final_metrics = EvaluationMetrics(
        strict_json_adherence_rate=(total_json_success / total_llm_calls) if total_llm_calls > 0 else 0.0,
        negative_constraint_adherence_rate=(total_neg_constraints / len(all_raw_metrics)),
        avg_inference_latency_tps=float(sum(all_tps) / len(all_tps)) if all_tps else 0.0,
        peak_vram_usage_gb=max(
            (r.stats.vram_used_gb for report in all_reports
             for r in report.agent_outputs.values()
             if r.stats.vram_used_gb is not None),
            default=0.0
        ),
        total_requests=len(all_reports),
        total_worker_in_tokens=total_worker_in,
        total_worker_out_tokens=total_worker_out,
        total_sup_in_tokens=total_sup_in,
        total_sup_out_tokens=total_sup_out,
    )
    
    end_time = time.time()
    total_processing_time = end_time - start_time
    
    print("\n" + "="*70)
    print(f"   SLM BENCHMARK RESULTS: {CONFIG.get('worker_model', 'Unknown Model')}")
    print("="*70)
    actual_end = (end_idx - 1) if end_idx is not None else len(data_df) - 1
    print(f"Batch Range:                      Rows {start_idx} - {actual_end}")
    print(f"Total Pipeline Executions:        {len(all_reports)}")
    print(f"Total LLM Calls (Supervisor):     {len(all_reports)}")
    print(f"Total SLM Calls (Workers):        {total_llm_calls - len(all_reports)}")
    print(f"Total System Invocations:         {total_llm_calls}")
    print("\n--- PILLAR 1: SYSTEM RELIABILITY ---")
    print(f" Strict JSON Adherence:           {final_metrics.strict_json_adherence_rate:.2%}")
    print(f" Negative Constraint Adherence:   {final_metrics.negative_constraint_adherence_rate:.2%}")
    print("\n--- COST TRACKING ---")
    print(f" Total Requests:                  {final_metrics.total_requests}")
    print(f" Worker Input Tokens:             {final_metrics.total_worker_in_tokens}")
    print(f" Worker Output Tokens:            {final_metrics.total_worker_out_tokens}")
    print(f" Supervisor Input Tokens:         {final_metrics.total_sup_in_tokens}")
    print(f" Supervisor Output Tokens:        {final_metrics.total_sup_out_tokens}")
    print("\n--- EDGE EFFICIENCY ---")
    print(f" Average Generation Speed (TPS):  {final_metrics.avg_inference_latency_tps:.1f} tokens/sec")
    print(f" Peak VRAM Usage:                 {final_metrics.peak_vram_usage_gb:.2f} GB")
    print(f" Total Processing Time:           {total_processing_time:.2f} seconds")
    
    current_model = CONFIG["worker_model"]
    safe_model_name = current_model.replace(":", "-").replace("/", "-")
    output_filename = f"run_{safe_model_name}_rows_{start_idx}_to_{end_idx}.json"

    reports_list_dict = [asdict(report) for report in all_reports]

    run_payload = {
        "metadata": {
            "worker_model": current_model,
            "supervisor_model": CONFIG["supervisor_model"],
            "start_idx": start_idx,
            "end_idx": end_idx,
            "total_processing_time_sec": round(total_processing_time, 2)
        },
        "system_metrics": asdict(final_metrics),
        "generated_reports": reports_list_dict
    }

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(run_payload, f, indent=2, ensure_ascii=False)
        
    print(f"\nRun complete, exported result to: {output_filename}")
    
    return all_reports, final_metrics

if __name__ == "__main__":
    generated_reports, benchmarking_results = run_benchmark(start_idx=0, end_idx=1000)