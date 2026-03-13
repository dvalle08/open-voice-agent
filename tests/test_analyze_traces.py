from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_module():
    path = Path(__file__).resolve().parents[1] / "dev" / "analyze_traces.py"
    spec = importlib.util.spec_from_file_location("dev_analyze_traces", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analyze_traces = _load_module()


def _observation(
    name: str,
    *,
    attributes: dict[str, Any] | None = None,
    start_time: str = "2026-03-13T00:00:00.000Z",
    end_time: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": f"{name}-{start_time}",
        "name": name,
        "startTime": start_time,
        "createdAt": start_time,
        "metadata": {"attributes": attributes or {}},
    }
    if end_time is not None:
        payload["endTime"] = end_time
    return payload


def test_extract_turn_metrics_prefers_phase_observations_for_simple_turn() -> None:
    detail = {
        "id": "trace-simple",
        "name": "turn",
        "sessionId": "session-1",
        "metadata": {
            "attributes": {
                "tool.execution_count": "0",
                "latency_ms.eou_delay": "540.5",
                "latency_ms.llm_ttft": "600.0",
                "latency_ms.tts_ttfb": "200.0",
                "latency_ms.perceived_first_audio": "1300.4",
            }
        },
        "observations": [
            _observation(
                "LLMMetrics",
                attributes={
                    "phase_index": "1",
                    "phase_call_index": "2",
                    "ttft_ms": "640.0",
                },
                start_time="2026-03-13T00:00:02.000Z",
            ),
            _observation(
                "LLMMetrics",
                attributes={
                    "phase_index": "1",
                    "phase_call_index": "1",
                    "ttft_ms": "510.0",
                },
                start_time="2026-03-13T00:00:01.000Z",
            ),
            _observation(
                "TTSMetrics",
                attributes={
                    "phase_index": 1,
                    "phase_call_index": 1,
                    "ttfb_ms": 215.5,
                },
                start_time="2026-03-13T00:00:03.000Z",
            ),
        ],
    }

    extracted = analyze_traces.extract_turn_metrics(detail)

    assert extracted.has_tools is False
    assert extracted.eou_delay_ms == 540.5
    assert extracted.llm_ttft_by_phase[1] == 510.0
    assert extracted.tts_ttfb_by_phase[1] == 215.5
    assert extracted.perceived_latency_first_audio_ms == 1300.4
    assert extracted.perceived_latency_second_audio_ms is None


def test_extract_turn_metrics_uses_root_fallbacks_for_missing_phase_and_tool_values() -> None:
    detail = {
        "id": "trace-tool-fallback",
        "name": "turn",
        "sessionId": "session-2",
        "metadata": {
            "attributes": {
                "tool.execution_count": "1",
                "latency_ms.eou_delay": "550.0",
                "latency_ms.llm_ttft": "800.0",
                "latency_ms.tts_ttfb": "180.0",
                "latency_ms.tool_calls_total": "425.0",
                "latency_ms.perceived_first_audio": "1450.0",
                "latency_ms.perceived_second_audio": "2100.0",
            }
        },
        "observations": [
            _observation(
                "LLMMetrics",
                attributes={"phase_index": "2", "phase_call_index": "1", "ttft_ms": "610.0"},
                start_time="2026-03-13T00:00:02.000Z",
            ),
            _observation(
                "TTSMetrics",
                attributes={"phase_index": "2", "phase_call_index": "1", "ttfb_ms": "205.0"},
                start_time="2026-03-13T00:00:03.000Z",
            ),
        ],
    }

    extracted = analyze_traces.extract_turn_metrics(detail)

    assert extracted.has_tools is True
    assert extracted.tool_round_count == 1
    assert extracted.llm_ttft_by_phase[1] == 800.0
    assert extracted.tts_ttfb_by_phase[1] == 180.0
    assert extracted.tool_execution_by_round[1] == 425.0
    assert extracted.llm_ttft_by_phase[2] == 610.0
    assert extracted.tts_ttfb_by_phase[2] == 205.0


def test_build_tool_rows_expands_for_multiple_tool_rounds() -> None:
    first = analyze_traces.TurnTraceMetrics(
        trace_id="trace-1",
        session_id="session-1",
        tool_round_count=2,
        eou_delay_ms=500.0,
        perceived_latency_first_audio_ms=1500.0,
        perceived_latency_second_audio_ms=2500.0,
        llm_ttft_by_phase={1: 800.0, 2: 600.0, 3: 550.0},
        tts_ttfb_by_phase={1: 180.0, 2: 210.0, 3: 205.0},
        tool_execution_by_round={1: 300.0, 2: 700.0},
    )
    second = analyze_traces.TurnTraceMetrics(
        trace_id="trace-2",
        session_id="session-1",
        tool_round_count=1,
        eou_delay_ms=700.0,
        perceived_latency_first_audio_ms=1700.0,
        perceived_latency_second_audio_ms=2100.0,
        llm_ttft_by_phase={1: 900.0, 2: 500.0},
        tts_ttfb_by_phase={1: 190.0, 2: 220.0},
        tool_execution_by_round={1: 400.0},
    )

    rows = analyze_traces.build_tool_rows([first, second])
    labels = [row.stage for row in rows]

    assert labels == [
        "eou_delay_ms",
        "llm_ttft (phase 1)_ms",
        "tts_ttfb (phase 1)_ms",
        "tool_execution_1_ms",
        "llm_ttft (phase 2)_ms",
        "tts_ttfb (phase 2)_ms",
        "tool_execution_2_ms",
        "llm_ttft (phase 3)_ms",
        "tts_ttfb (phase 3)_ms",
        "perceived_latency_first_audio_ms",
        "perceived_latency_second_audio_ms",
    ]

    row_map = {row.stage: row.summary for row in rows}
    assert row_map["tool_execution_1_ms"].p50 == 350.0
    assert row_map["tool_execution_2_ms"].p50 == 700.0
    assert row_map["tool_execution_2_ms"].p95 == 700.0
    assert row_map["llm_ttft (phase 3)_ms"].p50 == 550.0
    assert row_map["tts_ttfb (phase 3)_ms"].p95 == 205.0


def test_render_report_outputs_markdown_and_na_for_missing_values() -> None:
    report = analyze_traces.AnalysisReport(
        session_ids=["session-a"],
        trace_count=1,
        simple_traces=[
            analyze_traces.TurnTraceMetrics(
                trace_id="trace-1",
                session_id="session-a",
                tool_round_count=0,
                eou_delay_ms=500.0,
                perceived_latency_first_audio_ms=1200.0,
                perceived_latency_second_audio_ms=None,
                llm_ttft_by_phase={1: None},
                tts_ttfb_by_phase={1: 200.0},
                tool_execution_by_round={},
            )
        ],
        tool_traces=[],
        skipped=analyze_traces.Counter(),
    )

    rendered = analyze_traces.render_report(report)

    assert "# Langfuse Session Analysis" in rendered
    assert "Sessions: `session-a`" in rendered
    assert "| llm_ttft_ms | n/a | n/a |" in rendered
    assert "| tool_execution_1_ms | n/a | n/a |" in rendered
