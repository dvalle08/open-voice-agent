from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


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


def _turn(
    trace_id: str,
    *,
    session_id: str | None = "session-1",
    tool_round_count: int = 0,
    eou_delay_ms: float | None = None,
    perceived_latency_first_audio_ms: float | None = None,
    perceived_latency_second_audio_ms: float | None = None,
    llm_ttft_by_phase: dict[int, float | None] | None = None,
    tts_ttfb_by_phase: dict[int, float | None] | None = None,
    tool_execution_by_round: dict[int, float | None] | None = None,
) -> Any:
    return analyze_traces.TurnTraceMetrics(
        trace_id=trace_id,
        session_id=session_id,
        tool_round_count=tool_round_count,
        eou_delay_ms=eou_delay_ms,
        perceived_latency_first_audio_ms=perceived_latency_first_audio_ms,
        perceived_latency_second_audio_ms=perceived_latency_second_audio_ms,
        llm_ttft_by_phase=llm_ttft_by_phase or {1: None},
        tts_ttfb_by_phase=tts_ttfb_by_phase or {1: None},
        tool_execution_by_round=tool_execution_by_round or {},
    )


def _report(
    *,
    session_ids: list[str],
    simple_traces: list[Any] | None = None,
    tool_traces: list[Any] | None = None,
    skipped: dict[str, int] | None = None,
    trace_count: int | None = None,
) -> Any:
    simple = simple_traces or []
    tool = tool_traces or []
    return analyze_traces.AnalysisReport(
        session_ids=session_ids,
        trace_count=trace_count if trace_count is not None else len(simple) + len(tool),
        simple_traces=simple,
        tool_traces=tool,
        skipped=analyze_traces.Counter(skipped or {}),
    )


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


def test_report_snapshot_roundtrip(tmp_path: Path) -> None:
    report = _report(
        session_ids=["session-a"],
        simple_traces=[
            _turn(
                "trace-simple",
                session_id="session-a",
                eou_delay_ms=120.0,
                perceived_latency_first_audio_ms=900.0,
                llm_ttft_by_phase={1: 320.0},
                tts_ttfb_by_phase={1: 140.0},
            )
        ],
        tool_traces=[
            _turn(
                "trace-tool",
                session_id="session-a",
                tool_round_count=1,
                eou_delay_ms=150.0,
                perceived_latency_first_audio_ms=1300.0,
                perceived_latency_second_audio_ms=1800.0,
                llm_ttft_by_phase={1: 410.0, 2: 520.0},
                tts_ttfb_by_phase={1: 160.0, 2: 220.0},
                tool_execution_by_round={1: 700.0},
            )
        ],
    )

    destination = tmp_path / "reports" / "session-a.json"
    analyze_traces.save_report_snapshot(report, destination)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["schema_version"] == analyze_traces.REPORT_SCHEMA_VERSION

    loaded = analyze_traces.load_report_snapshot(destination)
    assert loaded == report


def test_merge_analysis_reports_recomputes_combined_percentiles() -> None:
    first = _report(
        session_ids=["session-a"],
        simple_traces=[
            _turn(
                "trace-simple-1",
                session_id="session-a",
                eou_delay_ms=100.0,
                perceived_latency_first_audio_ms=1000.0,
                llm_ttft_by_phase={1: 200.0},
                tts_ttfb_by_phase={1: 80.0},
            )
        ],
        tool_traces=[
            _turn(
                "trace-tool-1",
                session_id="session-a",
                tool_round_count=1,
                eou_delay_ms=210.0,
                perceived_latency_first_audio_ms=1400.0,
                perceived_latency_second_audio_ms=1900.0,
                llm_ttft_by_phase={1: 300.0, 2: 450.0},
                tts_ttfb_by_phase={1: 100.0, 2: 130.0},
                tool_execution_by_round={1: 300.0},
            )
        ],
    )
    second = _report(
        session_ids=["session-b"],
        simple_traces=[
            _turn(
                "trace-simple-2",
                session_id="session-b",
                eou_delay_ms=300.0,
                perceived_latency_first_audio_ms=1200.0,
                llm_ttft_by_phase={1: 250.0},
                tts_ttfb_by_phase={1: 90.0},
            ),
            _turn(
                "trace-simple-3",
                session_id="session-b",
                eou_delay_ms=500.0,
                perceived_latency_first_audio_ms=1700.0,
                llm_ttft_by_phase={1: 350.0},
                tts_ttfb_by_phase={1: 110.0},
            ),
        ],
        tool_traces=[
            _turn(
                "trace-tool-2",
                session_id="session-b",
                tool_round_count=1,
                eou_delay_ms=260.0,
                perceived_latency_first_audio_ms=1600.0,
                perceived_latency_second_audio_ms=2300.0,
                llm_ttft_by_phase={1: 340.0, 2: 550.0},
                tts_ttfb_by_phase={1: 120.0, 2: 150.0},
                tool_execution_by_round={1: 700.0},
            )
        ],
    )

    merged = analyze_traces.merge_analysis_reports([first, second])

    assert merged.session_ids == ["session-a", "session-b"]
    assert merged.trace_count == 5
    assert merged.skipped == analyze_traces.Counter()

    simple_rows = {
        row.stage: row.summary
        for row in analyze_traces.build_simple_rows(merged.simple_traces)
    }
    assert simple_rows["eou_delay_ms"].p50 == 300.0
    assert simple_rows["eou_delay_ms"].p95 == 480.0

    tool_rows = {
        row.stage: row.summary
        for row in analyze_traces.build_tool_rows(merged.tool_traces)
    }
    assert tool_rows["tool_execution_1_ms"].p50 == 500.0
    assert tool_rows["tool_execution_1_ms"].p95 == 680.0


def test_merge_analysis_reports_rejects_duplicate_trace_ids() -> None:
    first = _report(
        session_ids=["session-a"],
        simple_traces=[_turn("trace-duplicate", session_id="session-a")],
    )
    second = _report(
        session_ids=["session-b"],
        tool_traces=[
            _turn(
                "trace-duplicate",
                session_id="session-b",
                tool_round_count=1,
                llm_ttft_by_phase={1: 120.0, 2: 180.0},
                tts_ttfb_by_phase={1: 90.0, 2: 120.0},
                tool_execution_by_round={1: 400.0},
            )
        ],
    )

    with pytest.raises(analyze_traces.ReportSnapshotError, match="duplicate trace_id"):
        analyze_traces.merge_analysis_reports([first, second])


def test_merge_analysis_reports_rejects_skipped_input_reports() -> None:
    clean = _report(
        session_ids=["session-a"],
        simple_traces=[_turn("trace-clean", session_id="session-a")],
    )
    incomplete = _report(
        session_ids=["session-b"],
        simple_traces=[_turn("trace-incomplete", session_id="session-b")],
        skipped={"missing observations": 1},
        trace_count=2,
    )

    with pytest.raises(analyze_traces.ReportSnapshotError, match="skipped traces"):
        analyze_traces.merge_analysis_reports([clean, incomplete])


def test_load_report_snapshot_rejects_unknown_schema_version(tmp_path: Path) -> None:
    report = _report(
        session_ids=["session-a"],
        simple_traces=[_turn("trace-simple", session_id="session-a")],
    )
    payload = analyze_traces.report_to_payload(report)
    payload["schema_version"] = analyze_traces.REPORT_SCHEMA_VERSION + 1
    path = tmp_path / "invalid-schema.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(analyze_traces.ReportSnapshotError, match="unsupported schema_version"):
        analyze_traces.load_report_snapshot(path)


def test_load_report_snapshot_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "malformed.json"
    path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(analyze_traces.ReportSnapshotError, match="invalid JSON"):
        analyze_traces.load_report_snapshot(path)


def test_parse_args_requires_a_mode() -> None:
    with pytest.raises(SystemExit):
        analyze_traces._parse_args([])


def test_parse_args_rejects_mixing_fetch_and_merge_modes() -> None:
    with pytest.raises(SystemExit):
        analyze_traces._parse_args(
            ["--session", "session-a", "--merge-report", "saved-report.json"]
        )


def test_parse_args_accepts_merge_mode_with_save_report() -> None:
    parsed = analyze_traces._parse_args(
        [
            "--merge-report",
            "first.json",
            "--merge-report",
            "second.json",
            "--save-report",
            "merged.json",
        ]
    )

    assert parsed.sessions is None
    assert parsed.merge_reports == ["first.json", "second.json"]
    assert parsed.save_report == "merged.json"
