# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the dashboard API server."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.client import HTTPConnection

from aevyra_reflex.run_store import (
    CheckpointState,
    IterationState,
    RunStore,
)
from aevyra_reflex.dashboard.server import _DashboardHandler
from http.server import ThreadingHTTPServer as HTTPServer


def _start_test_server(store, port):
    """Start a server on a background thread, return (server, thread)."""
    handler_class = type(
        "_TestHandler",
        (_DashboardHandler,),
        {"store": store},
    )
    server = HTTPServer(("127.0.0.1", port), handler_class)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _get(conn, path):
    """Helper to GET a path and return (status, parsed_json_or_text)."""
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read().decode("utf-8")
    content_type = resp.getheader("Content-Type", "")
    if "application/json" in content_type:
        return resp.status, json.loads(body)
    return resp.status, body


class TestDashboardAPI(unittest.TestCase):
    """Test the JSON API endpoints."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = RunStore(root=self.tmpdir)

        # Create a run with some data
        self.run = self.store.create_run(
            config={"strategy": "auto", "reasoning_model": "claude-sonnet-4-20250514", "max_iterations": 10, "score_threshold": 0.85},
            dataset_path="data.jsonl",
            prompt_path="prompt.md",
            initial_prompt="You are a helpful assistant.",
        )
        self.run.save_baseline({"mean_score": 0.58, "scores_by_metric": {"rouge": 0.58}})
        self.run.save_iteration(IterationState(
            iteration=1,
            system_prompt="You are a highly capable assistant.",
            score=0.72,
            scores_by_metric={"rouge": 0.72},
            reasoning="The prompt lacks specificity. Adding capability framing improves responses.",
        ))
        self.run.save_iteration(IterationState(
            iteration=2,
            system_prompt="You are an expert assistant. Be precise and thorough.",
            score=0.86,
            scores_by_metric={"rouge": 0.86},
            reasoning="Adding expert framing and precision instructions further improved scores.",
        ))
        self.run.save_checkpoint(CheckpointState(
            run_id=self.run.run_id,
            initial_prompt="You are a helpful assistant.",
            current_prompt="You are an expert assistant. Be precise and thorough.",
            completed_iterations=2,
            best_prompt="You are an expert assistant. Be precise and thorough.",
            best_score=0.86,
            score_trajectory=[0.72, 0.86],
        ))

        # Start server
        self.port = 18337  # use a high port to avoid conflicts
        self.server, self.thread = _start_test_server(self.store, self.port)
        self.conn = HTTPConnection("127.0.0.1", self.port, timeout=5)

    def tearDown(self):
        self.conn.close()
        self.server.shutdown()

    def test_list_runs(self):
        status, data = _get(self.conn, "/api/runs")
        self.assertEqual(status, 200)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["run_id"], "001")
        self.assertEqual(data[0]["status"], "interrupted")
        self.assertAlmostEqual(data[0]["baseline_score"], 0.58)
        self.assertAlmostEqual(data[0]["best_score"], 0.86)

    def test_get_run(self):
        status, data = _get(self.conn, "/api/runs/001")
        self.assertEqual(status, 200)
        self.assertEqual(data["run_id"], "001")
        self.assertFalse(data["is_complete"])
        self.assertEqual(len(data["iterations"]), 2)
        self.assertAlmostEqual(data["iterations"][0]["score"], 0.72)
        self.assertAlmostEqual(data["iterations"][1]["score"], 0.86)
        self.assertIsNotNone(data["baseline"])
        self.assertIsNotNone(data["checkpoint"])
        self.assertAlmostEqual(data["checkpoint"]["best_score"], 0.86)

    def test_get_run_not_found(self):
        status, data = _get(self.conn, "/api/runs/999")
        self.assertEqual(status, 404)
        self.assertIn("error", data)

    def test_get_iterations(self):
        status, data = _get(self.conn, "/api/runs/001/iterations")
        self.assertEqual(status, 200)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["iteration"], 1)
        self.assertIn("reasoning", data[0])
        self.assertIn("system_prompt", data[1])

    def test_get_single_iteration(self):
        status, data = _get(self.conn, "/api/runs/001/iterations/2")
        self.assertEqual(status, 200)
        self.assertEqual(data["iteration"], 2)
        self.assertAlmostEqual(data["score"], 0.86)
        self.assertIn("expert assistant", data["system_prompt"])

    def test_get_iteration_not_found(self):
        status, data = _get(self.conn, "/api/runs/001/iterations/99")
        self.assertEqual(status, 404)

    def test_spa_served(self):
        status, body = _get(self.conn, "/")
        self.assertEqual(status, 200)
        self.assertIn("aevyra reflex", body)

    def test_spa_served_for_unknown_paths(self):
        """Any non-API path serves the SPA for client-side routing."""
        status, body = _get(self.conn, "/runs/001")
        self.assertEqual(status, 200)
        self.assertIn("aevyra reflex", body)

    def test_cors_header(self):
        self.conn.request("GET", "/api/runs")
        resp = self.conn.getresponse()
        resp.read()
        self.assertEqual(resp.getheader("Access-Control-Allow-Origin"), "*")


class TestDashboardEmpty(unittest.TestCase):
    """Test with an empty run store."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = RunStore(root=self.tmpdir)
        self.port = 18338
        self.server, self.thread = _start_test_server(self.store, self.port)
        self.conn = HTTPConnection("127.0.0.1", self.port, timeout=5)

    def tearDown(self):
        self.conn.close()
        self.server.shutdown()

    def test_empty_runs_list(self):
        status, data = _get(self.conn, "/api/runs")
        self.assertEqual(status, 200)
        self.assertEqual(data, [])


class TestDashboardCompletedRun(unittest.TestCase):
    """Test with a completed run."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = RunStore(root=self.tmpdir)
        run = self.store.create_run(
            config={"strategy": "iterative", "reasoning_model": "qwen3:8b"},
            dataset_path="test.jsonl",
            prompt_path="prompt.md",
            initial_prompt="Hello.",
        )
        run.save_baseline({"mean_score": 0.50})
        run.save_iteration(IterationState(iteration=1, system_prompt="Better.", score=0.90))
        run.save_checkpoint(CheckpointState(
            run_id=run.run_id, initial_prompt="Hello.", current_prompt="Better.",
            completed_iterations=1, best_prompt="Better.", best_score=0.90,
        ))
        run.save_result({
            "best_score": 0.90,
            "best_prompt": "Better.",
            "iterations": [{"iteration": 1, "score": 0.90}],
            "final": {"mean_score": 0.90},
        })

        self.port = 18339
        self.server, self.thread = _start_test_server(self.store, self.port)
        self.conn = HTTPConnection("127.0.0.1", self.port, timeout=5)

    def tearDown(self):
        self.conn.close()
        self.server.shutdown()

    def test_completed_run_status(self):
        status, data = _get(self.conn, "/api/runs")
        self.assertEqual(status, 200)
        self.assertEqual(data[0]["status"], "completed")
        self.assertAlmostEqual(data[0]["final_score"], 0.90)

    def test_completed_run_detail(self):
        status, data = _get(self.conn, "/api/runs/001")
        self.assertEqual(status, 200)
        self.assertTrue(data["is_complete"])
        self.assertIsNotNone(data["result"])


def _post(conn, path, body):
    """Helper to POST JSON and return (status, parsed_json)."""
    data = json.dumps(body).encode("utf-8")
    conn.request("POST", path, body=data, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    return resp.status, json.loads(resp.read().decode("utf-8"))


class TestBranchEndpoint(unittest.TestCase):
    """Tests for POST /api/jobs/branch."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = RunStore(root=self.tmpdir)

        # Create a parent run with iterations and a baseline
        self.parent = self.store.create_run(
            config={"strategy": "iterative", "reasoning_model": "claude-sonnet-4-20250514",
                    "max_iterations": 10, "score_threshold": 0.85, "_cli_models": ["openai/gpt-4o-mini"]},
            dataset_path="data.jsonl",
            prompt_path="prompt.md",
            initial_prompt="You are helpful.",
        )
        self.parent.save_baseline({"mean_score": 0.58, "scores_by_metric": {"rouge": 0.58}})
        self.parent.save_iteration(IterationState(
            iteration=1, system_prompt="You are very helpful.", score=0.70,
        ))
        self.parent.save_iteration(IterationState(
            iteration=2, system_prompt="You are an expert assistant.", score=0.82,
        ))
        self.parent.save_iteration(IterationState(
            iteration=3, system_prompt="You are a precise expert assistant.", score=0.88,
        ))
        self.parent.save_result({"best_score": 0.88, "best_prompt": "precise expert", "iterations": []})

        self.port = 18340
        self.server, self.thread = _start_test_server(self.store, self.port)
        self.conn = HTTPConnection("127.0.0.1", self.port, timeout=5)

    def tearDown(self):
        self.conn.close()
        self.server.shutdown()

    def test_missing_parent_run_id_returns_400(self):
        status, data = _post(self.conn, "/api/jobs/branch", {
            "parent_iteration": 2, "strategy": "fewshot", "max_iterations": 5,
        })
        self.assertEqual(status, 400)
        self.assertIn("parent_run_id", data["error"])

    def test_missing_parent_iteration_returns_400(self):
        status, data = _post(self.conn, "/api/jobs/branch", {
            "parent_run_id": "001", "strategy": "fewshot", "max_iterations": 5,
        })
        self.assertEqual(status, 400)
        self.assertIn("parent_iteration", data["error"])

    def test_nonexistent_parent_run_returns_404(self):
        status, data = _post(self.conn, "/api/jobs/branch", {
            "parent_run_id": "999", "parent_iteration": 1,
            "strategy": "iterative", "max_iterations": 5,
        })
        self.assertEqual(status, 404)

    def test_nonexistent_iteration_returns_404(self):
        status, data = _post(self.conn, "/api/jobs/branch", {
            "parent_run_id": "001", "parent_iteration": 99,
            "strategy": "iterative", "max_iterations": 5,
        })
        self.assertEqual(status, 404)

    def test_valid_branch_returns_202_with_job_id(self):
        status, data = _post(self.conn, "/api/jobs/branch", {
            "parent_run_id": "001", "parent_iteration": 2,
            "strategy": "fewshot", "max_iterations": 5,
        })
        self.assertEqual(status, 202)
        self.assertIn("job_id", data)
        self.assertIsInstance(data["job_id"], str)
        self.assertGreater(len(data["job_id"]), 0)


class TestTokenFieldsInAPI(unittest.TestCase):
    """Token fields (eval_tokens, reasoning_tokens) surfaced via the API."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = RunStore(root=self.tmpdir)

        run = self.store.create_run(
            config={"strategy": "iterative"},
            dataset_path="data.jsonl",
            prompt_path="prompt.md",
            initial_prompt="You are helpful.",
        )
        run.save_baseline({"mean_score": 0.58})
        run.save_iteration(IterationState(
            iteration=1, system_prompt="Better prompt.", score=0.75,
            eval_tokens=8200, reasoning_tokens=3100,
        ))
        run.save_iteration(IterationState(
            iteration=2, system_prompt="Even better prompt.", score=0.85,
            eval_tokens=9400, reasoning_tokens=2800,
        ))

        self.port = 18341
        self.server, self.thread = _start_test_server(self.store, self.port)
        self.conn = HTTPConnection("127.0.0.1", self.port, timeout=5)

    def tearDown(self):
        self.conn.close()
        self.server.shutdown()

    def test_run_detail_iterations_include_token_fields(self):
        status, data = _get(self.conn, "/api/runs/001")
        self.assertEqual(status, 200)
        iters = data["iterations"]
        self.assertEqual(len(iters), 2)
        self.assertEqual(iters[0]["eval_tokens"], 8200)
        self.assertEqual(iters[0]["reasoning_tokens"], 3100)
        self.assertEqual(iters[1]["eval_tokens"], 9400)
        self.assertEqual(iters[1]["reasoning_tokens"], 2800)

    def test_iterations_endpoint_includes_token_fields(self):
        status, data = _get(self.conn, "/api/runs/001/iterations")
        self.assertEqual(status, 200)
        self.assertEqual(data[0]["eval_tokens"], 8200)
        self.assertEqual(data[0]["reasoning_tokens"], 3100)

    def test_single_iteration_endpoint_includes_token_fields(self):
        status, data = _get(self.conn, "/api/runs/001/iterations/2")
        self.assertEqual(status, 200)
        self.assertEqual(data["eval_tokens"], 9400)
        self.assertEqual(data["reasoning_tokens"], 2800)

    def test_token_fields_default_to_zero_for_old_iterations(self):
        """Iterations saved before token tracking was added have eval/reasoning_tokens = 0."""
        # Write a raw iteration file without token fields
        run = self.store.get_run("001")
        iter_file = run.iterations_dir / "003.json"
        iter_file.write_text(json.dumps({
            "iteration": 3, "system_prompt": "old", "score": 0.60,
            "scores_by_metric": {}, "reasoning": "", "timestamp": "",
        }))

        status, data = _get(self.conn, "/api/runs/001/iterations/3")
        self.assertEqual(status, 200)
        self.assertEqual(data["eval_tokens"], 0)
        self.assertEqual(data["reasoning_tokens"], 0)


class TestRunSummaryBranchFields(unittest.TestCase):
    """parent_run_id / parent_iteration appear in /api/runs list for branch runs."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = RunStore(root=self.tmpdir)

        # Root run
        self.store.create_run(
            config={}, dataset_path="d.jsonl", prompt_path="p.md", initial_prompt="a",
        )
        # Branch run
        self.store.create_run(
            config={}, dataset_path="d.jsonl", prompt_path="p.md", initial_prompt="b",
            branched_from={"run_id": "001", "iteration": 2},
        )

        self.port = 18342
        self.server, self.thread = _start_test_server(self.store, self.port)
        self.conn = HTTPConnection("127.0.0.1", self.port, timeout=5)

    def tearDown(self):
        self.conn.close()
        self.server.shutdown()

    def test_root_run_has_no_parent_fields(self):
        status, data = _get(self.conn, "/api/runs")
        self.assertEqual(status, 200)
        root = data[0]
        self.assertIsNone(root["parent_run_id"])
        self.assertIsNone(root["parent_iteration"])

    def test_branch_run_has_parent_fields(self):
        status, data = _get(self.conn, "/api/runs")
        self.assertEqual(status, 200)
        branch = data[1]
        self.assertEqual(branch["parent_run_id"], "001")
        self.assertEqual(branch["parent_iteration"], 2)


if __name__ == "__main__":
    unittest.main()
