# tests/test_api.py
from __future__ import annotations

import json
from pathlib import Path


def test_health_stats(client):
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    # best-effort stats present
    assert "num_chunks" in data

    r = client.get("/stats")
    assert r.status_code == 200
    s = r.json()
    assert s["num_docs"] >= 1
    assert s["num_chunks"] >= 1
    assert isinstance(s["example_doc_ids"], list)


def test_version(client):
    r = client.get("/version")
    assert r.status_code == 200
    v = r.json()
    assert "app_version" in v
    assert "encoder" in v


def test_search_and_open_page(client):
    # basic search
    r = client.post("/search", json={"query": "search", "k": 3})
    assert r.status_code == 200
    payload = r.json()
    assert "results" in payload and "took_ms" in payload
    results = payload["results"]
    if results:
        # try open-page on the first result
        h0 = results[0]
        r2 = client.get(f"/open-page/{h0['doc_id']}/{h0['page']}")
        assert r2.status_code == 200
        op = r2.json()
        assert "path" in op and "page" in op


def test_admin_ingest_rebuild(client, tmp_path, monkeypatch):
    """
    Hit /admin/ingest-rebuild; the endpoint now schedules the rebuild as a
    FastAPI BackgroundTask and returns immediately with 202 Accepted. We
    only assert the route is reachable and returns the expected envelope
    — the actual rebuild runs in the background and reports failures via
    the server log, not the HTTP response.
    """
    pdfs_dir = tmp_path / "pdfs"
    pdfs_dir.mkdir()

    r = client.post("/admin/ingest-rebuild", json={"pdfs_dir": str(pdfs_dir)})
    assert r.status_code == 202
    data = r.json()
    assert data["status"] == "accepted"
    assert "message" in data
