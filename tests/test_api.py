"""The standalone API (temis/api.py) against the active model; skipped when none is exported."""
import os

import pytest

from temis.service import active_model_dir

MODEL = active_model_dir()
pytestmark = pytest.mark.skipif(MODEL is None, reason="no exported model")
TOKEN = "test-token-0123456789abcdefghij"


@pytest.fixture(scope="module")
def client():
    os.environ["TEMIS_TOKEN"] = TOKEN
    os.environ["TEMIS_MODEL_DIR"] = str(MODEL)
    from fastapi.testclient import TestClient

    from temis.api import app
    return TestClient(app)


def test_api_auth_and_answers(client):
    from temis.api import EXAMPLE
    assert client.get("/health").json()["status"] == "ok"
    assert client.post("/v1/decide", json=EXAMPLE).status_code == 401
    assert client.post("/v1/decide", json=EXAMPLE, headers={"Authorization": "Bearer nope"}).status_code == 401
    r = client.post("/v1/systemone", json=EXAMPLE, headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200 and r.headers["x-temis-run"]
    out = r.json()
    assert list(out) == ["model", "answers", "usage"] and out["usage"]["output_tokens"] == 0
    a = out["answers"]
    # fixed keys and order per answer type
    assert list(a["operation"]) == ["type", "choice", "confidence", "probabilities"]
    assert list(a["contradicts"]) == ["type", "noul"]
    assert list(a["usefulness"]) == ["type", "score", "confidence", "legend", "probabilities"]
    assert a["usefulness"]["legend"] == {"0": "Nada útil", "1": "Algo útil", "2": "Muy útil"}
    assert a["operation"]["choice"] in EXAMPLE["questions"]["operation"]["criteria"]
    bad = {"state": "x", "questions": {"q": {"type": "choice", "instructions": "a", "criteria": {"a": "b"}}}}
    r = client.post("/v1/decide", json=bad, headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 400 and r.json()["detail"]["error_type"] == "api_usage_error"
    r = client.post("/v1/decide", json={**EXAMPLE, "model": "other-latest"}, headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 400
    assert client.get("/v1/model").json()["detail"]["error_type"] == "authentication_error"
