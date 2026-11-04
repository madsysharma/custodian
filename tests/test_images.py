"""Screenshots are downloaded and inlined, and a provider that rejects images does not
stop triage or an evaluation run."""

import httpx

from custodian.images import fetch_image_data_url
from custodian.llm import ChatMessage
from custodian.workflows import triage

from conftest import final

PNG = b"\x89PNG\r\n\x1a\nfake"


def make_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_follows_github_redirect_to_signed_s3_link():
    def handler(request):
        if request.url.host == "github.com":
            return httpx.Response(302, headers={"Location": "https://s3.example.com/img.png?sig=1"})
        assert "authorization" not in request.headers  # the token never leaks to S3
        return httpx.Response(200, headers={"content-type": "image/png"}, content=PNG)

    url = fetch_image_data_url("https://github.com/user-attachments/assets/abc", "tok", make_client(handler))
    assert url and url.startswith("data:image/png;base64,")


def test_bad_images_are_skipped_not_fatal():
    not_found = make_client(lambda r: httpx.Response(404))
    html = make_client(lambda r: httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html>"))
    huge = make_client(lambda r: httpx.Response(200, headers={"content-type": "image/png"}, content=b"0" * (5 * 1024 * 1024)))
    for client in (not_found, html, huge):
        assert fetch_image_data_url("https://github.com/x.png", "", client) is None


class RejectsImages:
    """A model that fails like Fireworks did whenever a message contains an image."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        content = messages[1]["content"]
        if isinstance(content, list) and any(p["type"] == "image_url" for p in content):
            raise RuntimeError("Model call failed: Error code: 400 - Cannot download image")
        return final({"summary": "ok", "labels": ["bug"], "difficulty": "intermediate"})


def test_triage_retries_without_images(make_ctx, monkeypatch):
    monkeypatch.setattr("custodian.workflows.fetch_image_data_url", lambda url, token="": "data:image/png;base64,AA")
    model = RejectsImages()
    result = triage(make_ctx(model))
    assert result.verified["labels"] == ["bug"]
    assert model.calls == 2
    assert any("text only" in w for w in result.warnings)