"""Generate image artifacts through Vita Ask UAT."""
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit
from uuid import uuid4
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "benchmark"
DEFAULT_SIZE = "1024x1024"
IMAGE_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")
ASK_ORIGIN = "https://ask.vita-uat.vsf.services"
ASK_API = "https://ask-api.vita-uat.vsf.services"
ASSET_PATH = "/api/mcp/vita-design/assets/"
ASSET_ID = r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"
ASSET_LINK = re.compile(re.escape(ASSET_PATH) + f"({ASSET_ID})(?![0-9a-f-])", re.IGNORECASE)
POLICY_CODES = {"content_policy_violation", "content_filter", "moderation_blocked", "safety_violation"}


def settings():
    values = {}
    path = ROOT / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.strip().removeprefix("export ").partition("=")
            if separator and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                clean = value.strip().strip("'\"")
                values[key] = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}",
                                     lambda match: os.environ.get(match[1], values.get(match[1], "")), clean)
    return values | dict(os.environ)


def atomic_write(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        part = Path(temporary.name)
        try:
            temporary.write(contents)
            temporary.close()
            part.replace(path)
        finally:
            part.unlink(missing_ok=True)


def detect_extension(data):
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    raise ValueError("Vita Design không trả dữ liệu ảnh PNG, JPEG hoặc WebP")


def stream_events(lines):
    data = []
    for line in chain(lines, ("",)):
        if line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
        elif not line and data:
            raw = "\n".join(data)
            data.clear()
            if raw == "[DONE]":
                return
            event = json.loads(raw)
            if isinstance(event, dict):
                yield event


def find_asset_id(value):
    if isinstance(value, dict):
        for key in ("asset_id", "assetId"):
            asset_id = value.get(key)
            if isinstance(asset_id, str) and re.fullmatch(ASSET_ID, asset_id, re.IGNORECASE):
                return asset_id
        children = value.values()
    elif isinstance(value, list):
        children = value
    elif isinstance(value, str):
        match = ASSET_LINK.search(value)
        if match:
            return match[1]
        if value.lstrip().startswith(("{", "[")):
            try:
                decoded = json.loads(value)
            except ValueError:
                return None
            return find_asset_id(decoded)
        return None
    else:
        return None
    for child in children:
        found = find_asset_id(child)
        if found:
            return found
    return None


def created_image_id(output):
    if not isinstance(output, dict) or output.get("status") not in {"done", "partial"}:
        return None
    images = output.get("outputs")
    if not isinstance(images, list):
        return None
    for image in images:
        if not isinstance(image, dict) or image.get("kind") in {"slides", "document", "pptx", "html"}:
            continue
        asset_id = image.get("asset_id")
        if isinstance(asset_id, str) and re.fullmatch(ASSET_ID, asset_id, re.IGNORECASE):
            return asset_id
    return None


RATE_LIMIT_MESSAGE = re.compile(r"\b(?:rate[-_\s]+limit(?:ed|ing)?|too[-_\s]+many[-_\s]+requests)\b", re.IGNORECASE)


class RateLimitReached(RuntimeError):
    def __init__(self):
        super().__init__("Vita Ask UAT báo rate limit. Đã dừng gửi request cho lượt chạy này.")


def is_rate_limit_error(value):
    if isinstance(value, dict):
        return any(is_rate_limit_error(value.get(field)) for field in (
            "error", "errorText", "error_code", "code", "status", "statusCode",
            "http_status", "message", "detail",
        ))
    if isinstance(value, str):
        if value.strip() == "429" or RATE_LIMIT_MESSAGE.search(value.replace("_", " ")):
            return True
    return value == 429


def image_from_stream(lines, *, on_rate_limit=None):
    image_tools = set()
    tool_outputs = []
    created_asset_id = None
    asset_id = None
    error_code = "NO_IMAGE"
    text_deltas = []
    for event in stream_events(lines):
        kind = event.get("type", "")
        call_id = event.get("toolCallId")
        if on_rate_limit is not None:
            error = event.get("output") if kind == "tool-output-available" else event
            if kind in {"error", "tool-output-error", "tool-output-available"} and is_rate_limit_error(error):
                on_rate_limit()
        if kind in {"tool-input-start", "tool-input-available"}:
            name = (event.get("toolName") or "").lower().replace("-", "_")
            if isinstance(call_id, str) and name.endswith(("generate_image", "generateimage")):
                image_tools.add(call_id)
        elif kind == "tool-output-available":
            tool_outputs.append((call_id, event.get("output")))
        elif kind == "data-vita-design":
            data = event.get("data")
            if isinstance(data, dict):
                if isinstance(data.get("toolCallId"), str):
                    image_tools.add(data["toolCallId"])
                created_asset_id = created_asset_id or created_image_id(data)
            asset_id = asset_id or find_asset_id(event)
        elif kind == "file" or kind.startswith("data-"):
            asset_id = asset_id or find_asset_id(event)
        elif kind == "text-delta":
            text_deltas.append(event.get("delta") or "")
        elif kind == "error" or (kind == "tool-output-error" and call_id in image_tools):
            error_code = "CHAT_STREAM_ERROR"

    for call_id, output in tool_outputs:
        if image_tools and call_id not in image_tools:
            continue
        output_asset = find_asset_id(output)
        if call_id in image_tools:
            created_asset_id = created_asset_id or created_image_id(output)
        asset_id = asset_id or output_asset
        if isinstance(output, dict) and isinstance(output.get("error"), dict):
            code = output["error"].get("code")
            if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", code):
                error_code = code

    match = ASSET_LINK.search("".join(text_deltas))
    selected_asset = created_asset_id or asset_id or (match[1] if match else None)
    return selected_asset, error_code, created_asset_id is not None


class ImageClient:
    base_url = ASK_API

    def __init__(self, config, *, use_configured_conversation=True, rate_limit=None):
        self._rate_limit = rate_limit
        self.model = config.get("VITA_ASK_MODEL") or "v-llm-v1-xlarge"
        self.conversation_id = config.get("VITA_ASK_CONVERSATION_ID", "") if use_configured_conversation else ""
        self.timeout = float(config.get("VITA_DESIGN_TIMEOUT", "300"))
        names = ("VITA_DESIGN_TOKEN", "VITA_ASK_CSRF_COOKIE", "VITA_ASK_CSRF_TOKEN")
        if use_configured_conversation:
            names += ("VITA_ASK_CONVERSATION_ID",)
        missing = [name for name in names if not config.get(name)]
        if missing:
            raise ValueError("Thiếu cấu hình Vita Ask UAT trong .env: " + ", ".join(missing))
        access, csrf_cookie, csrf_token = (config[name] for name in names[:3])
        if access == "dev" or not all(re.fullmatch(r"[A-Za-z0-9._~+/%=-]+", value)
                                      for value in (access, csrf_cookie, csrf_token)):
            raise ValueError("Token và cookie Vita Ask UAT có định dạng không hợp lệ")
        if use_configured_conversation and not self.conversation_id.isdecimal():
            raise ValueError("VITA_ASK_CONVERSATION_ID phải là ID hội thoại dạng số")
        if not 0 < self.timeout < float("inf"):
            raise ValueError("VITA_DESIGN_TIMEOUT must be positive and finite")
        self._headers = {
            "Origin": ASK_ORIGIN,
            "Referer": ASK_ORIGIN + "/",
            "Cookie": f"vita-suite-at={access}; vita-suite-csrf={csrf_cookie}",
            "X-Csrf-Token": csrf_token,
        }

        try:
            import httpx
        except ImportError as error:
            raise ValueError(
                f"Thiếu httpx trong {sys.executable}. Cài bằng python3 -m pip install httpx."
            ) from error
        self._httpx = httpx

    @property
    def rate_limited(self):
        return self._rate_limit is not None and self._rate_limit.is_set()

    def _stop_on_rate_limit(self):
        self._rate_limit.set()
        raise RateLimitReached()

    def _check_request(self, request):
        if self.rate_limited:
            raise RateLimitReached()

    def _check_response(self, response):
        if response.status_code == 429:
            self._stop_on_rate_limit()

    def _client(self, timeout):
        hooks = None
        if self._rate_limit is not None:
            hooks = {"request": [self._check_request], "response": [self._check_response]}
        return self._httpx.Client(timeout=timeout, follow_redirects=False, event_hooks=hooks)

    def create_conversation(self):
        try:
            with self._client(20) as http:
                response = http.post(f"{self.base_url}/api/conversations", headers=self._headers,
                                     json={"model": self.model})
        except self._httpx.RequestError:
            raise ValueError("Không kết nối được Vita Ask UAT để tạo hội thoại.") from None
        if response.status_code == 401:
            raise ValueError(
                "Tạo hội thoại UAT thất bại (HTTP 401). Đăng nhập lại Vita Ask UAT, "
                "cập nhật VITA_DESIGN_TOKEN, VITA_ASK_CSRF_COOKIE và VITA_ASK_CSRF_TOKEN "
                "từ cùng một phiên web trong .env, rồi chạy lại cùng --run."
            )
        if not response.is_success:
            raise ValueError(f"Tạo hội thoại UAT thất bại (HTTP {response.status_code}).")
        try:
            body = response.json()
        except json.JSONDecodeError:
            raise ValueError("Vita Ask UAT trả phản hồi hội thoại không hợp lệ.") from None
        conversation_id = str(body.get("id", "")) if isinstance(body, dict) else ""
        if not re.fullmatch(r"[0-9]+", conversation_id):
            raise ValueError("Vita Ask UAT không trả về ID hội thoại mới hợp lệ.")
        return conversation_id

    def delete_conversation(self, conversation_id):
        if not re.fullmatch(r"[0-9]+", conversation_id):
            raise ValueError("ID hội thoại không hợp lệ.")
        try:
            with self._client(20) as http:
                response = http.delete(f"{self.base_url}/api/conversations/{conversation_id}", headers=self._headers)
        except self._httpx.RequestError:
            raise ValueError("Không kết nối được Vita Ask UAT để xóa hội thoại benchmark.") from None
        if response.status_code not in {204, 404}:
            raise ValueError(f"Xóa hội thoại UAT thất bại (HTTP {response.status_code}).")
        return response.status_code

    def request(self, prompt):
        body = {
            "model": self.model,
            "conversationId": self.conversation_id,
            "messages": [{
                "id": uuid4().hex[:16],
                "role": "user",
                "parts": [{"type": "text", "text": prompt}],
                "metadata": {"custom": {"submittedAt": datetime.now(timezone.utc).isoformat(timespec="milliseconds")}},
            }],
            "trigger": "submit-message",
        }
        timeout = self._httpx.Timeout(30, read=self.timeout)
        with self._client(timeout) as http:
            with http.stream("POST", self.base_url + "/api/chat/mastra", json=body,
                             headers=self._headers | {"Accept": "text/event-stream"}) as response:
                response.raise_for_status()
                if response.headers.get("content-type", "").split(";", 1)[0] != "text/event-stream":
                    return {"error_code": "UNEXPECTED_CHAT_RESPONSE"}, None
                on_rate_limit = self._stop_on_rate_limit if self._rate_limit is not None else None
                asset_id, error_code, asset_created = image_from_stream(
                    response.iter_lines(), on_rate_limit=on_rate_limit
                )
            if not asset_id:
                return {"error_code": error_code}, None

            result = {
                "asset_id": asset_id,
                "asset_created": asset_created,
                "generated_at": datetime.now(IMAGE_TIMEZONE).isoformat(timespec="seconds"),
            }
            asset = http.get(self.base_url + ASSET_PATH + asset_id, headers=self._headers,
                             params={"disposition": "image"})
            if not asset.is_redirect:
                asset.raise_for_status()
                return result, asset.content
            location = asset.headers.get("location", "")
            download_url = urlsplit(location)
            host = download_url.hostname or ""
            if (download_url.scheme != "https" or not host.endswith(".amazonaws.com")
                    or download_url.username or download_url.password):
                return {"error_code": "UNEXPECTED_ASSET_REDIRECT"}, None

        with self._client(timeout) as download:
            image = download.get(location)
            image.raise_for_status()
        return result, image.content

    def delete_asset(self, asset_id):
        if not isinstance(asset_id, str) or not re.fullmatch(ASSET_ID, asset_id, re.IGNORECASE):
            raise ValueError("ID ảnh Vita Design không hợp lệ")
        with self._client(30) as http:
            try:
                response = http.delete(self.base_url + ASSET_PATH + asset_id, headers=self._headers)
            except self._httpx.RequestError as error:
                raise ValueError(f"Không kết nối được Vita Design để xóa ảnh {asset_id}") from error
        if not response.is_success and response.status_code != 404:
            raise ValueError(f"Vita Ask UAT không xóa được ảnh {asset_id} (HTTP {response.status_code})")
        return datetime.now(IMAGE_TIMEZONE).isoformat(timespec="seconds")

    def image(self, prompt, size=None):
        if size is not None:
            prompt = f"{prompt}\n\nGenerate exactly one image from the description above at a resolution of {size} pixels."
        start = time.perf_counter()
        try:
            result, image = self.request(prompt)
        except self._httpx.HTTPStatusError as error:
            result = {"http_status": error.response.status_code}
            if error.request.url.path == "/api/chat/mastra":
                result["error_code"] = "WEB_CHAT_HTTP_ERROR"
            elif error.request.url.path.startswith(ASSET_PATH):
                result["error_code"] = "WEB_ASSET_HTTP_ERROR"
            else:
                result["error_code"] = "WEB_DOWNLOAD_HTTP_ERROR"
            request_id = error.response.headers.get("x-request-id")
            if request_id and re.fullmatch(ASSET_ID, request_id, re.IGNORECASE):
                result["request_id"] = request_id
            image = None
        except self._httpx.TimeoutException:
            result, image = {"error_code": "WEB_TIMEOUT"}, None
        except self._httpx.RequestError:
            result, image = {"error_code": "WEB_CONNECTION_ERROR"}, None
        except json.JSONDecodeError:
            result, image = {"error_code": "INVALID_CHAT_STREAM"}, None
        elapsed = time.perf_counter() - start
        status = "success"
        if image is None:
            status = "rejected" if result.get("error_code") in POLICY_CODES else "error"
        return result | {"status": status, "elapsed_seconds": elapsed, "request_prompt": prompt}, image
