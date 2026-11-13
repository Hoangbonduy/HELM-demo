#!/usr/bin/env python3
"""Discover muse2api models and generate local HELM configuration (stdlib only)."""
import argparse
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent


def api(base_url, key, path, payload=None):
    request = Request(
        base_url + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    with urlopen(request, timeout=120 if payload else 15) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18610/v1")
    parser.add_argument("--list", action="store_true", help="Only list available model IDs")
    parser.add_argument("--model", action="append", help="Model ID to register; repeat as needed (default: all)")
    parser.add_argument("--smoke-test", metavar="MODEL", help="Send one short chat request; may incur usage")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    local = ROOT / "prod_env"
    credentials_path = local / "credentials.conf"
    credentials = json.loads(credentials_path.read_text()) if credentials_path.exists() else {}
    key = os.environ.get("MUSE2API_API_KEY") or credentials.get("muse2apiApiKey")
    if not key:
        parser.error("Set MUSE2API_API_KEY or prod_env/credentials.conf: muse2apiApiKey")
    models = api(base_url, key, "/models")["data"]
    available = {model["id"]: model for model in models if model.get("kind") == "chat" and not model.get("alias_of")}
    if args.list:
        print("\n".join(available))
        return
    selected = args.model or list(available)
    unknown = set(selected) - available.keys()
    if unknown:
        parser.error("Unknown model IDs: " + ", ".join(sorted(unknown)))
    if args.smoke_test:
        if args.smoke_test not in available:
            parser.error("Smoke-test model is not in /models")
        result = api(base_url, key, "/chat/completions", {
            "model": args.smoke_test,
            "messages": [{"role": "user", "content": "Reply with OK only."}],
            "max_tokens": 32, "stream": False, "temperature": 0, "top_p": 1, "n": 1,
            "presence_penalty": 0, "frequency_penalty": 0,
        })
        content = result["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Gateway returned no text completion")
        print("Chat smoke test passed: " + content)
    deployments, metadata = [], []
    for model_id in dict.fromkeys(selected):
        # Keep one slash for HELM's organization/engine naming convention.
        name = "muse2api/" + model_id.replace("%", "%25").replace("/", "%2F")
        deployments.append({
            "name": name, "model_name": name,
            "client_spec": {"class_name": "helm.clients.openai_client.OpenAIClient",
                            "args": {"base_url": base_url, "openai_model_name": model_id}},
        })
        metadata.append({
            "name": name, "creator_organization_name": available[model_id].get("owned_by", "muse2api"),
            "display_name": model_id, "description": "Chat model served through local muse2api.",
            "access": "limited", "release_date": None,
            "tags": ["TEXT_MODEL_TAG", "LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG", "INSTRUCTION_FOLLOWING_MODEL_TAG"],
        })
    local.mkdir(exist_ok=True)
    # JSON is valid YAML and HOCON. No additional package is needed for setup.
    for filename, data in (("model_deployments.yaml", {"model_deployments": deployments}),
                           ("model_metadata.yaml", {"models": metadata})):
        path = local / filename
        if path.exists():
            existing = json.loads(path.read_text())
            field = next(iter(data))
            retained = [entry for entry in existing[field] if not entry["name"].startswith("muse2api/")]
            data[field] = retained + data[field]
        path.write_text(json.dumps(data, indent=2) + "\n")
    credentials["muse2apiApiKey"] = key
    # Create a private file before writing the key.
    fd = os.open(credentials_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(json.dumps(credentials, indent=2) + "\n")
    print(f"Configured {len(deployments)} models at {base_url}")
    print("Example deployment: " + deployments[0]["name"])


if __name__ == "__main__":
    try:
        main()
    except (HTTPError, URLError, ValueError, KeyError, OSError) as error:
        print(f"muse2api setup failed: {error}", file=sys.stderr)
        sys.exit(1)
