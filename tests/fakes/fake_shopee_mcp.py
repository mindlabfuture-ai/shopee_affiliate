"""Minimal stdio MCP server standing in for shopee-mcp in tests (newline-delimited JSON-RPC, no SDK)."""
import json
import os
import sys

TOOLS = ["search_products", "get_product_detail", "get_product_reviews", "add_to_cart"]
URLS = ["https://shopee.ph/product/111/222", "https://shopee.ph/product/333/444"]


def reply(i, result):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": i, "result": result}) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    msg = json.loads(line)
    method, i = msg.get("method"), msg.get("id")
    if method == "initialize":
        reply(i, {"protocolVersion": msg["params"]["protocolVersion"], "capabilities": {"tools": {}},
                  "serverInfo": {"name": "fake-shopee", "version": "0"}})
    elif method == "tools/list":
        reply(i, {"tools": [{"name": t, "inputSchema": {"type": "object"}} for t in TOOLS]})
    elif method == "tools/call":
        name, args = msg["params"]["name"], msg["params"].get("arguments", {})
        if name == "search_products":
            text = "Results:\n" + "\n".join(f"- Dress {n} 🔗 {u}" for n, u in enumerate(URLS))
        else:
            env_keys = sorted(k for k in os.environ if k.startswith(("SHOPEE_", "CLOAKBROWSER_")) or "KEY" in k)
            text = f"{name} {json.dumps(args)} env={env_keys} " \
                   f"account={os.environ.get('SHOPEE_ACCOUNT_TOOLS')} tz={os.environ.get('SHOPEE_TIMEZONE')}"
        reply(i, {"content": [{"type": "text", "text": text}], "isError": False})
