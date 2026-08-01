import json
import subprocess
import sys

def send_request(proc, request):
    payload = json.dumps(request) + "\n"
    proc.stdin.write(payload)
    proc.stdin.flush()
    return proc.stdout.readline().decode('utf-8').strip()

def test_memory():
    proc = subprocess.Popen(
        ["/Users/hodorii/dev/turbo-memory-mcp/.venv/bin/python", "/Users/hodorii/dev/turbo-memory-mcp/server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    # 1. Initialize
    init_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"}
        }
    }
    send_request(proc, init_req)
    
    # Skip logs and find JSON
    while True:
        line = proc.stdout.readline()
        if not line: break
        if line.startswith('{'):
            break

    # 2. Call memory_stats
    stats_req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "memory_stats", "arguments": {}}
    }
    send_request(proc, stats_req)
    
    while True:
        line = proc.stdout.readline()
        if not line: break
        if line.startswith('{'):
            print("Stats Result:")
            print(line)
            break

    # 3. Call recall
    recall_req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "recall", "arguments": {"query": "Sisyphus", "top_k": 5}}
    }
    send_request(proc, recall_req)
    
    while True:
        line = proc.stdout.readline()
        if not line: break
        if line.startswith('{'):
            print("\nRecall Result:")
            print(line)
            break

    proc.terminate()

if __name__ == "__main__":
    test_memory()
