"""
Launcher that starts Streamlit, warms up the model cache by hitting the app
once internally, and only prints the clickable URL when the app is fully
ready. Run with: poetry run python run.py
"""
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HOST = "localhost"
PORT = 8501
HEALTH_URL = f"http://{HOST}:{PORT}/_stcore/health"
APP_URL = f"http://{HOST}:{PORT}"


def wait_for(url: str, timeout: float, label: str) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.5)
    print(f"[run.py] Gave up waiting for {label} after {timeout:.0f}s")
    return False


def main() -> int:
    print("[run.py] Starting Streamlit...")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "app/app.py",
            "--server.port", str(PORT),
            "--server.address", HOST,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        if not wait_for(HEALTH_URL, timeout=60, label="Streamlit server"):
            proc.terminate()
            return 1

        print("[run.py] Server up. Warming up models (this loads torch + sentence-transformers)...")
        if not wait_for(APP_URL, timeout=300, label="first page render"):
            proc.terminate()
            return 1

        bar = "=" * 60
        print(f"\n{bar}\n  READY  →  {APP_URL}\n{bar}\n")
        print("(Ctrl+C to stop the server)")

        proc.wait()
        return proc.returncode or 0
    except KeyboardInterrupt:
        print("\n[run.py] Stopping...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
