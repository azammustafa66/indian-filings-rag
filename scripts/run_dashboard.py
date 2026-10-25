"""Start FastAPI and Streamlit together locally; Ctrl+C stops both child processes."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    """Launch both services with shared port settings and clean them up on exit."""
    parser = argparse.ArgumentParser(description='Run the local filings dashboard and API')
    parser.add_argument('--api-port', type=int, default=8000)
    parser.add_argument('--ui-port', type=int, default=8501)
    args = parser.parse_args()
    if not all(1 <= port <= 65535 for port in (args.api_port, args.ui_port)) or args.api_port == args.ui_port:
        parser.error('Choose two distinct ports between 1 and 65535.')
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, 'FILINGS_API_URL': f'http://127.0.0.1:{args.api_port}'}
    commands = [
        [
            sys.executable,
            '-m',
            'uvicorn',
            'filings_rag.api:app',
            '--host',
            '127.0.0.1',
            '--port',
            str(args.api_port),
        ],
        [
            sys.executable,
            '-m',
            'streamlit',
            'run',
            str(root / 'streamlit_app.py'),
            '--server.address',
            '127.0.0.1',
            '--server.port',
            str(args.ui_port),
            '--server.headless',
            'true',
        ],
    ]
    processes = []
    try:
        for command in commands:
            processes.append(subprocess.Popen(command, cwd=root, env=environment))
        print(f'\nDashboard: http://127.0.0.1:{args.ui_port}', flush=True)
        print(f'API docs:  http://127.0.0.1:{args.api_port}/docs\n', flush=True)
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
        raise SystemExit('A service stopped. Check the logs above or choose unused ports.')
    except KeyboardInterrupt:
        print('\nStopping the dashboard and API…', flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    main()
