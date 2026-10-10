#!/usr/bin/env python3
import http.server
import socketserver
import json
import re
import subprocess
import os
import time
import argparse
import glob
import socket
from threading import Thread
from pathlib import Path
from datetime import datetime

class Dashboard:
    def __init__(self, mlp_log, cnn_log, stream_log):
        self.mlp_log = mlp_log
        self.cnn_log = cnn_log
        self.stream_log = stream_log
        self.script_dir = Path(__file__).parent
        self.process_cache = {}
        self.process_cache_time = 0

    def get_lan_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"

    def get_processes(self):
        now = time.time()
        if now - self.process_cache_time < 10:
            return self.process_cache
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_Process | Where-Object {$_.Name -in 'python.exe','Mini Motorways.exe'} | Select-Object Name,CommandLine | ConvertTo-Json -Compress"],
                capture_output=True, text=True, timeout=5
            )
            data = json.loads(result.stdout) if result.stdout.strip() else []
            if not isinstance(data, list):
                data = [data] if data else []

            game_count = sum(1 for p in data if p.get("Name") == "Mini Motorways.exe")
            train_ppo = {"mlp": False, "cnn": False}
            best_tracker = False
            stream_showcase = False

            for p in data:
                cmd = p.get("CommandLine", "")
                if "ppo_ckpt_connected_mlp" in cmd:
                    train_ppo["mlp"] = True
                if "ppo_ckpt_connected_cnn" in cmd:
                    train_ppo["cnn"] = True
                if "best_tracker" in cmd:
                    best_tracker = True
                if "stream_showcase" in cmd:
                    stream_showcase = True

            self.process_cache = {
                "game_instances": game_count,
                "train_ppo": train_ppo,
                "best_tracker": best_tracker,
                "stream_showcase": stream_showcase
            }
            self.process_cache_time = now
            return self.process_cache
        except:
            return {"game_instances": 0, "train_ppo": {"mlp": False, "cnn": False}, "best_tracker": False, "stream_showcase": False}

    def get_best_model(self):
        path = self.script_dir / "best_models" / "overall_best.json"
        try:
            with open(path, encoding='utf-8', errors='replace') as f:
                data = json.load(f)
            data["evaluated_ago"] = int(time.time() - data.get("evaluatedAt", 0))
            return data
        except:
            return None

    def parse_logs(self):
        def parse_log_file(path):
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                if len(content) > 400000:
                    content = content[-400000:]
            except:
                return []

            records = []
            current = {}
            dash_pattern = re.compile(r'^-+$', re.MULTILINE)
            parts = dash_pattern.split(content)

            for part in parts:
                current = {}
                for line in part.split('\n'):
                    match = re.search(r'\|\s+(\w+(?:/\w+)?)\s*\|\s*([-\d.]+)\s*\|', line)
                    if match:
                        key, val = match.groups()
                        try:
                            current[key] = float(val)
                        except:
                            pass

                if "total_timesteps" in current:
                    records.append(current)

            return records

        mlp_records = parse_log_file(self.mlp_log)
        cnn_records = parse_log_file(self.cnn_log)

        return {"mlp": mlp_records, "cnn": cnn_records}

    def get_validation_curves(self):
        curves = {"mlp": [], "cnn": []}
        for model_type in ["mlp", "cnn"]:
            path = self.script_dir / "best_models" / f"{model_type}_val_curve.jsonl"
            try:
                with open(path, encoding='utf-8', errors='replace') as f:
                    for line in f:
                        try:
                            curves[model_type].append(json.loads(line))
                        except:
                            pass
            except:
                pass
        return curves

    def get_baselines(self):
        configs = [
            ("wait", "NORMAL_START"),
            ("wait", "CONNECTED_START"),
            ("random", "NORMAL_START"),
            ("random", "CONNECTED_START")
        ]
        baselines = {}

        for policy, start in configs:
            path = self.script_dir.parent / "research" / f"phase11a_{policy}_{start}.jsonl"
            try:
                with open(path, encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()
                    if lines:
                        last = json.loads(lines[-1])
                        if "SUMMARY" in last:
                            s = last["SUMMARY"]
                            baselines[f"{policy}/{start}"] = {
                                "fractionWithPositiveReward": s.get("fractionWithPositiveReward", 0) * 100,
                                "meanFinalScore": s.get("meanFinalScore", 0),
                                "meanReturn": s.get("meanReturn", 0),
                                "meanLength": s.get("meanLength", 0)
                            }
                        else:
                            baselines[f"{policy}/{start}"] = None
                    else:
                        baselines[f"{policy}/{start}"] = None
            except:
                baselines[f"{policy}/{start}"] = None

        return baselines

    def get_stream_lines(self):
        try:
            with open(self.stream_log, encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            stream_lines = [l.strip() for l in lines if '[stream]' in l]
            return stream_lines[-6:] if stream_lines else []
        except:
            return []

    def get_val_reference(self):
        out = {}
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research")
        for pol in ("wait", "random"):
            try:
                with open(os.path.join(base, f"phase11a_val_{pol}_CONNECTED_START.jsonl"), encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if '"SUMMARY"' in line:
                            out[pol] = json.loads(line)["SUMMARY"]["meanReturn"]
            except Exception:
                pass
        return out

    def get_state(self):
        return {
            "timestamp": time.time(),
            "processes": self.get_processes(),
            "best_model": self.get_best_model(),
            "training_curves": self.parse_logs(),
            "validation_curves": self.get_validation_curves(),
            "baselines": self.get_baselines(),
            "val_reference": self.get_val_reference(),
            "stream_lines": self.get_stream_lines()
        }

class DashboardHandler(http.server.BaseHTTPRequestHandler):
    dashboard = None

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(self.get_html().encode('utf-8'))
        elif self.path == "/api/state":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            state = self.dashboard.get_state()
            self.wfile.write(json.dumps(state).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

    def get_html(self):
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mini Motorways RL Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root {
            --bg: #ffffff;
            --text: #000000;
            --card-bg: #f5f5f5;
            --border: #e0e0e0;
            --green: #4caf50;
            --red: #f44336;
            --blue: #2196f3;
            --orange: #ff9800;
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --bg: #121212;
                --text: #ffffff;
                --card-bg: #1e1e1e;
                --border: #333333;
            }
        }
        body {
            font-family: system-ui, -apple-system, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            padding: 16px;
            font-size: 14px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            flex-wrap: wrap;
            gap: 12px;
        }
        .header h1 { font-size: 20px; font-weight: 600; }
        .header-right { display: flex; gap: 12px; align-items: center; }
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--green);
        }
        .status-dot.offline { background: var(--red); }
        .updated { font-size: 12px; opacity: 0.7; }
        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
        }
        .card h2 { font-size: 16px; font-weight: 600; margin-bottom: 12px; }
        .pills { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
        .pill {
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
            background: var(--red);
            color: white;
        }
        .pill.active { background: var(--green); }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        th, td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        th { font-weight: 600; opacity: 0.8; }
        .number { text-align: right; font-variant-numeric: tabular-nums; }
        svg { max-width: 100%; height: auto; margin: 12px 0; }
        .chart-label { fill: var(--text); opacity: 0.8; }
        .stream-log {
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 12px;
            font-family: monospace;
            font-size: 12px;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 200px;
            overflow-y: auto;
        }
        .metric-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 12px;
            font-size: 13px;
        }
        .metric { padding: 8px; background: var(--bg); border-radius: 4px; }
        .metric-label { opacity: 0.7; font-size: 11px; }
        .metric-value { font-weight: 600; font-size: 14px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Mini Motorways RL</h1>
        <div class="header-right">
            <span class="updated" id="updated">updated just now</span>
            <div class="status-dot" id="status"></div>
        </div>
    </div>

    <div id="dashboard"></div>

    <script>
        const dashboard = document.getElementById('dashboard');
        const statusDot = document.getElementById('status');
        const updatedSpan = document.getElementById('updated');
        let lastUpdate = Date.now();

        function formatTime(secs) {
            if (secs < 60) return Math.round(secs) + 's';
            if (secs < 3600) return Math.round(secs / 60) + 'm';
            return Math.round(secs / 3600) + 'h';
        }

        function drawLineChart(data, title, xKey, yKey, yLabel, refLine = null, refLabel = '') {
            if (!data || data.length === 0) return '';
            const mlp = data.mlp || [];
            const cnn = data.cnn || [];
            const allData = [...mlp, ...cnn];

            if (allData.length === 0) return '';

            const okPts = allData.filter(d => d[yKey] !== undefined && d[yKey] !== null && isFinite(d[yKey]));
            if (okPts.length === 0) return '';
            const xMin = Math.min(...okPts.map(d => d[xKey]));
            const xMax = Math.max(...okPts.map(d => d[xKey]));
            const yVals = okPts.map(d => d[yKey]);
            const yMin = Math.min(...yVals);
            const yMax = Math.max(...yVals);
            const yPadding = (yMax - yMin) * 0.1 || 1;

            const w = 300, h = 150, pad = 40;
            const scaleX = (xMax === xMin) ? 0 : (w - 2 * pad) / (xMax - xMin);
            const yRange = yMax + yPadding - (yMin - yPadding);
            const scaleY = (yRange === 0) ? 0 : (h - 2 * pad) / yRange;

            const toSvgX = x => pad + (x - xMin) * scaleX;
            const toSvgY = y => h - pad - (y - (yMin - yPadding)) * scaleY;

            let svg = `<svg viewBox="0 0 ${w} ${h}" style="max-width:100%;">
                <text x="5" y="15" class="chart-label" style="font-size:11px">${title}</text>`;

            // Reference line
            if (refLine !== null) {
                const refY = toSvgY(refLine);
                svg += `<line x1="${pad}" y1="${refY}" x2="${w - pad}" y2="${refY}" stroke="rgba(100,100,100,0.3)" stroke-dasharray="4" stroke-width="1"/>`;
                svg += `<text x="${w - pad - 60}" y="${refY - 5}" class="chart-label" style="font-size:10px">${refLabel}</text>`;
            }

            // Data lines
            const colors = ['#ff6b6b', '#4ecdc4'];
            for (const [idx, dataset] of [mlp, cnn].entries()) {
                if (dataset.length === 0) continue;
                const sorted = dataset.filter(d => d[yKey] !== undefined && d[yKey] !== null && isFinite(d[yKey]) && isFinite(d[xKey])).sort((a, b) => a[xKey] - b[xKey]);
                if (sorted.length === 0) continue;
                const points = sorted.map(d => `${toSvgX(d[xKey])},${toSvgY(d[yKey])}`).join(' ');
                svg += `<polyline points="${points}" fill="none" stroke="${colors[idx]}" stroke-width="2" opacity="0.8"/>`;

                // Markers
                for (const d of sorted) {
                    svg += `<circle cx="${toSvgX(d[xKey])}" cy="${toSvgY(d[yKey])}" r="2" fill="${colors[idx]}"/>`;
                }
            }

            // Y-axis value labels (data range, not the padded range)
            const fmt = v => Math.abs(v) >= 100 ? Math.round(v) : v.toFixed(1);
            svg += `<text x="${pad - 4}" y="${toSvgY(yMax) + 3}" text-anchor="end" class="chart-label" style="font-size:9px">${fmt(yMax)}</text>`;
            svg += `<text x="${pad - 4}" y="${toSvgY(yMin) + 3}" text-anchor="end" class="chart-label" style="font-size:9px">${fmt(yMin)}</text>`;

            // Axes
            svg += `<line x1="${pad}" y1="${h - pad}" x2="${w - pad}" y2="${h - pad}" stroke="currentColor" stroke-width="1" opacity="0.3"/>`;
            svg += `<line x1="${pad}" y1="${pad}" x2="${pad}" y2="${h - pad}" stroke="currentColor" stroke-width="1" opacity="0.3"/>`;
            svg += `<text x="${w/2}" y="${h - 10}" text-anchor="middle" class="chart-label" style="font-size:11px">${xKey}</text>`;
            svg += `<text x="15" y="${h/2}" text-anchor="middle" class="chart-label" style="font-size:11px" transform="rotate(-90 15 ${h/2})">${yLabel}</text>`;

            svg += `<rect x="${w - 112}" y="6" width="8" height="8" fill="#ff6b6b"/><text x="${w - 100}" y="14" class="chart-label" style="font-size:10px">MLP</text>`;
            svg += `<rect x="${w - 62}" y="6" width="8" height="8" fill="#4ecdc4"/><text x="${w - 50}" y="14" class="chart-label" style="font-size:10px">CNN</text>`;

            svg += `</svg>`;
            return svg;
        }

        async function update() {
            try {
                const resp = await fetch('/api/state');
                const state = await resp.json();
                lastUpdate = Date.now();
                statusDot.classList.remove('offline');

                const { processes, best_model, training_curves, validation_curves, baselines, stream_lines } = state;

                let html = '';

                // Status card
                html += `<div class="card">
                    <h2>Status</h2>
                    <div class="pills">
                        <div class="pill ${processes.game_instances > 0 ? 'active' : ''}">
                            Game: ${processes.game_instances}
                        </div>
                        <div class="pill ${processes.train_ppo.mlp ? 'active' : ''}">
                            Train MLP
                        </div>
                        <div class="pill ${processes.train_ppo.cnn ? 'active' : ''}">
                            Train CNN
                        </div>
                        <div class="pill ${processes.best_tracker ? 'active' : ''}">
                            Tracker
                        </div>
                        <div class="pill ${processes.stream_showcase ? 'active' : ''}">
                            Stream
                        </div>
                    </div>
                </div>`;

                // Best model card
                if (best_model) {
                    const ago = formatTime(best_model.evaluated_ago);
                    html += `<div class="card">
                        <h2>Best Model</h2>
                        <table>
                            <tr><th>Label</th><td>${(best_model.label || 'N/A').toUpperCase()}</td></tr>
                            <tr><th>Steps</th><td class="number">${best_model.checkpointSteps || 0}</td></tr>
                            <tr><th>Val Return</th><td class="number">${(best_model.valMeanReturn || 0).toFixed(2)}</td></tr>
                            <tr><th>Val Score</th><td class="number">${(best_model.valMeanScore || 0).toFixed(2)}</td></tr>
                            <tr><th>Length</th><td class="number">${(best_model.valMeanLength || 0).toFixed(1)}</td></tr>
                            <tr><th>Evaluated</th><td>${ago} ago</td></tr>
                        </table>
                    </div>`;
                }

                // Training curves card
                if (training_curves.mlp && training_curves.mlp.length > 0 || training_curves.cnn && training_curves.cnn.length > 0) {
                    html += `<div class="card">
                        <h2>Training Curves</h2>
                        ${drawLineChart(training_curves, 'Episode Reward', 'total_timesteps', 'ep_rew_mean', 'Reward')}
                        ${drawLineChart(training_curves, 'Episode Length', 'total_timesteps', 'ep_len_mean', 'Steps')}
                        <table>
                            <tr><th>Model</th><th class="number">Steps</th><th class="number">Reward</th><th class="number">Length</th><th class="number">Entropy</th><th class="number">FPS</th></tr>`;

                    for (const model of ['mlp', 'cnn']) {
                        const recs = training_curves[model] || [];
                        if (recs.length > 0) {
                            const latest = recs[recs.length - 1];
                            html += `<tr><td>${model.toUpperCase()}</td>
                                <td class="number">${latest.total_timesteps || 0}</td>
                                <td class="number">${(latest.ep_rew_mean || 0).toFixed(2)}</td>
                                <td class="number">${(latest.ep_len_mean || 0).toFixed(1)}</td>
                                <td class="number">${(latest.entropy_loss || 0).toFixed(3)}</td>
                                <td class="number">${(latest.fps || 0).toFixed(1)}</td></tr>`;
                        }
                    }
                    html += `</table></div>`;
                }

                // Validation curves card
                if ((validation_curves.mlp && validation_curves.mlp.length > 0) || (validation_curves.cnn && validation_curves.cnn.length > 0)) {
                    html += `<div class="card">
                        <h2>Held-Out Validation</h2>
                        ${drawLineChart(validation_curves, 'Validation Return (dashed = WAIT-only, same seeds)', 'checkpointSteps', 'valMeanReturn', 'Return', (state.val_reference || {}).wait ?? null, (state.val_reference && state.val_reference.wait !== undefined) ? 'WAIT ' + state.val_reference.wait.toFixed(1) : '')}
                        <table>
                            <tr><th>Model</th><th class="number">Steps</th><th class="number">Return</th><th class="number">Score</th><th class="number">Length</th></tr>`;

                    for (const model of ['mlp', 'cnn']) {
                        const recs = (validation_curves[model] || []).slice(-4);
                        for (const rec of recs) {
                            html += `<tr><td>${model.toUpperCase()}</td>
                                <td class="number">${rec.checkpointSteps || 0}</td>
                                <td class="number">${(rec.valMeanReturn || 0).toFixed(2)}</td>
                                <td class="number">${(rec.valMeanScore || 0).toFixed(2)}</td>
                                <td class="number">${(rec.valMeanLength || 0).toFixed(1)}</td></tr>`;
                        }
                    }
                    html += `</table></div>`;
                }

                // Baselines card
                html += `<div class="card">
                    <h2>Baselines</h2>
                    <table>
                        <tr><th>Policy / Start</th><th class="number">+Reward %</th><th class="number">Score</th><th class="number">Return</th><th class="number">Length</th></tr>`;

                for (const key of ['wait/NORMAL_START', 'wait/CONNECTED_START', 'random/NORMAL_START', 'random/CONNECTED_START']) {
                    const data = baselines[key];
                    if (data === null) {
                        html += `<tr><td>${key.replace('_', ' ')}</td><td colspan="4">running...</td></tr>`;
                    } else if (data) {
                        html += `<tr><td>${key.replace('_', ' ')}</td>
                            <td class="number">${data.fractionWithPositiveReward.toFixed(1)}</td>
                            <td class="number">${data.meanFinalScore.toFixed(2)}</td>
                            <td class="number">${data.meanReturn.toFixed(2)}</td>
                            <td class="number">${data.meanLength.toFixed(1)}</td></tr>`;
                    }
                }
                html += `</table></div>`;

                // Stream card
                if (stream_lines && stream_lines.length > 0) {
                    html += `<div class="card">
                        <h2>Stream</h2>
                        <div class="stream-log">${stream_lines.map(l => l.replace(/</g, '&lt;').replace(/>/g, '&gt;')).join('\\n')}</div>
                    </div>`;
                }

                dashboard.innerHTML = html;
                const age = Math.round((Date.now() - lastUpdate) / 1000);
                updatedSpan.textContent = `updated ${formatTime(age)} ago`;

            } catch (err) {
                statusDot.classList.add('offline');
                updatedSpan.textContent = 'connection lost';
            }
        }

        update();
        setInterval(update, 8000);
    </script>
</body>
</html>"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--mlp-log', default='C:/Users/TKA/AppData/Local/Temp/claude/C--Users-TKA-Desktop-mini-moto/cc5f0315-07aa-4a41-a13e-d71216b9ed18/tasks/bdyyiejqf.output')
    parser.add_argument('--cnn-log', default='C:/Users/TKA/AppData/Local/Temp/claude/C--Users-TKA-Desktop-mini-moto/cc5f0315-07aa-4a41-a13e-d71216b9ed18/tasks/bnlakigo6.output')
    parser.add_argument('--stream-log', default='C:/Users/TKA/AppData/Local/Temp/claude/C--Users-TKA-Desktop-mini-moto/cc5f0315-07aa-4a41-a13e-d71216b9ed18/tasks/bakk185t4.output')
    args = parser.parse_args()

    dashboard = Dashboard(args.mlp_log, args.cnn_log, args.stream_log)
    DashboardHandler.dashboard = dashboard

    server = http.server.ThreadingHTTPServer(("0.0.0.0", args.port), DashboardHandler)

    lan_ip = dashboard.get_lan_ip()
    print(f"Dashboard listening on http://localhost:{args.port} and http://{lan_ip}:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()

if __name__ == "__main__":
    main()
