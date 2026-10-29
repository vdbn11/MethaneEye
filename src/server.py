import argparse
import asyncio
import json
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import DEFAULT_HOST, DEFAULT_PORT, REPORTS_DIR, WEB_DIR
from src.simulation import SimulationManager

manager: SimulationManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager
    manager = SimulationManager()
    yield
    manager.executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="MethaneEye Simulator", version="1.0.0", lifespan=lifespan)


@app.get("/api/health")
async def health():
    return {"ok": True, "model": manager.detector.checkpoint.name if manager else None}


@app.get("/api/state")
async def state():
    return manager.snapshot()


@app.post("/api/start")
async def start():
    manager.start()
    return {"ok": True, "phase": manager.phase}


@app.post("/api/pause")
async def pause():
    manager.pause()
    return {"ok": True, "phase": manager.phase}


@app.post("/api/launch/{drone_id}")
async def launch(drone_id: str):
    if drone_id.strip().lower() == "all":
        manager.launch_all()
        return {"ok": True, "phase": manager.phase}
    ok = manager.launch(drone_id)
    return {"ok": ok, "phase": manager.phase, "drone": drone_id.upper()}


@app.post("/api/hold/{drone_id}")
async def hold(drone_id: str):
    state = manager.hold(drone_id)
    return {"ok": state is not None, "hold": bool(state), "phase": manager.phase, "drone": drone_id.upper()}


@app.post("/api/export")
async def export():
    manager.persist_run()
    return {"ok": True}


@app.post("/api/automation/{value}")
async def automation(value: str):
    enabled = value.strip().lower() in ("auto", "true", "1", "on")
    manager.set_automation(enabled)
    return {"ok": True, "auto_mode": manager.auto_mode}


@app.post("/api/dispatch/{leak_id}")
async def dispatch(leak_id: int):
    result = manager.dispatch_leak(leak_id)
    result["leak_id"] = leak_id
    return result


@app.post("/api/verify")
async def verify():
    ok = manager.start_verification()
    return {"ok": ok}


@app.post("/api/reset")
async def reset():
    manager.reset()
    return {"ok": True, "phase": manager.phase}


@app.post("/api/speed/{value}")
async def speed(value: float):
    manager.set_speed(value)
    return {"ok": True, "speed": manager.speed}


@app.post("/api/mode/{mode}")
async def mode(mode: str):
    manager.set_mode(mode)
    return {"ok": True, "mode": manager.mode}


@app.post("/api/optimize")
async def optimize():
    result = manager.rerun_ga()
    return {
        "ok": True,
        "distance_km": result["distance_m"] / 1000.0,
        "baseline_km": result["baseline_distance_m"] / 1000.0,
        "improvement_pct": result["improvement_pct"],
    }


@app.get("/api/model")
async def model_info():
    return manager._model_info()


@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")


@app.websocket("/ws")
async def websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            payload = manager.snapshot()
            await websocket.send_text(json.dumps(payload, separators=(",", ":")))
            await asyncio.sleep(0.22)
    except (WebSocketDisconnect, RuntimeError):
        return


app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

FRAMES_DIR = REPORTS_DIR / "frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/frames", StaticFiles(directory=FRAMES_DIR), name="frames")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
