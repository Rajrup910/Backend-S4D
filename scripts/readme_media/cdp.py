"""Minimal Chrome DevTools rig: headless Chrome + websockets. Frame-by-frame capture of a local page on a virtual clock."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
REPO = Path(__file__).resolve().parents[2]
FRAMES = Path(os.environ.get("README_MEDIA_TMP", Path(tempfile.gettempdir()) / "readme_media")) / "frames"
PORT = 9333
FPS = 25
OUT_W = 1600  # recording width: about 2x the README column, so text stays sharp on high-resolution monitors

# Installed before any page script. While recording, time only moves when tick() is called: performance.now, Date.now,
# requestAnimationFrame, timers and CSS animations all read one virtual clock, so every frame is exactly 1/FPS later than
# the last, however long the page took to compute it. A slow frame costs capture time, never smoothness.
CLOCK_JS = r"""
(() => {
  const P = performance, real = {now: P.now.bind(P), dnow: Date.now.bind(Date), raf: window.requestAnimationFrame.bind(window), caf: window.cancelAnimationFrame.bind(window),
    st: window.setTimeout.bind(window), ct: window.clearTimeout.bind(window), si: window.setInterval.bind(window), ci: window.clearInterval.bind(window)};
  let on = false, vt = 0, offset = 0, nextId = 0;
  const dateOffset = real.dnow() - real.now();
  const now = () => on ? vt : real.now() + offset;
  const rafs = new Map(), timers = new Map(), remap = new Map();
  let started = new WeakMap(), seen = new Set();
  P.now = now; Date.now = () => now() + dateOffset;
  window.requestAnimationFrame = cb => { if (!on) return real.raf(() => cb(now())); const id = -(++nextId); rafs.set(id, cb); return id; };
  window.cancelAnimationFrame = id => { if (id < 0) { rafs.delete(id); if (remap.has(id)) real.caf(remap.get(id)); } else real.caf(id); };
  const addTimer = (fn, ms, every, args) => { const id = -(++nextId); timers.set(id, {due: vt + Math.max(ms || 0, 0), fn, every, args}); return id; };
  window.setTimeout = (fn, ms, ...a) => on ? addTimer(fn, ms, 0, a) : real.st(fn, ms, ...a);
  window.setInterval = (fn, ms, ...a) => on ? addTimer(fn, Math.max(ms || 1, 1), Math.max(ms || 1, 1), a) : real.si(fn, ms, ...a);
  const clear = id => { if (id < 0) { timers.delete(id); if (remap.has(id)) { real.ct(remap.get(id)); real.ci(remap.get(id)); } } else { real.ct(id); real.ci(id); } };
  window.clearTimeout = clear; window.clearInterval = clear;
  const call = (fn, args) => { try { typeof fn === 'function' ? fn(...(args || [])) : (0, eval)(fn); } catch (e) { console.error(e); } };
  function seek() {
    for (const a of document.getAnimations()) {
      if (!started.has(a)) { started.set(a, vt - (a.currentTime || 0)); seen.add(a); try { a.pause(); } catch (e) {} }
      if (a.playState === 'finished') continue;
      const t = vt - started.get(a), end = a.effect ? a.effect.getComputedTiming().endTime : Infinity;
      try { if (end !== Infinity && t >= end) a.finish(); else a.currentTime = t; } catch (e) {}
    }
  }
  window.__vc = {
    enable() { vt = now(); on = true; seek(); },
    tick(ms) {
      const target = vt + ms;
      for (;;) {
        let pick = null, pickId = null;
        for (const [id, t] of timers) if (t.due <= target && (!pick || t.due < pick.due)) { pick = t; pickId = id; }
        if (!pick) break;
        vt = Math.max(vt, pick.due);
        if (pick.every) pick.due += pick.every; else timers.delete(pickId);
        call(pick.fn, pick.args);
      }
      vt = target;
      const batch = [...rafs.values()]; rafs.clear();
      for (const cb of batch) call(cb, [vt]);
      seek();
      return new Promise(res => real.raf(() => real.raf(() => res(1))));
    },
    disable() {
      offset = vt - real.now(); on = false;
      for (const [id, cb] of rafs) remap.set(id, real.raf(() => cb(now())));
      rafs.clear();
      for (const [id, t] of timers) remap.set(id, t.every ? real.si(t.fn, t.every, ...t.args) : real.st(t.fn, Math.max(t.due - vt, 0), ...t.args));
      timers.clear();
      for (const a of seen) { try { if (a.playState === 'paused') a.play(); } catch (e) {} }
      seen = new Set(); started = new WeakMap();
    },
  };
})();
"""



class Tab:
    def __init__(self, ws):
        self.ws, self._id, self._pending, self._reader = ws, 0, {}, None

    async def start(self):
        self._reader = asyncio.create_task(self._read())
        for domain in ("Page", "Runtime"):
            await self.send(f"{domain}.enable")
        await self.send("Page.addScriptToEvaluateOnNewDocument", source=CLOCK_JS)
        return self

    async def _read(self):
        async for raw in self.ws:
            msg = json.loads(raw)
            if "id" in msg and msg["id"] in self._pending:
                self._pending.pop(msg["id"]).set_result(msg)

    async def send(self, method, **params):
        self._id += 1
        fut = asyncio.get_running_loop().create_future()
        self._pending[self._id] = fut
        await self.ws.send(json.dumps({"id": self._id, "method": method, "params": params}))
        msg = await asyncio.wait_for(fut, 60)
        if "error" in msg:
            raise RuntimeError(f"{method}: {msg['error']}")
        return msg["result"]

    async def js(self, expr, await_promise=True):
        r = await self.send("Runtime.evaluate", expression=expr, awaitPromise=await_promise, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(json.dumps(r["exceptionDetails"])[:800])
        return r["result"].get("value")

    async def size(self, w, h, dpr=1.0):
        await self.send("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=dpr, mobile=False)

    async def goto(self, url, settle=2.0):
        await self.send("Page.navigate", url=url)
        for _ in range(160):
            await asyncio.sleep(0.25)
            if await self.js("document.readyState") == "complete":
                break
        await self.js("document.fonts.ready.then(()=>1)")
        await asyncio.sleep(settle)

    async def shot(self, path):
        data = base64.b64decode((await self.send("Page.captureScreenshot", format="png"))["data"])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)


class Rec:
    """async with Rec(tab, 'name') as rec: ... await rec.play(seconds) captures exactly FPS frames per virtual second."""

    def __init__(self, tab, name):
        self.t, self.dir = tab, FRAMES / name
        self.n = 0

    async def __aenter__(self):
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True)
        await self.t.js("__vc.enable()")
        return self

    async def play(self, seconds: float) -> None:
        for _ in range(round(seconds * FPS)):
            await self.t.js(f"__vc.tick({1000 / FPS})")
            data = (await self.t.send("Page.captureScreenshot", format="png", optimizeForSpeed=True))["data"]
            (self.dir / f"{self.n:05d}.png").write_bytes(base64.b64decode(data))
            self.n += 1

    async def __aexit__(self, *exc):
        await self.t.js("__vc.disable()")
        stamps = [i / FPS for i in range(self.n)]
        (self.dir / "stamps.json").write_text(json.dumps({"start": 0, "end": self.n / FPS, "stamps": stamps}))
        print(f"  {self.dir.name}: {self.n} frames, {self.n / FPS:.1f} s at {FPS} fps", flush=True)


class Browser:
    async def __aenter__(self):
        self.dir = tempfile.mkdtemp(prefix="cdp_chrome_")
        self.proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={self.dir}",
                                      "--hide-scrollbars", "--no-first-run", "--no-default-browser-check", "--disable-extensions",
                                      "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
                                      "--force-color-profile=srgb", "--window-size=1440,900", "about:blank"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            try:
                targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
                page = next(t for t in targets if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.25)
        self.ws = await websockets.connect(page["webSocketDebuggerUrl"], max_size=None, ping_interval=None)
        self.tab = await Tab(self.ws).start()
        return self.tab

    async def __aexit__(self, *exc):
        await self.ws.close()
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except Exception:
            self.proc.kill()
        shutil.rmtree(self.dir, ignore_errors=True)
