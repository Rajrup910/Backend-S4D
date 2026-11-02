"""Record the big-type scenes to frames. Usage: capture_scenes.py [scene-name-prefix ...]"""
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cdp import OUT_W, Browser, Rec
from scenes import H, PAGE, W, all_scenes


async def main(prefixes):
    scenes = all_scenes()
    base = Path(tempfile.gettempdir()) / "readme_media_scene.html"
    base.write_text(PAGE, encoding="utf-8")
    async with Browser() as t:
        await t.size(W, H, OUT_W / W)
        await t.goto(base.as_uri())
        for name, (html, dur) in scenes.items():
            if prefixes and not any(name.startswith(p) for p in prefixes):
                continue
            await t.js("document.getElementById('root').innerHTML=''")
            await asyncio.sleep(0.1)
            async with Rec(t, name) as rec:
                await t.js(f"show({repr(html)})".replace("\n", " "))
                await rec.play(dur)
            overflow = await t.js("""(()=>{const r=document.querySelector('.scene');const out=[];
              r.querySelectorAll('*').forEach(e=>{const b=e.getBoundingClientRect();if(b.width&&(b.bottom>481||b.right>1001||b.left<-1))out.push(e.tagName+':'+(e.innerText||'').slice(0,40))});
              const src=r.querySelector('.src');const cs=src?src.getBoundingClientRect().top:480;
              let maxb=0;[...r.children].forEach(c=>{if(!c.classList.contains('src')&&!c.classList.contains('top'))maxb=Math.max(maxb,c.getBoundingClientRect().bottom)});
              return {out:out.slice(0,4),bottom:Math.round(maxb),srcTop:Math.round(cs)}})()""")
            flag = " OVERFLOW" if overflow["out"] or overflow["bottom"] > overflow["srcTop"] else ""
            print(f"    {name}: content bottom {overflow['bottom']}, source top {overflow['srcTop']}{flag} {overflow['out'] or ''}", flush=True)


asyncio.run(main(sys.argv[1:]))
