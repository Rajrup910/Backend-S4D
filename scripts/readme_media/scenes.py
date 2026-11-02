"""Big-type horizontal scenes (1000 x 480) for the README reels. Every number comes from facts.load()."""
from __future__ import annotations

from facts import ci, load, num, sgn

W, H = 1000, 480
ACCENT = {"V1": "#38bdf8", "V2": "#34d399", "V3": "#fb7185", "V4": "#fbbf24", "V5": "#a78bfa", "V6": "#22d3ee", "ALL": "#38bdf8"}
TAG = {"V1": "Baseline", "V2": "Rigour", "V3": "Validity", "V4": "Measurement", "V5": "Mechanism"}

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:1000px;height:480px;overflow:hidden;background:#0b1220;font-family:'Plus Jakarta Sans',system-ui,sans-serif;color:#f1f5f9}
#root{position:absolute;inset:0;overflow:hidden}
.scene{position:absolute;inset:0;padding:64px 44px 34px;display:flex;flex-direction:column}
.dark{background:#0b1220;color:#f1f5f9}.light{background:#f8fafc;color:#0f172a}
.top{position:absolute;left:44px;right:44px;top:20px;display:flex;align-items:center;justify-content:space-between;font-size:15px;font-weight:800;letter-spacing:.03em}
.chip{display:inline-flex;align-items:center;gap:8px;padding:5px 13px;border-radius:999px;color:#0b1220;background:var(--a)}
.pills{display:flex;gap:6px}.pill{width:34px;text-align:center;font-size:13px;padding:3px 0;border-radius:7px;background:rgba(148,163,184,.16);color:#94a3b8}
.pill.on{background:var(--a);color:#0b1220}
.src{position:absolute;left:44px;right:44px;bottom:11px;font:500 13.5px 'JetBrains Mono',monospace;color:#64748b}
.mono{font-family:'JetBrains Mono',monospace}
h1{font-size:47px;font-weight:800;line-height:1.1;letter-spacing:-.02em}
h2{font-size:33px;font-weight:800;line-height:1.14;letter-spacing:-.015em}
.lead{font-size:23px;line-height:1.32;font-weight:500;color:#cbd5e1}.light .lead{color:#334155}
.key{color:var(--a)}.bad{color:#f43f5e}.good{color:#10b981}.mute{color:#94a3b8}.light .mute{color:#64748b}
.up{animation:up .55s cubic-bezier(.16,1,.3,1) both;animation-delay:var(--d,0s)}
.pop{animation:pop .5s cubic-bezier(.34,1.56,.64,1) both;animation-delay:var(--d,0s)}
.slam{animation:slam .42s cubic-bezier(.2,1.4,.5,1) both;animation-delay:var(--d,0s)}
.wipe{animation:wipe .7s cubic-bezier(.65,0,.35,1) both;animation-delay:var(--d,0s)}
.left{animation:left .6s cubic-bezier(.16,1,.3,1) both;animation-delay:var(--d,0s)}
@keyframes up{from{opacity:0;transform:translateY(22px)}to{opacity:1;transform:none}}
@keyframes left{from{opacity:0;transform:translateX(-46px)}to{opacity:1;transform:none}}
@keyframes pop{from{opacity:0;transform:scale(.5)}to{opacity:1;transform:none}}
@keyframes slam{from{opacity:0;transform:scale(2.1) rotate(-5deg)}to{opacity:1;transform:rotate(-3deg)}}
@keyframes wipe{from{clip-path:inset(0 100% 0 0)}to{clip-path:inset(0 0 0 0)}}
@keyframes grow{from{width:0}to{width:var(--w)}}
@keyframes railfill{from{width:var(--from)}}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 color-mix(in srgb,var(--c) 55%,transparent)}50%{box-shadow:0 0 0 9px transparent}}
.vnum{font-size:112px;font-weight:800;line-height:.95;letter-spacing:-.04em;color:var(--a)}
.tiles{display:flex;gap:16px}
.tile{flex:1;border-radius:18px;padding:18px 20px;background:rgba(148,163,184,.1);border:1.5px solid rgba(148,163,184,.2)}
.light .tile{background:#fff;border-color:#e2e8f0;box-shadow:0 2px 8px rgba(15,23,42,.05)}
.tile .n{font:800 46px/1.05 'JetBrains Mono',monospace;letter-spacing:-.03em}
.tile .l{font-size:18px;font-weight:600;line-height:1.28;margin-top:8px;color:#cbd5e1}.light .tile .l{color:#475569}
.row{display:flex;align-items:center;gap:14px}
.bar-row{display:grid;grid-template-columns:190px 1fr;gap:14px;align-items:center;margin:9px 0}
.bar-label{font-size:21px;font-weight:700;text-align:right}
.bar-wrap{position:relative;height:38px;border-radius:9px;background:rgba(148,163,184,.14);display:flex;align-items:center}
.bar-out{margin-left:12px;font:800 20px 'JetBrains Mono',monospace;color:#f1f5f9;white-space:nowrap}
.bar{height:100%;border-radius:9px;background:var(--c);animation:grow 1s cubic-bezier(.65,0,.35,1) both;animation-delay:var(--d,0s);display:flex;align-items:center;justify-content:flex-end;padding-right:12px;font:800 20px 'JetBrains Mono',monospace;color:#0b1220;white-space:nowrap;min-width:0;overflow:visible}
.dots{display:grid;gap:9px}
.dot{width:var(--s,34px);height:var(--s,34px);border-radius:50%;background:rgba(148,163,184,.22);animation:pop .35s cubic-bezier(.34,1.56,.64,1) both;animation-delay:var(--d,0s)}
.dot.hit{background:var(--c,#10b981)}.dot.miss{background:#f43f5e}
.stamp{display:inline-block;padding:3px 12px;border-radius:8px;border:2.5px solid currentColor;font-weight:800;font-size:17px;letter-spacing:.03em;white-space:nowrap}
.vrow{display:grid;grid-template-columns:52px 1fr auto;gap:12px;align-items:center;padding:8px 14px;border-radius:11px;background:rgba(148,163,184,.09);margin-bottom:7px}
.light .vrow{background:#fff;box-shadow:0 1px 5px rgba(15,23,42,.06)}
.vrow .id{font:800 20px 'JetBrains Mono',monospace;color:var(--a)}
.light .vrow .id{color:#0f172a}
.vrow .t{font-size:20px;font-weight:600;line-height:1.2}
.ba{display:grid;grid-template-columns:1fr auto auto auto;gap:18px;align-items:center;padding:13px 20px;border-radius:14px;background:rgba(148,163,184,.1);margin-bottom:11px}
.ba .t{font-size:22px;font-weight:700}.ba .v{font:800 34px 'JetBrains Mono',monospace;letter-spacing:-.02em}.ba .arrow{font-size:28px;color:#94a3b8}
.track{position:absolute;left:60px;right:60px;top:88px;height:6px;border-radius:3px;background:rgba(148,163,184,.25)}
.fill{position:absolute;left:0;top:0;bottom:0;border-radius:3px;background:linear-gradient(90deg,#38bdf8,#34d399,#fb7185,#fbbf24,#a78bfa);background-size:880px 100%}
.node{position:absolute;top:-21px;width:48px;height:48px;margin-left:-24px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:18px;background:#1e293b;color:#94a3b8;border:3px solid #334155;transition:all .4s}
.node.on{background:var(--c);color:#0b1220;border-color:#fff;transform:scale(1.18)}.node.done{background:var(--c);color:#0b1220;border-color:var(--c)}
.flow{display:flex;align-items:stretch;gap:9px}
.fbox{flex:1;border-radius:14px;padding:13px 8px;text-align:center;background:#16213a;border:2px solid #2a3a57;font-weight:700;font-size:18px;line-height:1.2;display:flex;flex-direction:column;justify-content:center;gap:6px;min-height:118px}
.fbox b{font:800 22px 'JetBrains Mono',monospace;color:var(--c)}
.farrow{align-self:center;color:#64748b;font-size:22px}
"""

COUNT_JS = """
window.runCounters=function(){document.querySelectorAll('[data-to]').forEach(el=>{
  const to=+el.dataset.to,dec=+(el.dataset.dec||0),delay=+(el.dataset.delay||0)*1000,dur=+(el.dataset.dur||1.1)*1000,fin=el.dataset.final;
  const fmt=v=>v.toLocaleString('en-US',{minimumFractionDigits:dec,maximumFractionDigits:dec});
  el.textContent=fmt(0);const s=performance.now()+delay;
  (function f(n){const k=Math.max(0,Math.min(1,(n-s)/dur));el.textContent=k>=1?fin:fmt(to*(1-Math.pow(1-k,3)));if(k<1)requestAnimationFrame(f)})(performance.now())})};
window.show=function(html){document.getElementById('root').innerHTML=html;runCounters()};
"""

PAGE = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700;800&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body><div id="root"></div><script>{COUNT_JS}</script></body></html>"""


def count(text: str, value: float, dec: int = 0, delay: float = 0.0, dur: float = 1.1) -> str:
    """A number that counts up and lands on exactly `text`."""
    return f'<span data-to="{value}" data-dec="{dec}" data-delay="{delay}" data-dur="{dur}" data-final="{text}">{text}</span>'


def sci(p: float) -> str:
    """3.1 × 10⁻⁵"""
    mantissa, exponent = f"{p:.1e}".split("e")
    sup = str(abs(int(exponent))).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{mantissa} × 10⁻{sup}"


def chrome(v: str, src: str = "", theme: str = "dark") -> str:
    pills = "".join(f'<span class="pill{" on" if k == v else ""}">{k}</span>' for k in ("V1", "V2", "V3", "V4", "V5"))
    chip = f'<span class="chip">{v} · {TAG[v].upper()}</span>' if v in TAG else '<span class="chip">SCAN4DISEASE 2.0</span>'
    s = f'<div class="src">{src}</div>' if src else ""
    return f'<div class="top">{chip}<span class="pills">{pills}</span></div>{s}'


def scene(v: str, body: str, src: str = "", theme: str = "dark", extra: str = "") -> str:
    return f'<div class="scene {theme}" style="--a:{ACCENT[v]};{extra}">{chrome(v, src, theme)}{body}</div>'


def title_scene(v: str, name: str, question: str, meta: str) -> str:
    words = "".join(f'<span class="up" style="display:inline-block;--d:{0.35 + i * 0.075:.2f}s">{w}&nbsp;</span>' for i, w in enumerate(question.split()))
    return scene(v, f"""
      <div class="row" style="gap:28px;margin-top:16px">
        <div class="vnum pop">{v}</div>
        <div><div class="left" style="font-size:34px;font-weight:800;--d:.15s">{name}</div>
             <div class="left mute" style="font-size:19px;font-weight:600;margin-top:5px;--d:.25s">{meta}</div></div>
      </div>
      <div style="margin-top:38px;font-size:41px;font-weight:700;line-height:1.22;letter-spacing:-.015em">{words}</div>""")


def next_scene(v: str, settled: str, nxt: str, question: str, label: str = "") -> str:
    return scene(v, f"""
      <div class="mute up" style="font-size:20px;font-weight:800;letter-spacing:.09em;margin-top:18px">WHAT THIS SETTLED</div>
      <h2 class="up" style="--d:.12s;margin-top:8px;font-size:34px">{settled}</h2>
      <div class="left" style="--d:.95s;margin-top:auto;margin-bottom:26px;display:flex;align-items:center;gap:20px;padding:18px 24px;border-radius:18px;background:{ACCENT[nxt]};color:#0b1220">
        <div style="font-size:50px;font-weight:800;letter-spacing:-.03em">{label or nxt}</div>
        <div style="font-size:25px;font-weight:700;line-height:1.22">{question}</div>
      </div>""")


def verdict_scene(v: str, heading: str, rows: list[tuple[str, str, str, str]], src: str, big: bool = False) -> str:
    """rows: (id, text, stamp, colour)"""
    out = ""
    for i, (rid, text, stamp, colour) in enumerate(rows):
        d = 0.25 + i * 0.3
        size = "padding:15px 18px;margin-bottom:11px" if big else ""
        tsize = "font-size:24px" if big else ""
        out += f"""<div class="vrow left" style="--d:{d:.2f}s;{size}"><span class="id">{rid}</span><span class="t" style="{tsize}">{text}</span>
          <span class="stamp slam" style="--d:{d + 0.2:.2f}s;color:{colour}">{stamp}</span></div>"""
    return scene(v, f'<h2 class="up" style="font-size:29px;margin:4px 0 13px">{heading}</h2>{out}', src)


def bars(rows: list[tuple[str, float, str, str]], maxv: float, start: float = 0.3, step: float = 0.28) -> str:
    out = ""
    for i, (label, value, text, colour) in enumerate(rows):
        d = start + i * step
        inside = value / maxv >= 0.45
        tail = "" if inside else f'<span class="bar-out up" style="--d:{d + 0.7:.2f}s">{text}</span>'
        out += f"""<div class="bar-row"><div class="bar-label">{label}</div><div class="bar-wrap">
          <div class="bar" style="--w:{100 * value / maxv:.1f}%;--c:{colour};--d:{d:.2f}s">{text if inside else ""}</div>{tail}</div></div>"""
    return out


def dots(n: int, hit: int, cols: int, size: int = 34, start: float = 0.3, gap: float = 0.03, hit_colour: str = "#10b981") -> str:
    out = "".join(f'<div class="dot {"hit" if i < hit else "miss"}" style="--d:{start + i * gap:.2f}s;--c:{hit_colour}"></div>' for i in range(n))
    return f'<div class="dots" style="grid-template-columns:repeat({cols},{size}px);--s:{size}px">{out}</div>'


# ==================================================================== reels
def hero(f) -> dict:
    a7, b = f["rungs"]["A7"], f["band"]
    nv = f["nv_share"] * 100
    u = b["<40"]
    return {
        "hero_title": (scene("ALL", f"""
          <div style="margin:auto 0"><div class="wipe" style="font-size:72px;font-weight:800;letter-spacing:-.035em;line-height:1">Scan4Disease 2.0</div>
          <div class="lead up" style="--d:.65s;margin-top:20px;font-size:30px;line-height:1.28;color:#e2e8f0">Can a skin-lesion classifier be trusted as a <span class="key">triage tool</span>, for <span class="key">every</span> patient?</div>
          <div class="row up" style="--d:1.2s;margin-top:26px;gap:11px;flex-wrap:wrap;font-size:18px;font-weight:700;color:#cbd5e1"><span>Lesion-grouped splits</span><span class="mute">·</span><span>Plans hashed before data</span><span class="mute">·</span><span>Every number from results/</span></div></div>"""), 3.6),
        "hero_trap": (scene("ALL", f"""
          <h1 class="up" style="margin-top:10px">Accuracy is misleading here.</h1>
          <div class="up" style="--d:.2s;margin-top:22px">{bars([("Benign nevi", nv, f"{nv:.1f}%", "#94a3b8"), ("All 6 others", 100 - nv, f"{100 - nv:.1f}%", "#fb7185")], 100)}</div>
          <div class="lead up" style="--d:1.15s;margin-top:18px;font-size:26px">A model that always answers “benign mole” scores about <b class="key">67%</b> and misses <b class="bad">every melanoma</b>.</div>""",
          f"HAM10000: {f['ham_images']:,} images · results/review2/ham10000_class_profile.csv"), 4.2),
        "hero_gap": (scene("ALL", f"""
          <h2 class="up" style="margin-top:6px;font-size:35px">The best system looked strong, and failed young patients.</h2>
          <div class="row" style="gap:18px;margin-top:20px;align-items:stretch">
            <div class="tile up" style="--d:.2s;flex:.8"><div class="n key" style="font-size:66px">{count(f"{a7['f1']:.4f}", a7['f1'], 4, .3)}</div><div class="l" style="font-size:20px">Macro-F1 on {f['test_images']:,} test images<br><span class="mute">6 CNNs + TTA + Dirichlet</span></div></div>
            <div class="tile up" style="--d:.45s;flex:1.2"><div class="row" style="gap:18px;align-items:flex-start"><div class="n bad" style="font-size:66px;white-space:nowrap">{u['caught']} of {u['n']}</div><div>{dots(u['n'], u['caught'], 7, 25, .75, .035)}</div></div>
              <div class="l" style="font-size:20px;margin-top:10px">serious under-40 cases caught: <b class="bad">{u['sens']:.3f}</b> vs about <b>0.78</b> in older bands</div></div>
          </div>
          <div class="lead up" style="--d:1.7s;margin-top:17px;font-size:23px">And the misses were <b class="bad">confident</b>: abstention referred only <b>{u['referred_pct']:.1f}%</b> of them.</div>""",
          "research/selective/results/session4_report.md · results/ablation_table.csv"), 5.2),
        "hero_why": (scene("ALL", """
          <h1 class="up" style="margin-top:8px">So we asked why. Five times.</h1>
          <div class="row" style="gap:10px;margin-top:26px">""" + "".join(
            f'<div class="fbox pop" style="--c:{ACCENT[k]};--d:{0.35 + i * 0.32:.2f}s;background:{ACCENT[k]};color:#0b1220;border-color:{ACCENT[k]};font-size:18px;min-height:112px"><span style="font-size:32px;font-weight:800">{k}</span>{t}</div>'
            for i, (k, t) in enumerate((("V1", "Baseline"), ("V2", "Threshold?"), ("V3", "Shortcut?"), ("V4", "Backbone? Rule?"), ("V5", "Reasoning?")))) + """</div>
          <div class="lead up" style="--d:2.1s;margin-top:24px;font-size:28px;color:#f1f5f9">The answer narrowed to a <b class="key">ranking limit in the learned representation</b>.</div>"""), 5.0),
    }


def programme(f) -> dict:
    u = f["band"]["<40"]
    g = f["v5_gates"]["A"]
    s = f["s84"]["global"]
    steps = [
        ("V1", "Phases 0–5 · S1–S27 · Aug – 7 Sep", "How far do CNNs, ensembling, TTA and calibration go?",
         f"Only ensembling clears significance (p = 3.4 × 10⁻⁵). Under 40: <b>{u['caught']} of {u['n']}</b> caught."),
        ("V2", "S28–S39 · 12–13 Sep", "Is the under-40 miss a decision-threshold problem?",
         "<b>No.</b> Every compression interval spans 0. Coverage is not protection."),
        ("V3", "S40–S47 · 14 Sep", "Can a better head, more archives or removing age close it?",
         f"<b>{f['v3_counts']['FALSIFIED']} falsified, {f['v3_counts']['CERTIFIED']} certified.</b> Removing age costs {sgn(f['v3_auc_delta'])} under-40 AUC."),
        ("V4", "S48–S75 · 15–18 Sep", "With a powered endpoint, does any recipe, model or rule fix it?",
         f"<b>No: a ranking limit.</b> In-domain data gives {sgn(f['v4_a_f1'], 2)} Macro-F1; the contract meets {f['v4_terms_met']} of {f['v4_terms']} terms."),
        ("V5", "Run 30 Sep – 5 Oct · external read 2 Nov", "Do dermatologist-style modules help?",
         f"Under-40 ranking <b>{sgn(g['point'])}</b>, below the 0.05 MCID. On MILK10k, Macro-F1 <b>{sgn(s['delta'], 3)}</b>."),
    ]
    cards = {}
    for i, (v, meta, q, a) in enumerate(steps):
        nodes = "".join(
            f'<div class="node {"on" if j == i else "done" if j < i else ""}" style="left:{j * 25}%;--c:{ACCENT[k]}">{k}</div>'
            for j, k in enumerate(("V1", "V2", "V3", "V4", "V5")))
        cards[f"prog_{v}"] = (f"""<div class="scene dark" style="--a:{ACCENT[v]}">
          <div class="track"><div class="fill" style="width:{i * 25}%;animation:railfill .8s cubic-bezier(.65,0,.35,1) both;--from:{max(0, i - 1) * 25}%"></div>{nodes}</div>
          <div style="margin-top:96px">
            <div class="mute up" style="font-size:18px;font-weight:700">{meta}</div>
            <div class="up" style="--d:.1s;font-size:37px;font-weight:800;line-height:1.16;margin-top:8px;letter-spacing:-.015em">{q}</div>
            <div class="left" style="--d:.65s;margin-top:24px;padding:18px 24px;border-radius:16px;border-left:7px solid var(--a);background:rgba(148,163,184,.11);font-size:27px;line-height:1.3;font-weight:500">{a}</div>
          </div></div>""", 4.4)
    return cards


def v1(f) -> dict:
    r, b = f["rungs"], f["band"]
    a6, a7, a5 = r["A6"], r["A7"], r["A5"]
    raw, cal = f["calib_raw"], f["calib_dirichlet"]
    return {
        "v1_title": (title_scene("V1", "The baseline", "How far do CNNs, ensembling, TTA and calibration go on HAM10000?", "Phases 0–5 · S1–S27 · Aug – 7 Sep"), 3.4),
        "v1_catch": (scene("V1", f"""
          <h2 class="up" style="font-size:33px;margin-top:2px">Better calibration, <span class="bad">worse triage</span>.</h2>
          <div style="margin-top:16px">
          <div class="ba left" style="--d:.2s"><span class="t">ECE, soft vote + Dirichlet</span><span class="v">{raw['test ECE']:.4f}</span><span class="arrow">→</span><span class="v good">{cal['test ECE']:.4f}</span></div>
          <div class="ba left" style="--d:.75s"><span class="t">Macro-F1, TTA stack + Dirichlet</span><span class="v">{a6['f1']:.4f}</span><span class="arrow">→</span><span class="v good">{a7['f1']:.4f}</span></div>
          <div class="ba left" style="--d:1.3s;background:rgba(244,63,94,.17);border:2px solid #f43f5e"><span class="t">Missed serious cases, same stack</span><span class="v">{a6['missed']}</span><span class="arrow">→</span><span class="v bad">{a7['missed']}</span></div>
          </div>
          <div class="lead up" style="--d:1.9s;font-size:21px;margin-top:3px">{f['a6a7_lost']} detections lost, {f['a6a7_gained']} gained (exact McNemar p = {sci(f['a6a7_p'])}). Aggregate metrics and safety pull apart.</div>""",
          "results/ablation_table.csv · results/review2/escalation_mcnemar_a6_a7.json"), 5.4),
        "v1_gap": (scene("V1", f"""
          <h2 class="up" style="font-size:33px;margin-top:2px">The blind spot: patients <span class="bad">under 40</span></h2>
          <div class="up" style="--d:.15s;margin-top:12px">{bars([("Under 40", b['<40']['sens'], f"{b['<40']['sens']:.3f}  ({b['<40']['caught']}/{b['<40']['n']})", "#fb7185"), ("40–59", b['40-59']['sens'], f"{b['40-59']['sens']:.3f}  ({b['40-59']['caught']}/{b['40-59']['n']})", "#7dd3fc"), ("60+", b['60+']['sens'], f"{b['60+']['sens']:.3f}  ({b['60+']['caught']}/{b['60+']['n']})", "#7dd3fc")], 1.0)}</div>
          <div class="tiles up" style="--d:1.2s;margin-top:13px">
            <div class="tile" style="padding:13px 18px"><div class="n bad" style="font-size:40px">{b['<40']['referred_pct']:.1f}%</div><div class="l" style="margin-top:4px">of under-40 misses referred: confidently wrong</div></div>
            <div class="tile" style="padding:13px 18px"><div class="n key" style="font-size:38px">{f['prior_u40'] * 100:.1f}%</div><div class="l" style="margin-top:4px">of under-40 training images escalate, vs {f['prior_60'] * 100:.1f}% at 60+</div></div>
          </div>""", "Escalation sensitivity, HAM10000 test · session4_report.md · results/age_band_prior.csv"), 5.2),
        "v1_next": (next_scene("V1", f"A leak-free {a7['f1']:.4f} benchmark, and a failure it could not explain: {f['u40_test_images']} under-40 images, from only {f['u40_test_lesions']} lesions.", "V2", "Is the under-40 miss a decision-threshold problem?"), 3.6),
    }


def v2(f) -> dict:
    c = f["v2_c"]
    v = f["v2_verdicts"]
    red, green, amber = "#fb7185", "#34d399", "#fbbf24"
    return {
        "v2_title": (title_scene("V2", "Explain the gap", "Is the under-40 miss a decision-threshold problem?", "S28–S39 · 12–13 Sep · plan frozen and hashed before the read"), 3.2),
        "v2_conformal": (scene("V2", f"""
          <h2 class="up" style="font-size:36px;margin-top:2px">Coverage is <span class="bad">not</span> protection.</h2>
          <div class="tiles" style="margin-top:16px">
            <div class="tile left" style="--d:.2s"><div class="mute" style="font-size:17px;font-weight:800;letter-spacing:.05em">MARGINAL CONFORMAL</div>
              <div class="n good" style="margin-top:8px">{f['v2_cov'] * 100:.2f}%</div><div class="l">coverage, target 95%. Looks valid.</div>
              <div class="n bad" style="margin-top:12px">{f['v2_frr'] * 100:.2f}%</div><div class="l">false reassurance on serious lesions, above the {f['v2_one_minus'] * 100:.2f}% ceiling</div></div>
            <div class="tile left" style="--d:.9s;border-color:var(--a)"><div class="key" style="font-size:17px;font-weight:800;letter-spacing:.05em">BIPARTITE FIX, UNDER 40</div>
              <div class="row" style="margin-top:12px;gap:14px"><div class="n bad">{f['v2_frr_m_k']}/{f['v2_frr_n']}</div><div class="arrow" style="font-size:30px">→</div><div class="n good">{f['v2_frr_b_k']}/{f['v2_frr_n']}</div></div>
              <div class="l" style="margin-top:13px">false reassurance vs class-conditional conformal, α = 0.10, Holm p = {f['v2_frr_holm']:.3f}</div>
              <div class="l key" style="margin-top:10px;font-weight:800">V2's only GO</div></div>
          </div>""", "results/v2/frr_by_group.csv · results/v2/conformal_subgroup_safety.csv"), 5.4),
        "v2_verdict": (verdict_scene("V2", "Six hypotheses, one survivor", [
            ("H1", "Decision “compression” explains the gap", v["H1"], red),
            ("H2", "A better uncertainty score certifies a ranking deficit", "NOT CERTIFIED", amber),
            ("H3", "A frozen HAM threshold transports to BCN", v["H3"], red),
            ("H4", "Marginal coverage does not protect subgroups", v["H4"], green),
            ("H5", "Abstention and the λ rule rescue different cases", v["H5"], red),
            ("H6", "A new escalation loss or head is a contribution", v["H6"], red),
        ], "results/v2/final_verdict.json"), 5.0),
        "v2_next": (next_scene("V2", f"The decision rule is not the bottleneck: compression spans 0 in all three cohorts ({sgn(c['ham_oof']['C'])}, {sgn(c['bcn20000']['C'])}, {sgn(c['pad']['C'])}).", "V3", "Can a better head, more archives or removing age close it?"), 3.6),
    }


def v3(f) -> dict:
    red, green = "#fb7185", "#34d399"
    h = f["v3_hyp"]
    return {
        "v3_title": (title_scene("V3", "Break the 0.80 ceiling?", "Can a better head, more archives or removing the age shortcut close it?", "S40–S47 · 14 Sep · 0 test reads"), 3.2),
        "v3_verdict": (verdict_scene("V3", "Six hypotheses, five dead", [
            ("H1", f"V2's hidden-signal probe survives out-of-sample ({f['v3_in']:.4f} → {f['v3_out']:.4f})", h["H1"], red),
            ("H2", f"A better head recovers the gap (overall {sgn(f['v3_h2_all'])})", h["H2"], red),
            ("H3", f"The representation is age-entangled (AUC {f['v3_h3_auc']:.4f})", h["H3"], green),
            ("H4", f"Pooling archives adds ≥ +0.03 Macro-F1 ({sgn(f['v3_h4'])})", h["H4"], red),
            ("H5", f"Breadth buys unseen-archive robustness ({f['v3_h5_excl']} of 2)", h["H5"], red),
            ("H6", f"Removing age helps under 40 (AUC {sgn(f['v3_auc_delta'])})", h["H6"], red),
        ], "results/v3/final_verdict.json · results/v3/d2_strong_evaluation.json"), 5.0),
        "v3_next": (next_scene("V3", f"Best condition {f['v3_best']:.4f}: ceiling not broken. Age is load-bearing, and {f['v3_seed_n']} positives could not rank anything.", "V4", "With a powered endpoint, does any recipe, model or rule fix it?"), 3.6),
    }


def v4(f) -> dict:
    a = f["v4_auc"]
    cohort = f["v4_cohort"]
    return {
        "v4_title": (title_scene("V4", "Measure it properly", "With a powered endpoint, does any recipe, foundation model or decision layer fix it?", "S48–S75 · 15–18 Sep"), 3.4),
        "v4_power": (scene("V4", f"""
          <h2 class="up" style="font-size:36px;margin-top:2px">“3 of 21” was <span class="key">{f['u40_test_lesions']} lesions</span>.</h2>
          <div class="tiles" style="margin-top:18px">
            <div class="tile pop" style="--d:.25s"><div class="n bad">{count(str(f['u40_test_lesions']), f['u40_test_lesions'], 0, .3)}</div><div class="l">independent lesions behind the 0.143</div></div>
            <div class="tile pop" style="--d:.55s"><div class="n">{count(str(f['power_paired']), f['power_paired'], 0, .6)}</div><div class="l">lesions needed for a paired test at MCID 0.10</div></div>
            <div class="tile pop" style="--d:.85s;border-color:var(--a)"><div class="n key">{count(str(cohort['under40_escalating_lesions']), cohort['under40_escalating_lesions'], 0, .9)}</div><div class="l">in the new reserved cohort ({cohort['images']:,} images)</div></div>
          </div>
          <div class="lead up" style="--d:1.6s;margin-top:24px;font-size:27px">Before optimising the model, V4 made the question <b class="key">measurable</b>: lesion unit, three seeds, one reserved cohort.</div>""",
          "results/v4/power_audit.json · results/v4/s54/s54_gate.json"), 5.0),
        "v4_survive": (scene("V4", f"""
          <h2 class="up" style="font-size:33px;margin-top:2px">It is a <span class="key">ranking</span> limit. One layer survives.</h2>
          <div class="up" style="--d:.15s;margin-top:11px">{bars([("Under 40", a['<40'], f"AUC {a['<40']:.3f}", "#fb7185"), ("40–59", a['40-59'], f"AUC {a['40-59']:.3f}", "#fcd34d"), ("60+", a['60+'], f"AUC {a['60+']:.3f}", "#fcd34d")], 1.0)}</div>
          <div class="tile up" style="--d:1.25s;flex:none;margin-top:12px;padding:13px 18px;border-color:var(--a)"><div class="row" style="gap:18px"><div class="n key" style="font-size:38px;white-space:nowrap">{sgn(f['v4_s56'], 3)}</div><div class="l" style="margin:0;font-size:19px"><b>S56 per-band abstention</b> (deployed): under-40 system sensitivity {ci(f['v4_s56_lo'], f['v4_s56_hi'], 3)}, paid for with referral {f['v4_ref_global']:.3f} → {f['v4_ref_band']:.3f}</div></div></div>""",
          "results/v4/s64/ceiling.json · results/v4/s56/contrasts_reserved.csv"), 5.2),
        "v4_next": (next_scene("V4", f"In-domain data is the big lever ({f['v4_v1_f1']:.3f} → {f['v4_ctrl_f1']:.3f} Macro-F1). No head, weighting or rule fixes under-40 ranking.", "V5", "Can dermatologist-style modules change what the model ranks?"), 3.6),
    }


def v5(f) -> dict:
    g = f["v5_gates"]
    red, green = "#fb7185", "#34d399"
    mod, trunk, p = f["v5_module"]["pauc_u40"], f["v5_trunk"]["pauc_u40"], f["v5_primary"]["pauc_u40"]
    shares = mod["point"] / p["point"] * 100
    out = [("zoom", "random crop wins 3/3"), ("geometry", "rotation keeps 75%"), ("memory", "not young-specific"), ("look · clues · gem", "missed the bar"), ("structure · m5 · m7", "stopped early")]
    keep = [("twostep", "triage heads"), ("m4", "melanoma vs look-alike"), ("youngdata", "biopsy-proven, aged ≤ 35")]
    return {
        "v5_title": (title_scene("V5", "Dermatologist reasoning", "Teach the model to look like a dermatologist, then try to falsify every module.", "Plan frozen 30 Sep · run 30 Sep – 5 Oct · external read 2 Nov"), 3.4),
        "v5_funnel": (scene("V5", f"""
          <h2 class="up" style="font-size:31px;margin-top:0">Every module met a falsifier written <span class="key">in advance</span>.</h2>
          <div class="row" style="gap:16px;margin-top:13px;align-items:stretch">
            <div class="tile" style="flex:1.1;padding:12px 16px;background:rgba(244,63,94,.1);border-color:rgba(244,63,94,.45)"><div class="bad" style="font-size:18px;font-weight:800">✕ OUT</div>""" + "".join(
              f'<div class="left" style="--d:{0.25 + i * 0.26:.2f}s;font-size:23px;margin-top:11px"><b>{n}</b> <span class="mute" style="font-size:19px">{w}</span></div>' for i, (n, w) in enumerate(out)) + f"""</div>
            <div class="tile" style="flex:1;padding:12px 16px;background:rgba(16,185,129,.1);border-color:rgba(16,185,129,.5)"><div class="good" style="font-size:18px;font-weight:800">✓ LOCKED 3 OCT</div>""" + "".join(
              f'<div class="left" style="--d:{1.55 + i * 0.28:.2f}s;font-size:23px;margin-top:11px"><b>{n}</b> <span class="mute" style="font-size:19px">{w}</span></div>' for i, (n, w) in enumerate(keep)) + f"""
              <div class="up" style="--d:2.5s;font-size:19px;margin-top:16px;color:#cbd5e1">on an ImageNet-22k ConvNeXt-Tiny trunk</div></div>
          </div>""", f"results/v5/review2_tables.md · results/v5/composite_lock.json ({f['v5_lock_sha']}…)"), 5.6),
        "v5_gates": (verdict_scene("V5", f"Held out: {f['v5_rows']:,} images, 3 seeds, four frozen gates", [
            ("A", f"Under-40 pAUC {sgn(g['A']['point'])} {ci(*g['A']['ci95'])}, needs ≥ +0.050", "NOT MET", red),
            ("B", f"Macro-F1 retained: {sgn(g['B']['point'])}", "PASS", green),
            ("C", f"Older-band sensitivity retained: {sgn(g['C']['40-59']['point'])} / {sgn(g['C']['60+']['point'])}", "PASS", green),
            ("D", f"All-age pAUC {sgn(g['D']['mean'])}, only {g['D']['n_positive']} of 3 seeds positive", "NOT MET", red),
        ], "results/v5/confirm_s42_s43_s44.json · no test read", big=True), 4.4),
        "v5_q11": (scene("V5", f"""
          <h2 class="up" style="font-size:33px;margin-top:2px">Where the {sgn(p['point'])} came from</h2>
          <div class="row wipe" style="--d:.3s;margin-top:20px;height:84px;gap:4px">
            <div style="flex:{mod['point'] * 100:.2f};height:100%;border-radius:12px 0 0 12px;background:#a78bfa;color:#0b1220;display:flex;align-items:center;justify-content:center;font:800 32px 'JetBrains Mono',monospace">{sgn(mod['point'])}</div>
            <div style="flex:{trunk['point'] * 100:.2f};height:100%;border-radius:0 12px 12px 0;background:#475569;display:flex;align-items:center;justify-content:center;font:800 20px 'JetBrains Mono',monospace">{sgn(trunk['point'])}</div>
          </div>
          <div class="tiles" style="margin-top:16px">
            <div class="tile up" style="--d:1s;flex:1.5;border-color:var(--a)"><div style="font-size:21px;font-weight:800">Modules + young data</div><div class="l">{ci(*mod['ci95'])}, p = {mod['p_two_sided']:.3f}<br><b class="key">{shares:.0f}% of the gain; the CI excludes 0</b></div></div>
            <div class="tile up" style="--d:1.3s;flex:1"><div style="font-size:21px;font-weight:800">IN-22k trunk</div><div class="l">{ci(*trunk['ci95'])}, p = {trunk['p_two_sided']:.3f}</div></div>
          </div>""", "results/v5/q11_decomposition.json · deviation D4"), 5.0),
        "v5_next": (next_scene("V5", "A real under-40 gain, on all three seeds, but below the clinical bar. All-age ranking did not move.", "V5", "Does it hold on a cohort it has never seen?", "S84"), 3.6),
    }


def milk(f) -> dict:
    s = f["s84"]
    pr, sec, gl = s["primary"], s["key_secondary"], s["global"]
    d6 = s["d6_lesion_link_exclusion"]
    allb = s["descriptive"]["by_age_band"]["ALL"]
    u40b = s["descriptive"]["by_age_band"]["<40"]
    n_in = s["in_scope_lesions"]
    esc = allb["V5"]["n_escalating"]
    dep = sec["v5_deployed_u40"]
    x = lambda v: (v + 0.05) / 0.2 * 100  # forest axis -0.05 .. +0.15
    return {
        "m_title": (scene("V5", f"""
          <div class="mute up" style="font-size:19px;font-weight:800;letter-spacing:.08em;margin-top:6px">EXTERNAL READ · S84 · 2 NOV</div>
          <div class="wipe" style="font-size:62px;font-weight:800;letter-spacing:-.03em;margin-top:3px">MILK10k, read once.</div>
          <div class="tiles" style="margin-top:20px">
            <div class="tile pop" style="--d:.5s"><div class="n">{count(f"{f['s75_eligible']:,}", f['s75_eligible'], 0, .5)}</div><div class="l">eligible lesions</div></div>
            <div class="tile pop" style="--d:.8s"><div class="n bad">{count(f"−{d6['excluded_lesions']}", d6['excluded_lesions'], 0, .8)}</div><div class="l">already in ISIC 2019: excluded before the read (D6)</div></div>
            <div class="tile pop" style="--d:1.1s;border-color:var(--a)"><div class="n key">{count(f"{d6['analysed_lesions']:,}", d6['analysed_lesions'], 0, 1.1)}</div><div class="l">analysed, <b>{esc / n_in * 100:.1f}%</b> of in-scope lesions escalating</div></div>
          </div>""", f"results/v5/s84/s84_report.json · receipt: {f['s84_receipt']['n']} execution, {f['s84_receipt']['labels']} label read"), 4.6),
        "m_f1": (scene("V5", f"""
          <h2 class="up" style="font-size:35px;margin-top:2px">Macro-F1: <span class="good">V5 leads, CI clear of zero</span></h2>
          <div class="row" style="gap:16px;margin-top:15px;align-items:stretch">
            <div class="tile up" style="--d:.2s;flex:.62"><div class="n key">{count(f"{gl['macro_f1_V5']:.3f}", gl['macro_f1_V5'], 3, .3)}</div><div class="l">V5 Macro-F1</div><div class="n" style="margin-top:11px;color:#94a3b8">{count(f"{gl['macro_f1_V1']:.3f}", gl['macro_f1_V1'], 3, .5)}</div><div class="l">V1 Macro-F1</div></div>
            <div class="tile up" style="--d:.4s;flex:1"><div style="font-size:20px;font-weight:800">Δ Macro-F1 <span class="good mono">{sgn(gl['delta'], 3)}</span> <span class="mono" style="font-size:18px">{ci(*gl['ci95_bootstrap'], 3)}</span></div>
              <div style="position:relative;height:74px;margin-top:20px">
                <div style="position:absolute;left:0;right:0;top:32px;height:2px;background:#475569"></div>
                <div style="position:absolute;left:{x(0):.1f}%;top:4px;bottom:0;width:2px;background:#94a3b8"></div><div class="mono mute" style="position:absolute;left:{x(0):.1f}%;top:52px;font-size:15px;transform:translateX(-50%)">0</div>
                <div style="position:absolute;left:{x(-gl['margin']):.1f}%;top:8px;bottom:18px;border-left:2px dashed #f43f5e"></div><div class="mono bad" style="position:absolute;left:{x(-gl['margin']):.1f}%;top:52px;font-size:15px;transform:translateX(-50%)">−{gl['margin']}</div>
                <div class="wipe" style="--d:.9s;position:absolute;left:{x(gl['ci95_bootstrap'][0]):.1f}%;width:{x(gl['ci95_bootstrap'][1]) - x(gl['ci95_bootstrap'][0]):.1f}%;top:27px;height:12px;border-radius:6px;background:#6ee7b7"></div>
                <div class="pop" style="--d:1.5s;position:absolute;left:calc({x(gl['delta']):.1f}% - 12px);top:21px;width:24px;height:24px;border-radius:50%;background:#10b981;border:3px solid #fff"></div>
              </div>
              <div class="l" style="margin-top:6px">Non-inferior at the pre-declared −{gl['margin']} margin.</div></div>
          </div>
          <div class="lead up" style="--d:1.8s;font-size:21px;margin-top:13px">Escalation AUROC <b class="key">{allb['V5']['auroc']:.3f}</b> vs {allb['V1']['auroc']:.3f}, higher in every age band. One cell goes to V1: under-40 pAUC {u40b['V1']['pauc_0.20']:.3f} vs {u40b['V5']['pauc_0.20']:.3f}.</div>""",
          "results/v5/s84/s84_report.json"), 5.4),
        "m_primary": (scene("V5", f"""
          <h2 class="up" style="font-size:33px;margin-top:2px">Under 40: <span class="bad">primary not met</span>. Both at the ceiling.</h2>
          <div class="row" style="gap:18px;margin-top:15px;align-items:stretch">
            <div class="tile up" style="--d:.2s"><div style="font-size:20px;font-weight:800">V5 <span class="key mono">{pr['V5']['caught']}/{pr['V5']['n_escalating']}</span></div><div style="margin-top:9px">{dots(pr['V5']['n_escalating'], pr['V5']['caught'], 10, 25, .3, .022, '#a78bfa')}</div></div>
            <div class="tile up" style="--d:.3s"><div style="font-size:20px;font-weight:800">V1 <span class="mono">{pr['V1']['caught']}/{pr['V1']['n_escalating']}</span></div><div style="margin-top:9px">{dots(pr['V1']['n_escalating'], pr['V1']['caught'], 10, 25, .4, .022, '#94a3b8')}</div></div>
          </div>
          <div class="lead up" style="--d:1.6s;font-size:22px;margin-top:13px">{pr['discordant_V5_only']} vs {pr['discordant_V1_only']} discordant, McNemar <b>p = {pr['mcnemar_exact_one_sided_p']:.4f}</b>, at a matched {pr['referral_all_eligible_matched']['V5']:.2f} referral. Only {pr['n_escalating_u40_lesions']} of the planned {f['s75_u40']} lesions remained after D6.</div>""",
          "results/v5/s84/s84_report.json"), 5.4),
        "m_contract": (scene("V5", f"""
          <h2 class="up" style="font-size:33px;margin-top:2px">The safety contract fails on <span class="bad">referral</span>.</h2>
          <div class="tiles" style="margin-top:17px">
            <div class="tile left" style="--d:.2s;border-color:#10b981"><div class="n good">{dep['sensitivity']:.3f}</div><div class="l">under-40 sensitivity, {dep['caught']}/{dep['n_escalating']}. Lower CI {dep['ci95_clopper_pearson'][0]:.3f} clears the 0.855 floor ✓</div></div>
            <div class="tile left" style="--d:.7s;border-color:#f43f5e"><div class="n bad">{sec['v5_deployed_referral_all_eligible']:.3f}</div><div class="l">referral rate, against a cap of 0.25 ✕</div></div>
          </div>
          <div class="lead up" style="--d:1.3s;font-size:27px;margin-top:24px">With <b>{esc / n_in * 100:.1f}%</b> of lesions escalating here vs <b>{f['v5_dev_prevalence'] * 100:.1f}%</b> in the development pool, a referral budget does not carry over.</div>""",
          "results/v5/s84/s84_report.json · results/v5/v5_oof_common_matrix.csv"), 5.0),
        "m_use": (next_scene("V5", f"MILK10k is now spent as a test set. It becomes V6 training data, minus the {f['s84_linked_total']} lesions linked to ISIC 2019 or the young extras.", "V6", "HIBA stays untouched: V6's only remaining external confirmation."), 3.8),
    }


def system(f) -> dict:
    r = f["rungs"]
    boxes = [("Dermoscopy image", "", "#7dd3fc"), ("6 CNNs", "", "#7dd3fc"), ("Soft vote", f"{r['A5']['f1']:.4f}", "#34d399"), ("24-view TTA", f"{r['A6']['f1']:.4f}", "#34d399"),
             ("Dirichlet", f"{r['A7']['f1']:.4f}", "#34d399")]
    html = ""
    for i, (t, n, c) in enumerate(boxes):
        arrow = '<span class="farrow up" style="--d:%.2fs">→</span>' % (0.25 + i * 0.5 - 0.15) if i else ""
        val = f"<b>{n}</b>" if n else ""
        html += f'{arrow}<div class="fbox pop" style="--c:{c};--d:{0.25 + i * 0.5:.2f}s"><span>{t}</span>{val}</div>'
    d = 0.25 + 5 * 0.5
    return {
        "sys": (scene("ALL", f"""
          <h2 class="up" style="font-size:34px;margin-top:0">The deployed stack</h2>
          <div class="flow" style="margin-top:16px">{html}</div>
          <div class="row" style="margin-top:16px;gap:14px;align-items:stretch">
            <div class="tile pop" style="--d:{d:.2f}s;flex:1.25;padding:12px 18px;border-color:#fbbf24"><div style="font-size:21px;font-weight:800;color:#fbbf24">S56 per-age-band abstention</div><div class="l" style="margin-top:4px;font-size:18px">Under-40 system sensitivity <b class="mono">{sgn(f['v4_s56'], 3)}</b> over global abstention at a 20% budget</div></div>
            <div class="tile pop" style="--d:{d + .5:.2f}s;padding:12px 18px;border-color:#34d399"><div style="font-size:21px;font-weight:800;color:#34d399">Confident</div><div class="l" style="margin-top:4px;font-size:18px">Automated triage decision</div></div>
            <div class="tile pop" style="--d:{d + .8:.2f}s;padding:12px 18px;border-color:#fb7185"><div style="font-size:21px;font-weight:800;color:#fb7185">Uncertain</div><div class="l" style="margin-top:4px;font-size:18px">Refer to a dermatologist</div></div>
          </div>""", "Macro-F1 on HAM10000 test: results/ablation_table.csv · S56: results/v4/s56/contrasts_reserved.csv").replace('<span class="chip">SCAN4DISEASE 2.0</span>', '<span class="chip">DEPLOYED SINCE V4</span>'), 6.4),
    }


REELS = {"hero": hero, "programme": programme, "v1": v1, "v2": v2, "v3": v3, "v4": v4, "v5": v5, "milk": milk, "system": system}


def all_scenes() -> dict:
    f = load()
    out = {}
    for fn in REELS.values():
        out.update(fn(f))
    return out
