"""Record the real infographic pieces (charts, pipeline strips, simulator, fold machine), each lifted out of the page
and shown alone at README-legible size. The page's own data and animation code run unchanged; only the framing is added.
Usage: capture_real.py [segment-name-prefix ...]"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cdp import REPO, Browser, Rec
from facts import ci, load, num, sgn
from scenes import ACCENT, TAG, sci

PAGE = (REPO / "review2_infographic.html").as_uri()

STAGE_JS = """
window.__stage=function(o){let s=document.getElementById('__stage');if(s)s.remove();
  s=document.createElement('div');s.id='__stage';
  const pills=['V1','V2','V3','V4','V5'].map(k=>`<span class="p${k===o.v?' on':''}">${k}</span>`).join('');
  s.innerHTML=`<div class="tp"><span class="ch">${o.v} · ${o.tag}</span><span class="ps">${pills}</span></div>
    <div class="tt">${o.title}</div><div id="__slot"></div>${o.take?`<div class="tk">${o.take}</div>`:''}${o.src?`<div class="sr">${o.src}</div>`:''}`;
  s.style.setProperty('--a',o.accent);document.body.appendChild(s);return true};
"""

STAGE_CSS = """
html,body{overflow:hidden!important}
#navPill,#toV6,#rrPeek,.rr-back,#topNav{display:none!important}
#__stage{position:fixed;inset:0;z-index:999999;background:#f8fafc;color:#0f172a;display:flex;flex-direction:column;padding:VPADpx HPADpx 6px;font-family:'Plus Jakarta Sans',sans-serif}
#__stage .tp{display:flex;justify-content:space-between;align-items:center;font-weight:800;font-size:CHIPpx;letter-spacing:.03em}
#__stage .ch{padding:3px 10px;border-radius:999px;background:var(--a);color:#0b1220}
#__stage .ps{display:flex;gap:4px}#__stage .p{width:PILLpx;text-align:center;font-size:PFONTpx;padding:2px 0;border-radius:5px;background:#e2e8f0;color:#64748b}#__stage .p.on{background:var(--a);color:#0b1220}
#__stage .tt{font-size:TITLEpx;font-weight:800;letter-spacing:-.015em;margin-top:GAPpx;line-height:1.15}
#__slot{flex:1;min-height:0;position:relative;margin-top:GAPpx}
#__stage .tk{font-size:TAKEpx;font-weight:700;line-height:1.25;margin-top:GAPpx;color:#0f172a;animation:__up .5s cubic-bezier(.16,1,.3,1) 1.2s both}
#__stage .sr{font:500 SRCpx 'JetBrains Mono',monospace;color:#64748b;margin-top:3px}
@keyframes __up{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
"""


def css(width: int) -> str:
    """Sizes are given in final pixels, so every stage has the same type size once captured at 1000 px wide."""
    values = {"VPAD": 20, "HPAD": 36, "CHIP": 15, "PILL": 34, "PFONT": 13, "TITLE": 29, "GAP": 9, "TAKE": 23, "SRC": 13.5}
    out = STAGE_CSS
    for key, value in values.items():
        out = out.replace(key + "px", f"{value * width / 1000:.2f}px")
    return out


async def setup(t, width, height):
    await t.size(width, height, 1000 / width)
    await t.goto(PAGE, settle=2.5)
    await t.js("""(()=>{const s=document.createElement('style');s.id='__rec';document.head.appendChild(s);})()""")
    await t.js(STAGE_JS)
    await t.js(f"document.getElementById('__rec').textContent={json.dumps(css(width))}")


async def stage(t, v, title, take="", src=""):
    await t.js(f"__stage({json.dumps(dict(v=v, tag=TAG[v].upper(), accent=ACCENT[v], title=title, take=take, src=src))})")


async def chart_scene(t, name, v, tab, canvas, title, take, src, hold=3.6, flatten=False):
    await t.js(f"switchTab('{tab}')")
    await asyncio.sleep(0.9)
    await t.js(f"document.getElementById('{canvas}').scrollIntoView({{block:'center'}})")
    for _ in range(40):
        if await t.js(f"!!Chart.getChart(document.getElementById('{canvas}'))"):
            break
        await asyncio.sleep(0.15)
    await asyncio.sleep(1.2)
    await stage(t, v, title, take, src)
    flatten = "true" if flatten else "false"
    await t.js(f"""(()=>{{const c=document.getElementById('{canvas}');const slot=document.getElementById('__slot');slot.appendChild(c);
        c.style.width='100%';c.style.height='100%';const ch=Chart.getChart(c);ch.options.maintainAspectRatio=false;ch.options.responsive=true;
        if({flatten}){{ch.options.scales.y.ticks.callback=function(v){{return this.getLabelForValue(v)}};ch.options.scales.y.afterFit=a=>{{a.width=262}}}};ch.resize();ch.update('none')}})()""")
    await asyncio.sleep(0.5)
    async with Rec(t, name) as rec:
        await t.js(f"(()=>{{const ch=Chart.getChart(document.getElementById('{canvas}'));ch.reset();ch.update()}})()")
        await rec.play(hold)


STRIP_CSS = """
#__slot .pipeline-step-card>div:first-child>:nth-child(n+4){display:none!important}
#__slot .pipeline-step-card h4{font-size:18px!important;line-height:1.16!important;margin:9px 0!important;min-height:104px!important;align-items:flex-start!important}
#__slot [id^=pipeline-grid]{display:grid!important;grid-template-columns:repeat(6,minmax(0,1fr))!important;gap:9px!important;height:100%;margin:0!important;padding:0!important}
#__slot .pipeline-step-card{padding:12px!important;min-width:0}
#__slot .pipeline-step-card>div:first-child>div:first-child span{font-size:13px!important;max-width:none!important}
#__slot .pipeline-step-card>div:first-child>div:first-child span:first-child{width:26px!important;height:26px!important;font-size:14px!important}
#__slot .pipeline-step-card>div:first-child>div:nth-child(3){padding:9px 0!important}
#__slot .pipeline-step-card>div:first-child>div:nth-child(3) svg{width:44px!important;height:44px!important}
#__slot .pipeline-step-card>div:last-child>div{display:block!important;font-size:14px!important;line-height:1.22!important;padding:6px 8px!important;overflow-wrap:anywhere}
#__slot .pipeline-step-card>div:last-child svg{display:none!important}
#__slot [class*=arrow],#__slot .pipeline-connector{display:none!important}
"""


async def strip_scene(t, name, v, tab, key, title, pills=True):
    await t.js(f"switchTab('{tab}')")
    await asyncio.sleep(1.0)
    await stage(t, v, title)
    await t.js(f"document.getElementById('__rec').textContent+={json.dumps(STRIP_CSS)}")
    await t.js(f"""(()=>{{const g=document.getElementById('pipeline-grid-{key}');document.getElementById('__slot').appendChild(g);
        g.querySelectorAll('.pipeline-step-card').forEach(c=>{{c.parentElement.style.height='100%'}})}})()""")
    # V2's pills list output filenames, which cannot wrap inside a card this narrow
    await t.js(f"document.querySelectorAll('#__slot .pipeline-step-card>div:last-child').forEach(e=>e.style.display='{'' if pills else 'none'}')")
    await asyncio.sleep(0.4)
    async with Rec(t, name) as rec:
        await t.js(f"triggerStepCascade(document.getElementById('pipeline-grid-{key}'))")
        await rec.play(1.0)
        await t.js(f"playStepSequence('{key}')")
        await rec.play(6.1)


async def sim_scene(t, f):
    b = f["rungs"]
    await t.js("switchTab('tab-simulators')")
    await asyncio.sleep(1.0)
    await stage(t, "V1", "Abstention: refer the uncertain, keep the confident", "", "results/ablation_table.csv · rungs B1–B4, margin score, HAM10000 test")
    await t.js("""(()=>{const sl=document.getElementById('abstentionSlider');const card=sl.closest('.space-y-5,.bg-white,[class*=rounded-2xl]');
        const slot=document.getElementById('__slot');slot.appendChild(card);card.style.height='100%';
        card.querySelectorAll('h3,p').forEach(e=>e.style.display='none');
        [...card.querySelectorAll('div')].filter(e=>/^Based on empirical/.test(e.textContent.trim())).forEach(e=>e.style.display='none');
        [...card.querySelectorAll('span,div')].forEach(e=>{if(/Abstention Gate|Deferral$/.test(e.textContent)&&e.children.length===0&&e.id!=='abstentionValText')e.style.display='none'});
        const grid=document.getElementById('simMacroF1').closest('.grid');grid.style.gridTemplateColumns='repeat(4,minmax(0,1fr))';
        // "malignancies" would be wrong for this set: AKIEC is pre-malignant. The numbers are untouched.
        [...grid.querySelectorAll('div,span')].forEach(e=>{if(e.children.length===0&&e.textContent.trim()==='Missed Malignancies')e.textContent='Missed serious cases'});
        const s=document.createElement('style');s.textContent='#simMacroF1,#simRetained,#simMissed,#simBalAcc{font-size:23px!important;white-space:nowrap}';document.head.appendChild(s);
        sl.step='any';sl.value=0;updateAbstentionSim('0')})()""")
    await asyncio.sleep(0.5)
    async with Rec(t, "v1_sim") as rec:
        await rec.play(0.9)
        for v in (5, 10, 15, 20):
            # the thumb glides to its next stop, then the page's own handler updates the tiles
            await t.js(f"""(()=>{{const s=document.getElementById('abstentionSlider');const from=+s.value,to={v},t0=performance.now();
                (function f(){{const k=Math.min(1,(performance.now()-t0)/380);const e=k<.5?2*k*k:1-Math.pow(-2*k+2,2)/2;s.value=from+(to-from)*e;
                if(k<1)requestAnimationFrame(f);else{{s.value=to;updateAbstentionSim(String(to))}}}})()}})()""")
            await rec.play(1.15)
        await rec.play(0.8)


async def fold_scene(t):
    await t.js("switchTab('tab-v5')")
    await asyncio.sleep(1.0)
    await t.js("document.getElementById('foldLab').scrollIntoView({block:'center'})")
    await asyncio.sleep(1.5)
    await stage(t, "V5", "")
    await t.js("""(()=>{const lab=document.getElementById('foldLab');document.getElementById('__slot').appendChild(lab);
        lab.style.cssText+=';height:100%;display:flex;flex-direction:column;padding:0;border:0;background:transparent;box-shadow:none;margin:0';
        lab.querySelector('.fl-head').style.display='none';
        const st=lab.querySelector('.fl-stage');st.style.cssText+=';flex:1;min-height:0';
        const svg=document.getElementById('foldLabSvg');svg.style.cssText+=';height:100%;width:100%';
        const s=document.createElement('style');s.textContent='#__stage .tt{display:none}#__stage{padding-bottom:12px!important}#flCap b{font-size:23px!important}#flCap span{font-size:18px!important;line-height:1.28!important}.fl-chip{padding:3px 8px!important;font-size:14px!important}.fl-ctrl{display:none!important}.fl-foot{margin-top:2px!important;min-height:82px}';document.head.appendChild(s);
        window.dispatchEvent(new Event('resize'))})()""")
    await asyncio.sleep(1.2)
    async with Rec(t, "v5_fold") as rec:
        await t.js("document.querySelector('#foldLab .fl-btn[data-act=restart]').click()")
        last = None
        for _ in range(60):
            await rec.play(2)
            cap = await t.js("document.getElementById('flCap').innerText.split('\\n')[0]")
            if cap != last:
                print("    fold:", cap, flush=True)
                last = cap
            label = await t.js("document.querySelector('#foldLab .fl-btn[data-act=play]').getAttribute('aria-label')")
            if label != "Pause":
                await rec.play(2.5)
                break


async def run_charts(t, f, only):
    r = f["rungs"]
    g = f["v5_gates"]
    c = f["v2_c"]
    charts = [
        ("v1_ladder", "V1", "tab-architecture", "ablationChart", "The ablation ladder: test Macro-F1 with 95% CIs",
         f"Only ensembling clears significance: McNemar χ² {f['ens_chi2']:.2f}, p = {sci(f['ens_p'])}", "results/ablation_table.csv · results/mcnemar_delong.json"),
        ("v2_c", "V2", "tab-v2", "chartV2C", "S34: the compression gap C, under 40",
         "Every interval crosses zero: not a decision problem.", "results/v2/bootstrap_intervals.json"),
        ("v3_leak", "V3", "tab-v3", "chartV3Leak", "S40: an inflated probe, caught",
         f"{f['v3_in']:.4f} collapses to {f['v3_out']:.4f} on features the model never trained on.", "results/v3/oos_probe_report.json"),
        ("v3_noise", "V3", "tab-v3", "chartV3Noise", "S44: the endpoint was noisier than the effect",
         f"Byte-identical data: {f['v3_seed_a']}/{f['v3_seed_n']} vs {f['v3_seed_b']}/{f['v3_seed_n']}. Noise beat the signal.", "results/v3/s44_control_and_multiplicity.json"),
        ("v3_age", "V3", "tab-v3", "chartV3Age", "S45: age can be removed, and removing it hurts",
         f"Macro-F1 {f['v3_f1_before']:.4f} → {f['v3_f1_after']:.4f}. Under-40 AUC {f['v3_auc_before']:.4f} → {f['v3_auc_after']:.4f}.", "results/v3/d2_strong_evaluation.json"),
        ("v4_shift", "V4", "tab-v4", "chartV4Shift", "Same V1 system, different rulers",
         f"Archives it never saw: {f['v4_v1_f1']:.3f}. Trained in domain: {f['v4_ctrl_f1']:.3f}, on the same rows.", "results/v5/same_footing.md · results/v4/s54/"),
        ("v4_probe", "V4", "tab-v4", "chartV4Probe", "Foundation models do not rank under-40 better",
         f"PanDerm − ConvNeXt = {sgn(f['v4_pan'])} {ci(f['v4_pan_lo'], f['v4_pan_hi'])}. Falsifier fired.", "results/v4/backbone_probe_deltas.csv"),
        ("v4_contract", "V4", "tab-v4", "chartV4Contract", "The safety contract, read once on the reserved cohort",
         f"{f['v4_terms_met']} of {f['v4_terms']} terms met. Referral {f['v4_referral']:.3f} vs a cap of {f['v4_referral_cap']:.2f}.", "results/v4/s59/s59_report.json"),
        ("v5_screens", "V5", "tab-v5", "chartV5Screens", "Fold-0 screens: Δ all-age escalation pAUC vs comparator",
         "", "results/v5/review2_tables.md", 3.6, True),
        ("v5_confirm", "V5", "tab-v5", "chartV5Confirm", "Confirmation: locked composite − in1k control, folds 1–4",
         f"Under-40 {sgn(g['A']['point'])} {ci(*g['A']['ci95'])}: real, but below the +0.050 MCID.", "results/v5/confirm_s42_s43_s44.json"),
        ("v5_slope", "V5", "tab-v5", "chartV5Slope", "Under-40 pAUC by seed: control to composite",
         "All three seeds move up. Each stays below the +0.050 bar.", "results/v5/review2_tables.md"),
    ]
    for item in charts:
        if only and not any(item[0].startswith(p) for p in only):
            continue
        await chart_scene(t, *item)


async def main(only):
    f = load()
    want = lambda n: not only or any(n.startswith(p) for p in only)
    async with Browser() as t:
        # charts and simulator: one column of page layout, captured at 1.33x
        await setup(t, 750, 360)
        await run_charts(t, f, only)
        if want("v1_sim"):
            await sim_scene(t, f)
        # six-card strips and the fold machine need the full width
        strips = [("v1_steps", "V1", "tab-architecture", "tab-v1-V1", "V1 in six steps"),
                  ("v2_steps", "V2", "tab-v2", "tab-v2-V2", "V2 in six steps: S28 – S39", False),
                  ("v3_steps", "V3", "tab-v3", "tab-v3-V3", "V3 in six steps: S40 – S47"),
                  ("v4_steps", "V4", "tab-v4", "tab-v4-V4", "V4 in six steps: S48 – S75")]
        if any(want(s[0]) for s in strips):
            await setup(t, 1000, 480)
            for s in strips:
                if want(s[0]):
                    await strip_scene(t, *s)
        if want("v5_fold"):
            await setup(t, 1000, 640)
            await fold_scene(t)


asyncio.run(main(sys.argv[1:]))
