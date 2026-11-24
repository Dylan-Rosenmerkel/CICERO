"""Histograms with automatically chosen binning in given ranges (run_CICERO --hist): pixel charge [e-] and cluster
energy [keV] per image and CCD.

    --hist pixels:LO:HI      calibrated pixel charge of the CCD's active area (no prescan / overscan / rows read after
                             the last CCD row / masked columns), LO..HI electrons
    --hist clusters:LO:HI    cluster energies of the image's CCD, LO..HI keV (the g(n)-corrected files when
                             --gain-correction made them)

Binning (written with every histogram):
  'comb'   pixel charge of a CCD whose electron peaks are resolved: edges halfway between whole electrons (the range is
           widened to the half-integers around LO and HI), k bins per electron with k odd (so a bin is centred on every
           whole electron and each peak sits in the middle of it), k in COMB_K chosen by Knuth's rule
  'knuth'  everything else: M equal-width bins over LO..HI, M chosen by Knuth's rule (the maximum of the posterior of
           a piecewise-constant density, Knuth 2006, arXiv:physics/0605197) among the divisors of FINE up to MAX_BINS

Outputs (write_all): hist/<image>_EXT<n>_<kind>_<LO>-<HI>.csv (low edge, high edge, count; '#' header lines with the
method), a PNG of each, and hist/index.html (one page: image / CCD / range selectors, log or linear, zoom, CSV).
"""
import csv
import html
import json
import math
import os

import numpy as np
from scipy.special import gammaln

KINDS = {"pixels": "e-", "clusters": "keV"}
COMB_K = (1, 3, 5, 9, 15, 45)               # bins per electron tried on resolved CCDs: odd (a bin centred on every
                                            # whole electron, edges still on the half-integers) and all divide 45
FINE = 5040                                  # fine bins for Knuth's rule (highly composite: many divisors)
MAX_BINS = 1000


# ---- the ranges --------------------------------------------------------------------------------------------------
def parse_spec(text):
    """'pixels:-1:5' / 'clusters:5:7' -> dict(kind, lo, hi, unit, tag). Raises ValueError."""
    parts = text.split(":")
    if len(parts) != 3 or parts[0] not in KINDS:
        raise ValueError(f"--hist {text}: expected pixels:LO:HI (electrons) or clusters:LO:HI (keV)")
    lo, hi = float(parts[1]), float(parts[2])
    if not hi > lo:
        raise ValueError(f"--hist {text}: HI must be larger than LO")
    return dict(kind=parts[0], lo=lo, hi=hi, unit=KINDS[parts[0]], tag=f"{parts[0]}_{_num(lo)}-{_num(hi)}")


def _num(v):
    return f"{v:g}".replace("-", "m")


# ---- binning -----------------------------------------------------------------------------------------------------
def knuth_logp(counts):
    """log posterior (up to a constant) of M equal-width bins with these counts (Knuth 2006, eq. 29)"""
    counts = np.asarray(counts, float)
    n, m = counts.sum(), counts.size
    return (n * math.log(m) + gammaln(m / 2.0) - m * gammaln(0.5) - gammaln(n + m / 2.0)
            + float(np.sum(gammaln(counts + 0.5))))


def _pick(fine, groups):
    """the grouping (fine bins per bin) with the largest Knuth posterior; fine counts must divide evenly"""
    best = None
    for g in groups:
        c = fine.reshape(-1, g).sum(1)
        lp = knuth_logp(c)
        if best is None or lp > best[0]:
            best = (lp, g, c)
    return best[1], best[2]


def comb_histogram(x, lo, hi):
    """resolved electron peaks: edges at n - 1/2 + j/k; returns (edges, counts, info)"""
    a = math.floor(lo - 0.5) + 0.5
    b = math.ceil(hi - 0.5) + 0.5
    kmax = max(COMB_K)
    nfine = int(round((b - a) * kmax))
    fine, _ = np.histogram(x, bins=nfine, range=(a, b))
    groups = [kmax // k for k in COMB_K if (b - a) * k <= MAX_BINS]
    g, counts = _pick(fine, groups or [kmax])
    k = kmax // g
    edges = a + np.arange(counts.size + 1) / k
    return edges, counts, dict(method="comb", bins_per_unit=k, width=1.0 / k, range=[a, b],
                               note=f"{k} bin(s) per electron, centred on whole electrons (Knuth)")


def knuth_histogram(x, lo, hi):
    """equal-width bins over lo..hi, the number of bins by Knuth's rule; returns (edges, counts, info)"""
    fine, _ = np.histogram(x, bins=FINE, range=(lo, hi))
    if fine.sum() == 0:
        m = 1
        counts = np.zeros(1, int)
    else:
        groups = [FINE // m for m in range(1, MAX_BINS + 1) if FINE % m == 0]
        g, counts = _pick(fine, groups)
        m = counts.size
    edges = np.linspace(lo, hi, m + 1)
    return edges, counts, dict(method="knuth", bins=m, width=(hi - lo) / m, range=[lo, hi],
                               note=f"{m} equal-width bins (Knuth)")


def histogram(x, spec, resolved=False):
    """dict(edges, counts, n, method, note, width, range) of the values x in the spec's range"""
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    if spec["kind"] == "pixels" and resolved:
        edges, counts, info = comb_histogram(x, spec["lo"], spec["hi"])
    else:
        edges, counts, info = knuth_histogram(x, spec["lo"], spec["hi"])
    return dict(info, edges=[float(e) for e in edges], counts=[int(c) for c in counts], n=int(np.sum(counts)))


# ---- inputs from a run -------------------------------------------------------------------------------------------
def pixel_part(s, geo, specs):
    """{EXTn: [histogram per pixel spec]} from a native summary with its arrays (worker, keep_arrays=True)"""
    import native_pipeline as NP
    out = {}
    for e, c in sorted(s["ccds"].items()):
        img = c.get("e_img")
        if img is None:
            continue
        act, _ = NP.regions(img.shape, geo)
        act[:geo.get("id_row_start", 0), :] = False
        for col in c.get("masked_columns") or []:
            act[:, col] = False
        x = img[act]
        out[e] = [dict(histogram(x, sp, bool(c.get("resolved"))), spec=sp["tag"]) for sp in specs]
    return out


def cluster_part(path, specs):
    """[histogram per cluster spec] of one panaSKImg-format cluster file (branch Energy, keV)"""
    import ROOT
    f = ROOT.TFile.Open(path)
    t = f.Get("clustersRec")
    e = []
    for ev in t:
        e.extend(list(ev.Energy))
    f.Close()
    return [dict(histogram(e, sp), spec=sp["tag"]) for sp in specs]


# ---- outputs -----------------------------------------------------------------------------------------------------
def write_all(out_dir, items, specs):
    """items: [dict(image, ext, histogram fields, spec)] -> CSV + PNG per item and index.html; returns summary lines"""
    os.makedirs(out_dir, exist_ok=True)
    by_tag = {sp["tag"]: sp for sp in specs}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for it in items:
        sp = by_tag[it["spec"]]
        name = f"{it['image']}_{it['ext']}_{sp['tag']}"
        it["file"] = name
        with open(os.path.join(out_dir, name + ".csv"), "w", newline="") as fh:
            fh.write(f"# {it['image']} {it['ext']}: {sp['kind']} {sp['lo']:g} to {sp['hi']:g} {sp['unit']}\n")
            fh.write(f"# binning: {it['method']} ({it['note']}); bin width {it['width']:.6g} {sp['unit']}; "
                     f"range {it['range'][0]:g} to {it['range'][1]:g}; {it['n']} entries\n")
            w = csv.writer(fh)
            w.writerow([f"low_{sp['unit']}", f"high_{sp['unit']}", "count"])
            for a, b, c in zip(it["edges"][:-1], it["edges"][1:], it["counts"]):
                w.writerow([f"{a:.6g}", f"{b:.6g}", c])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.stairs(it["counts"], it["edges"], color="#2a6fb0")
        if any(it["counts"]):
            ax.set_yscale("log")
        ax.set_xlabel(f"{'pixel charge' if sp['kind'] == 'pixels' else 'cluster energy'} [{sp['unit']}]")
        ax.set_ylabel("count per bin")
        ax.set_title(f"{it['image']} {it['ext']}\n{it['n']} entries; {it['note']}", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, name + ".png"), dpi=110)
        plt.close(fig)
    with open(os.path.join(out_dir, "index.html"), "w") as fh:
        fh.write(page(items, specs))
    lines = [f"histograms (--hist): {len(items)} in {out_dir} (index.html; one CSV and PNG each):"]
    for sp in specs:
        sel = [it for it in items if it["spec"] == sp["tag"]]
        meth = sorted({it["method"] for it in sel})
        lines.append(f"  {sp['kind']} {sp['lo']:g} to {sp['hi']:g} {sp['unit']}: {len(sel)} histogram(s), binning "
                     f"{', '.join(meth) or '-'}")
    return lines


def page(items, specs):
    """one self-contained page: image / CCD / range selectors, log or linear, zoom (drag; double-click resets), CSV"""
    data = [dict(image=it["image"], ext=it["ext"], spec=it["spec"], edges=it["edges"], counts=it["counts"], n=it["n"],
                 note=it["note"], method=it["method"], file=it["file"]) for it in items]
    units = {sp["tag"]: [sp["kind"], sp["unit"], sp["lo"], sp["hi"]] for sp in specs}
    return PAGE.replace("__DATA__", json.dumps(data)).replace("__UNITS__", json.dumps(units)).replace(
        "__TITLE__", html.escape("Histograms (--hist)"))


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#fff;--fg:#1d2330;--muted:#5d6676;--line:#d7dce3;--bar:#2a6fb0}
@media (prefers-color-scheme:dark){:root{--bg:#15181e;--fg:#e4e7ec;--muted:#9aa3b1;--line:#343a44;--bar:#6aa6e0}}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif}
h1{font-size:18px;margin:0 0 4px} .muted{color:var(--muted)} select,button{font:inherit;margin:0 8px 8px 0}
#plot{width:100%;height:420px;border:1px solid var(--line);border-radius:6px} #info{margin:8px 0}
a{color:var(--bar)}
</style></head><body>
<h1>Histograms</h1><div class="muted">run_CICERO --hist: binning chosen per histogram (comb: edges halfway between
whole electrons; knuth: equal-width bins, number by Knuth's rule). Drag on the plot to zoom, double-click to reset.</div>
<div style="margin-top:12px"><select id="img"></select><select id="ext"></select><select id="spec"></select>
<button id="log">linear y</button><a id="csv" href="#">CSV</a> · <a id="png" href="#">PNG</a></div>
<svg id="plot"></svg><div id="info" class="muted"></div>
<script>
const D=__DATA__, U=__UNITS__; let logy=true, view=null;
const $=id=>document.getElementById(id), uniq=a=>[...new Set(a)];
function fill(sel,vals){const v=sel.value;sel.innerHTML=vals.map(x=>`<option>${x}</option>`).join("");if(vals.includes(v))sel.value=v;}
function cur(){return D.find(d=>d.image===$("img").value&&d.ext===$("ext").value&&d.spec===$("spec").value);}
function refresh(){fill($("img"),uniq(D.map(d=>d.image)));
 fill($("ext"),uniq(D.filter(d=>d.image===$("img").value).map(d=>d.ext)));
 fill($("spec"),uniq(D.filter(d=>d.image===$("img").value&&d.ext===$("ext").value).map(d=>d.spec)));view=null;draw();}
function draw(){const d=cur(),s=$("plot");if(!d){s.innerHTML="";$("info").textContent="no histogram";return;}
 const u=U[d.spec],W=s.clientWidth,H=s.clientHeight,L=64,R=12,T=12,B=40,e=d.edges,c=d.counts;
 const x0=view?view[0]:e[0],x1=view?view[1]:e[e.length-1];
 const vis=c.filter((v,i)=>e[i+1]>x0&&e[i]<x1),cmax=Math.max(1,...vis);
 const fy=v=>logy?(v>0?Math.log10(v):-0.3):v, y0=logy?-0.3:0, y1=logy?Math.log10(cmax)+0.15:cmax*1.05;
 const X=v=>L+(v-x0)/(x1-x0)*(W-L-R), Y=v=>H-B-(fy(v)-y0)/(y1-y0)*(H-T-B);
 let g=`<g stroke="var(--line)">`;for(let k=0;k<=5;k++){const v=x0+k*(x1-x0)/5;g+=`<line x1="${X(v)}" x2="${X(v)}" y1="${T}" y2="${H-B}"/>`}
 g+=`</g><g fill="var(--muted)" font-size="11">`;for(let k=0;k<=5;k++){const v=x0+k*(x1-x0)/5;g+=`<text x="${X(v)}" y="${H-B+16}" text-anchor="middle">${+v.toPrecision(4)}</text>`}
 const yt=logy?[...Array(Math.floor(y1)+1).keys()].map(k=>10**k):[0,cmax/2,cmax].map(Math.round);
 for(const v of yt){if(v>0||!logy)g+=`<text x="${L-6}" y="${Y(v)+4}" text-anchor="end">${v>=1e4?v.toExponential(0):v}</text>`}
 g+=`<text x="${(L+W-R)/2}" y="${H-6}" text-anchor="middle">${u[0]==="pixels"?"pixel charge":"cluster energy"} [${u[1]}]</text></g>`;
 let p="";for(let i=0;i<c.length;i++){if(e[i+1]<=x0||e[i]>=x1)continue;const a=Math.max(X(e[i]),L),b=Math.min(X(e[i+1]),W-R);
  if(c[i]>0||!logy)p+=`<rect x="${a}" y="${Y(c[i])}" width="${Math.max(b-a,0.5)}" height="${Math.max(H-B-Y(c[i]),0)}" fill="var(--bar)" fill-opacity="0.75"><title>${+e[i].toPrecision(6)} to ${+e[i+1].toPrecision(6)} ${u[1]}: ${c[i]}</title></rect>`}
 s.innerHTML=g+p+`<rect id="sel" fill="var(--bar)" fill-opacity="0.15" height="${H-T-B}" y="${T}" width="0"/>`;
 $("info").textContent=`${d.image} ${d.ext}: ${d.n} entries, ${c.length} bins; ${d.note}`;
 $("csv").href=d.file+".csv";$("png").href=d.file+".png";
 let start=null;const inv=px=>x0+(px-L)/(W-L-R)*(x1-x0),pos=ev=>ev.clientX-s.getBoundingClientRect().left;
 s.onmousedown=ev=>{start=pos(ev)};s.onmousemove=ev=>{if(start===null)return;const a=Math.min(start,pos(ev)),r=document.getElementById("sel");r.setAttribute("x",a);r.setAttribute("width",Math.abs(pos(ev)-start))};
 s.onmouseup=ev=>{if(start===null)return;const a=inv(Math.min(start,pos(ev))),b=inv(Math.max(start,pos(ev)));start=null;if(b-a>(x1-x0)/200){view=[a,b]};draw()};
 s.ondblclick=()=>{view=null;draw()};}
$("img").onchange=refresh;$("ext").onchange=()=>{fill($("spec"),uniq(D.filter(d=>d.image===$("img").value&&d.ext===$("ext").value).map(d=>d.spec)));view=null;draw()};
$("spec").onchange=()=>{view=null;draw()};$("log").onclick=()=>{logy=!logy;$("log").textContent=logy?"linear y":"log y";draw()};
window.onresize=draw;refresh();
</script></body></html>
"""
