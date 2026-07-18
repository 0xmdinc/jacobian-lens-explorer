#!/usr/bin/env python3
"""
Little web app for poking at a Jacobian lens.

Give it a model plus a lens you fit earlier (fit_lens.py) and you get a page where
you type a sentence and see two things: what the model actually predicts next, and
the jlens grid of what each layer is "leaning toward". Handy for building intuition
without living in a notebook.

    python serve.py                               # Qwen3-8B, out/lens.pt
    python serve.py --model Qwen/Qwen3-8B --lens out/lens.pt --port 5005
"""
import argparse
import io
import os
import threading
import urllib.request
import html as _html

import torch
import transformers
import jlens
from jlens import vis
from flask import Flask, request, Response, jsonify

HERE = os.path.dirname(os.path.abspath(__file__))
D3_PATH = os.path.join(HERE, "assets", "d3.min.js")
DARK_CSS_PATH = os.path.join(HERE, "static", "dark.css")


def parse_args():
    p = argparse.ArgumentParser(description="Jacobian Lens Explorer web app")
    p.add_argument("--model", default=os.environ.get("JLENS_MODEL", "Qwen/Qwen3-8B"),
                   help="HF id or local path. Needs a plain attention model. Hybrid "
                        "mamba/linear-attention ones (Qwen3.5, Qwen3-Next) won't work.")
    p.add_argument("--lens", default=os.environ.get("JLENS_LENS", os.path.join(HERE, "out", "lens.pt")),
                   help="Path to a fitted lens (.pt) produced by fit_lens.py")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("JLENS_PORT", "5005")))
    p.add_argument("--device", default="cuda", help="cuda | cuda:0 | auto")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


ARGS = parse_args()

# build_page normally fetches d3 from a CDN. Hand it our bundled copy instead so
# the whole thing works offline (and doesn't randomly break when a CDN is flaky).
with open(D3_PATH, "rb") as _f:
    _D3 = _f.read()
_orig_urlopen = urllib.request.urlopen
def _urlopen(url, *a, **k):
    u = url if isinstance(url, str) else getattr(url, "full_url", "")
    if "d3.min.js" in str(u):
        return io.BytesIO(_D3)
    return _orig_urlopen(url, *a, **k)
urllib.request.urlopen = _urlopen

if not os.path.exists(ARGS.lens):
    raise SystemExit(
        f"\nLens not found: {ARGS.lens}\n"
        f"Fit one first, e.g.:\n"
        f"    python fit_lens.py --model {ARGS.model} --out {ARGS.lens}\n")

_DTYPE = getattr(torch, ARGS.dtype)
print(f"Loading model '{ARGS.model}' ({ARGS.dtype}) ... this can take a minute", flush=True)
hf = transformers.AutoModelForCausalLM.from_pretrained(
    ARGS.model, torch_dtype=_DTYPE, device_map=ARGS.device)
tok = transformers.AutoTokenizer.from_pretrained(ARGS.model)
model = jlens.from_hf(hf, tok)
lens = jlens.JacobianLens.load(ARGS.lens)
MODEL_LABEL = ARGS.model.rstrip("/").split("/")[-1]
print(f"READY -> open http://localhost:{ARGS.port} in your browser", flush=True)

gpu_lock = threading.Lock()
app = Flask(__name__)


def dark_inject():
    try:
        with open(DARK_CSS_PATH, "r", encoding="utf-8") as f:
            css = f.read()
    except OSError:
        css = ""
    return "<meta name='color-scheme' content='dark'><style>" + css + "</style>"


HOME = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<meta name=color-scheme content=dark>
<title>Jacobian Lens Explorer</title>
<style>
  :root{--bg:#0d1220;--panel:#141b29;--panel2:#1b2334;--line:#2a3446;--fg:#e8ecf4;--dim:#9aa7bd;--accent:#8ab4ff;--accent2:#3b82f6;--violet:#b98cff;--good:#34d399;}
  *{box-sizing:border-box}
  body{margin:0;background:radial-gradient(1100px 520px at 15% -10%,#141d31 0%,#0d1220 60%);color:var(--fg);font-family:system-ui,'Segoe UI',Roboto,sans-serif;line-height:1.55}
  .wrap{max-width:1432px;margin:0 auto;padding:26px 22px 60px}
  h1{font-size:28px;margin:0 0 6px} .tag{color:var(--dim);font-size:14px;margin:0 0 22px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px 22px;margin:16px 0}
  .card h3{margin:0 0 8px;font-size:16px;color:#fff}
  .card p{margin:8px 0;color:#d5dae2} .grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  @media(max-width:680px){.grid2{grid-template-columns:1fr}}
  ul{margin:8px 0 0;padding-left:20px} li{margin:5px 0;color:#d5dae2} .k{color:var(--accent);font-weight:600}
  form{position:sticky;top:0;z-index:20;background:linear-gradient(180deg,#0d1220 72%,rgba(13,18,32,0));padding:14px 0 12px}
  .row{display:flex;gap:10px}
  input[type=text]{flex:1;padding:14px 16px;font-size:17px;background:var(--panel2);color:var(--fg);border:1px solid var(--line);border-radius:10px}
  input[type=text]:focus{outline:none;border-color:var(--accent2)}
  button{padding:14px 24px;font-size:16px;font-weight:600;border:0;border-radius:10px;background:var(--accent2);color:#fff;cursor:pointer}
  button:disabled{opacity:.5;cursor:default}
  .ex{margin:6px 0 0;font-size:13px;color:var(--dim)} .ex a{color:var(--accent);text-decoration:none;margin-right:14px;cursor:pointer}
  #status{margin:8px 0;color:var(--violet);font-size:14px;min-height:20px}
  #answer{display:none;background:linear-gradient(180deg,#16203a,#121a2c);border:1px solid #2b3a5c;border-radius:14px;padding:18px 22px;margin:10px 0 4px}
  .ahdr{font-size:13px;letter-spacing:.4px;text-transform:uppercase;color:#8fa6d6;margin-bottom:12px}
  .abars{display:flex;flex-direction:column;gap:7px;max-width:640px}
  .abar{display:flex;align-items:center;gap:10px;font-size:15px}
  .atok{flex:none;min-width:120px;font-family:ui-monospace,Menlo,Consolas,monospace;color:#dfe6f5;white-space:pre;overflow:hidden;text-overflow:ellipsis}
  .abar.top .atok{color:#fff;font-weight:700}
  .atrack{flex:1;height:14px;background:#0e1626;border-radius:7px;overflow:hidden;border:1px solid #223052}
  .afill{display:block;height:100%;background:linear-gradient(90deg,#3b82f6,#8ab4ff);border-radius:7px}
  .abar.top .afill{background:linear-gradient(90deg,#34d399,#a7f3d0)}
  .apct{flex:none;width:56px;text-align:right;color:var(--dim);font-variant-numeric:tabular-nums}
  .acont{margin-top:14px;padding-top:14px;border-top:1px dashed #2b3a5c;font-size:15px;line-height:1.7}
  .alabel{color:#8fa6d6;font-size:13px;text-transform:uppercase;letter-spacing:.4px;margin-right:8px}
  .aprompt{color:#aeb9cf} .agen{color:#7ef0c0;font-weight:600}
  #frameWrap{display:none;margin-top:8px}
  iframe{width:100%;height:80vh;border:1px solid var(--line);border-radius:12px;background:#0d1220}
  .legend{display:flex;gap:22px;flex-wrap:wrap;font-size:13px;color:var(--dim);margin-top:10px}
  .note{font-size:13px;color:var(--dim);background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-top:14px}
</style></head>
<body><div class=wrap>
  <h1>Jacobian Lens Explorer</h1>
  <p class=tag>Model: __MODEL__ &nbsp;&middot;&nbsp; watch an idea form inside the network, then see what the model actually says</p>

  <div class=card>
    <h3>What is this?</h3>
    <p>When an AI reads your sentence it processes it through many stacked <b>layers</b>, holding a hidden numeric "state" at every layer that is normally unreadable to humans.</p>
    <p>The <b>Jacobian Lens</b> translates that hidden state back into the word the model is most <i>leaning toward</i> at each spot, <b>even if that word never appears in your text</b>. Below you'll see two things: the model's <b>actual next-word answer</b>, and a grid of its internal "thoughts" layer by layer.</p>
  </div>

  <form id=f>
    <div class=row>
      <input id=p type=text placeholder="Type a sentence, e.g. The capital of France is..." autofocus>
      <button id=go type=submit>Analyze</button>
    </div>
    <div class=ex>Try:
      <a data-ex="The capital of France is">capital of France</a>
      <a data-ex="The Eiffel Tower is located in the city of">Eiffel Tower</a>
      <a data-ex="Water is a molecule made of two atoms of hydrogen and one atom of">what water is made of</a>
    </div>
  </form>

  <div id=status></div>
  <div id=answer></div>

  <div id=frameWrap>
    <iframe id=frame title="Jacobian Lens result"></iframe>
    <div class=legend>
      <span><span class=k>Columns</span> = words of your sentence</span>
      <span><span class=k>Rows</span> = layers (bottom = early, top = late)</span>
      <span><span class=k>Each cell</span> = the word the lens reads there</span>
      <span><span class=k>Click a cell</span> to pin &amp; track a word</span>
    </div>
  </div>

  <div class=card>
    <h3>How to read the grid</h3>
    <div class=grid2>
      <div><ul>
        <li><span class=k>Left to right</span> = each word (token) of your sentence.</li>
        <li><span class=k>Bottom to top</span> = the model's layers, shallow up to final.</li>
        <li>Each square = the <b>top word</b> the lens decodes there; small number = its rank.</li>
      </ul></div>
      <div><ul>
        <li><b>Hover</b> a square to see the full top-10 behind it.</li>
        <li><b>Click</b> a square to pin a word and chart its rank across layers.</li>
        <li>The eventual answer often <b>"lights up" in the middle layers</b> before it's said.</li>
      </ul></div>
    </div>
    <div class=note>The grid can be noisy depending on how thoroughly the lens was fitted, so it is best for intuition rather than a precise measurement. The <b>answer panel</b> above is the model's real output. Each analysis runs a live pass on the model, so it takes a few seconds.</div>
  </div>
</div>
<script>
  var f=document.getElementById('f'),p=document.getElementById('p'),go=document.getElementById('go'),
      status=document.getElementById('status'),ans=document.getElementById('answer'),
      wrap=document.getElementById('frameWrap'),frame=document.getElementById('frame');
  function esc(s){return (s==null?'':String(s)).replace(/[&<>"]/g,function(c){return{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
  function showTok(s){ if(s==='')return '∅'; return s.replace(/ /g,'·'); }
  function renderAnswer(d,q){
    var top=d.top||[]; var h='<div class=ahdr>The model’s actual answer: most likely next word</div><div class=abars>';
    top.forEach(function(t,i){
      var pct=t[1]*100, w=Math.max(2,pct);
      h+='<div class="abar'+(i===0?' top':'')+'"><span class=atok>'+esc(showTok(t[0]))+'</span>'
        +'<span class=atrack><span class=afill style="width:'+w+'%"></span></span>'
        +'<span class=apct>'+pct.toFixed(1)+'%</span></div>';
    });
    h+='</div>';
    if(d.continuation){ h+='<div class=acont><span class=alabel>If it kept writing</span> <span class=aprompt>'+esc(q)+'</span><span class=agen>'+esc(d.continuation)+'</span></div>'; }
    ans.innerHTML=h; ans.style.display='block';
  }
  f.addEventListener('submit',function(e){
    e.preventDefault();
    var q=p.value.trim(); if(!q){return;}
    go.disabled=true; status.textContent='Thinking… running the model (a few seconds)';
    fetch('/answer?prompt='+encodeURIComponent(q)).then(function(r){return r.json();})
      .then(function(d){ renderAnswer(d,q); }).catch(function(){});
    fetch('/view?prompt='+encodeURIComponent(q)).then(function(r){return r.text();}).then(function(html){
      frame.srcdoc=html; wrap.style.display='block'; status.textContent='Showing results for: “'+q+'”'; go.disabled=false;
    }).catch(function(err){ status.textContent='Error: '+err; go.disabled=false; });
  });
  Array.prototype.forEach.call(document.querySelectorAll('[data-ex]'),function(a){
    a.addEventListener('click',function(e){e.preventDefault(); p.value=a.getAttribute('data-ex'); f.requestSubmit();});
  });
</script>
</body></html>"""


@app.route("/")
def home():
    return HOME.replace("__MODEL__", _html.escape(MODEL_LABEL))


@app.route("/answer")
def answer():
    prompt = (request.args.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"top": [], "continuation": ""})
    try:
        with gpu_lock:
            ids = tok(prompt, return_tensors="pt").input_ids.to(hf.device)
            with torch.no_grad():
                logits = hf(ids).logits[0, -1].float()
                probs = torch.softmax(logits, dim=-1)
                pv, pi = probs.topk(8)
                top = [[tok.decode([int(i)]), float(v)] for v, i in zip(pv, pi)]
                gen = hf.generate(ids, max_new_tokens=24, do_sample=False,
                                  pad_token_id=(tok.eos_token_id or tok.pad_token_id))
                cont = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        return jsonify({"top": top, "continuation": cont})
    except Exception as e:  # noqa: BLE001
        return jsonify({"top": [], "continuation": "", "error": str(e)})


@app.route("/view")
def view():
    prompt = (request.args.get("prompt") or "").strip()
    if not prompt:
        return Response("<p>empty prompt</p>", mimetype="text/html")
    try:
        with gpu_lock:
            sl = vis.compute_slice(model, lens, prompt, top_n=10)
            page, _, _ = vis.build_page(sl, prompt, title=f"{MODEL_LABEL} Jacobian Lens",
                                        description=prompt, mode="embed")
        # The main grid is drawn on a <canvas>; CSS cannot recolor canvas pixels,
        # so patch the hardcoded light-theme colors for a dark background.
        page = page.replace("(r > 0 ? '#888' : '#000')", "(r > 0 ? '#9aa7bd' : '#eef2f8')")
        page = page.replace("hmCtx.strokeStyle = '#eee'", "hmCtx.strokeStyle = '#28313f'")
        page = page.replace("hmCtx.fillStyle = '#b00'", "hmCtx.fillStyle = '#f87171'")
        page = page.replace("</head>", dark_inject() + "</head>", 1)
        return Response(page, mimetype="text/html")
    except Exception as e:  # noqa: BLE001
        return Response("<pre style='color:#eee;background:#0d1220;padding:16px'>error: "
                        + _html.escape(str(e)) + "</pre>", mimetype="text/html", status=500)


if __name__ == "__main__":
    app.run(host=ARGS.host, port=ARGS.port, threaded=True)
