"""Independent registered arithmetic from authenticated saved metrics; no model execution."""
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import statistics
import time

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
RUN = EXP/'2026-09-09_n3a_v2_vps_cpu'
ACCEPT = HERE.parent/'v2'
EXPECTED = '875bc76f564d63ded5dae403f7077f4fbf6e6ed530a189ac45e5035c9222f1f1'
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def main():
    result_path = ACCEPT/'SCIENTIFIC_DECISION.json'
    if digest(result_path) != EXPECTED:
        raise RuntimeError('decision digest mismatch')
    target = read(result_path)
    receipt_path = ACCEPT/'REMOTE_RECEIPT.json'
    if digest(receipt_path) != 'a1bd3201acee8bab74f989b51d24bc64df4b37bc8f98da03a8adc90aef7bfc47':
        raise RuntimeError('transport receipt digest mismatch')
    inventory = read(receipt_path)['members']
    used = {}
    def authenticated(p):
        name = p.relative_to(EXP).as_posix()
        raw = p.read_bytes()
        h = hashlib.sha256(raw).hexdigest()
        if h != inventory[name]['sha256'] or len(raw) != inventory[name]['bytes']:
            raise RuntimeError('input mismatch: '+name)
        used[name] = h
        return json.loads(raw)
    cfg = authenticated(RUN/'config.json')
    seeds, arches = cfg['seeds'], cfg['architectures']
    if seeds != list(range(64000,64064)) or arches != ['query','transformer']:
        raise RuntimeError('registered census mismatch')
    margin = .005
    def ce(arch,r,seed,arm,step,parent=None):
        unit = f'{arch}_r{r}_seed{seed}_{arm}' if parent is None else f'{arch}_r{r}_seed{seed}_S_at{parent}_{arm}'
        folder = 'main' if parent is None else 'forks'
        value = authenticated(RUN/'outputs'/folder/unit/'metrics'/f'step_{step:04d}.json')['validation_cross_entropy']
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError('invalid CE')
        return value
    rank = max(j for j in range(1,33) if Fraction(2*sum(math.comb(64,k) for k in range(j)),2**64) <= Fraction(1,400))
    def stats(values):
        if len(values) != 64:
            raise RuntimeError('incomplete paired contrast')
        x = sorted(values)
        successes = sum(v > margin for v in values)
        p = float(Fraction(sum(math.comb(64,k) for k in range(successes,65)),2**64))
        return dict(n=64,median=statistics.median(x),strictly_above_margin=successes,
                    sign_p=p,ci_lower=x[rank-1],ci_upper=x[-rank],excludes_required_direction=x[-rank]<=margin)
    def adjust(ps):
        order = sorted(enumerate(ps),key=lambda t:t[1])
        running = 0
        out = [None]*len(ps)
        for j,(idx,p) in enumerate(order):
            running = max(running,min(1.,p*(len(ps)-j)))
            out[idx] = running
        return out
    context = {}
    for r in (16,24):
        components = []
        for arch in arches:
            for phase,steps,orientation in [('early_sparse_benefit',(32,64,128),-1),('late_sparse_cost',(768,1024),1)]:
                values = []
                for seed in seeds:
                    ratios = []
                    for t in steps:
                        d,s = ce(arch,r,seed,'D',t),ce(arch,r,seed,'S',t)
                        ratios.append(orientation*(s-d)/d)
                    values.append(math.fsum(ratios)/len(ratios))
                components.append(dict(architecture=arch,phase=phase,**stats(values)))
        context[str(r)] = dict(components=components,p=max(c['sign_p'] for c in components))
    for group,p in zip(context.values(),adjust([v['p'] for v in context.values()])):
        group['holm_p'] = p
    effects = {}
    for arch in arches:
        for r in (1,4,16,24):
            for seed in seeds:
                for parent in (128,768):
                    den = ce(arch,r,seed,'S',parent)
                    for effect,left,right in [('B','N','G'),('F','P','L')]:
                        effects[arch,r,seed,parent,effect] = (ce(arch,r,seed,left,32,parent)-ce(arch,r,seed,right,32,parent))/den
    identification = []
    for effect in ('B','F'):
        for density in (16,24,'interaction'):
            parts = []
            for arch in arches:
                values = []
                for seed in seeds:
                    if density == 'interaction':
                        deltas = {r:effects[arch,r,seed,768,effect]-effects[arch,r,seed,128,effect] for r in (1,4,16,24)}
                        value = (deltas[16]+deltas[24])/2-(deltas[1]+deltas[4])/2
                    else:
                        value = effects[arch,density,seed,768,effect]
                    values.append(value)
                parts.append(dict(architecture=arch,**stats(values)))
            identification.append(dict(effect=effect,target=density,components=parts,p=max(c['sign_p'] for c in parts)))
    for group,p in zip(identification,adjust([v['p'] for v in identification])):
        group['holm_p'] = p
    context_pass = any(g['holm_p']<=.05 for g in context.values())
    backward = context_pass and all(g['holm_p']<=.05 for g in identification[:3])
    forward = context_pass and all(g['holm_p']<=.05 for g in identification[3:])
    excluded_context = all(any(c['ci_upper']<=margin for c in g['components']) for g in context.values())
    excluded_branches = all(any(c['ci_upper']<=margin for g in groups for c in g['components']) for groups in (identification[:3],identification[3:]))
    verdict = 'BACKWARD_SUPPORTED_SYNTHETIC' if backward else 'FORWARD_ONLY_SUPPORTED_SYNTHETIC' if forward else 'KILLED_FOR_REGISTERED_EPSILON' if excluded_context or excluded_branches else 'INCONCLUSIVE'
    recomputed = dict(status=verdict,context_pass=context_pass,context=context,identification=identification,margin=margin,component_ci_alpha=.0025)
    comparisons = 0
    def compare(a,b,path):
        nonlocal comparisons
        if isinstance(a,dict):
            for k,v in a.items(): compare(v,b[k],path+'/'+k)
        elif isinstance(a,list):
            if len(a)!=len(b): raise RuntimeError(path)
            for i,v in enumerate(a): compare(v,b[i],path+'/'+str(i))
        else:
            comparisons += 1
            equal = math.isclose(a,b,rel_tol=1e-13,abs_tol=1e-15) if isinstance(a,float) else a==b
            if not equal: raise RuntimeError(f'{path}: {a} != {b}')
    compare(recomputed,target,'result')
    output = dict(status='PASS',decision_sha256=EXPECTED,script_sha256=digest(Path(__file__)),
                  independent_implementation=True,model_execution=False,verified_unix=time.time(),
                  authenticated_inputs=len(used),input_hashes=used,scalar_comparisons=comparisons,
                  median_interval_order_statistic_rank=rank,component_family_size=20,
                  recomputed=recomputed,registered_b_compute_allowed=backward)
    with (HERE/'VERIFICATION.json').open('x',encoding='utf-8') as f:
        json.dump(output,f,indent=2,sort_keys=True)
    print(json.dumps({k:v for k,v in output.items() if k not in ('input_hashes','recomputed')}))
    print('receipt_sha256='+digest(HERE/'VERIFICATION.json'))
if __name__ == '__main__':
    main()
