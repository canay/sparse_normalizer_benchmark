"""Independent read-only reconstruction of registered endpoints from saved rows."""
import csv
import itertools
import json
import math
import platform
import time
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import ttest_1samp
import argparse,hashlib
from datetime import datetime,timezone
PROJECT=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8')
def json_write(p,v):write(p,json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def stamp(op):return now()+'; saved-output verification; '+op
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--output-dir',type=Path,default=PROJECT/'verification_output')
args=ap.parse_args()


OP = 'snd-weekend-saved-check-20260905'
ROOT = PROJECT / 'experiments/2026-08-26_codex_mta_crossover-s2d-extension'
OUT = args.output_dir.resolve()
started = now()
clock0 = time.monotonic()
assert not (OUT / 'VERIFICATION.json').exists(), 'Preserve previous result'
OUT.mkdir(parents=True, exist_ok=True)
raw_hashes = {}
verified = {}
attempt_status_counts = {}


def read(rel):
    p = ROOT / rel
    verified[rel] = sha(p)
    return json.loads(p.read_bytes().decode('utf-8-sig'))


def eq(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12), (a,b)


def rows(phase):
    out = {}
    statuses = Counter()
    for p in sorted((ROOT / f'outputs/{phase}/raw').glob('*.jsonl')):
        raw_hashes[p.relative_to(ROOT).as_posix()] = sha(p)
        for line in p.read_text(encoding='utf-8-sig').splitlines():
            r = json.loads(line)
            statuses[str(r.get('status'))] += 1
            if r.get('status') != 'completed':
                assert r.get('status') == 'failed', (p.name, r.get('status'))
                continue
            key = (r['dataset'],r['architecture'],r['tier'],int(r['seed']),r['method'])
            assert key not in out, ('duplicate', phase, key)
            out[key] = r
    attempt_status_counts[phase] = dict(statuses)
    return out


def mean(values):
    values = list(values)
    return math.fsum(values)/len(values)


def bootstrap(values, seed):
    values = np.asarray(values, dtype=float)
    draws = np.random.default_rng(seed).integers(0,len(values),(20000,len(values)))
    return np.quantile(values[draws].mean(axis=1),[.025,.975]).tolist()


datasets = ['mnist','fashion_mnist','kmnist','cifar10','cifar100','twenty_news','synthetic_marker','emnist_letters','emnist_digits','usps']
archs = ['plain_transformer','local_hybrid']
tiers = ['compact','deep','wide']
m1 = rows('m1')
assert set(m1) == set(itertools.product(datasets,archs,tiers,range(5),['softmax','topk_softmax_025']))
m1_result = read('outputs/m1_analysis/m1_result.json')
crossings = []
for d,a,t in itertools.product(datasets,archs,tiers):
    deltas = [mean(m1[d,a,t,s,'topk_softmax_025']['epoch_log'][e]['val_accuracy']-m1[d,a,t,s,'softmax']['epoch_log'][e]['val_accuracy'] for s in range(5)) for e in range(30)]
    early = mean(deltas[:3])
    target, reversal = 31, False
    if early <= 0:
        target = 3
    else:
        for e in range(4,29):
            if all(deltas[j-1] <= 0 for j in (e,e+1,e+2)):
                target,reversal=e,True
                break
    crossings.append({'dataset':d,'architecture':a,'tier':t,'early_difference':early,
                      'final_difference':deltas[-1],'target_switch_epoch':target,
                      'persistent_reversal':reversal,'delta_by_epoch':deltas})
assert sum(c['persistent_reversal'] for c in crossings)==m1_result['persistent_reversal_cells']==39
with (ROOT/'outputs/m1_analysis/m1_cell_features.csv').open(encoding='utf-8-sig',newline='') as f:
    saved_cells=list(csv.DictReader(f))
assert len(saved_cells)==60
saved_index={(r['dataset'],r['architecture'],r['tier']):r for r in saved_cells}
for c in crossings:
    r=saved_index[c['dataset'],c['architecture'],c['tier']]
    assert int(r['target_switch_epoch'])==c['target_switch_epoch']
    assert bool(int(r['persistent_reversal']))==c['persistent_reversal']
    for actual,expected in zip(c['delta_by_epoch'],json.loads(r['delta_by_epoch'])):
        eq(actual,expected)
with (ROOT/'outputs/m1_analysis/m1_lodo_predictions.csv').open(encoding='utf-8-sig',newline='') as f:
    predictions=list(csv.DictReader(f))
assert len(predictions)==60
for method, expected in m1_result['macro_mae'].items():
    eq(mean(mean(abs(int(p['pred_'+method])-int(p['target_switch_epoch'])) for p in predictions if p['dataset']==d) for d in datasets),expected)
predictor_diffs=[mean(int(p['abs_error_mechanistic'])-int(p['abs_error_early_accuracy']) for p in predictions if p['dataset']==d) for d in datasets]
for a,b in zip(bootstrap(predictor_diffs,260826),m1_result['proposed_minus_strongest_bootstrap_ci95']):eq(a,b)

m2=rows('m2')
m2_result=read('outputs/m2_analysis/m2_result.json')
m2methods=['softmax','topk_softmax_025','median_switch','early_accuracy_switch','entropy_switch','mechanistic_switch','headwise_adaptive_entmax','cosine_alpha']
assert set(m2)==set(itertools.product(datasets,archs,tiers,range(5,10),m2methods))
confirmation=['emnist_balanced','k49','svhn']
methods=['softmax','topk_softmax_025','mechanistic_switch','median_switch']
m3=rows('m3_validation')
test=rows('m3_test')
expected=set(itertools.product(confirmation,archs,tiers,range(10,20),methods))
assert set(m3)==expected and set(test)==expected
m3_result=read('outputs/m3_analysis/m3_validation_result.json')
test_result=read('outputs/m3_test_analysis/m3_test_result.json')


def ttt(log,threshold):
    return next((int(r['epoch']) for r in log if float(r['val_accuracy'])>=threshold),31)


confirmation_rows=[]
paired=[]
for d in confirmation:
    red,vdiff,tdiff,simple=[],[],[],[]
    for a,t,s in itertools.product(archs,tiers,range(10,20)):
        key=d,a,t,s
        dense=m3[*key,'softmax']; mech=m3[*key,'mechanistic_switch']; ref=m3[*key,'median_switch']
        threshold=.95*dense['epoch_log'][-1]['val_accuracy']
        dt=ttt(dense['epoch_log'],threshold); mt=ttt(mech['epoch_log'],threshold); rt=ttt(ref['epoch_log'],threshold)
        red.append((dt-mt)/dt);simple.append((rt-mt)/rt)
        vdiff.append(mech['epoch_log'][-1]['val_accuracy']-dense['epoch_log'][-1]['val_accuracy'])
        diff=test[*key,'mechanistic_switch']['test_accuracy']-test[*key,'softmax']['test_accuracy']
        tdiff.append(diff)
        paired.append({'dataset':d,'architecture':a,'tier':t,'seed':s,'relative_epoch_reduction':red[-1],
                       'test_difference':diff,'mechanistic_minus_simple_test':test[*key,'mechanistic_switch']['test_accuracy']-test[*key,'median_switch']['test_accuracy']})
    eq(mean(red),m3_result['mean_relative_reduction_by_dataset'][d]);eq(mean(tdiff),test_result['test_difference_by_dataset'][d])
    confirmation_rows.append({'dataset':d,'n_paired':len(tdiff),'relative_epoch_reduction':mean(red),
                             'validation_accuracy_difference':mean(vdiff),'test_accuracy_difference':mean(tdiff),
                             'relative_reduction_vs_simple':mean(simple),'predeclared_pooled_t_p':float(ttest_1samp(tdiff,0).pvalue)})
pvals=[r['predeclared_pooled_t_p'] for r in confirmation_rows]
adjusted=[0.]*3; running=0.
for rank,i in enumerate(sorted(range(3),key=pvals.__getitem__)):
    running=max(running,(3-rank)*pvals[i]);adjusted[i]=min(1.,running)
for i,r in enumerate(confirmation_rows):
    eq(adjusted[i],test_result['task_tests'][i]['p_holm'])
    r['predeclared_pooled_t_p_holm']=adjusted[i]
for a,b in zip(bootstrap([r['test_accuracy_difference'] for r in confirmation_rows],260829),test_result['dataset_bootstrap_ci95_test_difference']):eq(a,b)
assert m1_result['status']=='M1_KILLED' and m2_result['status']=='M2_METHOD_CLAIM_KILLED'
assert m3_result['status']=='M3_CONFIRMATION_KILLED' and test_result['final_method_status']=='KILLED'
for rel in ['outputs/m1_analysis/m1_cell_features.csv','outputs/m1_analysis/m1_lodo_predictions.csv','outputs/m3_test_analysis/m3_test_pairwise.csv']:
    verified[rel]=sha(ROOT/rel)
counts={'m1':len(m1),'m2':len(m2),'m3_validation':len(m3),'m3_test':len(test)}
assert sum(counts.values())==4440
synthetic=[c for c in crossings if c['dataset']=='synthetic_marker']
non_synthetic=[c for c in crossings if c['dataset']!='synthetic_marker']
report={'status':'PASS','operation_id':OP,'started_at':started,'completed_at':now(),
 'elapsed_seconds':time.monotonic()-clock0,'host':platform.node(),'python':platform.python_version(),
 'scope':'Existing saved-row census and arithmetic only; no model training, evaluation, refit or new test read',
 'verified_artifacts':[{'path':(ROOT/rel).relative_to(PROJECT).as_posix(),'sha256':h} for rel,h in verified.items()],
 'raw_input_digest':{'manifest_sha256':__import__('hashlib').sha256(json.dumps(raw_hashes,sort_keys=True).encode()).hexdigest(),
                     'cell_count':4440,'files':raw_hashes},
 'script_sha256':sha(Path(__file__)),'counts':counts,'attempt_status_counts':attempt_status_counts,
 'population_contract':{'aggregation_unit':'dataset x architecture x capacity, paired seeds within each cell',
    'population_rule':'All registered completed cells, exact grid equality asserted; failed infrastructure attempts retained and counted separately under original validator status selection',
    'included_population':counts,'excluded_population':'Failed attempts do not replace complete cells; smoke/pilot folders are outside registered phase directories; no completed cell excluded',
    'weighting_and_denominator':'M1 60 aggregate cells, five seeds each; confirmation 3 dataset clusters, 60 paired conditions per dataset',
    'sensitivity_structural_control':{'all':{'reversals':39,'cells':60},
      'without_synthetic_marker':{'reversals':sum(c['persistent_reversal'] for c in non_synthetic),'cells':len(non_synthetic)},
      'synthetic_marker_only':{'reversals':sum(c['persistent_reversal'] for c in synthetic),'cells':len(synthetic)}}},
 'm1_reversal_cells':crossings,'m1_macro_mae_saved_predictions':m1_result['macro_mae'],
 'confirmation':confirmation_rows,'final_method_status':'KILLED',
 'simple_test_pairs_identical':sum(r['mechanistic_minus_simple_test']==0 for r in paired),
 'caveats':['Crossover is the predeclared descriptive three-epoch sign criterion, not a significance claim.',
 'M1 prediction outputs were arithmetically checked, not refitted.',
 'Pooled task t-tests reproduce the frozen analysis but do not establish independence of architecture/capacity conditions sharing a seed.',
 'Only three confirmation dataset clusters; noninferiority permits loss and is not equal accuracy.',
 'Time-to-target is in epochs, not wall-clock or GPU speedup.']}
json_write(OUT/'VERIFICATION.json',report)
write(OUT/'STATUS.md','# Saved-output verification\n\n'+stamp(OP)+'\n\nAll 4,440 registered completed cells and the reported arithmetic were checked from immutable saved records. No model training, test evaluation, predictor refit, or endpoint change was performed. See VERIFICATION.json.\n')
json_write(OUT/'RUN_MANIFEST.json',{'run_id':OUT.name,'operation_id':OP,'run_kind':'verification_only',
 'started_at':started,'completed_at':now(),'host':platform.node(),'platform':platform.platform(),
 'python':platform.python_version(),'script':str(Path(__file__).relative_to(PROJECT)),'script_sha256':sha(Path(__file__)),
 'input_run':ROOT.relative_to(PROJECT).as_posix(),'output':'VERIFICATION.json','output_sha256':sha(OUT/'VERIFICATION.json'),
 'compute_allowed':'saved_outputs_only','scientific_endpoints_changed':False,'exit_code':0})
print(json.dumps({'status':report['status'],'counts':counts,'sensitivity':report['population_contract']['sensitivity_structural_control'],
 'confirmation':confirmation_rows,'seconds':report['elapsed_seconds']},ensure_ascii=False,indent=2))
