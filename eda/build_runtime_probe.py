"""Measure actual dual-GPU reader parity and runtime on 58 training studies."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = [Path('/tmp/knee-top10/effnet-output'), Path('/tmp/knee-top10/queue/highres-training'),
           Path('/tmp/knee-top10/queue/convnext-training')]
FOLDERS = ['effnet-study-train', 'effnet-highres-train', 'convnext-study-train']
KERNELS = ['knee-effnet-study-fold0', 'knee-effnet-high-resolution-fold1', 'knee-convnext-study-fold0']


def build():
    old = json.loads(next((ROOT / 'kaggle/convnext-stack').glob('*.ipynb')).read_text())
    public = {}
    for cell in old['cells']:
        code = ''.join(cell['source'])
        if code.startswith('%%writefile /kaggle/working/own_src/'):
            name = code.split('\n', 1)[0].rsplit('/', 1)[-1]
            public[name] = code.split('\n', 1)[1]
    assert set(public) == {'infer.py', 'preprocess.py', 'knee.py'}
    helpers = {name: (ROOT / 'kaggle/effnet-study-infer' / name).read_text()
               for name in ('fast_infer.py', 'parallel_readers.py')}
    public.update(helpers)
    contract = (ROOT / 'kaggle/effnet-cache-check/check.py').read_text().split('\nids = np.load(find("all_ids.npy")', 1)[0]
    readers = []
    for folder, output in zip(FOLDERS, OUTPUTS):
        training = (ROOT / 'kaggle' / folder / 'train.py').read_text()
        run = json.loads((output / 'run.json').read_text())
        assert run['status'] == 'complete' and run['source_sha256'] == hashlib.sha256(training.encode()).hexdigest()
        readers.append({'source': {'study_train.py': training, 'cache_contract.py': contract,
                                  'infer.py': (ROOT / 'kaggle/effnet-study-infer/infer.py').read_text(), **helpers},
                        'sha256': hashlib.sha256((output / 'effnet_best.pt').read_bytes()).hexdigest(),
                        'epoch': run['best_epoch'], 'auc': run['best_validation_auc']})
    script = '''import hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
WORK = Path('/kaggle/working')
def find(name):
    hits = sorted({p for prefix in ('', '*/', '*/*/', '*/*/*/') for p in Path('/kaggle/input').glob(prefix + name)})
    assert len(hits) == 1, (name, hits)
    return hits[0]
assert torch.cuda.device_count() == 2 and all(torch.cuda.get_device_name(i) == 'Tesla T4' for i in range(2))
for gpu in range(2):
    x = torch.arange(64, device=f'cuda:{gpu}', dtype=torch.float32).reshape(8, 8)
    assert float((x @ x.T).sum()) == 510720
print('both T4 computation witnesses passed', flush=True)
root, assets = find('train.csv').parent, find('cnxt_v0_fold0.pt').parent
gold = pd.read_csv(root / 'train.csv', dtype={'StudyInstanceUID': str})
labels = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
gold = gold.loc[gold[labels].notna().all(1)]
assert len(gold) == 58
studies = WORK / 'studies.csv'; gold[['StudyInstanceUID']].to_csv(studies, index=False)
private = WORK / 'decoder_env'
wheel = find('timm-1.0.22-py3-none-any.whl')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', '--target', str(private),
                '--find-links', str(assets), 'pylibjpeg', 'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg', str(wheel)], check=True)
env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
env['PYTHONPATH'] = str(private) + os.pathsep + env.get('PYTHONPATH', '')
def write(folder, modules):
    folder.mkdir(exist_ok=True)
    for name, code in modules.items(): (folder / name).write_text(code)
def execute(command, output, gpu=None):
    process_env = dict(env)
    if gpu is not None: process_env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    started = time.monotonic()
    subprocess.run([sys.executable, *map(str, command)], env=process_env, check=True)
    duration = time.monotonic() - started
    print(output.name, duration, flush=True)
    return duration
def compare(reference, actual):
    a = pd.read_csv(reference, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    b = pd.read_csv(actual, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    assert a.index.is_unique and b.index.is_unique and set(a.index) == set(b.index) == set(gold.StudyInstanceUID)
    assert a.columns.tolist() == b.columns.tolist() == labels
    error = float(np.abs(a.to_numpy() - b.loc[a.index].to_numpy()).max())
    assert np.isfinite(b.to_numpy()).all() and error <= 1e-6, error
    return error
timing = {}
'''
    script += f"public = WORK / 'public_reader'; write(public, {public!r})\n"
    script += '''checkpoints = sorted(assets.glob('cnxt_v0_fold*.pt')); assert len(checkpoints) == 3
reference, fast = WORK / 'public_reference.csv', WORK / 'public_fast.csv'
command = [public / 'parallel_readers.py', 'public', root, 'train', reference, *checkpoints, '--studies', studies]
reference_seconds = execute([*command, '--worker'], reference, gpu=0)
command[4] = fast
fast_seconds = execute(command, fast)
timing['public'] = {'reference_seconds': reference_seconds, 'fast_seconds': fast_seconds,
                    'maximum_absolute_error': compare(reference, fast)}
'''
    script += f'readers = {readers!r}\n'
    script += '''paths = sorted({p for prefix in ('', '*/', '*/*/', '*/*/*/') for p in Path('/kaggle/input').glob(prefix + 'effnet_best.pt')})
cache = Path('/tmp/knee_runtime_cache')
for index, reader in enumerate(readers):
    source = WORK / f'reader{index}'; write(source, reader['source'])
    matches = [p for p in paths if hashlib.sha256(p.read_bytes()).hexdigest() == reader['sha256']]
    assert len(matches) == 1
    checkpoint = matches[0]
    common = ['--sha256', reader['sha256'], '--expected-epoch', str(reader['epoch']), '--expected-auc', str(reader['auc'])]
    reference, fast = WORK / f'reader{index}_reference.csv', WORK / f'reader{index}_fast.csv'
    reference_seconds = execute([source / 'infer.py', root, 'train', checkpoint, reference, *common, '--studies', studies], reference, gpu=0)
    fast_seconds = execute([source / 'parallel_readers.py', 'study', root, 'train', fast, checkpoint, *common,
                            '--studies', studies, '--cache-dir', cache], fast)
    assert len(list(cache.glob('*.npz'))) == 58
    cached = checkpoint.parent / 'gold.csv'
    timing[f'reader{index}'] = {'reference_seconds': reference_seconds, 'fast_seconds': fast_seconds,
                               'checkpoint_sha256': reader['sha256'], 'maximum_absolute_error': compare(reference, fast),
                               'cached_prediction_maximum_error': compare(cached, fast),
                               'cache_files': len(list(cache.glob('*.npz')))}
    print('reader parity witness passed', index, timing[f'reader{index}'], flush=True)
report = {'status': 'complete', 'studies': 58, 'parity_tolerance': 1e-6, 'timing': timing,
          'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'scope': 'added readers only; full parent and hidden test runtime are not measured by this probe'}
(WORK / 'runtime_probe.json').write_text(json.dumps(report, indent=2))
print('all four actual parity witnesses passed', json.dumps(timing), flush=True)
'''
    ast.parse(script)
    destination = ROOT / 'kaggle/reader-runtime-probe'
    destination.mkdir(exist_ok=True)
    (destination / 'probe.py').write_text(script)
    metadata = json.loads((ROOT / 'kaggle/effnet-stack/kernel-metadata.json').read_text())
    metadata.update(id='dk2lone/knee-reader-runtime-probe', title='knee reader runtime probe', kernel_type='script',
                    code_file='probe.py', model_sources=[], dataset_sources=['goodpjw2008/rsna-knee-2-5d-convnext-reader', 'mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2'],
                    kernel_sources=[f'dk2lone/{k}' for k in KERNELS])
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    spec = {'base': metadata['docker_image'], 'machine_shape': metadata['machine_shape'], 'enable_internet': False,
            'system_pkgs': [], 'pip_phases': [{'source': 'attached D4 timm and public reader decoder wheels', 'no_index': True, 'no_deps': True,
                                             'packages': ['timm==1.0.22', 'pylibjpeg', 'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg']}],
            'env': {'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'},
            'smoke': {'import_names': ['torch', 'torchvision', 'numpy', 'pandas', 'pylibjpeg'],
                      'gpu_tests': [{'cmd': 'python3 /kaggle/src/script.py', 'expect': 'all four actual parity witnesses passed'}],
                      'cli_checks': ['kaggle kernels status dk2lone/knee-reader-runtime-probe']}}
    (destination / 'env-spec.json').write_text(json.dumps(spec, indent=2) + '\n')
    print('runtime probe built; source', hashlib.sha256(script.encode()).hexdigest())
    print('spec', hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:8])


if __name__ == '__main__':
    build()
