"""Advance the already-running knee experiments through verified submissions."""
import csv
import datetime
import fcntl
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import time
import zipfile
from kaggle import api

ROOT = Path(__file__).resolve().parents[1]
WORK = Path('/tmp/knee-top10')
NUMERIC = WORK / 'venv/bin/python'
PYTHON = Path('/Users/dz/.local/share/uv/tools/kaggle/bin/python')
KAGGLE = Path('/Users/dz/.local/bin/kaggle')
COMPETITION = 'rsna-knee-abnormality-detection'
STATE = WORK / 'submission_queue.json'
SOURCES = ['eda/advance_top10.py', 'eda/check_candidate.py', 'eda/build_reader_probe.py', 'eda/build_effnet_stack.py',
           'eda/build_supervised_head_stack.py', 'eda/build_reader_combination.py', 'kaggle/effnet-study-infer/infer.py',
           'kaggle/effnet-study-train/train.py', 'kaggle/effnet-highres-train/train.py', 'kaggle/convnext-study-train/train.py',
           'kaggle/convnext-head-train/train.py', 'kaggle/effnet-cache-check/check.py',
           'kaggle/convnext-stack/kernel-metadata.json', 'kaggle/effnet-stack/kernel-metadata.json']
SOURCES += [str(p.relative_to(ROOT)) for p in (ROOT / 'kaggle/convnext-stack').glob('*.ipynb')]
SOURCES += [str(p.relative_to(ROOT)) for p in (ROOT / 'kaggle/effnet-stack').glob('*.ipynb')]
TRAINING = {'highres': ('knee-effnet-high-resolution-fold1', 'effnet-highres'),
            'convnext': ('knee-convnext-study-fold0', 'convnext-reader')}
PATTERN = r'^(run.json|validation.csv|gold.csv|gold_truth.csv|effnet_best.pt)$'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args):
    result = subprocess.run(list(map(str, args)), cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def save(state):
    state['at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(STATE)


def guard(state):
    assert command(['/usr/bin/git', 'branch', '--show-current']).strip() == 'public-stack-baseline', 'working branch changed'
    assert all(digest(ROOT / p) == h for p, h in state['source_sha256'].items()), 'queue source changed'
    assert not (WORK / 'STOP_SUBMISSION_QUEUE').exists(), 'submission queue stop requested'


def status(kernel):
    return str(api.kernels_status(kernel).status).split('.')[-1]


def collect(kernel, destination, pattern):
    destination.mkdir(parents=True, exist_ok=True)
    api.kernels_output(kernel, str(destination), file_pattern=pattern, page_size=100)


def push(state, job, folder, kind):
    guard(state)
    metadata = json.loads((ROOT / folder / 'kernel-metadata.json').read_text())
    assert metadata['id'].startswith('dk2lone/')
    job.update(stage='launching', active_kind=kind, kernel=metadata['id'], folder=folder)
    save(state)
    result = command([KAGGLE, 'kernels', 'push', '-p', folder])
    match = re.search(r'Kernel version (\d+) successfully pushed', result)
    assert match and f'https://www.kaggle.com/code/{metadata["id"]}' in result, result
    assert int(match[1]) == 1, 'notebook ID already had another version; require manual review'
    job.update(stage='active', version=1, launched_at=state['at'])
    save(state)


def submit(state, job):
    guard(state)
    job['stage'] = 'submitting'
    save(state)
    response = api.competition_submit_code('submission.csv', job['description'], COMPETITION,
                                          job['kernel'], job['version'], quiet=True)
    assert response.ref > 0, response.message
    job.update(stage='submitted', submission_ref=response.ref, submit_message=response.message)
    save(state)


def leaderboard(state):
    directory = WORK / 'queue/leaderboard'
    directory.mkdir(parents=True, exist_ok=True)
    command([KAGGLE, 'competitions', 'leaderboard', COMPETITION, '--download', '-p', directory])
    with zipfile.ZipFile(directory / f'{COMPETITION}.zip') as archive:
        rows = list(csv.DictReader(io.TextIOWrapper(archive.open(archive.namelist()[0]), encoding='utf-8-sig')))
    own = next(r for r in rows if r['TeamId'] == '16715496')
    assert 'dk2lone' in own['TeamMemberUserNames'].split(',')
    state['leaderboard'] = {'rank': int(own['Rank']), 'score': float(own['Score']), 'teams': len(rows),
                            'tenth_score': float(rows[9]['Score']), 'at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    state['top10_achieved'] = int(own['Rank']) <= 10


def run():
    lock = (WORK / 'submission_queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (WORK / 'submission_queue.pid').write_text(str(__import__('os').getpid()))
    if STATE.exists():
        state = json.loads(STATE.read_text())
    else:
        state = {'status': 'running', 'target_score': 0.961, 'top10_achieved': False,
                 'source_sha256': {p: digest(ROOT / p) for p in SOURCES},
                 'first_submission_ref': 56897663, 'best_score': 0.944, 'jobs': {
                     'head': {'stage': 'waiting_verifier', 'description': 'supervised ConvNeXt classifiers, original 30 percent blend'},
                     'highres': {'stage': 'training', 'description': 'high-resolution reader, fixed 15 percent blend'},
                     'convnext': {'stage': 'training', 'description': 'ConvNeXt study reader, fixed 15 percent blend'},
                     'combination': {'stage': 'waiting_readers', 'description': 'three study readers, fixed five percent each'}}}
    save(state)
    while True:
        try:
            guard(state)
        except Exception as error:
            state.update(status='stopped', reason=str(error)); save(state); return
        try:
            submissions = {s.ref: s for s in api.competition_submissions(COMPETITION, page_size=100) if s}
            train_status = {name: status(f'dk2lone/{kernel}') for name, (kernel, _) in TRAINING.items()}
            active = {name: status(j['kernel']) for name, j in state['jobs'].items() if j['stage'] == 'active'}
        except Exception as error:
            state['read_error'] = str(error); save(state)
            print('status read failed; retrying', str(error), flush=True)
            time.sleep(60); continue
        score_changed = False
        refs = [state['first_submission_ref'], *[j['submission_ref'] for j in state['jobs'].values() if 'submission_ref' in j]]
        state['submissions'] = [{'ref': ref, 'status': str(submissions[ref].status), 'score': submissions[ref].public_score}
                                for ref in refs if ref in submissions]
        for row in state['submissions']:
            if row['score'] and float(row['score']) > state['best_score']:
                state['best_score'] = float(row['score']); score_changed = True
        if score_changed or state.get('leaderboard_retry'):
            try:
                leaderboard(state); state.pop('leaderboard_retry', None)
            except Exception as error:
                state['leaderboard_retry'] = str(error)
        state['training_status'] = train_status
        slots = 2 - sum(s in ('RUNNING', 'PENDING', 'QUEUED') for s in [*train_status.values(), *active.values()])
        for name, job in state['jobs'].items():
            try:
                if job['stage'] in ('launching', 'submitting', 'submission_unknown'):
                    # A restart after an ambiguous write must never repeat that write.
                    match = next((s for s in submissions.values() if s.description == job['description']), None)
                    if job['stage'] in ('submitting', 'submission_unknown') and match:
                        job.update(stage='submitted', submission_ref=match.ref)
                    elif job['stage'] == 'launching':
                        job.update(stage='failed', error='interrupted kernel push requires manual review')
                    else:
                        job.update(stage='submission_unknown', error='awaiting confirmation; never resubmit an ambiguous request')
                if job['stage'] == 'training':
                    if train_status[name] in ('ERROR', 'CANCELLED'):
                        job.update(stage='failed', error='training failed')
                    elif train_status[name] == 'COMPLETE':
                        destination = WORK / f'queue/{name}-training'
                        collect(f'dk2lone/{TRAINING[name][0]}', destination, PATTERN)
                        command([NUMERIC, 'eda/score_reader.py', destination, WORK / 'dense/convnext_baseline.csv'])
                        job.update(training=str(destination), stage='ready_raw')
                if job['stage'] == 'waiting_verifier' and (WORK / 'head_env_verified.json').exists():
                    verified = json.loads((WORK / 'head_env_verified.json').read_text())
                    assert verified['status'] == 'passed', verified
                    training = WORK / 'head-output'
                    assert verified['head_delta_sha256'] == digest(training / 'head_delta.npz')
                    record = json.loads((training / 'head_fit.json').read_text())
                    job['training'] = str(training)
                    if record['metrics']['oof']['macro_auc'] <= record['metrics']['baseline']['macro_auc']:
                        job.update(stage='rejected', reason='supervised classifier CV did not improve')
                    else:
                        job['stage'] = 'ready_preview'
                if job['stage'] == 'ready_raw' and slots > 0:
                    command([PYTHON, 'eda/build_reader_probe.py', job['training'], '--variant', name])
                    push(state, job, f'kaggle/{TRAINING[name][1]}-raw-probe', 'raw'); slots -= 1
                elif job['stage'] == 'ready_preview' and slots > 0:
                    if name == 'head':
                        command([PYTHON, 'eda/build_supervised_head_stack.py', job['training']])
                        folder = 'kaggle/convnext-supervised-head-stack'
                    else:
                        training = Path(job['training'])
                        command([PYTHON, 'eda/build_effnet_stack.py', training / 'effnet_best.pt', training / 'run.json', '--variant', name])
                        folder = f'kaggle/{TRAINING[name][1]}-stack'
                    push(state, job, folder, 'preview'); slots -= 1
                elif job['stage'] == 'active':
                    worker = active.get(name, 'PENDING')
                    if worker in ('ERROR', 'CANCELLED'):
                        job.update(stage='failed', error=f'{job["active_kind"]} failed')
                    elif worker == 'COMPLETE':
                        destination = WORK / f'queue/{name}-{job["active_kind"]}'
                        if job['active_kind'] == 'raw':
                            collect(job['kernel'], destination, r'^(raw_gold.csv|raw_gold.receipt.json)$')
                            command([NUMERIC, 'eda/check_candidate.py', 'raw', destination, '--training', job['training']])
                            job.update(stage='ready_preview', raw_verified=str(destination / 'verified.json'))
                        else:
                            kind = 'head' if name == 'head' else ('combination' if name == 'combination' else 'reader')
                            collect(job['kernel'], destination, r'^(submission.csv|_effnet.csv|_effnet.receipt.json|_convnext_stack.csv|_head_receipt.json|_public_stack.csv|_own.csv|_combination_base.csv|_reader[0-2].csv|_reader[0-2].receipt.json|combination_receipt.json)$')
                            args = [NUMERIC, 'eda/check_candidate.py', kind, destination]
                            if name != 'combination': args += ['--training', job['training']]
                            command(args)
                            job['preview_verified'] = str(destination / 'verified.json')
                            submit(state, job)
                if job['stage'] == 'submitted':
                    submission = submissions.get(job['submission_ref'])
                    if submission and str(submission.status).endswith('COMPLETE') and submission.public_score:
                        job.update(stage='scored', score=submission.public_score)
                    elif submission and str(submission.status).endswith('ERROR'):
                        job.update(stage='failed', error='Kaggle scoring error')
            except Exception as error:
                if job['stage'] == 'submitting':
                    job.update(stage='submission_unknown', error=str(error))
                elif '429' in str(error):
                    job['retry_error'] = str(error)
                else:
                    job.update(stage='failed', error=str(error))
                print(name, str(error), flush=True)
            save(state)
        combination = state['jobs']['combination']
        readers = [state['jobs'][name] for name in ('highres', 'convnext')]
        if combination['stage'] == 'waiting_readers':
            if any(j['stage'] in ('failed', 'rejected') for j in readers):
                combination.update(stage='rejected', reason='a required reader failed')
            elif all('raw_verified' in j for j in readers) and state['jobs']['head']['stage'] not in ('waiting_verifier', 'ready_preview', 'active') and slots > 0:
                args = [PYTHON, 'eda/build_reader_combination.py', WORK / 'effnet-output', *[j['training'] for j in readers]]
                if 'preview_verified' in state['jobs']['head']: args.append('--use-head')
                try:
                    command(args)
                    push(state, combination, 'kaggle/reader-combination-stack', 'preview')
                except Exception as error:
                    combination.update(stage='failed', error=str(error))
        terminal = all(j['stage'] in ('scored', 'rejected', 'failed') for j in state['jobs'].values())
        first = submissions.get(state['first_submission_ref'])
        if terminal and first and str(first.status).endswith(('COMPLETE', 'ERROR')):
            try:
                leaderboard(state)
            except Exception as error:
                state['leaderboard_retry'] = str(error); save(state)
                time.sleep(60); continue
            state['status'] = 'top10_achieved' if state['top10_achieved'] else 'trials_finished_target_unmet'
            save(state)
            (ROOT / 'docs/top10-results/submission_queue.json').write_text(json.dumps(state, indent=2) + '\n')
            return
        save(state)
        print(json.dumps({'at': state['at'], 'stages': {n: j['stage'] for n, j in state['jobs'].items()}, 'best_score': state['best_score']}), flush=True)
        time.sleep(60)


if __name__ == '__main__':
    run()
