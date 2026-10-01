"""Download, prepare, train and evaluate the five USPTO benchmarks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    'fg': 'functional_group_prediction',
    'properties': 'physicochemical_property_prediction',
    'structure': 'molecular_structure_elucidation',
    'detection': 'targeted_component_detection',
    'fraction': 'targeted_fractional_contribution_estimation',
}
RECORD = 'https://zenodo.org/api/records/16417648'
ENCODERS = ('ultrair_pretraining_general_epoch5.pt', 'ultrair_pretraining_molstrelu_epoch5.pt')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def run_logged(command, log, append=False):
    """Stream child progress to the terminal and preserve the complete log."""
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    with log.open('ab' if append else 'wb') as output:
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, env=env) as child:
            try:
                while block := child.stdout.read1(8192):
                    output.write(block)
                    output.flush()
                    sys.stdout.write(block.decode('utf-8', errors='replace'))
                    sys.stdout.flush()
                code = child.wait()
            except BaseException:
                child.terminate()
                child.wait()
                raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def md5(path):
    digest = hashlib.md5()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch(url, path, size=None, checksum=None):
    """Resume into a partial file; publish only after integrity checks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    def valid(p):
        return p.is_file() and (size is None or p.stat().st_size == size) and (
            checksum is None or md5(p) == checksum.removeprefix('md5:'))
    if valid(path):
        print(f'[download] verified {path}', flush=True)
        return
    if path.exists():
        raise ValueError(f'Existing file fails integrity checks: {path}; move it aside before retrying')
    partial = path.with_name(path.name + '.part')
    if valid(partial):
        partial.replace(path)
        return
    subprocess.run(['curl', '--location', '--fail', '--retry', '5', '--retry-delay', '3',
                    '--connect-timeout', '30', '--speed-limit', '1024', '--speed-time', '120',
                    '--continue-at', '-', '--output', str(partial), url], check=True)
    if not valid(partial):
        raise ValueError(f'Download fails integrity checks: {partial}')
    partial.replace(path)


def download(raw, manifest_path=None, read_only=False):
    with urlopen(RECORD, timeout=60) as response:
        record = json.load(response)
    files = sorted((f for f in record['files'] if re.fullmatch(r'IR_data_chunk00[1-9]_of_009\.parquet', f['key'])), key=lambda f: f['key'])
    if len(files) != 9:
        raise ValueError('Expected exactly nine IR Parquet shards in Zenodo record 16417648')
    for index, entry in enumerate(files, 1):
        print(f"[download] shard {index}/9: {entry['key']} (file ETA below)", flush=True)
        if read_only and not (raw / entry['key']).is_file():
            raise FileNotFoundError(f"Missing shard in supplied raw directory: {entry['key']}")
        fetch(entry['links']['self'], raw / entry['key'], entry['size'], entry['checksum'])
    write_json(manifest_path or raw / 'source.json', {'record': RECORD, 'doi': record['metadata'].get('doi'), 'files': files})


def prepare(args):
    import numpy as np
    import rdkit
    from data.common.parquet_ir import prepare as convert
    from data.common.molecular import prepare as label
    from data.common.scaffold_split import scaffold_split, scaffold_key, NO_SCAFFOLD
    work = args.data_dir
    target = work / 'prepared'
    shards = sorted(args.raw_dir.glob('IR_data_chunk00[1-9]_of_009.parquet'))
    if not args.max_molecules and len(shards) != 9:
        raise ValueError('Full preparation requires all nine USPTO IR shards')
    settings = {'seed': args.seed, 'max_molecules': args.max_molecules, 'k': 5, 'valid_fraction': .1,
                'range_cm-1': [400, 4000], 'points': 3600, 'protocol': 1,
                'rdkit_version': rdkit.__version__}
    marker = target / 'complete.json'
    if marker.exists():
        if json.loads(marker.read_text())['settings'] != settings:
            raise ValueError('Preparation settings changed; choose a new --data-dir')
        print(f'[prepare] reuse {target}', flush=True)
        return
    conversion_marker = work / 'converted/manifest.json'
    conversion = json.loads(conversion_marker.read_text()) if conversion_marker.exists() else {}
    reuse_conversion = (conversion.get('max_rows') == (args.max_molecules or None)
                        and conversion.get('signal_length') == 3600
                        and conversion.get('grid_from_source') is True
                        and conversion.get('source_files') == [p.name for p in shards])
    if not reuse_conversion:
        convert(args.raw_dir, work / 'converted', pattern='IR_data_chunk00[1-9]_of_009.parquet', batch_size=32, max_rows=args.max_molecules, grid_from_source=True, progress=True)
    if not args.max_molecules and json.loads((work / 'converted/manifest.json').read_text())['rows'] != 177461:
        raise ValueError('The full release should contain 177461 molecules')
    label(work / 'converted/ir_norm.npy', work / 'converted/smiles.npy', target / 'full', make_folds=False, progress=True)
    full = target / 'full'
    smiles = np.load(full / 'smiles.npy', allow_pickle=False)
    source_ids = np.load(work / 'converted/ids.npy', allow_pickle=False)
    from rdkit import Chem, rdBase
    original = np.load(work / 'converted/smiles.npy', allow_pickle=False)
    with rdBase.BlockLogs():
        source_valid = np.array([Chem.MolFromSmiles(str(s)) is not None for s in original])
    np.save(full / 'ids.npy', source_ids[source_valid])
    print("[prepare] Creating five scaffold folds; ETA estimating", flush=True)
    split_started = time.time()
    folds = scaffold_split(smiles, k=5, seed=args.seed, valid_fraction=.1)
    keys = np.asarray([scaffold_key(s) for s in smiles])
    reports = []
    for fold, (train, valid, test) in enumerate(folds, 1):
        indices = {'train': train, 'valid': valid, 'test': test}
        nonempty_scaffolds = {split: set(keys[idx]) - {NO_SCAFFOLD} for split, idx in indices.items()}
        for a, b in [('train','valid'), ('train','test'), ('valid','test')]:
            if nonempty_scaffolds[a] & nonempty_scaffolds[b]:
                raise ValueError(f'Scaffold leakage in fold {fold}: {a}/{b}')
        for split, idx in indices.items():
            if len(idx) < 6:
                raise ValueError(f'fold-{fold}/{split} has fewer than six molecules; increase --max-molecules')
            out = target / f'fold-{fold}' / split
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / 'source_rows.npy', np.flatnonzero(source_valid)[idx])
            for path in sorted(full.glob('*.npy')):
                array = np.load(path, allow_pickle=False)
                np.save(out / path.name, array[idx])
        reports.append({'fold': fold, 'rows': {k: len(v) for k,v in indices.items()}, 'scaffold_overlap': 0})
        eta = (time.time() - split_started) / fold * (5 - fold)
        print(f'[prepare] Scaffold folds {fold}/5 ({fold/5:.0%}); ETA {duration(eta)}', flush=True)
    write_json(marker, {'settings': settings, 'rows': len(smiles), 'folds': reports})


def checkpoints(args):
    paths = {}
    for name in ENCODERS:
        print(f"[encoders] {'reuse' if (args.checkpoint_dir / name).is_file() else 'download'} {name}", flush=True)
        path = args.checkpoint_dir / name
        if not path.is_file():
            fetch('https://huggingface.co/yusentan/UltraIR/resolve/main/checkpoints/pretraining/' + name, path)
        paths[name] = path
    return paths


def reusable_result(config_path, config, report_dir, task, fold):
    """Reuse only a completed prediction/evaluation with matching effective settings."""
    import yaml
    import copy
    report_path = report_dir / f"test_{config['run']['ckpt_tag']}.json"
    if not config_path.is_file() or not report_path.is_file() or not (report_dir / 'example_predictions.json').is_file():
        return False
    def comparable(value):
        value = copy.deepcopy(value)
        value['train'].pop('save_root', None)
        value['results'].pop('root', None)
        value['data']['root'] = str(Path(value['data']['root']).resolve())
        return value
    previous = yaml.safe_load(config_path.read_text())
    if comparable(previous) != comparable(config):
        raise ValueError(f'Completed result configuration differs: {report_dir}; use a new work directory')
    report = json.loads(report_path.read_text())
    return (report.get('task') == task and report.get('fold') == fold
            and bool(report.get('metrics')) and Path(report['checkpoint']).is_file())


def run(args):
    import yaml
    import numpy as np
    import torch
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Run this stage on a GPU machine or explicitly use --device cpu.')
    preparation = json.loads((args.data_dir / 'prepared/complete.json').read_text())
    if preparation['settings']['seed'] != args.seed or preparation['settings']['max_molecules'] != args.max_molecules:
        raise ValueError('Run settings differ from prepared folds; choose matching settings or a new --data-dir')
    paths = checkpoints(args)
    log_root = args.work_dir / 'logs'
    log_root.mkdir(parents=True, exist_ok=True)
    summary = []
    total_runs = len(args.tasks) * len(args.folds)
    timings = {alias: [] for alias in args.tasks}
    def show_progress():
        remaining = {alias: sum(not any(x["task"] == TASKS[alias] and x["fold"] == fold for x in summary) for fold in args.folds) for alias in args.tasks}
        unknown = any(count and not timings[alias] for alias, count in remaining.items())
        eta = "estimating" if unknown else duration(sum(count * (sum(timings[alias])/len(timings[alias]) if timings[alias] else 0) for alias, count in remaining.items()))
        print(f"[run] Completed {len(summary)}/{total_runs} task-folds ({len(summary)/total_runs:.0%}); remaining ETA {eta}", flush=True)
    show_progress()
    write_json(args.work_dir / 'run_summary.json', {'smoke': args.smoke, 'status': 'running', 'completed': summary})
    for fold in args.folds:
        for alias in args.tasks:
            task = TASKS[alias]
            print(f"[run] Task {alias}, fold {fold}/5; completed {len(summary)}/{total_runs}", flush=True)
            config = yaml.safe_load((ROOT / 'configs' / task / 'uspto.yaml').read_text())
            config['data']['root'] = str(args.data_dir / 'prepared')
            if alias in ('detection', 'fraction'):
                from data.common.pairs import generate_pairs_to_disk
                config['data']['root'] = str(args.pair_dir)
            config['device'] = args.device
            config['seed'] = args.seed
            if args.num_workers is not None:
                config['loader']['num_workers'] = args.num_workers
            if args.batch_size:
                config['loader']['batch_size'] = args.batch_size
            config['train']['save_root'] = str(args.work_dir / 'checkpoints')
            config['results']['root'] = str(args.work_dir / 'results')
            encoder = ENCODERS[1] if alias == 'structure' else ENCODERS[0]
            config['run']['init_ckpt'] = str(paths[encoder])
            config['augment']['path'] = str(ROOT / 'configs/aug.yaml')
            if args.epochs:
                config['train']['epochs'] = args.epochs
            if args.smoke:
                config['task']['args'].update({'generation_max_len': 16, 'beam_size': 2, 'num_return_sequences': 2, 'eval_topk': [1,2]}) if alias == 'structure' else None
                if alias == 'structure':
                    config['model']['args'].update({'beam_size': 2, 'num_return_sequences': 2})
            config_path = args.work_dir / 'configs' / task / f'fold-{fold}.yaml'
            report_dir = args.work_dir / 'results' / task / config['run']['method_name'] / f'fold-{fold}'
            if args.reuse_completed and reusable_result(config_path, config, report_dir, task, fold):
                print(f'[reuse] {task} fold-{fold}', flush=True)
                summary.append({'task': task, 'fold': fold, 'reused': True})
                show_progress()
                write_json(args.work_dir / 'run_summary.json', {'smoke': args.smoke, 'status': 'running', 'completed': summary})
                continue
            if alias in ('detection', 'fraction'):
                print(f'[pairs] {alias} fold {fold}: preparing train/valid/test mixtures', flush=True)
                pair_started = time.time()
                for split_index, split in enumerate(('train', 'valid', 'test'), 1):
                    print(f'[pairs] {alias} fold {fold}: {split} started', flush=True)
                    source = args.data_dir / 'prepared' / f'fold-{fold}' / split
                    directory = args.pair_dir / f'fold-{fold}' / split
                    pair_seed = args.seed + fold * 100 + ['train','valid','test'].index(split)
                    marker = directory / 'pairs_complete.json'
                    if marker.exists():
                        cached = json.loads(marker.read_text())
                        if cached['seed'] != pair_seed or cached['augmentations'] != 4 or cached['molecules'] != len(np.load(source / 'ids.npy')):
                            raise ValueError(f'Pair settings changed; remove regenerable cache at {directory}')
                    else:
                        generate_pairs_to_disk(np.load(source / 'ir_norm.npy'), directory,
                                               augmentations=4, seed=pair_seed)
                    print(f'[pairs] {alias} fold {fold}: {split} complete; splits {split_index}/3 ({split_index/3:.0%}); elapsed {duration(time.time()-pair_started)}', flush=True)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            command = [sys.executable, '-m', 'scripts.run', '--config', str(config_path), '--fold', str(fold)]
            log = log_root / f'{alias}-fold-{fold}.log'
            print('[train_eval]', ' '.join(command), flush=True)
            started = time.time()
            print(f'[train] {alias} fold {fold}; epoch/batch progress and ETA below; log: {log}', flush=True)
            run_logged(command, log)
            # Independently reload the saved checkpoint and run evaluation.
            evaluation = command + ['--mode', 'infer_eval', '--strict']
            print(f'[evaluate] {alias} fold {fold}: reload checkpoint; batch progress and ETA below', flush=True)
            run_logged(evaluation, log, append=True)
            report_dir = args.work_dir / 'results' / task / config['run']['method_name'] / f'fold-{fold}'
            report = json.loads((report_dir / f"test_{config['run']['ckpt_tag']}.json").read_text())
            prediction_input = report_dir / 'example_input.npy'
            test_dir = Path(config['data']['root']) / f'fold-{fold}' / 'test'
            # Small unlabeled prediction export verifies the standalone inference entry point.
            np.save(prediction_input, np.load(test_dir / config['data']['ir_name'])[:2])
            predict = [sys.executable, '-m', 'scripts.predict', '--config', str(config_path),
                       '--ckpt', report['checkpoint'], '--input', str(prediction_input),
                       '--output', str(report_dir / 'example_predictions.json'), '--stats-fold', str(fold),
                       '--batch-size', '2']
            if alias == 'structure':
                formula_path = report_dir / 'example_formula.npy'
                np.save(formula_path, np.load(test_dir / 'formula.npy')[:2])
                predict.extend(['--formula', str(formula_path)])
            print(f'[predict] {alias} fold {fold}: example predictions', flush=True)
            run_logged(predict, log, append=True)
            timings[alias].append(time.time()-started)
            summary.append({'task': task, 'fold': fold, 'seconds': round(time.time()-started, 1), 'log': str(log)})
            show_progress()
            write_json(args.work_dir / 'run_summary.json', {'smoke': args.smoke, 'status': 'running', 'completed': summary})
        if not args.keep_pairs and any(t in args.tasks for t in ('detection','fraction')):
            for split in ('train','valid','test'):
                directory = args.pair_dir / f'fold-{fold}' / split
                for name in ('mixture_set.npy','mixture_labels.npy','mixture_ref_idx.npy','mixture_ref_weight.npy','pairs_complete.json'):
                    (directory / name).unlink(missing_ok=True)
    print("[aggregate] Writing cross-validation summaries", flush=True)
    from scripts.run import aggregate_fold_metrics
    for alias in args.tasks:
        task = TASKS[alias]
        root = args.work_dir / 'results' / task / 'ultrair_pretrained_uspto'
        reports = []
        for fold in args.folds:
            report_paths = list((root / f'fold-{fold}').glob('test_*.json'))
            if len(report_paths) != 1:
                raise ValueError(f'Expected one selected-checkpoint report for {task}, fold {fold}')
            reports.append(json.loads(report_paths[0].read_text()))
        write_json(root / 'cross_validation.json', {
            'task': task, 'folds': args.folds, 'smoke': args.smoke,
            'metrics': aggregate_fold_metrics(reports),
        })
    write_json(args.work_dir / 'run_summary.json', {'smoke': args.smoke, 'status': 'complete', 'completed': summary})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all','download','prepare','run'], default='all')
    parser.add_argument('--work-dir', type=Path, default=ROOT / 'runs/uspto')
    parser.add_argument('--data-dir', type=Path, help='Dataset cache; defaults to data/uspto (data/uspto-smoke for smoke checks)')
    parser.add_argument('--raw-dir', type=Path, help='Existing raw shards; verified against Zenodo checksums')
    parser.add_argument('--checkpoint-dir', type=Path, help='Encoder-only checkpoint directory; defaults to checkpoints/pretraining in the repository')
    parser.add_argument('--pair-dir', type=Path, help='Scratch directory for regenerable mixtures; defaults to data-dir/pairs')
    parser.add_argument('--tasks', nargs='+', choices=list(TASKS), default=list(TASKS))
    parser.add_argument('--folds', nargs='+', type=int, choices=range(1,6), default=list(range(1,6)))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-workers', type=int, help='Override YAML loader workers')
    parser.add_argument('--max-molecules', type=int, default=0)
    parser.add_argument('--smoke', action='store_true', help='256 real molecules, fold 1, one epoch, short structure decoding; not paper metrics')
    parser.add_argument('--reuse-completed', action='store_true', help='Skip completed task/folds with matching configuration, checkpoint, evaluation and predictions')
    parser.add_argument('--keep-pairs', action='store_true', help='Retain generated pair arrays (about 40 GB per full fold)')
    args = parser.parse_args()
    args.work_dir = args.work_dir.resolve()
    raw_supplied = args.raw_dir is not None
    args.data_dir = (args.data_dir or ROOT / 'data' / ('uspto-smoke' if args.smoke else 'uspto')).resolve()
    args.raw_dir = (args.raw_dir or args.data_dir / 'raw').resolve()
    args.checkpoint_dir = (args.checkpoint_dir or ROOT / 'checkpoints/pretraining').resolve()
    args.pair_dir = (args.pair_dir or args.data_dir / 'pairs').resolve()
    if args.smoke:
        args.max_molecules = args.max_molecules or 256
        args.epochs = args.epochs or 1
        args.batch_size = args.batch_size or 8
        args.folds = [1]
    if args.epochs is not None and args.epochs < 1 or args.max_molecules < 0 or (args.num_workers is not None and args.num_workers < 0):
        parser.error('epochs must be positive; max-molecules and num-workers must be non-negative')
    if args.batch_size is not None and args.batch_size < 1:
        parser.error('batch-size must be positive')
    os.environ.setdefault('PYTHONPATH', str(ROOT / 'src'))
    sys.path.insert(0, str(ROOT / 'src'))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.work_dir / 'invocation.json', {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
    stages = ['download', 'prepare', 'run'] if args.stage == 'all' else [args.stage]
    for index, stage in enumerate(stages, 1):
        print(f'[stage {index}/{len(stages)}] {stage.upper()} started', flush=True)
        started = time.time()
        if stage == 'download':
            download(args.raw_dir, args.data_dir / 'source.json', read_only=raw_supplied)
            checkpoints(args)
        elif stage == 'prepare':
            prepare(args)
        else:
            run(args)
        print(f'[stage {index}/{len(stages)}] {stage.upper()} complete; elapsed {duration(time.time()-started)}', flush=True)
    print(f'[pipeline] stage {args.stage} completed; outputs: {args.work_dir}', flush=True)

if __name__ == '__main__':
    main()
