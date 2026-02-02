import os, sys
import sqlite3
import hashlib, base64
import argparse
import shutil
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import quote
from Bio import SeqIO
from tqdm import tqdm
import queue
import itertools
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor as TPE
from contextlib import contextmanager
import subprocess
import re

# --- Helper Functions ---

@contextmanager
def get_log(log_path):
    if not log_path:
        yield subprocess.DEVNULL
        return
    f = open(log_path, 'w', encoding='utf-8')
    try:
        yield f
    finally:
        f.close()

def clean_seq(seq, keep_case):
    seq = seq.strip().replace("*", "").replace("-", "").replace(" ", "")
    if not keep_case:
        seq = seq.upper()
    return seq

def get_hash(seq, keep_case):
    raw_hash = hashlib.sha256(str(seq).encode('ascii')).digest()
    return base64.urlsafe_b64encode(raw_hash).decode().rstrip("=")

def get_rle(seq):
    return "".join(f"{len(list(group))}{char}" for char, group in itertools.groupby(seq.strip()))

def decode_rle(data):
    return "".join(int(count) * char for count, char in re.findall(r"(\d+)(.)", data))

def download_file(url, output_path):
    """Downloads a file using urllib.request."""
    if not output_path.exists():
        output_path.parent.mkdir(parents = True, exist_ok = True)
        print(f"[*] Downloading: {url} -> {output_path}")
        try:
            with urllib.request.urlopen(url) as response, open(output_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        except Exception as e:
            if output_path.exists():
                output_path.unlink()
            print(f"[!] Failed to download {url}: {e}")
            raise e

# --- Initialization Logic ---

def data_paths(data_dir):
    data_path = Path(data_dir).resolve()
    torch_path = data_path / "torch"
    db_path = data_path / "data.sq3"
    return data_path, torch_path, db_path

def launch_init(data_dir):
    """Phase 1: General environment setup and 1.6B latent model."""
    data_path, torch_path, db_path = data_paths(data_dir)
    esm_ckpt_path = torch_path / "hub" / "checkpoints"

    # 2. Build General Download Queue
    file_tasks = [
        ("https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt", esm_ckpt_path / "esm2_t36_3B_UR50D.pt"),
        ("https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t36_3B_UR50D-contact-regression.pt", esm_ckpt_path / "esm2_t36_3B_UR50D-contact-regression.pt")
    ]
    # 3. Parallel Downloads for General Init
    print(f"[*] Starting General Init parallel downloads...")
    with TPE(max_workers = len(file_tasks)) as executor:
        futures = []
        for url, target in file_tasks:
            futures.append(executor.submit(download_file, url, target))
        with tqdm(total = len(file_tasks), leave = True) as progress:
            for future in concurrent.futures.as_completed(futures):
                progress.update(1)

    print(f"[*] Initializing database at {db_path}")
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS deep_tmhmm (seq_hash TEXT PRIMARY KEY, rle TEXT)")

    print("[✔] Initialization complete.")

def fetch_rle(conn, seq_hash):
    found = conn.execute("SELECT rle FROM deep_tmhmm WHERE seq_hash = ?", (seq_hash,)).fetchone()
    return found[0] if found else None

# --- Inference Logic ---

def run_gpu_worker(batch, gpu, cpus, data_path, torch_path, dt_path, log):

    assert isinstance(cpus, int)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        fasta_path = tmp_path / "input.fasta"
        output_path = tmp_path / "output"

        customize_path = tmp_path / "sitecustomize.py"
        customize_path.write_text(f"import os; os.cpu_count = lambda: {cpus}")

        for item in dt_path.iterdir():
            if item.name not in [ fasta_path.name, output_path.name ]:
                (tmp_path / item.name).symlink_to(item.resolve(), target_is_directory = item.is_dir())

        seq_hashes = {}
        seq_names = {}
        with open(fasta_path, 'w') as file:
            for seq_hash, record in batch:
                quoted_name = quote(record.id, safe = "-_")
                seq_names[quoted_name] = record.id
                seq_hashes[quoted_name] = seq_hash
                record.id = quoted_name
                SeqIO.write(record, file, "fasta")

        env = os.environ.copy()
        pythonpath = [ str(tmp_path) ]
        if "PYTHONPATH" in env:
            pythonpath.add(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu) if gpu is not None else ''
        env["TORCH_HOME"] = str(torch_path)
        cmd = [ "python", "predict.py", "--fasta", fasta_path.name, "--output-dir", output_path.name ]
        subprocess.run(cmd, env = env, check = True, cwd = tmp_path, stdout = log, stderr = log)
        output = []
        pred_path = output_path / "predicted_topologies.3line"
        if pred_path.exists():
            with open(pred_path) as file:
                for header, seq, pred in zip(*[file] * 3):
                    quoted_name = header[1:].split()[0]
                    seq_name = seq_names[quoted_name]
                    seq_hash = seq_hashes[quoted_name]
                    rle = get_rle(pred)
                    output.append((seq_hash, seq_name, rle))
        return output

class GPUPool:
    def __init__(self, gpus, workers):
        self.assigned_gpus = list(itertools.islice(itertools.cycle(gpus), workers))
        self._queue = queue.Queue()
        for gpu in self.assigned_gpus:
            self._queue.put(gpu)
    def get(self):
        return self._queue.get() if self.assigned_gpus else None
    def put(self, gpu):
        if gpu is not None:
            self._queue.put(gpu)

def launch_run(input_file, output_file, batch_size, data_dir, dt_dir, log_file, keep_case, gpus, cpus, workers):
    output_path = Path(output_file).resolve()
    data_path, torch_path, db_path = data_paths(data_dir)
    dt_path = Path(dt_dir)

    if cpus <= 0:
        raise ValueError(f"Specify a positive number of CPUs")
    if workers <= 0:
        raise ValueError(f"Specify a positive number of workers")

    if not data_path.exists():
        raise FileNotFoundError(f"[✘] Data directory {data_dir} not found. Run 'init' mode first.")
    if not dt_path.exists() or not (dt_path / "predict.py").exists():
        raise FileNotFoundError(f"[✘] DeepTMHMM not found in {dt_dir}. Download it from https://dtu.biolib.com/DeepTMHMM.")
    if not db_path.exists():
        raise FileNotFoundError(f"[✘] Database {db_path} not found. Run 'init' mode first.")
    if not torch_path.exists():
        raise FileNotFoundError(f"[✘] Torch directory {torch_path} not found. Run 'init' mode first.")

    print(f"[*] Checking the fasta file")
    to_analyze = {}
    all_names = []
    unique = set()
    results = {}
    with sqlite3.connect(db_path) as conn:
        for record in SeqIO.parse(input_file, "fasta"):
            if record.id in unique:
                print(f"[✘] Duplicated record id {record.id}")
                sys.exit(1)
            unique.add(record.id)
            seq = clean_seq(record.seq, keep_case)
            seq_hash = get_hash(seq, keep_case)
            rle = fetch_rle(conn, seq_hash)
            if rle:
                results[record.id] = rle
            else:
                to_analyze[record.id] = seq_hash
            all_names.append(record.id)
    print(f"[✔] A total of {len(all_names)} sequences, {len(to_analyze)} to analyze")

    def process_fasta():
        to_predict = {}
        for record in SeqIO.parse(input_file, "fasta"):
            seq_hash = to_analyze.pop(record.id, None)
            if seq_hash:
                record.seq = clean_seq(record.seq, keep_case)
                to_predict[record.id] = seq_hash, record
                if len(to_predict) == batch_size:
                    yield to_predict
                    to_predict = {}
        if to_predict:
            yield to_predict

    gpu_pool = GPUPool(gpus, workers)

    def wrapper(batch, log):
        gpu = gpu_pool.get()
        try:
            return run_gpu_worker(batch, gpu, max(1, cpus // workers), data_path, torch_path, dt_path, log)
        except Exception as e:
            print(f"[✘] Error: Got exception: {e}")
        finally:
            gpu_pool.put(gpu)

    def check_futures(futures, conn, all_results, progress_bar, max_num = 1):
        assert max_num > 0
        results = {}
        while len(futures) >= max_num:
            futures_done, futures = concurrent.futures.wait(futures, return_when = concurrent.futures.FIRST_COMPLETED)
            for future in futures_done:
                for seq_hash, seq_name, rle in future.result():
                    conn.execute("INSERT OR IGNORE INTO deep_tmhmm (seq_hash, rle) VALUES (?, ?)", (seq_hash, rle))
                    results[seq_name] = rle
                conn.commit()
        progress_bar.update(len(results))
        all_results.update(results)
        return futures

    with get_log(log_file) as log, TPE(max_workers = workers) as executor, sqlite3.connect(db_path) as conn, tqdm(total = len(to_analyze)) as progress_bar:
        futures = set()
        for to_predict in process_fasta():
            futures = check_futures(futures, conn, results, progress_bar, workers)
            futures.add(executor.submit(wrapper, to_predict.values(), log))
        check_futures(futures, conn, results, progress_bar)

    ok = True
    with open(output_path, 'w') as file:
        for seq_name in all_names:
            if rle := results.pop(seq_name, None):
                pred = decode_rle(rle)
                file.write(f">{seq_name}\n{pred}\n")
            else:
                print(f"No results obtained for {seq_name}")
                ok = False
    if not ok:
        print(f"Error: No results were returned for some of the records")
        sys.exit(1)

# --- Main CLI Entry ---

def init_cli():
    parser = argparse.ArgumentParser(description = "DeepTMHMM wrapper: initialize the data")
    
    # --- General Arguments ---
    parser.add_argument("-D", "--data-dir", required = True, help = "Base directory for data")
    args = parser.parse_args()
    launch_init(args.data_dir)

def run_cli():
    parser = argparse.ArgumentParser(description = "DeepTMHMM wrapper: run the inference")
    def set_of_int(arg):
        return set(int(x) for x in arg.split(',')) if arg != '' else []

    BATCH_SIZE = 100

    parser.add_argument("-D", "--data-dir", required = True, help = "Base directory for data")
    parser.add_argument("-i", "--input", required = True, help = "Path to input sequences")
    parser.add_argument("-o", "--output", required = True, help = "Output file")
    parser.add_argument("-E", "--deep-tmhmm", required = True, help = "DeepTMHMM executable directory (from https://dtu.biolib.com/DeepTMHMM)")
    parser.add_argument("-b", "--batch", type = int, default = BATCH_SIZE, help = f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("-g", "--gpus", type = set_of_int, default = [0], help = "Comma-separated list of indices of available GPUs or empty (default: 0 [first GPU])")
    parser.add_argument("-c", "--cpus", type = int, default = 1, help = "Total number of CPUs to use (default: 1)")
    parser.add_argument("-w", "--workers", type = int, default = 1, help = "Total number of parallel workers to spawn (default: 1)")
    parser.add_argument("-k", "--keep-case", action = 'store_true', help = "Do not change the case of the sequences")
    parser.add_argument("-l", "--log", type = str, help = "Raw log file")
    args = parser.parse_args()
    launch_run(
        args.input, args.output, args.batch, args.data_dir, args.deep_tmhmm, args.log,
        gpus = args.gpus, cpus = args.cpus, workers = args.workers, keep_case = args.keep_case
    )
