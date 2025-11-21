#!/usr/bin/env python3
import os, sys, json, subprocess, random, time

# --- 兜底：确保优先导入 /opt/nemo ---
_p = "/opt/nemo"
if os.path.isdir(_p) and _p not in sys.path:
    sys.path.insert(0, _p)

def env(name, default=None, required=False, cast=str):
    v = os.environ.get(name, default)
    if required and (v is None or v == ""):
        print(f"[X] missing env: {name}")
        sys.exit(2)
    if v is not None and cast is not str:
        try:
            v = cast(v)
        except Exception:
            pass
    return v

# === SageMaker Script Mode 环境（容器内） ===
CH_TRAIN      = env("SM_CHANNEL_TRAIN",      required=True)   # /opt/ml/input/data/train  或 train_s3 等
CH_VAL        = env("SM_CHANNEL_VAL",        required=True)   # /opt/ml/input/data/val
CH_TOKENIZER  = env("SM_CHANNEL_TOKENIZER",  required=True)   # /opt/ml/input/data/tokenizer
CH_PRETRAINED = env("SM_CHANNEL_PRETRAINED", required=True)   # /opt/ml/input/data/pretrained
MODEL_DIR     = env("SM_MODEL_DIR",          required=True)   # /opt/ml/model
OUT_DIR       = env("SM_OUTPUT_DATA_DIR",    required=True)   # /opt/ml/output
TB_DIR        = os.path.join("/opt/ml/output", "tensorboard")

# 训练/评估文件
TRAIN_MANIFEST       = env("TRAIN_MANIFEST", required=True)
VAL_MANIFEST         = env("VAL_MANIFEST",   required=True)
PRETRAINED_FILENAME  = env("PRETRAINED_FILENAME", "parakeet-tdt_ctc-110m.nemo")

# 配置文件名称
CONFIG_NAME          = env("CONFIG_NAME", "fastconformer_hybrid_tdt_ctc_bpe_110m")

# 训练控制
DEVICES              = env("DEVICES",            8,  cast=int)
PRECISION            = env("PRECISION",          "bf16")
VAL_CHECK_INTERVAL   = env("VAL_CHECK_INTERVAL", 2000, cast=int)
RESUME               = env("RESUME",             0,   cast=int)   # 0/1
MAX_EPOCHS           = env("MAX_EPOCHS",         20,  cast=int)
MAX_STEPS_ENV        = os.environ.get("MAX_STEPS", "")

# 批量与 Loader
TRAIN_BATCH          = env("TRAIN_BATCH",        "", cast=str)    # 若空则自动/不覆写
VAL_BATCH            = env("VAL_BATCH",          "", cast=str)
GRAD_ACCUM           = env("GRAD_ACCUM",         1,   cast=int)
NUM_WORKERS          = env("NUM_WORKERS",        8,   cast=int)
AUTO_BATCH_PER_GPU   = env("AUTO_BATCH_PER_GPU", 1,   cast=int)   # 1=自动给默认 per-GPU batch

# 时长裁剪（留空表示不覆写）
MAX_DURATION_TRAIN   = os.environ.get("MAX_DURATION_TRAIN", "")
MIN_DURATION_TRAIN   = os.environ.get("MIN_DURATION_TRAIN", "")
MAX_DURATION_VAL     = os.environ.get("MAX_DURATION_VAL", "")
MIN_DURATION_VAL     = os.environ.get("MIN_DURATION_VAL", "")

# 抽样检查
SAMPLE_K             = env("SAMPLE_K", 64, cast=int)
SAMPLE_SEED_ENV      = os.environ.get("SAMPLE_SEED")

# Checkpoint 频率（二选一；若 steps>0 则 steps 优先）
CKPT_EVERY_STEPS     = env("CKPT_EVERY_STEPS",  0, cast=int)
CKPT_EVERY_EPOCHS    = env("CKPT_EVERY_EPOCHS", 0, cast=int)

# 信息提示（由 entrance.py 传入）
S3_OUT = os.environ.get("S3_OUT", "")
S3_TB  = os.environ.get("S3_TB", "")
S3_CKPT= os.environ.get("S3_CKPT", "")
REGION = os.environ.get("REGION", "")

# === AUGMENTOR 环境变量 ===
AUG_ON_VAL          = env("AUG_ON_VAL",           0,    cast=int)

# noise
AUG_NOISE_PROB      = env("AUG_NOISE_PROB",       0.0,  cast=float)
AUG_NOISE_MANIFEST  = env("AUG_NOISE_MANIFEST",   "",   cast=str)  # 可相对 CH_TRAIN
AUG_NOISE_MIN_SNR   = env("AUG_NOISE_MIN_SNR_DB", 0.0,  cast=float)
AUG_NOISE_MAX_SNR   = env("AUG_NOISE_MAX_SNR_DB", 15.0, cast=float)

# speed
AUG_SPEED_PROB      = env("AUG_SPEED_PROB",       0.0,  cast=float)
AUG_SPEED_SR        = env("AUG_SPEED_SR",         16000, cast=int)
AUG_SPEED_RESAMPLE  = env("AUG_SPEED_RESAMPLE",   "kaiser_fast", cast=str)
AUG_SPEED_MIN_RATE  = env("AUG_SPEED_MIN_RATE",   0.95, cast=float)
AUG_SPEED_MAX_RATE  = env("AUG_SPEED_MAX_RATE",   1.05, cast=float)

# gain（与 entrance.py 对齐：*_DBFS）
AUG_GAIN_PROB       = env("AUG_GAIN_PROB",        0.0,  cast=float)
AUG_GAIN_MIN_DB     = env("AUG_GAIN_MIN_DBFS",   -10.0, cast=float)
AUG_GAIN_MAX_DB     = env("AUG_GAIN_MAX_DBFS",    10.0, cast=float)

def resolve_path(base_dir: str, p: str) -> str:
    if not p:
        return ""
    return p if p.startswith("/") else os.path.join(base_dir, p)

print("== PRE-FLIGHT (Script Mode) ==")
print("SM_CHANNELS:", {
    "train": CH_TRAIN, "val": CH_VAL,
    "tokenizer": CH_TOKENIZER, "pretrained": CH_PRETRAINED
})

# 0) 打印容器级提示（方便在 CloudWatch 日志里直接看到 S3 去向）
print("\n--- WHERE TO WATCH (printed by container) ---")
print(f"[TensorBoard S3]  {S3_TB}")
print(f"[Checkpoints S3]  {S3_CKPT}")
print(f"[Output S3]       {S3_OUT}")
print("------------------------------------------------\n")

# 1) GPU / Torch
subprocess.run(["nvidia-smi"], check=False)
py = (
    "import sys, torch; "
    "print('[Torch]', torch.__version__, 'cuda?', torch.cuda.is_available(), 'gpus:', torch.cuda.device_count()); "
    "assert torch.cuda.is_available(), 'CUDA not available'"
)
subprocess.check_call([sys.executable, "-c", py])

# 2) 关键文件存在性
tok_dir   = CH_TOKENIZER
nemo_path = os.path.join(CH_PRETRAINED, PRETRAINED_FILENAME)
for p in [os.path.join(CH_TRAIN, TRAIN_MANIFEST),
          os.path.join(CH_VAL,   VAL_MANIFEST),
          tok_dir, nemo_path]:
    if not (os.path.isdir(p) or os.path.isfile(p)):
        print("[X] missing:", p); sys.exit(2)

# 3) 随机抽样检查 manifest（Reservoir Sampling）
def sample_check_random(manifest_path, base_dir, k=64, seed=None, show_n=10):
    if seed is None:
        if SAMPLE_SEED_ENV:
            try: seed = int(SAMPLE_SEED_ENV)
            except Exception: seed = None
    if seed is None:
        seed = (int(time.time()) & 0xFFFFFFFF)
    rng = random.Random(seed)

    reservoir = []
    total = 0
    bad_json = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            try:
                ex = json.loads(line)
            except Exception:
                bad_json += 1
                continue
            if len(reservoir) < k:
                reservoir.append((total, ex))
            else:
                j = rng.randint(1, total)
                if j <= k:
                    reservoir[j - 1] = (total, ex)

    misses, abs_paths, missing_field = [], [], []
    for idx, ex in reservoir:
        rel = ex.get("audio_filepath")
        if not isinstance(rel, str) or not rel:
            missing_field.append((idx, "audio_filepath missing/invalid"))
            continue
        if rel.startswith("/"):
            abs_paths.append((idx, rel))
            continue
        fp = os.path.join(base_dir, rel) if not rel.startswith("s3://") else rel
        if not rel.startswith("s3://") and not os.path.exists(fp):
            misses.append((idx, fp))

    print(f"[RANDSAMPLE] manifest={manifest_path} seed={seed} picked={len(reservoir)} "
          f"/ seen={total} bad_json={bad_json}")

    if missing_field:
        print(f"[RANDSAMPLE] entries missing field: {len(missing_field)} (show {min(show_n,len(missing_field))})")
        for t in missing_field[:show_n]:
            print("  - line#", t[0], t[1])
        raise SystemExit("manifest sample has entries missing audio_filepath")

    if abs_paths:
        print(f"[RANDSAMPLE] absolute-path entries: {len(abs_paths)} (show {min(show_n,len(abs_paths))})")
        for t in abs_paths[:show_n]:
            print("  - line#", t[0], t[1])
        # S3 直读允许 s3://，这里只是防止本地绝对路径
        # raise SystemExit("absolute paths found in manifest sample")

    if misses:
        print(f"[RANDSAMPLE] missing local files (ignoring s3://): {len(misses)} (show {min(show_n,len(misses))})")
        for t in misses[:show_n]:
            print("  - line#", t[0], t[1])
        # 若 audio_filepath 是 s3://，不会进入 misses；本地不存在只是提示

    print("[OK] random sample existence check passed (local paths)")

sample_check_random(os.path.join(CH_TRAIN, TRAIN_MANIFEST), CH_TRAIN, k=SAMPLE_K)
sample_check_random(os.path.join(CH_VAL,   VAL_MANIFEST),   CH_VAL,   k=SAMPLE_K)

# 4) TB & model 可写（预写入一个事件文件，避免 TB 空白）
py = (
    f"from torch.utils.tensorboard import SummaryWriter; "
    f"open('{MODEL_DIR}/_write_ok.txt','w').write('ok'); "
    f"w=SummaryWriter('{TB_DIR}'); "
    f"w.add_text('hello', 'tensorboard is ready'); "
    f"w.add_scalar('sanity/ok',1.0,0); "
    f"w.flush(); w.close(); "
    f"print('[OK] TB & model dir writable (wrote hello + sanity/ok)')"
)
subprocess.check_call([sys.executable, "-c", py])

# 4.5) 计算带时间戳的 checkpoint 目录（如 ts-20251012-153045）
TS = time.strftime("%Y%m%d-%H%M%S", time.localtime())
CKPT_BASE = "/opt/ml/checkpoints"
CKPT_DIR_TS = os.path.join(CKPT_BASE, f"ts-{TS}")
os.makedirs(CKPT_DIR_TS, exist_ok=True)

# 尝试拿到 JobName（便于打印 S3 最终去向）
job_name = ""
try:
    sm_env = os.environ.get("SM_TRAINING_ENV")
    if sm_env:
        job_name = json.loads(sm_env).get("job_name", "")
except Exception:
    pass

print(f"[CKPT] local dir : {CKPT_DIR_TS}")
if S3_CKPT:
    if job_name:
        print(f"[CKPT] S3 dir    : {S3_CKPT.rstrip('/')}/{job_name}/ts-{TS}/")
    else:
        print(f"[CKPT] S3 parent : {S3_CKPT} (actual under <JobName>/ts-{TS}/)")

# 5) 批量自动/手工设定 + 汇总打印
def detect_gpu_name():
    try:
        py = "import torch; print(torch.cuda.get_device_name(0))"
        out = subprocess.check_output([sys.executable, "-c", py], text=True).strip()
        return out
    except Exception:
        return ""

gpu_name = detect_gpu_name()

def default_batch_per_gpu_by_gpu(gpu_name: str) -> int:
    g = gpu_name.lower()
    if "h100" in g:                  return 112
    if "a100" in g and "80" in g:    return 96
    if "a100" in g:                  return 64
    if "a10"  in g:                  return 32
    return -1  # -1 表示不覆写，用 YAML

train_bs_override = None
val_bs_override   = None
if TRAIN_BATCH.strip():
    train_bs_override = int(TRAIN_BATCH)
elif AUTO_BATCH_PER_GPU:
    guess = default_batch_per_gpu_by_gpu(gpu_name)
    if guess > 0:
        train_bs_override = guess

if VAL_BATCH.strip():
    val_bs_override = int(VAL_BATCH)
elif train_bs_override is not None:
    val_bs_override = max(8, min(train_bs_override, 64))

try:
    global_bs_val = (train_bs_override if train_bs_override is not None else None)
    if isinstance(global_bs_val, int):
        global_bs_val = global_bs_val * DEVICES * GRAD_ACCUM
except Exception:
    global_bs_val = None

print("== BATCH CONFIG SUMMARY ==")
print(f"[GPU]        {gpu_name}")
print(f"[devices]    {DEVICES}")
print(f"[train_bs]   {train_bs_override if train_bs_override is not None else '(use YAML)'}  (per GPU)")
print(f"[val_bs]     {val_bs_override   if val_bs_override   is not None else '(use YAML)'}  (per GPU)")
print(f"[grad_accum] {GRAD_ACCUM}")
print(f"[global_bs]  {global_bs_val if global_bs_val is not None else '(YAML or unknown)'}\n")

# === AUGMENTOR 覆盖组装 ===
def build_augmentor_overrides(for_val: bool):
    ds_key = "model.validation_ds" if for_val else "model.train_ds"
    overrides = []
    # noise
    if AUG_NOISE_PROB and AUG_NOISE_PROB > 0:
        manifest = resolve_path(CH_TRAIN, AUG_NOISE_MANIFEST)
        if manifest and os.path.isfile(manifest):
            overrides += [
                f"++{ds_key}.augmentor.noise.prob={AUG_NOISE_PROB}",
                f"++{ds_key}.augmentor.noise.manifest_path={manifest}",
                f"++{ds_key}.augmentor.noise.min_snr_db={AUG_NOISE_MIN_SNR}",
                f"++{ds_key}.augmentor.noise.max_snr_db={AUG_NOISE_MAX_SNR}",
            ]
        else:
            print(f"[WARN] noise.prob={AUG_NOISE_PROB} 但未找到 manifest 文件：{manifest!r}，将跳过 noise 覆盖。")
    # speed
    if AUG_SPEED_PROB and AUG_SPEED_PROB > 0:
        overrides += [
            f"++{ds_key}.augmentor.speed.prob={AUG_SPEED_PROB}",
            f"++{ds_key}.augmentor.speed.sr={AUG_SPEED_SR}",
            f"++{ds_key}.augmentor.speed.resample_type={AUG_SPEED_RESAMPLE}",
            f"++{ds_key}.augmentor.speed.min_speed_rate={AUG_SPEED_MIN_RATE}",
            f"++{ds_key}.augmentor.speed.max_speed_rate={AUG_SPEED_MAX_RATE}",
        ]
    # gain
    if AUG_GAIN_PROB and AUG_GAIN_PROB > 0:
        overrides += [
            f"++{ds_key}.augmentor.gain.prob={AUG_GAIN_PROB}",
            f"++{ds_key}.augmentor.gain.min_gain_dbfs={AUG_GAIN_MIN_DB}",
            f"++{ds_key}.augmentor.gain.max_gain_dbfs={AUG_GAIN_MAX_DB}",
        ]
    return overrides

aug_train_overrides = build_augmentor_overrides(for_val=False)
aug_val_overrides   = build_augmentor_overrides(for_val=True) if AUG_ON_VAL else []

print("== AUGMENTOR SUMMARY ==")
def _yn(x): return "ON" if x else "OFF"
print(f"[train] noise={_yn(any('augmentor.noise' in s for s in aug_train_overrides))} "
      f"speed={_yn(any('augmentor.speed' in s for s in aug_train_overrides))} "
      f"gain={_yn(any('augmentor.gain'  in s for s in aug_train_overrides))}")
print(f"[valid] enabled={bool(AUG_ON_VAL)} | "
      f"noise={_yn(any('augmentor.noise' in s for s in aug_val_overrides))} "
      f"speed={_yn(any('augmentor.speed' in s for s in aug_val_overrides))} "
      f"gain={_yn(any('augmentor.gain'  in s for s in aug_val_overrides))}\n")

print("== PRE-FLIGHT PASSED ==\n")

# 6) 组装 Hydra 覆写并启动 NeMo 训练
nemo_script = "/opt/nemo/examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py"
args = [
    sys.executable, nemo_script,
    "--config-path=/opt/nemo/examples/asr/conf/fastconformer/hybrid_transducer_ctc",
    f"--config-name={CONFIG_NAME}",

    # 数据路径
    f"model.train_ds.manifest_filepath={os.path.join(CH_TRAIN, TRAIN_MANIFEST)}",
    f"model.validation_ds.manifest_filepath={os.path.join(CH_VAL,   VAL_MANIFEST)}",
    f"model.tokenizer.dir={CH_TOKENIZER}",

    # change lr
    "model.optim.lr=1.0e-3",

    # 预训练 .nemo
    f"init_from_nemo_model.model0.path={nemo_path}",
    "init_from_nemo_model.model0.exclude=[decoder,joint,aux_ctc]",

    # Trainer
    "trainer.num_nodes=1",
    "trainer.accelerator=gpu",
    "trainer.strategy._target_=lightning.pytorch.strategies.DDPStrategy",
    "trainer.strategy.gradient_as_bucket_view=true",
    f"trainer.devices={DEVICES}",
    f"trainer.precision={PRECISION}",
    f"trainer.val_check_interval={VAL_CHECK_INTERVAL}",
    f"trainer.max_epochs={MAX_EPOCHS}",
    f"trainer.accumulate_grad_batches={GRAD_ACCUM}",

    # 强制打开 logger & 提高写频率，避免 TB 空白
    "trainer.log_every_n_steps=50",

    # ema 参数
    "++exp_manager.ema.enable=True",
    "++exp_manager.ema.decay=0.999",

    # ===== 新增：禁用 TDT greedy 解码的 CUDA graphs 和 loop_labels =====
    "++model.decoding.greedy.loop_labels=false",
    "++model.decoding.greedy.use_cuda_graph_decoder=false",
]

# 若设置 max_steps，则覆盖 max_epochs
if MAX_STEPS_ENV and MAX_STEPS_ENV.strip():
    args += [f"trainer.max_steps={MAX_STEPS_ENV.strip()}", "trainer.max_epochs=null"]

# 批量覆写
if train_bs_override is not None:
    args += [f"model.train_ds.batch_size={train_bs_override}"]
if val_bs_override is not None:
    args += [f"model.validation_ds.batch_size={val_bs_override}"]

# DataLoader worker
args += [
    f"model.train_ds.num_workers={NUM_WORKERS}",
    f"model.validation_ds.num_workers={NUM_WORKERS}",
]

# 时长裁剪（仅当设置了值时才覆写）
if MAX_DURATION_TRAIN.strip():
    args += [f"model.train_ds.max_duration={MAX_DURATION_TRAIN.strip()}"]
if MIN_DURATION_TRAIN.strip():
    args += [f"model.train_ds.min_duration={MIN_DURATION_TRAIN.strip()}"]
if MAX_DURATION_VAL.strip():
    args += [f"model.validation_ds.max_duration={MAX_DURATION_VAL.strip()}"]
if MIN_DURATION_VAL.strip():
    args += [f"++model.validation_ds.min_duration={MIN_DURATION_VAL.strip()}"]

# Checkpoint：写到 /opt/ml/checkpoints/ts-<TS>（用于实时 S3 同步）
args += [f"+exp_manager.checkpoint_callback_params.dirpath={CKPT_DIR_TS}"]
if CKPT_EVERY_STEPS > 0:
    args += [f"+exp_manager.checkpoint_callback_params.every_n_train_steps={CKPT_EVERY_STEPS}"]
elif CKPT_EVERY_EPOCHS > 0:
    args += [f"+exp_manager.checkpoint_callback_params.every_n_epochs={CKPT_EVERY_EPOCHS}"]
if CKPT_EVERY_EPOCHS == 0:
    args += [f"++exp_manager.checkpoint_callback_params.every_n_epochs=0"]

# 始终保存 last
args += ["+exp_manager.checkpoint_callback_params.save_last=true"]

# TensorBoard
args += [
    "exp_manager.create_tensorboard_logger=true",
    f"exp_manager.exp_dir={TB_DIR}",
    "exp_manager.name=hausa-en_fastconformer",
]

# 续训策略
if RESUME:
    args += [
        "exp_manager.resume_if_exists=true",
        "exp_manager.resume_ignore_no_checkpoint=true",
    ]
else:
    args += ["exp_manager.resume_if_exists=false"]

# === 将 AUGMENTOR 覆盖注入到 args ===
if aug_train_overrides:
    args += aug_train_overrides
if aug_val_overrides:
    args += aug_val_overrides

print("Launch:", " ".join(args))
subprocess.check_call(args)
