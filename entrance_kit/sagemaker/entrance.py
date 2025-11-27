import argparse
import io
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError
from sagemaker import get_execution_role
from sagemaker.debugger import TensorBoardOutputConfig
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch
from sagemaker.session import Session

# entrance.py

# ===== 基本配置 =====
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
image_uri = "457411337639.dkr.ecr.us-east-1.amazonaws.com/sagemaker-studio-mayufeng:nemo-b3-20251120"
region = re.search(r"ecr\.([a-z0-9-]+)\.amazonaws\.com", image_uri).group(1)
boto_sess = boto3.Session(region_name=region)
sess      = Session(boto_session=boto_sess)
role      = get_execution_role()

DEFAULT_BUCKET = "asr-sagemakers"
DEFAULT_BASE   = "users/yufeng.ma"


def ensure_trailing_slash(u: str) -> str:
    if not u:
        return u
    return u if u.endswith("/") else u + "/"


@dataclass
class LanguageProfile:
    """Language-dependent S3 materials and naming."""

    name: str
    train_prefix: str
    val_prefix: str
    tokenizer_prefix: str
    pretrained_prefix: str
    train_manifest: str
    val_manifest: str
    pretrained_filename: str
    output_prefix: str
    tensorboard_prefix: str
    checkpoint_prefix: str
    config_name: str = "fastconformer_hybrid_tdt_ctc_bpe_110m"
    base_job_name: Optional[str] = None
    aug_noise_manifest: str = "nemo_noise_manifest_final.jsonl"

    def _apply_template(self, v: str) -> str:
        """Allow {name}/{language} placeholders in prefixes/file names."""
        if not isinstance(v, str):
            return v
        try:
            return v.format(name=self.name, language=self.name)
        except Exception:
            return v

    def __post_init__(self):
        self.train_prefix = ensure_trailing_slash(self._apply_template(self.train_prefix))
        self.val_prefix = ensure_trailing_slash(self._apply_template(self.val_prefix))
        self.tokenizer_prefix = ensure_trailing_slash(self._apply_template(self.tokenizer_prefix))
        self.pretrained_prefix = ensure_trailing_slash(self._apply_template(self.pretrained_prefix))
        self.output_prefix = ensure_trailing_slash(self._apply_template(self.output_prefix))
        self.tensorboard_prefix = ensure_trailing_slash(self._apply_template(self.tensorboard_prefix))
        self.checkpoint_prefix = ensure_trailing_slash(self._apply_template(self.checkpoint_prefix))
        self.train_manifest = self._apply_template(self.train_manifest)
        self.val_manifest = self._apply_template(self.val_manifest)
        self.pretrained_filename = self._apply_template(self.pretrained_filename)
        self.config_name = self._apply_template(self.config_name)
        self.aug_noise_manifest = self._apply_template(self.aug_noise_manifest)
        if not self.base_job_name:
            self.base_job_name = f"nemo-{self.name}-fastconformer"
        self.base_job_name = self._apply_template(self.base_job_name)


LANGUAGE_PRESETS: Dict[str, LanguageProfile] = {
    "swahili": LanguageProfile(
        name="swahili",
        train_prefix=f"s3://{DEFAULT_BUCKET}/{DEFAULT_BASE}/nemo_train/",
        val_prefix=f"s3://{DEFAULT_BUCKET}/{DEFAULT_BASE}/nemo_val/",
        tokenizer_prefix=(
            f"s3://{DEFAULT_BUCKET}/{DEFAULT_BASE}/nemo_tokenizer/"
            "english_swahili_bilingual/tokenizer_spe_bpe_v2048/"
        ),
        pretrained_prefix=f"s3://{DEFAULT_BUCKET}/{DEFAULT_BASE}/nemo_pretrained/parakeet110m_fromnemo/models/",
        train_manifest="swahili_v5.manifest",
        val_manifest="return_data_manifest_final.jsonl",
        pretrained_filename="parakeet-tdt_ctc-110m.nemo",
        output_prefix=f"s3://{DEFAULT_BUCKET}/{DEFAULT_BASE}/nemo_exps/{{name}}/outputs/",
        tensorboard_prefix=f"s3://{DEFAULT_BUCKET}/{DEFAULT_BASE}/nemo_exps/{{name}}/tensorboard/",
        checkpoint_prefix=f"s3://{DEFAULT_BUCKET}/{DEFAULT_BASE}/nemo_exps/{{name}}/checkpoints/",
        base_job_name="nemo-{name}-fastconformer",
        config_name="fastconformer_hybrid_tdt_ctc_bpe_110m",
        aug_noise_manifest="nemo_noise_manifest_final.jsonl",
    ),
}


def load_language_profiles(extra_config_path: Optional[str]) -> Dict[str, LanguageProfile]:
    """Load language profiles; external JSON can override/add presets."""
    profiles = dict(LANGUAGE_PRESETS)
    if not extra_config_path:
        return profiles

    if not os.path.isfile(extra_config_path):
        raise SystemExit(f"lang-config file not found: {extra_config_path}")

    with open(extra_config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "train_prefix" in payload:
        lp = LanguageProfile(**payload)
        profiles[lp.name] = lp
        return profiles

    if not isinstance(payload, dict):
        raise SystemExit("lang-config expects a dict keyed by language or a single profile object")

    for name, cfg in payload.items():
        cfg = dict(cfg)
        cfg.setdefault("name", name)
        profiles[name] = LanguageProfile(**cfg)
    return profiles


def parse_args():
    parser = argparse.ArgumentParser(description="Submit NeMo SageMaker training with language materials.")
    parser.add_argument(
        "--language",
        default=os.environ.get("LANGUAGE_PROFILE", "swahili"),
        help="Language key/preset name. Defaults to env LANGUAGE_PROFILE or 'swahili'.",
    )
    parser.add_argument(
        "--lang-config",
        default=os.environ.get("LANG_CONFIG_PATH", ""),
        help="Optional JSON file with language profiles (dict keyed by language or single object).",
    )
    return parser.parse_args()


args = parse_args()
language_profiles = load_language_profiles(args.lang_config or None)
if args.language not in language_profiles:
    raise SystemExit(f"language preset '{args.language}' not found; available: {list(language_profiles)}")
lang = language_profiles[args.language]

# ===== 配置文件选择 =====
CONFIG_NAME = lang.config_name

# ===== S3 通道（前缀以 / 结尾）=====
s3_train = lang.train_prefix
s3_val   = lang.val_prefix
s3_tok   = lang.tokenizer_prefix
s3_pre   = lang.pretrained_prefix

# 输出与监控
s3_out   = lang.output_prefix
s3_tb    = lang.tensorboard_prefix
s3_ckpt  = lang.checkpoint_prefix

# Manifest / 预训练文件
TRAIN_MANIFEST = lang.train_manifest
VAL_MANIFEST   = lang.val_manifest
PRETRAINED_FILENAME = lang.pretrained_filename

# ===== Tarred dataset 配置（可选）=====
TRAIN_IS_TARRED   = 0
TRAIN_TAR_PATTERN = ""   # 例如 "shard_{000000..000127}.tar"（相对 train 通道）
VAL_IS_TARRED     = 0
VAL_TAR_PATTERN   = ""

# ===== S3 直读模式（manifest.audio_filepath 写 s3://...）=====
TRAIN_AUDIO_S3 = 0   # 1 = train 不挂 channel，只走 S3 直读
VAL_AUDIO_S3   = 0   # 暂时先不特殊处理 val，用不到也没关系

# 训练数据载入模式说明（3 种）
# 1) 常规模式（默认）：TRAIN_AUDIO_S3=0 且 TRAIN_IS_TARRED=0
#    - train/val 都挂 channel；manifest.audio_filepath 为相对路径，指向 channel 里的文件。
# 2) Tarred 模式：TRAIN_IS_TARRED=1（可选 VAL_IS_TARRED=1）
#    - 仍挂 channel；manifest 内为元数据，真实音频在 tar shard 中，tar pattern 通过 *_TAR_PATTERN 指定。
# 3) S3 直读模式（训练集）：TRAIN_AUDIO_S3=1
#    - train 不挂 channel，只下载 manifest；manifest.audio_filepath 写绝对 s3://...，音频直接由 dataloader 读 S3。
# 注意：TRAIN_IS_TARRED 与 TRAIN_AUDIO_S3 不能同时开启；val 侧 S3 直读暂未开启（VAL_AUDIO_S3 保留位）。

# ===== 训练控制 =====
INSTANCE_TYPE = "ml.g6e.48xlarge"
DEVICES = 8
PRECISION = "bf16"
VAL_CHECK_INTERVAL = 2000
RESUME = 0
MAX_EPOCHS = 20
MAX_STEPS  = ""
BASE_JOB_NAME = lang.base_job_name
LANGUAGE_TAG = lang.name
LR = "1.0e-3"
EMA_ENABLE = "1"
EMA_DECAY = "0.999"

# 批量 & Loader
TRAIN_BATCH = "32"
VAL_BATCH   = "32"
GRAD_ACCUM  = "4"
NUM_WORKERS = "8"
AUTO_BATCH_PER_GPU = "1"

# 时长裁剪
MAX_DURATION_TRAIN = "32"
MIN_DURATION_TRAIN = "1.0"
MAX_DURATION_VAL   = ""
MIN_DURATION_VAL   = "1.0"

# manifest 抽样检查
SAMPLE_K = 64
SAMPLE_SEED = ""

# Checkpoint 频率
CKPT_EVERY_STEPS  = 4000
CKPT_EPOCHS       = 0

# ======== Augmentor 默认 ========
AUG_ON_VAL = "0"

# noise
AUG_NOISE_PROB      = "0.1"
AUG_NOISE_MANIFEST  = lang.aug_noise_manifest
AUG_NOISE_MIN_SNR_DB= "0.0"
AUG_NOISE_MAX_SNR_DB= "15.0"

# speed
AUG_SPEED_PROB      = "0.2"
AUG_SPEED_SR        = "16000"
AUG_SPEED_RESAMPLE  = "kaiser_fast"
AUG_SPEED_MIN_RATE  = "0.95"
AUG_SPEED_MAX_RATE  = "1.05"

# gain
AUG_GAIN_PROB       = "0.2"
AUG_GAIN_MIN_DB     = "-10.0"
AUG_GAIN_MAX_DB     = "10.0"

# ===== S3 工具 =====
s3 = boto_sess.client("s3")

def parse_s3_uri(u):
    p = urlparse(u); assert p.scheme=="s3"
    bucket = p.netloc; key = p.path.lstrip("/")
    if u.endswith("/") and key and not key.endswith("/"): key += "/"
    return bucket, key

def s3_prefix_preview(uri, max_keys=5):
    try:
        b, k = parse_s3_uri(uri)
        resp = s3.list_objects_v2(Bucket=b, Prefix=k, MaxKeys=max_keys)
        cnt = resp.get("KeyCount", 0)
        print(f"[S3] prefix exists? {'YES' if cnt>0 else 'NO '}  count~{cnt}  uri={uri}")
        for obj in resp.get("Contents", [])[:max_keys]:
            print("      -", obj["Key"])
    except Exception as e:
        print(f"[S3] preview error for {uri}: {e}")

def s3_head(uri):
    try:
        b, k = parse_s3_uri(uri)
        if uri.endswith("/"):
            print(f"[HEAD] {uri} looks like a prefix, skip")
            return
        resp = s3.head_object(Bucket=b, Key=k)
        print(f"[HEAD] 200 OK size={resp.get('ContentLength')}  uri={uri}")
    except ClientError as e:
        print(f"[HEAD] {uri} -> {e.response.get('Error',{}).get('Code')}: {e}")
    except Exception as e:
        print(f"[HEAD] {uri} -> {e}")

def s3_print_manifest_head(uri, n=3):
    try:
        b, k = parse_s3_uri(uri)
        body = s3.get_object(Bucket=b, Key=k)["Body"].read().decode("utf-8", errors="ignore").splitlines()
        print(f"[MANIFEST] head {n} lines of {uri}:")
        for i, ln in enumerate(body[:n]):
            try:
                ex = json.loads(ln)
                print(f"  {i+1:02d} audio={ex.get('audio_filepath')} text={ex.get('text','')[:60]}")
            except Exception:
                print(f"  {i+1:02d} {ln[:120]}")
    except Exception as e:
        print(f"[MANIFEST] {uri} -> {e}")


def s3_sample_manifest_s3paths(manifest_uri: str, max_check: int = 8, show_n: int = 5, seed_env: str = ""):
    """For S3 direct-read mode: ensure audio_filepath startswith s3:// and sampled keys exist (streaming)."""
    b, k = parse_s3_uri(manifest_uri)
    resp = s3.get_object(Bucket=b, Key=k)
    stream = resp["Body"].iter_lines()
    reservoir = []
    total = 0
    seed = None
    if seed_env:
        try:
            seed = int(seed_env)
        except Exception:
            seed = None
    if seed is None:
        seed = (int(time.time()) & 0xFFFFFFFF)
    rng = random.Random(seed)

    for line_no, raw in enumerate(stream, start=1):
        if not raw:
            continue
        try:
            ln = raw.decode("utf-8", errors="ignore")
        except Exception:
            ln = str(raw)
        if not ln.strip():
            continue
        total += 1
        try:
            ex = json.loads(ln)
        except Exception:
            raise SystemExit(f"[S3-PATH] bad json at line {line_no} in {manifest_uri}")
        p = ex.get("audio_filepath")
        if not isinstance(p, str) or not p.startswith("s3://"):
            raise SystemExit(f"[S3-PATH] line {line_no} audio_filepath 非 s3:// 开头: {p}")

        if len(reservoir) < max_check:
            reservoir.append((line_no, p))
        else:
            j = rng.randint(1, total)
            if j <= max_check:
                reservoir[j - 1] = (line_no, p)

    if total == 0:
        raise SystemExit(f"[S3-PATH] manifest empty: {manifest_uri}")

    head_n = min(show_n, len(reservoir))
    print(f"[S3-PATH] sampled {len(reservoir)} entries (seed={seed}) from {total}; head-check {head_n}")
    for lineno, uri in reservoir[:head_n]:
        try:
            bb, kk = parse_s3_uri(uri)
            s3.head_object(Bucket=bb, Key=kk)
            print(f"  [OK] line {lineno} {uri}")
        except Exception as e:
            raise SystemExit(f"[S3-PATH] head_object fail at line {lineno} uri={uri} err={e}")


def s3_join(prefix, rel):
    if not prefix.endswith("/"): prefix += "/"
    return f"{prefix}{rel}"

# ===== 提交前可见日志 =====
print("== PRE-SUBMIT SUMMARY ==")
print(f"[region] {region}")
print(f"[role]   {role}")
print(f"[image]  {image_uri}")
print(f"[lang]   {LANGUAGE_TAG}")
print(f"[config] {CONFIG_NAME}")
print(f"[job]    {BASE_JOB_NAME}")
print(f"[inst]   {INSTANCE_TYPE}")
print(f"[output] {s3_out}")
print(f"[tb]     {s3_tb}")
print(f"[ckpt]   {s3_ckpt}")

print("\n-- S3 channels preview --")
for uri in [s3_train, s3_val, s3_tok, s3_pre]:
    s3_prefix_preview(uri)
s3_head(s3_join(s3_pre, PRETRAINED_FILENAME))

print("\n-- Manifest head --")
s3_print_manifest_head(s3_join(s3_train, TRAIN_MANIFEST))
s3_print_manifest_head(s3_join(s3_val,   VAL_MANIFEST))

print("\n-- DATA MODE --")
print(f"[train] is_tarred={bool(TRAIN_IS_TARRED)}  audio_s3={bool(TRAIN_AUDIO_S3)}")
print(f"[val]   is_tarred={bool(VAL_IS_TARRED)}    audio_s3={bool(VAL_AUDIO_S3)}")
if TRAIN_IS_TARRED and TRAIN_AUDIO_S3:
    print("[X] 配置错误：TRAIN_IS_TARRED=1 和 TRAIN_AUDIO_S3=1 不可同时开启")
    raise SystemExit(2)

# 计算 manifest 的 S3 绝对路径（给 S3 模式用）
train_manifest_s3_uri = s3_join(s3_train, TRAIN_MANIFEST)
val_manifest_s3_uri   = s3_join(s3_val,   VAL_MANIFEST)
if TRAIN_AUDIO_S3:
    s3_sample_manifest_s3paths(
        train_manifest_s3_uri,
        max_check=min(8, SAMPLE_K),
        show_n=5,
        seed_env=SAMPLE_SEED,
    )

# ===== 训练输入 =====
# 关键点：当 TRAIN_AUDIO_S3 = 1 时，不创建 "train" channel，
# SageMaker 不会从 s3_train 前缀拷贝任何数据到 /opt/ml/input/data/train。
inputs = {
    "val":        TrainingInput(s3_val,   distribution="FullyReplicated", input_mode="FastFile"),
    "tokenizer":  TrainingInput(s3_tok,   distribution="FullyReplicated"),
    "pretrained": TrainingInput(s3_pre,   distribution="FullyReplicated"),
}
if not TRAIN_AUDIO_S3:
    # 只有在非 S3 直读模式下，才挂 train channel
    inputs["train"] = TrainingInput(s3_train, distribution="FullyReplicated", input_mode="FastFile")

tb_cfg = TensorBoardOutputConfig(
    s3_output_path=s3_tb,
    container_local_output_path="/opt/ml/output/tensorboard"
)

# ===== 构建 PyTorch Estimator =====
est = PyTorch(
    image_uri=image_uri,
    entry_point="train_entry.py",
    source_dir=THIS_DIR,
    role=role,
    instance_count=1,
    instance_type=INSTANCE_TYPE,
    volume_size=300,
    max_run=144*3600,
    output_path=s3_out,

    checkpoint_s3_uri=s3_ckpt,
    checkpoint_local_path="/opt/ml/checkpoints",

    tensorboard_output_config=tb_cfg,
    debugger_hook_config=False,
    disable_profiler=True,
    enable_sagemaker_metrics=False,
    base_job_name=BASE_JOB_NAME,

    environment={
        "HYDRA_FULL_ERROR": "1",
        "LANGUAGE": LANGUAGE_TAG,
        "RUN_NAME": BASE_JOB_NAME,

        # 数据与预训练
        "TRAIN_MANIFEST": TRAIN_MANIFEST,
        "VAL_MANIFEST":   VAL_MANIFEST,
        "PRETRAINED_FILENAME": PRETRAINED_FILENAME,
        "CONFIG_NAME": CONFIG_NAME,

        # Tarred dataset 控制
        "TRAIN_IS_TARRED": str(int(TRAIN_IS_TARRED)),
        "TRAIN_TAR_PATTERN": TRAIN_TAR_PATTERN,
        "VAL_IS_TARRED": str(int(VAL_IS_TARRED)),
        "VAL_TAR_PATTERN": VAL_TAR_PATTERN,

        # S3 直读控制（train 可以没有 channel）
        "TRAIN_AUDIO_S3": str(int(TRAIN_AUDIO_S3)),
        "VAL_AUDIO_S3":   str(int(VAL_AUDIO_S3)),
        "TRAIN_MANIFEST_S3_URI": train_manifest_s3_uri if TRAIN_AUDIO_S3 else "",
        "VAL_MANIFEST_S3_URI":   val_manifest_s3_uri   if VAL_AUDIO_S3   else "",

        # 训练控制
        "DEVICES": str(DEVICES),
        "PRECISION": PRECISION,
        "VAL_CHECK_INTERVAL": str(VAL_CHECK_INTERVAL),
        "RESUME": str(RESUME),
        "MAX_EPOCHS": str(MAX_EPOCHS),
        "MAX_STEPS":  str(MAX_STEPS),
        "LR": LR,
        "EMA_ENABLE": EMA_ENABLE,
        "EMA_DECAY": EMA_DECAY,

        # 批量与 Loader
        "TRAIN_BATCH": TRAIN_BATCH,
        "VAL_BATCH":   VAL_BATCH,
        "GRAD_ACCUM":  GRAD_ACCUM,
        "NUM_WORKERS": NUM_WORKERS,
        "AUTO_BATCH_PER_GPU": AUTO_BATCH_PER_GPU,

        # 时长裁剪
        "MAX_DURATION_TRAIN": MAX_DURATION_TRAIN,
        "MIN_DURATION_TRAIN": MIN_DURATION_TRAIN,
        "MAX_DURATION_VAL":   MAX_DURATION_VAL,
        "MIN_DURATION_VAL":   MIN_DURATION_VAL,

        # 抽样检查
        "SAMPLE_K": str(SAMPLE_K),
        "SAMPLE_SEED": SAMPLE_SEED,

        # Checkpoint 频率
        "CKPT_EVERY_STEPS":  str(CKPT_EVERY_STEPS),
        "CKPT_EVERY_EPOCHS": str(CKPT_EPOCHS),

        # 信息提示
        "S3_OUT":  s3_out,
        "S3_TB":   s3_tb,
        "S3_CKPT": s3_ckpt,
        "REGION":  region,

        # Augmentor
        "AUG_ON_VAL": AUG_ON_VAL,

        "AUG_NOISE_PROB":       AUG_NOISE_PROB,
        "AUG_NOISE_MANIFEST":   AUG_NOISE_MANIFEST,
        "AUG_NOISE_MIN_SNR_DB": AUG_NOISE_MIN_SNR_DB,
        "AUG_NOISE_MAX_SNR_DB": AUG_NOISE_MAX_SNR_DB,

        "AUG_SPEED_PROB":      AUG_SPEED_PROB,
        "AUG_SPEED_SR":        AUG_SPEED_SR,
        "AUG_SPEED_RESAMPLE":  AUG_SPEED_RESAMPLE,
        "AUG_SPEED_MIN_RATE":  AUG_SPEED_MIN_RATE,
        "AUG_SPEED_MAX_RATE":  AUG_SPEED_MAX_RATE,

        "AUG_GAIN_PROB":   AUG_GAIN_PROB,
        "AUG_GAIN_MIN_DB": AUG_GAIN_MIN_DB,
        "AUG_GAIN_MAX_DB": AUG_GAIN_MAX_DB,
    },
    sagemaker_session=sess,
)

print("\n== SUBMIT TRAINING JOB ==")
est.fit(inputs, logs=False, wait=False)
job_name = est.latest_training_job.name

# 常用路径
s3_model_tar = f"{s3_out.rstrip('/')}/{job_name}/output/model.tar.gz"
s3_ckpt_job  = f"{s3_ckpt.rstrip('/')}/{job_name}/"
s3_tb_job    = f"{s3_tb.rstrip('/')}/{job_name}/"

sm_console = f"https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{job_name}"
cw_logs    = (f"https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}"
              f"#logsV2:log-groups/log-group/%252Faws%252Fsagemaker%252FTrainingJobs/log-events/{job_name}")
tb_console = f"https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/tensorboard"

print("\n================ WHERE TO WATCH ================")
print(f"[JobName]           {job_name}")
print(f"[Training Console]  {sm_console}")
print(f"[CloudWatch Logs]   {cw_logs}")
print(f"[TensorBoard (S3)]  {s3_tb_job}")
print(f"[TensorBoard (UI)]  {tb_console}")
print(f"[Checkpoints (S3)]  {s3_ckpt_job}")
print(f"[Final Model S3]    {s3_model_tar}")
print("------------------------------------------------")
print("List checkpoints:")
print(f"  aws s3 ls {s3_ckpt_job} --recursive | tail")
print("Download final model when ready:")
print(f"  aws s3 cp {s3_model_tar} . && tar -tzf model.tar.gz | head")
print("================================================\n")

print("== STREAMING CLOUDWATCH LOGS ==")
sess.logs_for_job(job_name, wait=True, log_type="All")

# ===== 结束状态 =====
sm = boto3.client("sagemaker", region_name=region)
desc = sm.describe_training_job(TrainingJobName=job_name)
print("\n== FINAL STATUS ==")
print("status=", desc["TrainingJobStatus"], " secondary=", desc.get("SecondaryStatus"))
if desc.get("FailureReason"): print("FailureReason:", desc["FailureReason"])
art = desc.get("ModelArtifacts", {}).get("S3ModelArtifacts")
if art: print("Artifacts:", art)

print("\n== INPUT DATACONFIG ==")
for ch in desc.get("InputDataConfig", []):
    print(f" - {ch['ChannelName']}: {ch['DataSource']['S3DataSource']['S3Uri']}")
