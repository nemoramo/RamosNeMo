# entrance.py
import re, os, io, json, time, boto3
from urllib.parse import urlparse
from sagemaker.session import Session
from sagemaker.inputs import TrainingInput
from sagemaker.debugger import TensorBoardOutputConfig
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role
from botocore.exceptions import ClientError

# ===== 基本配置 =====
image_uri = "457411337639.dkr.ecr.us-east-1.amazonaws.com/sagemaker-studio-mayufeng:nemo-b3-20251120"
region = re.search(r"ecr\.([a-z0-9-]+)\.amazonaws\.com", image_uri).group(1)
boto_sess = boto3.Session(region_name=region)
sess      = Session(boto_session=boto_sess)
role      = get_execution_role()

BUCKET, BASE = "asr-sagemakers", "users/yufeng.ma"

# ===== 配置文件选择 =====
CONFIG_NAME = "fastconformer_hybrid_tdt_ctc_bpe_110m"

# ===== S3 通道（前缀以 / 结尾）=====
# 注意：在 S3 直读模式下，这里的 s3_train 只需要包含 manifest 即可，
# 真正的音频可以在任意别的 bucket/prefix（manifest 里直接写 s3://...）
s3_train = f"s3://{BUCKET}/{BASE}/nemo_train/"
s3_val   = f"s3://{BUCKET}/{BASE}/nemo_val/"
s3_tok   = f"s3://{BUCKET}/{BASE}/nemo_tokenizer/english_swahili_bilingual/tokenizer_spe_bpe_v2048/"
s3_pre   = f"s3://{BUCKET}/{BASE}/nemo_pretrained/parakeet110m_fromnemo/models/"

# 输出与监控
s3_out   = f"s3://{BUCKET}/{BASE}/nemo_exps/swahili/outputs/"
s3_tb    = f"s3://{BUCKET}/{BASE}/nemo_exps/swahili/tensorboard/"
s3_ckpt  = f"s3://{BUCKET}/{BASE}/nemo_exps/swahili/checkpoints/"

# Manifest / 预训练文件
TRAIN_MANIFEST = "swahili_v5.manifest"
VAL_MANIFEST   = "return_data_manifest_final.jsonl"
PRETRAINED_FILENAME = "parakeet-tdt_ctc-110m.nemo"

# ===== Tarred dataset 配置（可选）=====
TRAIN_IS_TARRED   = 0
TRAIN_TAR_PATTERN = ""   # 例如 "shard_{000000..000127}.tar"（相对 train 通道）
VAL_IS_TARRED     = 0
VAL_TAR_PATTERN   = ""

# ===== S3 直读模式（manifest.audio_filepath 写 s3://...）=====
TRAIN_AUDIO_S3 = 0   # 1 = train 不挂 channel，只走 S3 直读
VAL_AUDIO_S3   = 0   # 暂时先不特殊处理 val，用不到也没关系

# ===== 训练控制 =====
INSTANCE_TYPE = "ml.g6e.48xlarge"
DEVICES = 8
PRECISION = "bf16"
VAL_CHECK_INTERVAL = 2000
RESUME = 0
MAX_EPOCHS = 20
MAX_STEPS  = ""

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
AUG_NOISE_MANIFEST  = "nemo_noise_manifest_final.jsonl"
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

def s3_join(prefix, rel):
    if not prefix.endswith("/"): prefix += "/"
    return f"{prefix}{rel}"

# ===== 提交前可见日志 =====
print("== PRE-SUBMIT SUMMARY ==")
print(f"[region] {region}")
print(f"[role]   {role}")
print(f"[image]  {image_uri}")
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
    source_dir=".",
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
    base_job_name="nemo-swahili-fastconformer",

    environment={
        "HYDRA_FULL_ERROR": "1",

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
