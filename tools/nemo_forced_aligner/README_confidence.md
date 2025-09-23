# NFA 对齐置信度（CTM 概率）说明

本文档说明 NeMo Forced Aligner（NFA）如何为 token / word / segment 计算置信度，并将其写入 CTM 的 `CONF` 列。

适用代码版本（相对路径）：
- 概率聚合：`nemo/collections/asr/parts/utils/aligner_utils.py`
- CTM 写出：`tools/nemo_forced_aligner/utils/make_ctm_files.py`
- 入口：`tools/nemo_forced_aligner/align.py`

## 背景与思路

对齐流程基于 CTC 模型输出的帧级对数后验 `log_probs[t, v]` 与 Viterbi 最优路径：

1) 运行 ASR 模型得到帧级对数概率（含 blank），形状约 `T × V`。
2) 用参考文本构建 CTC 标签序列（含 blank）。
3) Viterbi 解码得到逐帧对齐路径 `alignment_utt`，即每一帧对应的标签位置索引（包括 blank 位置）。
4) 利用路径中“某帧归属到的标签位置”，提取该帧该标签的对数后验并转为普通概率，最后对 token / word / segment 分别做聚合平均，得到置信度。

该方法参考了 torchaudio 强制对齐教程（CTC segmentation）中的思路：对路径上每一帧取相应标签的后验，做时间维度平均/加权平均来得到段级分数。

## 计算公式与实现要点

记：
- `alignment_utt[t]` 为第 `t` 帧对应的标签位置索引（含 blank）。
- `token_ids_with_blanks[pos]` 为该位置的 token id（含 blank）。
- `log_probs[t, token_id]` 是该帧该 token 的对数后验。
- `p = exp(log_prob)` 表示将对数概率转回普通概率。

1) Token 概率（逐帧平均）

对一个 token 位置 `pos`，统计所有满足 `alignment_utt[t] == pos` 的帧集合 `F_pos`：

```
P_token(pos) = mean_{t ∈ F_pos} exp(log_probs[t, token_ids_with_blanks[pos]])
```

实现：在 `add_probabilities_to_utt_obj()` 中累计每个位置的 `prob_sums[pos]` 与 `frame_counts[pos]`，最后 `prob_sums[pos]/frame_counts[pos]` 赋值到 `Token.probability`。

2) Word 概率（按帧数加权的 token 聚合，忽略 blank）

将一个词内的非 blank token 的“概率总和与帧数”做加总，再整体相除：

```
P_word = (Σ_token Σ_{t ∈ F_token} exp(log_prob[t, token_id])) / (Σ_token |F_token|)
```

3) Segment 概率（同理，聚合其内部的 word / 非 blank token）

```
P_segment = (Σ_word Σ_token Σ_{t ∈ F_token} exp(log_prob[t, token_id])) / (Σ_word Σ_token |F_token|)
```

4) 边界与缺省值

- 若某个 token/word/segment 没有任何帧对齐（计数为 0），其 `probability` 置为 `None`。
- 写 CTM 时，若存在 `probability`，则作为 CTM 的 `CONF` 列输出，并在 `[0,1]` 之间钳制；否则按 NIST CTM 规范使用 `NA`。

## 简化示意图

下面用 ASCII 图直观说明“帧 → 路径位置 → token/word/segment”的归并关系：

```
时间帧:   t=0   t=1   t=2   t=3   t=4   t=5   t=6
路径位:    0     0     1     2     2     3     3        (alignment_utt)
标签序列: [<b>] [A] [<b>] [B] [<b>] [<space>] [<b>] ... (token_ids_with_blanks 对应)

提取每帧对应标签的后验：p_t = exp(log_probs[t, token_id_at(alignment_utt[t])])

Token A 的帧集合 F_A = { t | alignment_utt[t] 指向 A 的位置 }
P_token(A) = mean_{t ∈ F_A} p_t

Word "AB" 的概率：聚合 A、B 的所有帧（忽略 blank），按帧数加权平均。
Segment 的概率：聚合其内部所有词（或非 blank token）的所有帧，按帧数加权平均。
```

## 输出示例（CTM）

CTM 每行字段：`SOURCE CHANNEL BEG-TIME DURATION TOKEN CONF TYPE SPEAKER`

```
utt_001 1 0.52 0.10 A 0.83 lex NA
utt_001 1 0.62 0.17 B 0.79 lex NA
utt_001 1 0.79 0.08 <space> 0.91 lex NA
...
```

说明：
- `TOKEN` 中的空格会被写成 `<space>`，以避免 CTM 解析时出现额外空白列。
- 若某项概率为 `None`，`CONF` 将写为 `NA`。
- 若配置 `minimum_timestamp_duration > 0`，时间戳可能被外扩；概率仅由对齐的帧决定，不受该外扩影响。

## 相关代码位置

- 概率聚合实现：
  - `nemo/collections/asr/parts/utils/aligner_utils.py`
    - `add_probabilities_to_utt_obj(...)`：逐帧收集并聚合到 `Token/Word/Segment.probability`
    - 数据类：`Token/Word/Segment` 新增 `probability: Optional[float]`
- 写 CTM：
  - `tools/nemo_forced_aligner/utils/make_ctm_files.py`
    - 读取 `boundary_info_.probability`，作为 `get_ctm_line(..., conf=...)` 的置信度
- 调用链：
  - `tools/nemo_forced_aligner/align.py`
    - Viterbi 后：先 `add_t_start_end_to_utt_obj(...)`，再 `add_probabilities_to_utt_obj(...)`，最后写 CTM

## 使用提示

- 若 `cfg.ctm_file_config.remove_blank_tokens=True`，token 级 CTM 会过滤 `<b>`（blank）；word/segment 不受影响。
- 概率属于“后验”含义，主要用于排序、筛选或可视化，不应直接视为校准后的置信区间。

## 参考

- Torchaudio: Forced Alignment with Wav2Vec2（CTC Segmentation）
  - https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html

