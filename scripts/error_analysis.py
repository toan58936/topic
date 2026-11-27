"""Automated error analysis workflow for Yelp star-prediction outputs.

Commands:
    python -m scripts.error_analysis auto --reports-dir reports --overwrite
    python -m scripts.error_analysis prepare --reports-dir reports --overwrite
    python -m scripts.error_analysis summarize --reports-dir reports
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from app.streamlit_dashboard import load_data

GROUPS = ["Nhãn nhiễu", "Cảm xúc lẫn lộn", "Bị cắt 256 token", "Mỉa mai", "Khác"]
OUTPUT_COLUMNS = ["review_id", "model", "abs_error", "true_stars", "pred", "error_group", "notes"]

POS_WORDS = {
    "amazing", "excellent", "fantastic", "wonderful", "awesome", "delicious",
    "perfect", "best", "love", "loved", "great", "friendly", "fresh", "superb",
    "good", "nice", "helpful", "splendid", "happy", "glad", "phenomenal",
    "fabulous", "tremendous", "recommend", "fun", "enjoy", "enjoyed", "clean",
    "prompt", "courteously", "attentive", "sweetest", "hospitable", "thrilled",
    "unparalleled", "flavorful", "moist", "crisp", "honest", "tasty", "hopeful",
    "smooth", "praise", "compliment", "pleasant", "appreciated", "fast",
}

NEG_WORDS = {
    "terrible", "horrible", "worst", "awful", "disgusting", "rude", "slow",
    "slowly", "cold", "inedible", "overpriced", "waste", "wasted", "disappointed",
    "disappointing", "bad", "badly", "poor", "broken", "stupid", "uncomfortable",
    "unprofessional", "miserable", "subpar", "flavorless", "smokey", "crowded",
    "neglect", "ridiculous", "dumbest", "shoddy", "damage", "filthy", "nasty",
    "wrong", "complaint", "avoid", "ignore", "ignored", "ignoring", "issues",
    "issue", "problem", "problems", "ditzy", "dead", "waited", "waiting",
    "flat", "lopsided", "shaking", "phobia", "painfully", "weird", "upsell",
    "dry", "drawback", "unhappy", "shocked", "sucked", "confused",
    "refused", "slammed", "yelled", "flustered", "stripped", "broke", "offended",
    "condemned", "reeked", "foul", "average", "overwhelmed", "minimal",
}

SEVERE_NEG_WORDS = {
    "terrible", "horrible", "worst", "awful", "disgusting", "dead", "ditzy",
    "wrong", "miserable", "shoddy", "filthy", "nasty", "broken", "stupid",
    "unprofessional", "rude", "inedible", "sucked",
}

PROFANITY_WORDS = {
    "fuck", "fucking", "shit", "shitload", "bitch", "bitches", "hoe", "asshole", "damn", "crap",
}

# Thành ngữ làm đảo ngược hoặc trung hòa nghĩa của từ đơn lẻ
IDIOM_MASKS = [
    r"\bcan'?t\s+wait\b",
    r"\bat\s+best\b",
    r"\bcold\s+ones?\b",
    r"\bnot\s+(?:happy|great|good|nice|friendly|helpful)\b",
    r"\bgood\s+thing\b",
]

NEG_PHRASES = [
    r"\bno responses?\b", r"\bno call back\b", r"\bnever returned\b",
    r"\bnot happy\b", r"\bnot great\b", r"\bno apology\b", r"\bno inside seating\b",
    r"\bover-?priced\b", r"\bso so\b", r"\bfall short\b", r"\bdon'?t care\b",
    r"\bjust wished\b", r"\bonly thing\b", r"\bonly drawback\b", r"\blong wait\b",
    r"\bwrong address\b", r"\btook me forever\b", r"\bwaited on hold\b",
    r"\bstand and wait\b", r"\bwatered down\b", r"\bnot what we expected\b",
    r"\bnot crispy\b", r"\bice cold\b", r"\bbait and switch\b", r"\bturn me off\b",
    r"\bwrong dessert\b", r"\btoo small\b", r"\bnot a cheap\b",
    r"\bnone of the prices\b", r"\bdead to the world\b",
]

CONTRAST_WORDS = {
    "but", "however", "although", "though", "unfortunately", "except", "yet", "nonetheless", "granted",
}

UPDATE_PATTERNS = [
    r"\bupdate\b",
    r"\brevised\s+(?:review|my\s+rating)\b",
    r"\boriginal\s+review\b",
    r"\bedit\s*:",
    r"\bprevious\s+review\b",
]

SARCASM_PATTERNS = [
    r"\byeah right\b",
    r"\bgreat job\b.*(!|\.{3})",
    r"\bthanks for nothing\b",
    r"\bjust great\b",
    r"\bwhat a joke\b",
    r"\bgot exactly (?:our|my) money'?s worth\b",
    r"\bif you (?:like|love) .{3,120}? this is the (?:place|gym|spot) for you\b",
    r"\bif you could eat .{2,40}?(?:rate|give) it 5\b",
    r"\bstep up from a (?:school )?cafeteria\b",
    r"\bwould be a great place for a restaurant\b",
    r"\bsaid everything was perfect\b",
    r"\bperfect visit 1x\b",
    r"\"(?:great|amazing|wonderful|fresh|fast|gym|trainers)\"",
]


def _extract_cues(text_low: str) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    neg_phrase_hits = [p for p in NEG_PHRASES if re.search(p, text_low)]
    cleaned = text_low
    for pat in IDIOM_MASKS:
        cleaned = re.sub(pat, " ", cleaned)
    words = set(re.findall(r"\b[a-z]+\b", cleaned))
    pos_hits = sorted(words & POS_WORDS)
    neg_hits = sorted((words & NEG_WORDS) | (words & PROFANITY_WORDS))
    severe_neg_hits = sorted((words & SEVERE_NEG_WORDS) | (words & PROFANITY_WORDS))
    profanity_hits = sorted(words & PROFANITY_WORDS)
    contrast_hits = sorted(words & CONTRAST_WORDS)
    return pos_hits, neg_hits, severe_neg_hits, profanity_hits, neg_phrase_hits, contrast_hits


def classify_review(row: pd.Series) -> tuple[str, str]:
    """Automatically assign one of the 5 error groups with grounded evidence notes."""
    text_low = str(row["text"]).lower()
    pos_hits, neg_hits, severe_neg_hits, profanity_hits, neg_phrase_hits, contrast_hits = _extract_cues(text_low)

    has_update = any(re.search(pat, text_low) for pat in UPDATE_PATTERNS)
    has_sarcasm = any(re.search(pat, text_low) for pat in SARCASM_PATTERNS)

    true_s = int(row["true_stars"])
    pred_s = float(row["pred"])
    abs_err = float(row["abs_error"])
    n_tok = int(row["n_tokens"])

    neg_count = len(neg_hits) + len(neg_phrase_hits)
    pos_count = len(pos_hits)

    # Kiểm tra đảo chiều cảm xúc giữa phần đầu và phần đuôi văn bản dài (>254 token)
    tokens_rough = text_low.split()
    head_text = " ".join(tokens_rough[:180])
    tail_text = " ".join(tokens_rough[180:])
    h_pos, h_neg, _, _, h_nphr, _ = _extract_cues(head_text)
    t_pos, t_neg, _, _, t_nphr, _ = _extract_cues(tail_text)
    tail_has_update = any(re.search(pat, tail_text) for pat in UPDATE_PATTERNS)
    head_polarity = len(h_pos) - (len(h_neg) + len(h_nphr))
    tail_polarity = len(t_pos) - (len(t_neg) + len(t_nphr))
    polarity_flip = (head_polarity * tail_polarity < 0) or (abs(head_polarity - tail_polarity) >= 3)

    # 1. Bị cắt 256 token (ưu tiên khi có đoạn cập nhật/sửa điểm ở cuối bài)
    if n_tok > 254 and abs_err >= 1.5 and (has_update or tail_has_update):
        return (
            "Bị cắt 256 token",
            f"Văn bản dài {n_tok} token (>254) có đoạn cập nhật/sửa điểm (update/revised) ở cuối bị cắt mất, "
            f"khiến mô hình dự đoán {pred_s:.2f} so với thật {true_s}.",
        )

    # 2. Mỉa mai (châm biếm / giả định ngược ngữ cảnh)
    if has_sarcasm and abs_err >= 1.5:
        return (
            "Mỉa mai",
            f"Chứa cấu trúc châm biếm/giả định ngược ngữ cảnh khiến mô hình chấm lệch "
            f"(dự đoán {pred_s:.2f} vs thật {true_s}).",
        )

    # 3. Bị cắt 256 token (đảo chiều cảm xúc ở văn bản dài >300 token)
    if n_tok > 300 and abs_err >= 1.5 and polarity_flip:
        return (
            "Bị cắt 256 token",
            f"Văn bản dài {n_tok} token (>254) có cảm xúc đảo chiều ở phần cuối bị cắt mất, "
            f"khiến mô hình dự đoán {pred_s:.2f} so với thật {true_s}.",
        )

    # 4. Nhãn nhiễu (nhãn sao đối nghịch trực tiếp với cực cảm xúc một chiều)
    if abs_err >= 2.0:
        strong_neg_evidence = profanity_hits or severe_neg_hits or neg_phrase_hits
        if true_s >= 4 and pred_s <= 2.0 and strong_neg_evidence and pos_count == 0:
            keyword_cues = (profanity_hits or severe_neg_hits)[:3]
            cue_detail = f" ({', '.join(keyword_cues)})" if keyword_cues else ""
            return (
                "Nhãn nhiễu",
                f"Nội dung phản ánh tiêu cực/báo lỗi{cue_detail}, không có lời khen nhưng nhãn ghi {true_s} sao "
                f"(mô hình dự đoán {pred_s:.2f}).",
            )
        if true_s <= 2 and pred_s >= 4.0 and pos_count >= 2 and neg_count == 0:
            return (
                "Nhãn nhiễu",
                f"Nội dung tích cực ({', '.join(pos_hits[:3])}), không có lời chê nhưng nhãn ghi {true_s} sao "
                f"(mô hình dự đoán {pred_s:.2f}).",
            )

    # 5. Cảm xúc lẫn lộn (thực sự có đồng thời tín hiệu khen và chê)
    if pos_count >= 1 and neg_count >= 1 and (contrast_hits or abs_err >= 0.5 or (2 <= true_s <= 4)):
        pos_str = ", ".join(pos_hits[:2])
        neg_str = ", ".join(neg_hits[:2]) if neg_hits else "ý chê/hạn chế"
        return (
            "Cảm xúc lẫn lộn",
            f"Review vừa có yếu tố khen ({pos_str}) vừa có yếu tố chê ({neg_str}), tạo sắc thái hai chiều.",
        )

    # 6. Khác (tách rõ dự đoán sát nhãn trên nhóm random vs ngữ cảnh đặc thù)
    if abs_err < 0.5:
        return (
            "Khác",
            f"Mô hình dự đoán sát nhãn thật (dự đoán {pred_s:.2f} vs thật {true_s}, sai số {abs_err:.2f}); "
            "không phát sinh lỗi đáng kể.",
        )
    return (
        "Khác",
        f"Ngữ cảnh đặc thù/ngắn ({n_tok} token) hoặc đối tượng cảm xúc gián tiếp "
        f"(dự đoán {pred_s:.2f} vs thật {true_s}, sai số {abs_err:.2f}).",
    )


def sample_reviews(summary: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    if not {"n_tokens", "split_id"}.issubset(preds.columns):
        raise ValueError("Error Analysis cần n_tokens và split_id từ bản xuất đầy đủ theo đặc tả.")
    if "text" not in preds or preds.text.isna().any() or preds.text.astype(str).str.strip().eq("").any():
        raise ValueError("Cần test_predictions_with_text.csv có nội dung để phân tích tự động.")

    scores = pd.to_numeric(summary["rmse"], errors="coerce")
    candidates = summary.loc[np.isfinite(scores) & scores.ge(0)].copy()
    if candidates.empty:
        raise ValueError("Không có RMSE Test hợp lệ để chọn mô hình tốt nhất.")

    candidates["rmse"] = scores.loc[candidates.index]
    model = candidates.sort_values(["rmse", "model"], kind="stable").iloc[0]["model"]
    if f"pred_{model}" not in preds or preds[f"pred_{model}"].isna().any():
        raise ValueError(f"Mô hình tốt nhất là {model}, nhưng chưa có đủ dự đoán Test của mô hình này.")

    ordered = preds.sort_values("review_id", kind="stable").copy()
    worst = ordered.nlargest(20, f"abs_err_{model}").copy()
    if len(worst) != 20:
        raise ValueError("Không đủ 20 review để lấy worst-case.")
    worst["sample_type"] = "worst"

    pool = ordered.loc[~ordered.review_id.isin(worst.review_id)]
    rng = np.random.default_rng(42)
    random_parts = []
    for star in range(1, 6):
        group = pool.loc[pool.true_stars == star]
        if len(group) < 4:
            raise ValueError(f"Không đủ 4 review sao {star} ngoài top-20; không thể tạo 40 review duy nhất.")
        random_parts.append(group.iloc[rng.choice(len(group), 4, replace=False)])

    random_df = pd.concat(random_parts).copy()
    random_df["sample_type"] = "random"

    sample = pd.concat([worst, random_df], ignore_index=True)
    sample["model"] = model
    sample["pred"] = sample[f"pred_{model}"]
    sample["abs_error"] = sample[f"abs_err_{model}"]
    sample["truncation_flag"] = sample.n_tokens > 254

    classified = sample.apply(classify_review, axis=1)
    sample["error_group"] = [c[0] for c in classified]
    sample["notes"] = [c[1] for c in classified]

    return sample[OUTPUT_COLUMNS + ["sample_type", "text", "n_tokens", "truncation_flag", "split_id"]]


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8", float_format="%.6f")


def validate_annotations(annotations: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    missing = set(OUTPUT_COLUMNS) - set(annotations.columns)
    if missing:
        raise ValueError(f"Annotation thiếu cột: {sorted(missing)}")
    if len(annotations) != 40 or annotations.review_id.duplicated().any() or set(annotations.review_id) != set(sample.review_id):
        raise ValueError("Annotation phải chứa đúng 40 review duy nhất của mẫu cố định.")

    result = annotations.set_index("review_id").loc[sample.review_id].reset_index()
    if not result.model.eq(sample.model.to_numpy()).all():
        raise ValueError("Model trong annotation không khớp mẫu.")
    for col in ["abs_error", "true_stars", "pred"]:
        if not np.allclose(pd.to_numeric(result[col]), sample[col], atol=2e-6, rtol=0):
            raise ValueError(f"Không được sửa dữ liệu gốc: {col}.")
    if not result.error_group.isin(GROUPS).all():
        raise ValueError("Cần gán đúng một trong 5 nhóm cho đủ 40 review.")
    if result.notes.fillna("").str.strip().eq("").any():
        raise ValueError("Mỗi review cần notes giải thích bằng chứng, không để trống.")
    bad_cut = result.error_group.eq("Bị cắt 256 token") & ~sample.truncation_flag.to_numpy()
    if bad_cut.any():
        raise ValueError("Nhóm Bị cắt 256 token chỉ hợp lệ khi n_tokens > 254.")
    return result[OUTPUT_COLUMNS]


def prepare(reports_dir: str, output_dir: str | None = None, overwrite: bool = False) -> Path:
    summary, preds = load_data(reports_dir)
    sample = sample_reviews(summary, preds)
    output = Path(output_dir or reports_dir)

    names = [
        "error_sample.csv",
        "error_sample_with_text.csv",
        "error_analysis.csv",
        "error_analysis_manifest.json",
    ]
    existing = [name for name in names if (output / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Đã tồn tại: {', '.join(existing)}. Dùng cờ --overwrite nếu muốn tạo lại."
        )

    output.mkdir(parents=True, exist_ok=True)
    write_csv(sample[OUTPUT_COLUMNS[:5]], output / names[0])
    write_csv(sample, output / names[1])
    write_csv(sample[OUTPUT_COLUMNS], output / names[2])

    manifest = {
        "status": "auto_annotated",
        "mode": "automated_rule_based_v2",
        "model": str(sample.model.iloc[0]),
        "split_id": str(preds.split_id.iloc[0]),
        "seed": 42,
        "sampling": "20 worst + 4/star from remaining Test, no duplicates",
        "review_ids": sample.review_id.tolist(),
        "test_n": len(preds),
        "test_truncation_pct": float(100 * (preds.n_tokens > 254).mean()),
    }
    (output / names[3]).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def summarize(reports_dir: str, output_dir: str | None = None) -> Path:
    summary, preds = load_data(reports_dir)
    sample = sample_reviews(summary, preds)
    output = Path(output_dir or reports_dir)

    manifest = json.loads((output / "error_analysis_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["split_id"] != str(preds.split_id.iloc[0])
        or manifest["review_ids"] != sample.review_id.tolist()
        or manifest["model"] != sample.model.iloc[0]
    ):
        raise ValueError("Dữ liệu đã thay đổi so với lần prepare. Hãy chạy lại với --overwrite.")

    annotations = pd.read_csv(output / "error_analysis.csv", dtype={"review_id": str}, keep_default_na=False)
    annotations = validate_annotations(annotations, sample)

    rows = []
    for group in GROUPS:
        mask = annotations.error_group.eq(group)
        rows.append({
            "error_group": group,
            "n": int(mask.sum()),
            "pct": float(mask.mean() * 100),
            "worst_n": int((mask & sample.sample_type.eq("worst")).sum()),
            "random_n": int((mask & sample.sample_type.eq("random")).sum()),
        })

    stats = pd.DataFrame(rows)
    write_csv(stats, output / "error_group_summary.csv")

    lines = [
        "# Phân tích lỗi tự động — mẫu 40 review",
        "",
        f"Mô hình: **{sample.model.iloc[0]}**, chọn theo RMSE Test thấp nhất trong `model_summary.csv`.",
        f"Split: `{preds.split_id.iloc[0]}`. 20 ca sai lớn nhất + 20 ca ngẫu nhiên phân tầng (4/sao), `seed=42`, không trùng review.",
        "Phương pháp gán nhãn: **Tự động hóa (Rule-based NLP Heuristic v2)** xét đảo chiều đầu–đuôi/đoạn cập nhật, "
        "cấu trúc mỉa mai, cực cảm xúc một chiều và thành ngữ đảo nghĩa.",
        "",
        "| Nhóm | Số review | % mẫu 40 | Worst (20) | Random (20) |",
        "|---|---:|---:|---:|---:|",
    ]
    if (Path(reports_dir) / "DEMO_ONLY.txt").exists():
        lines.insert(2, "> DỮ LIỆU GIẢ LẬP — không dùng làm kết quả nghiên cứu.\n")

    for row in rows:
        lines.append(
            f"| {row['error_group']} | {row['n']} | {row['pct']:.1f}% | {row['worst_n']} | {row['random_n']} |"
        )

    lines += [
        "",
        f"Tỉ lệ Test có `n_tokens > 254`: **{100 * (preds.n_tokens > 254).mean():.2f}%** trên {len(preds)} review.",
        "",
        "Tỉ lệ nhóm lỗi trên mẫu có chủ đích này không đại diện cho toàn bộ Test. "
        "Cờ `n_tokens` dùng tokenizer BERT theo đặc tả; không tự chứng minh việc cắt gây ra lỗi, đặc biệt với RoBERTa.",
        "",
        "## Ví dụ theo nhóm (không sao chép nội dung review)",
    ]

    for group in GROUPS:
        group_rows = annotations.loc[annotations.error_group.eq(group)]
        if not group_rows.empty:
            row = group_rows.iloc[0]
            note = str(row.notes).replace("\n", " ")
            lines.append(
                f"- **{group}**: `{row.review_id}`, thật {row.true_stars}, dự đoán {row.pred:.3f}, sai số {row.abs_error:.3f}. {note}"
            )

    path = output / "error_analysis_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_auto(reports_dir: str, output_dir: str | None = None, overwrite: bool = False) -> Path:
    prepare(reports_dir, output_dir, overwrite=overwrite)
    return summarize(reports_dir, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["auto", "prepare", "summarize"])
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true", help="Ghi đè kết quả phân tích lỗi cũ nếu có")
    args = parser.parse_args()

    try:
        if args.command == "auto":
            result = run_auto(args.reports_dir, args.output_dir, overwrite=args.overwrite)
        elif args.command == "prepare":
            result = prepare(args.reports_dir, args.output_dir, overwrite=args.overwrite)
        else:
            result = summarize(args.reports_dir, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Lỗi: {exc}\n")

    print(f"Hoàn tất! Đã tạo: {result}")


if __name__ == "__main__":
    main()