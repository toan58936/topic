"""Automated error analysis workflow for Yelp star-prediction outputs.

Commands:
    python -m scripts.error_analysis auto --reports-dir data
    python -m scripts.error_analysis prepare --reports-dir data
    python -m scripts.error_analysis summarize --reports-dir data
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
}
NEG_WORDS = {
    "terrible", "horrible", "worst", "awful", "disgusting", "rude", "slow",
    "cold", "inedible", "overpriced", "waste", "disappointed", "disappointing",
}
CONTRAST_WORDS = {
    "but", "however", "although", "though", "unfortunately", "except", "yet",
}
SARCASM_PATTERNS = [
    r"\byeah right\b",
    r"\bgreat job\b.*(!|\.{3})",
    r"\bthanks for nothing\b",
    r"\bjust great\b",
    r"\bwhat a joke\b",
    r"\bsure\b.*\bwait(ed|ing)?\b",
    r"\"(great|amazing|wonderful|fresh|fast)\"",
]


def classify_review(row: pd.Series) -> tuple[str, str]:
    """Automatically assign one of the 5 error groups and generate evidence notes."""
    text = str(row["text"])
    text_low = text.lower()
    words = set(re.findall(r"\b[a-z]+\b", text_low))

    pos_hits = sorted(words & POS_WORDS)
    neg_hits = sorted(words & NEG_WORDS)
    has_contrast = bool(words & CONTRAST_WORDS)
    has_sarcasm = any(re.search(pat, text_low) for pat in SARCASM_PATTERNS)

    true_s = int(row["true_stars"])
    pred_s = float(row["pred"])
    abs_err = float(row["abs_error"])
    n_tok = int(row["n_tokens"])

    # 1. Bị cắt 256 token (chỉ hợp lệ khi n_tokens > 254)
    if n_tok > 254 and abs_err >= 1.0:
        return (
            "Bị cắt 256 token",
            f"Văn bản dài {n_tok} token (>254), phần nội dung cuối bị cắt mất dẫn đến sai số {abs_err:.2f}.",
        )

    # 2. Mỉa mai
    if has_sarcasm or (true_s <= 2 and pred_s >= 3.8 and len(pos_hits) >= 2 and not neg_hits):
        evidence = ", ".join(pos_hits[:3]) if pos_hits else "mẫu câu châm biếm"
        return (
            "Mỉa mai",
            f"Biểu hiện mỉa mai/từ tích cực ngược ngữ cảnh ({evidence}) khiến mô hình dự đoán {pred_s:.2f} sao.",
        )

    # 3. Nhãn nhiễu
    if (true_s == 1 and len(pos_hits) >= 2 and not neg_hits and not has_contrast) or (
        true_s == 5 and len(neg_hits) >= 2 and not pos_hits and not has_contrast
    ):
        evidence = ", ".join((pos_hits or neg_hits)[:3])
        return (
            "Nhãn nhiễu",
            f"Nhãn {true_s} sao mâu thuẫn trực tiếp với từ ngữ cảm xúc rõ ràng trong bài ({evidence}).",
        )

    # 4. Cảm xúc lẫn lộn
    if (pos_hits and neg_hits) or (has_contrast and (len(pos_hits) + len(neg_hits) >= 1)):
        return (
            "Cảm xúc lẫn lộn",
            f"Review pha trộn khen ({pos_hits[:2]}) và chê ({neg_hits[:2]}) kèm từ nối tương phản.",
        )

    # 5. Khác
    return (
        "Khác",
        f"Review ngắn/trung tính ({n_tok} token) hoặc sắc thái mơ hồ; dự đoán {pred_s:.2f} so với thật {true_s}.",
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


def prepare(reports_dir: str = "data", output_dir: str | None = None, overwrite: bool = False) -> Path:
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
        "mode": "automated_rule_based",
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


def summarize(reports_dir: str = "data", output_dir: str | None = None) -> Path:
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
        "Phương pháp gán nhãn: **Tự động hóa (Rule-based NLP Heuristic)** dựa trên độ dài token, từ khóa cảm xúc và cấu trúc tương phản.",
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


def run_auto(reports_dir: str = "data", output_dir: str | None = None, overwrite: bool = False) -> Path:
    prepare(reports_dir, output_dir, overwrite=overwrite)
    return summarize(reports_dir, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["auto", "prepare", "summarize"])
    parser.add_argument("--reports-dir", default="data")
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