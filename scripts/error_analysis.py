"""Prepare a reproducible review worksheet, then validate and summarize annotations.

python -m scripts.error_analysis prepare --reports-dir reports
python -m scripts.error_analysis summarize --reports-dir reports
No model loading. Human labels and Leader agreement are never invented.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from app.streamlit_dashboard import load_data

GROUPS = ["Nhãn nhiễu", "Cảm xúc lẫn lộn", "Bị cắt 256 token", "Mỉa mai", "Khác"]
OUTPUT_COLUMNS = ["review_id", "model", "abs_error", "true_stars", "pred", "error_group", "notes"]


def sample_reviews(summary: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    if not {"n_tokens", "split_id"}.issubset(preds.columns):
        raise ValueError("Error Analysis cần n_tokens và split_id từ bản xuất đầy đủ theo đặc tả.")
    if "text" not in preds or preds.text.isna().any() or preds.text.astype(str).str.strip().eq("").any():
        raise ValueError("Cần test_predictions_with_text.csv có nội dung để đọc và gán nhãn.")
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
    random = pd.concat(random_parts).copy()
    random["sample_type"] = "random"
    sample = pd.concat([worst, random], ignore_index=True)
    sample["model"] = model
    sample["pred"] = sample[f"pred_{model}"]
    sample["abs_error"] = sample[f"abs_err_{model}"]
    sample["error_group"] = ""
    sample["notes"] = ""
    sample["truncation_flag"] = sample.n_tokens > 254
    return sample[OUTPUT_COLUMNS + ["sample_type", "text", "n_tokens", "truncation_flag", "split_id"]]


def leader_ids(sample: pd.DataFrame) -> list[str]:
    # Separate random seed stream; fixed set, cannot choose easy cases after labeling.
    ordered = sample.sort_values("review_id").reset_index(drop=True)
    return ordered.iloc[np.random.default_rng(42).choice(len(ordered), 4, replace=False)].review_id.tolist()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8", float_format="%.6f")


def prepare(reports_dir: str, output_dir: str | None = None) -> Path:
    summary, preds = load_data(reports_dir)
    sample = sample_reviews(summary, preds)
    output = Path(output_dir or reports_dir)
    names = ["error_sample.csv", "error_sample_with_text.csv", "error_analysis.csv",
             "leader_crosscheck_with_text.csv", "error_analysis_manifest.json"]
    existing = [name for name in names if (output / name).exists()]
    if existing:
        raise FileExistsError(f"Không ghi đè phần đã làm: {', '.join(existing)}. Chọn --output-dir mới.")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(sample[OUTPUT_COLUMNS[:5]], output / names[0])
    write_csv(sample, output / names[1])
    write_csv(sample[OUTPUT_COLUMNS], output / names[2])
    leader = sample.set_index("review_id").loc[leader_ids(sample)].reset_index()
    leader = leader[["review_id", "model", "true_stars", "pred", "text", "n_tokens"]].copy()
    leader["leader_error_group"] = ""
    leader["leader_notes"] = ""
    write_csv(leader, output / names[3])
    manifest = {"status": "pending_annotation", "model": str(sample.model.iloc[0]),
                "split_id": str(preds.split_id.iloc[0]), "seed": 42,
                "sampling": "20 worst + 4/star from remaining Test, no duplicates",
                "review_ids": sample.review_id.tolist(), "leader_review_ids": leader_ids(sample),
                "test_n": len(preds), "test_truncation_pct": float(100 * (preds.n_tokens > 254).mean())}
    (output / names[4]).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


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


def crosscheck(annotations: pd.DataFrame, leader: pd.DataFrame, sample: pd.DataFrame) -> tuple[int, float]:
    expected = leader_ids(sample)
    if len(leader) != 4 or leader.review_id.duplicated().any() or set(leader.review_id) != set(expected):
        raise ValueError("Leader phải đối chiếu đúng 4 review đã chọn bằng seed=42.")
    if not leader.leader_error_group.isin(GROUPS).all():
        raise ValueError("Leader chưa gán đủ 4 nhãn hợp lệ.")
    e_labels = annotations.set_index("review_id").loc[expected, "error_group"]
    l_labels = leader.set_index("review_id").loc[expected, "leader_error_group"]
    count = int(e_labels.eq(l_labels).sum())
    return count, count / 4


def summarize(reports_dir: str, output_dir: str | None = None) -> Path:
    summary, preds = load_data(reports_dir)
    sample = sample_reviews(summary, preds)
    output = Path(output_dir or reports_dir)
    manifest = json.loads((output / "error_analysis_manifest.json").read_text(encoding="utf-8"))
    if (manifest["split_id"] != str(preds.split_id.iloc[0])
            or manifest["review_ids"] != sample.review_id.tolist()
            or manifest["model"] != sample.model.iloc[0]):
        raise ValueError("Dữ liệu đã thay đổi so với lần prepare. Tạo bộ mẫu mới trong thư mục khác.")
    annotations = pd.read_csv(output / "error_analysis.csv", dtype={"review_id": str}, keep_default_na=False)
    annotations = validate_annotations(annotations, sample)
    leader = pd.read_csv(output / "leader_crosscheck_with_text.csv", dtype={"review_id": str}, keep_default_na=False)
    matches, agreement = crosscheck(annotations, leader, sample)
    if agreement < .75:
        raise ValueError(f"Cross-check FAIL: {matches}/4 ({agreement:.0%}). Thảo luận và gán nhãn lại trước khi tổng hợp.")
    rows = []
    for group in GROUPS:
        mask = annotations.error_group.eq(group)
        rows.append({"error_group": group, "n": int(mask.sum()), "pct": float(mask.mean() * 100),
                     "worst_n": int((mask & sample.sample_type.eq("worst")).sum()),
                     "random_n": int((mask & sample.sample_type.eq("random")).sum())})
    stats = pd.DataFrame(rows)
    write_csv(stats, output / "error_group_summary.csv")
    lines = ["# Phân tích lỗi — mẫu 40 review", "",
             f"Mô hình: **{sample.model.iloc[0]}**, chọn theo RMSE Test thấp nhất trong model_summary.csv.",
             f"Split: `{preds.split_id.iloc[0]}`. 20 ca sai lớn nhất + 20 ca ngẫu nhiên phân tầng (4/sao), seed=42, không trùng review.",
             f"Leader cross-check: **{matches}/4 = {agreement:.0%} — PASS** theo nhãn đã nhập.", "",
             "| Nhóm | Số review | % mẫu 40 | Worst (20) | Random (20) |", "|---|---:|---:|---:|---:|"]
    if (Path(reports_dir) / "DEMO_ONLY.txt").exists():
        lines.insert(2, "> DỮ LIỆU GIẢ LẬP — không dùng làm kết quả nghiên cứu.\n")
    for row in rows:
        lines.append(f"| {row['error_group']} | {row['n']} | {row['pct']:.1f}% | {row['worst_n']} | {row['random_n']} |")
    lines += ["", f"Tỉ lệ Test có n_tokens > 254: **{100 * (preds.n_tokens > 254).mean():.2f}%** trên {len(preds)} review.",
              "", "Tỉ lệ nhóm lỗi trên mẫu có chủ đích này không đại diện cho toàn bộ Test. "
              "Cờ n_tokens dùng tokenizer BERT theo đặc tả; không tự chứng minh việc cắt gây ra lỗi, đặc biệt với RoBERTa.",
              "", "## Ví dụ theo nhóm (không sao chép nội dung review)"]
    for group in GROUPS:
        group_rows = annotations.loc[annotations.error_group.eq(group)]
        if not group_rows.empty:
            row = group_rows.iloc[0]
            note = str(row.notes).replace("\n", " ")
            lines.append(f"- **{group}**: `{row.review_id}`, thật {row.true_stars}, dự đoán {row.pred:.3f}, sai số {row.abs_error:.3f}. {note}")
    path = output / "error_analysis_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "summarize"])
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    try:
        result = (prepare if args.command == "prepare" else summarize)(args.reports_dir, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Lỗi: {exc}\n")
    print(f"Đã tạo: {result}")


if __name__ == "__main__":
    main()
