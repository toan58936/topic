"""CSV-only Yelp dashboard. Run: streamlit run app/streamlit_dashboard.py."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
from src.metrics import per_star_metrics

COMPARISON_COLUMNS = [
    "model", "mae", "rmse", "rmse_raw", "r2", "pearson", "spearman",
    "macro_mae", "train_seconds", "seconds_per_epoch", "params_total",
    "rmse_reduction_vs_baseline_pct", "inference_seconds",
]


def get_model_columns(preds_df: pd.DataFrame) -> list[str]:
    return [c[5:] for c in preds_df.columns
            if c.startswith("pred_") and c != "pred_mean_predictor"]


def validate_predictions(df: pd.DataFrame) -> None:
    """Fail clearly on corrupt exports instead of displaying misleading metrics."""
    required = {"review_id", "true_stars"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions thiếu cột: {', '.join(sorted(missing))}")
    if df.empty or df.review_id.isna().any() or df.review_id.duplicated().any():
        raise ValueError("Predictions phải có review_id duy nhất, không rỗng.")
    if not df.true_stars.isin(range(1, 6)).all():
        raise ValueError("true_stars phải là số nguyên từ 1 đến 5.")
    if "split_id" in df and (df.split_id.isna().any() or df.split_id.nunique() != 1):
        raise ValueError("Predictions phải thuộc đúng một split_id.")
    if "n_tokens" in df and not (np.isfinite(df.n_tokens) & (df.n_tokens >= 0)
            & (df.n_tokens % 1 == 0)).all():
        raise ValueError("n_tokens phải là số nguyên không âm.")
    for model in get_model_columns(df) + (["mean_predictor"] if "pred_mean_predictor" in df else []):
        pred, error = f"pred_{model}", f"abs_err_{model}"
        if error not in df:
            raise ValueError(f"Predictions thiếu cột {error}.")
        valid = df[pred].notna()
        if not df.loc[valid, pred].between(1, 5).all():
            raise ValueError(f"{pred} phải nằm trong [1, 5].")
        if not valid.equals(df[error].notna()) or not np.allclose(
            df.loc[valid, error],
            (df.loc[valid, pred] - df.loc[valid, "true_stars"]).abs(), atol=2e-6, rtol=0,
        ):
            raise ValueError(f"{error} không khớp sai số tuyệt đối của {pred}.")


def load_data(reports_dir: str = "reports") -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = Path(reports_dir)
    summary = pd.read_csv(directory / "model_summary.csv", dtype={"split_id": str})
    path = directory / "test_predictions_with_text.csv"
    if not path.exists():
        path = directory / "test_predictions.csv"
    preds = pd.read_csv(path, dtype={"review_id": str, "split_id": str})
    if "model" not in summary or summary.model.isna().any() or summary.model.duplicated().any():
        raise ValueError("model_summary.csv cần cột model với tên duy nhất, không rỗng.")
    validate_predictions(preds)
    if "split_id" in summary and "split_id" in preds:
        split_ids = summary.split_id.dropna()
        if not split_ids.eq(preds.split_id.iloc[0]).all():
            raise ValueError("split_id của summary và predictions không khớp.")
    if path.name.endswith("_with_text.csv") and "text" not in preds:
        raise ValueError("File with_text thiếu cột text.")
    return summary, preds


def filter_predictions(preds_df: pd.DataFrame, model: str, stars: list[int]) -> pd.DataFrame:
    if f"pred_{model}" not in preds_df:
        raise ValueError(f"Không có dự đoán cho model '{model}'.")
    if not stars or not set(stars).issubset(range(1, 6)):
        raise ValueError("Chọn ít nhất một mức sao trong 1–5.")
    return preds_df.loc[preds_df.true_stars.isin(stars)].copy()


def comparison_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    columns = [c for c in COMPARISON_COLUMNS if c in summary_df and summary_df[c].notna().any()]
    return summary_df.loc[:, columns].copy()


def render_comparison_table(summary_df: pd.DataFrame) -> None:
    st.subheader("So sánh mô hình · toàn bộ Test")
    st.caption("Chỉ số và chi phí từ model_summary.csv; ô trống là chưa có kết quả.")
    st.dataframe(comparison_table(summary_df), hide_index=True, width="stretch")


def render_scatter_plot(preds_df: pd.DataFrame, model: str) -> None:
    st.subheader("Dự đoán so với nhãn thật")
    # Vega-Lite via Streamlit: identity line and points, no extra chart dependency.
    st.vega_lite_chart(preds_df, {
        "layer": [
            {"mark": {"type": "circle", "opacity": 0.25, "size": 35},
             "encoding": {
                 "x": {"field": "true_stars", "type": "quantitative", "scale": {"domain": [1, 5]}},
                 "y": {"field": f"pred_{model}", "type": "quantitative", "scale": {"domain": [1, 5]}},
                 "tooltip": [{"field": "review_id"}, {"field": "true_stars"}, {"field": f"pred_{model}"}]}},
            {"data": {"values": [{"x": 1, "y": 1}, {"x": 5, "y": 5}]},
             "mark": {"type": "line", "color": "#ef4444", "strokeDash": [5, 5]},
             "encoding": {"x": {"field": "x", "type": "quantitative"},
                          "y": {"field": "y", "type": "quantitative"}}},
        ], "height": 300,
    }, width="stretch")


def render_error_histogram(preds_df: pd.DataFrame, model: str) -> None:
    st.subheader("Phân phối sai số tuyệt đối")
    counts, edges = np.histogram(preds_df[f"abs_err_{model}"].dropna(), bins=20, range=(0, 4))
    hist = pd.DataFrame({"Sai số": (edges[:-1] + edges[1:]) / 2, "Số review": counts})
    st.bar_chart(hist, x="Sai số", y="Số review")


def compute_per_star(preds_df: pd.DataFrame, model: str) -> tuple[pd.DataFrame, float]:
    # Current src.metrics already clips internally and accepts only two arguments.
    result = per_star_metrics(preds_df.true_stars.to_numpy(), preds_df[f"pred_{model}"].to_numpy())
    table = pd.DataFrame([{"stars": star, **values} for star, values in result["per_star"].items()],
                         columns=["stars", "mae", "rmse", "n"])
    return table, result["macro_mae"]


def render_per_star_metrics(preds_df: pd.DataFrame, model: str) -> None:
    st.subheader("Sai số theo mức sao · dữ liệu đang lọc")
    table, macro = compute_per_star(preds_df, model)
    st.metric("Macro-MAE", f"{macro:.4f}" if np.isfinite(macro) else "—")
    st.dataframe(table, hide_index=True, width="stretch")
    if not table.empty:
        st.bar_chart(table, x="stars", y=["mae", "rmse"], stack=False)


def get_worst_cases(preds_df: pd.DataFrame, model: str, top_k: int = 10) -> pd.DataFrame:
    if top_k < 1:
        raise ValueError("top_k phải lớn hơn 0.")
    columns = ["review_id", "true_stars", f"pred_{model}", f"abs_err_{model}"]
    if "text" in preds_df:
        columns.append("text")
    return preds_df.nlargest(top_k, f"abs_err_{model}")[columns].copy()


def render_worst_cases(preds_df: pd.DataFrame, model: str, top_k: int = 10) -> None:
    st.subheader(f"Top {top_k} ca đoán sai · dữ liệu đang lọc")
    worst = get_worst_cases(preds_df, model, top_k).reset_index(drop=True)
    event = st.dataframe(worst, hide_index=True, width="stretch",
                         on_select="rerun", selection_mode="single-row", key=f"worst_{model}_{tuple(worst.review_id)}")
    if worst.empty:
        return
    selected = event.selection.rows
    if selected and selected[0] < len(worst):
        row = worst.iloc[selected[0]]
    else:
        review_id = st.selectbox("Xem chi tiết review", worst.review_id.tolist(), key=f"detail_{model}")
        row = worst.loc[worst.review_id == review_id].iloc[0]
    with st.expander(f"Chi tiết · {row.review_id}", expanded=True):
        st.write(f"Nhãn thật: {row.true_stars} · Dự đoán: {row[f'pred_{model}']:.4f} · "
                 f"Sai số: {row[f'abs_err_{model}']:.4f}")
        if "text" in row and pd.notna(row["text"]):
            st.text(str(row["text"]))
        else:
            st.info("Chưa có nội dung review. Thêm test_predictions_with_text.csv vào thư mục dữ liệu.")


def main() -> None:
    st.set_page_config(page_title="Yelp · Dashboard kết quả", page_icon="⭐", layout="wide")
    st.title("Yelp · Dự đoán điểm đánh giá")
    default_dir = os.environ.get("YELP_REPORTS_DIR", str(ROOT / "reports"))
    reports_dir = st.sidebar.text_input("Thư mục CSV", value=default_dir)
    if (Path(reports_dir) / "DEMO_ONLY.txt").exists():
        st.warning("DỮ LIỆU GIẢ LẬP — chỉ kiểm tra giao diện, không dùng làm kết quả báo cáo.")
    try:
        summary, preds = load_data(reports_dir)
    except (OSError, ValueError, TypeError, pd.errors.ParserError) as exc:
        st.error(f"Chưa thể đọc dữ liệu: {exc}")
        st.info("Cần model_summary.csv và test_predictions.csv (hoặc bản _with_text) do Leader xuất sau training.")
        return
    missing_metadata = {"split_id", "n_tokens"} - set(preds.columns)
    if missing_metadata:
        st.warning("CSV chưa đủ metadata: " + ", ".join(sorted(missing_metadata)) +
                   ". Vẫn xem được dashboard; Error Analysis cần bản xuất đầy đủ theo đặc tả.")
    render_comparison_table(summary)
    models = [m for m in get_model_columns(preds) if preds[f"pred_{m}"].notna().any()]
    if not models:
        st.warning("Chưa có model với dự đoán hợp lệ để vẽ biểu đồ.")
        return
    model = st.sidebar.radio("Chọn model", models)
    stars = st.sidebar.multiselect("Lọc mức sao", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    sort_error = st.sidebar.checkbox("Sắp xếp theo sai số giảm dần", value=True)
    if not stars:
        st.info("Chọn ít nhất một mức sao để xem kết quả.")
        return
    filtered = filter_predictions(preds, model, stars)
    missing = int(filtered[f"pred_{model}"].isna().sum())
    if missing:
        st.warning(f"Bỏ qua {missing} review chưa có dự đoán của {model} trong phần đang lọc.")
    filtered = filtered.dropna(subset=[f"pred_{model}"])
    if filtered.empty:
        st.info("Không có review phù hợp với bộ lọc.")
        return
    if sort_error:
        filtered = filtered.sort_values(f"abs_err_{model}", ascending=False, kind="stable")
    split_label = preds.split_id.iloc[0] if "split_id" in preds else "chưa có"
    st.caption(f"{len(filtered):,} / {len(preds):,} review · model {model} · split {split_label}")
    left, right = st.columns(2)
    with left:
        render_scatter_plot(filtered, model)
    with right:
        render_error_histogram(filtered, model)
    render_per_star_metrics(filtered, model)
    render_worst_cases(filtered, model)
    with st.expander("Tất cả dự đoán đang lọc"):
        columns = ["review_id", "true_stars", f"pred_{model}", f"abs_err_{model}"]
        st.dataframe(filtered[columns], hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
