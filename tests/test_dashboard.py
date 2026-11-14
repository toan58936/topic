"""Acceptance tests with synthetic CSVs, CPU only, no training dependencies."""
import subprocess
import sys
from unittest.mock import MagicMock
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from app import streamlit_dashboard as d
from scripts.create_dashboard_demo import create_demo
from src.metrics import per_star_metrics

@pytest.fixture
def reports(tmp_path):
    return create_demo(tmp_path / 'demo')

@pytest.fixture
def preds(reports):
    return d.load_data(reports)[1]


def test_prefers_text_and_fallback(reports):
    summary, preds = d.load_data(reports)
    assert len(summary) == 5 and len(preds) == 200 and 'text' in preds
    (reports / 'test_predictions_with_text.csv').unlink()
    assert 'text' not in d.load_data(reports)[1]


def test_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        d.load_data(tmp_path)


def test_model_columns(preds):
    preds['pred_mean_predictor'] = 3
    assert d.get_model_columns(preds) == ['baseline', 'bert', 'distilbert', 'roberta']


def test_filter_copy(preds):
    result = d.filter_predictions(preds, 'bert', [1, 5])
    assert len(result) == 80 and set(result.true_stars) == {1, 5}
    result.loc[:, 'true_stars'] = 2
    assert preds.true_stars.iloc[0] == 1

@pytest.mark.parametrize('model,stars', [('absent',[1]), ('bert',[]), ('bert',[6])])
def test_invalid_filter(preds, model, stars):
    with pytest.raises(ValueError):
        d.filter_predictions(preds, model, stars)


def test_render_per_star_matches_source(preds, monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(d, 'st', mock)
    d.render_per_star_metrics(preds, 'bert')
    actual = mock.dataframe.call_args.args[0].set_index('stars')
    expected = per_star_metrics(preds.true_stars, preds.pred_bert)
    for star, metrics in expected['per_star'].items():
        for key, value in metrics.items():
            assert actual.loc[star, key] == pytest.approx(value, abs=1e-6)
    assert mock.metric.call_args.args[1] == f"{expected['macro_mae']:.4f}"


def test_missing_star(preds):
    table, macro = d.compute_per_star(preds.loc[preds.true_stars != 1], 'bert')
    assert table.stars.tolist() == [2, 3, 4, 5]
    assert macro == pytest.approx(table.mae.mean())


def test_top_k_text_alignment(preds):
    subset = d.filter_predictions(preds, 'bert', [1, 5])
    result = d.get_worst_cases(subset, 'bert', 5)
    expected = subset.sort_values('abs_err_bert', ascending=False).head(5)
    assert result.review_id.tolist() == expected.review_id.tolist()
    assert result.text.tolist() == expected.text.tolist()
    assert result.abs_err_bert.is_monotonic_decreasing
    assert len(d.get_worst_cases(subset.head(3), 'bert')) == 3


def test_comparison_missing_metrics(reports, monkeypatch):
    summary, _ = d.load_data(reports)
    summary.loc[2, 'params_total'] = 123
    mock = MagicMock()
    monkeypatch.setattr(d, 'st', mock)
    d.render_comparison_table(summary)
    table = mock.dataframe.call_args.args[0]
    assert len(table) == 5 and 'params_total' in table
    assert pd.isna(table.loc[0, 'params_total']) and 'train_seconds' not in table


def test_chart_data(preds, monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(d, 'st', mock)
    d.render_error_histogram(preds, 'bert')
    assert mock.bar_chart.call_args.args[0]['Số review'].sum() == len(preds)
    d.render_scatter_plot(preds, 'bert')
    spec = mock.vega_lite_chart.call_args.args[1]
    assert spec['layer'][1]['data']['values'] == [{'x':1,'y':1}, {'x':5,'y':5}]

@pytest.mark.parametrize('column,value', [('true_stars',6), ('pred_bert',8), ('abs_err_bert',99), ('n_tokens',-1)])
def test_bad_values(preds, column, value):
    preds.loc[0, column] = value
    with pytest.raises(ValueError):
        d.validate_predictions(preds)


def test_duplicate_and_split_mismatch(reports, preds):
    preds.loc[1, 'review_id'] = preds.loc[0, 'review_id']
    with pytest.raises(ValueError, match='review_id'):
        d.validate_predictions(preds)
    summary = pd.read_csv(reports / 'model_summary.csv')
    summary['split_id'] = 'different'
    summary.to_csv(reports / 'model_summary.csv', index=False)
    with pytest.raises(ValueError, match='split_id'):
        d.load_data(reports)


def test_numeric_id_preserved(reports):
    path = reports / 'test_predictions_with_text.csv'
    df = pd.read_csv(path)
    df.loc[0, 'review_id'] = '00001'
    df.to_csv(path, index=False)
    assert d.load_data(reports)[1].review_id.iloc[0] == '00001'

@pytest.mark.parametrize('with_text', [True, False])
def test_app_runs_without_crash(reports, monkeypatch, with_text):
    if not with_text:
        (reports / 'test_predictions_with_text.csv').unlink()
    monkeypatch.setenv('YELP_REPORTS_DIR', str(reports))
    at = AppTest.from_file(d.__file__).run(timeout=30)
    assert not at.exception and len(at.radio) == 1 and len(at.multiselect) == 1
    at.radio[0].set_value('bert').run()
    at.multiselect[0].set_value([1,5]).run()
    assert not at.exception
    worst = at.dataframe[2].value
    assert set(worst.true_stars).issubset({1,5}) and worst.abs_err_bert.is_monotonic_decreasing
    at.selectbox[0].set_value(worst.review_id.iloc[-1]).run()
    assert not at.exception
    if with_text:
        assert any(worst.iloc[-1].text.strip() in t.value for t in at.text)
    else:
        assert any('Chưa có nội dung' in t.value for t in at.info)
    at.checkbox[0].uncheck().run()
    assert not at.exception and not at.dataframe[-1].value.abs_err_bert.is_monotonic_decreasing
    at.multiselect[0].set_value([]).run()
    assert not at.exception and at.info


def test_missing_input_ui(tmp_path, monkeypatch):
    monkeypatch.setenv('YELP_REPORTS_DIR', str(tmp_path))
    at = AppTest.from_file(d.__file__).run(timeout=30)
    assert not at.exception and at.error and at.info


def test_no_torch_import():
    subprocess.run([sys.executable, '-c', "import sys; import app.streamlit_dashboard; assert 'torch' not in sys.modules"], cwd=d.ROOT, check=True)


def test_legacy_producer_metadata_is_optional(reports):
    path = reports / 'test_predictions_with_text.csv'
    df = pd.read_csv(path).drop(columns=['n_tokens', 'split_id'])
    df.to_csv(path, index=False)
    assert len(d.load_data(reports)[1]) == 200


def test_selected_table_row_opens_detail(preds, monkeypatch):
    from types import SimpleNamespace
    mock = MagicMock()
    mock.dataframe.return_value = SimpleNamespace(selection=SimpleNamespace(rows=[2]))
    monkeypatch.setattr(d, 'st', mock)
    d.render_worst_cases(preds, 'bert')
    expected = d.get_worst_cases(preds, 'bert').iloc[2]
    assert mock.text.call_args.args[0] == expected.text
    assert expected.review_id in mock.expander.call_args.args[0]
    mock.selectbox.assert_not_called()
