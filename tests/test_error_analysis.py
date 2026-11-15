import pandas as pd
import pytest
from app.streamlit_dashboard import load_data
from scripts.create_dashboard_demo import create_demo
from scripts.error_analysis import GROUPS, OUTPUT_COLUMNS, sample_reviews, leader_ids, prepare, validate_annotations, crosscheck, summarize

@pytest.fixture
def reports(tmp_path):
    return create_demo(tmp_path / 'demo')

@pytest.fixture
def sample(reports):
    return sample_reviews(*load_data(reports))


def labeled(sample):
    result = sample[OUTPUT_COLUMNS].copy()
    result['error_group'] = 'Khác'
    result['notes'] = 'Synthetic test annotation only.'
    return result


def test_counts_strata_and_worst(reports, sample):
    _, preds = load_data(reports)
    assert len(sample) == 40 and sample.review_id.nunique() == 40
    assert sample.model.iloc[0] == 'bert'
    assert sample.loc[sample.sample_type == 'worst', 'review_id'].tolist() == preds.nlargest(20, 'abs_err_bert').review_id.tolist()
    assert sample.loc[sample.sample_type == 'random'].groupby('true_stars').size().to_dict() == dict.fromkeys(range(1,6), 4)
    assert sample.error_group.eq('').all()


def test_reproducible_input_order(reports, sample):
    summary, preds = load_data(reports)
    second = sample_reviews(summary.sample(frac=1, random_state=9), preds.sample(frac=1, random_state=8))
    assert sample.review_id.tolist() == second.review_id.tolist()
    assert leader_ids(sample) == leader_ids(second)


def test_need_text(reports):
    summary, preds = load_data(reports)
    with pytest.raises(ValueError, match='nội dung'):
        sample_reviews(summary, preds.drop(columns='text'))


def test_insufficient_strata(reports):
    summary, preds = load_data(reports)
    with pytest.raises(ValueError, match='sao 1'):
        sample_reviews(summary, preds.loc[preds.true_stars != 1])


def test_missing_best_not_replaced(reports):
    summary, preds = load_data(reports)
    with pytest.raises(ValueError, match='tốt nhất'):
        sample_reviews(summary, preds.drop(columns='pred_bert'))


def test_prepare_no_overwrite_blind_leader(reports, sample):
    prepare(str(reports))
    annotation = pd.read_csv(reports / 'error_analysis.csv')
    assert annotation.columns.tolist() == OUTPUT_COLUMNS
    assert len(annotation) == 40 and annotation.error_group.isna().all()
    leader = pd.read_csv(reports / 'leader_crosscheck_with_text.csv')
    assert leader.review_id.tolist() == leader_ids(sample)
    assert 'error_group' not in leader and leader.leader_error_group.isna().all()
    with pytest.raises(FileExistsError):
        prepare(str(reports))

@pytest.mark.parametrize('mutation', ['blank','invalid','duplicate','changed_pred','changed_model','empty_notes','false_cut'])
def test_bad_annotations(sample, mutation):
    a = labeled(sample)
    if mutation == 'blank': a.loc[0, 'error_group'] = ''
    elif mutation == 'invalid': a.loc[0, 'error_group'] = 'Invented'
    elif mutation == 'duplicate': a.loc[0, 'review_id'] = a.loc[1, 'review_id']
    elif mutation == 'changed_pred': a.loc[0, 'pred'] += .1
    elif mutation == 'changed_model': a.loc[0, 'model'] = 'roberta'
    elif mutation == 'empty_notes': a.loc[0, 'notes'] = ' '
    else: a.loc[sample.index[~sample.truncation_flag][0], 'error_group'] = 'Bị cắt 256 token'
    with pytest.raises(ValueError): validate_annotations(a, sample)


def test_crosscheck_threshold(sample):
    leader = pd.DataFrame({'review_id':leader_ids(sample), 'leader_error_group':['Khác']*3 + ['Mỉa mai']})
    assert crosscheck(labeled(sample), leader, sample) == (3,.75)
    leader.loc[0, 'review_id'] = 'wrong'
    with pytest.raises(ValueError): crosscheck(labeled(sample), leader, sample)


def test_complete_workflow_requires_agreement(reports, sample):
    prepare(str(reports))
    with pytest.raises(ValueError, match='40 review'): summarize(str(reports))
    labeled(sample).to_csv(reports / 'error_analysis.csv', index=False)
    leader = pd.read_csv(reports / 'leader_crosscheck_with_text.csv')
    leader['leader_error_group'] = ['Khác','Khác','Mỉa mai','Mỉa mai']
    leader.to_csv(reports / 'leader_crosscheck_with_text.csv', index=False)
    with pytest.raises(ValueError, match='FAIL'): summarize(str(reports))
    assert not (reports / 'error_analysis_summary.md').exists()
    leader.loc[2, 'leader_error_group'] = 'Khác'
    leader.to_csv(reports / 'leader_crosscheck_with_text.csv', index=False)
    assert '3/4 = 75%' in summarize(str(reports)).read_text()
    stats = pd.read_csv(reports / 'error_group_summary.csv')
    assert stats.n.sum() == 40 and stats.pct.sum() == 100
    assert stats.error_group.tolist() == GROUPS
    assert stats.worst_n.sum() == 20 and stats.random_n.sum() == 20
