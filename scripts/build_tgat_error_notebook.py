"""Update notebook 07 with the locked error-attribution and TGAT-ablation results."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "07_tgat_error_analysis.ipynb"


def markdown(source: str, cell_id: str):
    cell = nbformat.v4.new_markdown_cell(source.strip() + "\n")
    cell["id"] = cell_id
    return cell


def code(source: str, cell_id: str):
    cell = nbformat.v4.new_code_cell(source.strip() + "\n")
    cell["id"] = cell_id
    return cell


def build() -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    generated_ids = {
        "attribution-heading", "attribution-result", "risk-heading", "risk-result",
        "ablation-heading", "ablation-result",
        "case-heading", "case-result", "final-conclusion",
    }
    notebook.cells = [cell for cell in notebook.cells if cell.get("id") not in generated_ids]
    notebook.cells[0].source = """# Phân tích lỗi dự đoán và độ nhạy của TGAT

Notebook này tiếp nối kết quả GNNExplainer bằng hai kiểm tra bổ sung: phân tích trực tiếp FP/FN và control đã ghép, cùng ablation đầu vào của frozen TGAT.

Mọi can thiệp chỉ đo độ nhạy của model trên dữ liệu quan sát. `prediction_push > 0` nghĩa là đầu vào đang hỗ trợ phía dự đoán hiện tại của model; nó không chứng minh quan hệ nhân quả với fraud hay với lỗi ngoài thực tế."""
    notebook.cells[1].source = """from pathlib import Path
import json
import os
import sys

def find_project_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / 'data').is_dir() and (candidate / 'notebooks').is_dir():
            return candidate
    raise FileNotFoundError('Không tìm thấy project root')

ROOT = find_project_root(Path.cwd())
SCRIPT_DIR = ROOT / 'scripts'
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sprint5_2_prepare_error_cohorts as preparation
import sprint5_2_error_attribution as attribution_runner
import sprint5_2_tgat_ablation as ablation_runner

RUN_PREPARATION = os.getenv('RUN_SPRINT5_2_PREPARATION', '0') == '1'
RUN_ERROR_ATTRIBUTION = os.getenv('RUN_ERROR_ATTRIBUTION', '0') == '1'
RUN_TGAT_ABLATION = os.getenv('RUN_TGAT_ABLATION', '0') == '1'

summary = preparation.run(force_relock=False) if RUN_PREPARATION else json.loads(
    preparation.SUMMARY_PATH.read_text(encoding='utf-8')
)
attribution = attribution_runner.run() if RUN_ERROR_ATTRIBUTION else json.loads(
    attribution_runner.RESULT_PATH.read_text(encoding='utf-8')
)
ablation = ablation_runner.run() if RUN_TGAT_ABLATION else json.loads(
    ablation_runner.RESULT_PATH.read_text(encoding='utf-8')
)

import numpy as np
import pandas as pd
from IPython.display import Image, display

lock = json.loads(preparation.LOCK_PATH.read_text(encoding='utf-8'))
predictions = pd.read_csv(preparation.PREDICTION_PATH)
targets = pd.read_csv(preparation.TARGET_PATH)
threshold_curve = pd.read_csv(preparation.CURVE_PATH)

print(f'Project root: {ROOT}')
print(f'Preparation status: {summary["status"]}')
print(f'Prediction rows: {len(predictions):,} | selected targets: {len(targets):,}')"""
    notebook.cells[14].source = """## 7. Dữ liệu phân tích đã khóa

Threshold validation là **0,75738740**. Tại ngưỡng này, validation có 620 TP, 11.528 FP, 1.706 FN và 170.008 TN; test có 610 TP, 11.589 FP, 1.716 FN và 169.925 TN. Precision thấp phản ánh class imbalance mạnh và cho thấy threshold tối đa F1 vẫn tạo nhiều cảnh báo nhầm.

Danh sách gồm 160 target: trên mỗi split có 20 FP, 20 FN và 40 control TP/TN đã ghép. Các frozen local batch, model hash và neighborhood seed không đổi trong các phân tích tiếp theo."""

    notebook.cells.extend([
        markdown("""## 8. Feature nào thực sự gắn riêng với dự đoán sai?

Mỗi FP được ghép với một TP cùng phía dự đoán fraud; mỗi FN được ghép với một TN cùng phía dự đoán normal. Control được chọn gần error theo score, degree và số event.

Với từng node, mỗi feature được thay riêng bằng mức tham chiếu rồi model được chạy lại. Notebook tính `prediction_push(error) - prediction_push(control)` trong từng cặp. Một feature chỉ được đưa vào bảng khi push trung vị trên error dương và chênh lệch trung vị error–control cũng dương. Điều kiện thứ nhất cho biết feature hỗ trợ dự đoán sai; điều kiện thứ hai cho biết mức hỗ trợ đó mạnh hơn ở error so với control. Giá trị 0 chỉ là mốc so sánh, không phải thao tác xóa feature hay một trạng thái trung tính.""", "attribution-heading"),
        code("""error_feature_rows = pd.DataFrame([
    row for row in attribution['target_summaries'] if row['cohort'] in {'FP', 'FN'}
])
pair_feature_rows = pd.DataFrame([
    {'split': group['split'], 'pair': group['error_cohort'] + '–' + group['control_cohort'], **feature}
    for group in attribution['error_control_summaries']
    for feature in group['feature_contrasts']
])
candidate_rows = []
for group in attribution['error_control_summaries']:
    pair = group['error_cohort'] + '–' + group['control_cohort']
    names = set(group['positive_error_specific_candidates'])
    if not names:
        candidate_rows.append({
            'split': group['split'],
            'pair': pair,
            'feature_name': 'Không có',
        })
        continue
    for feature in group['feature_contrasts']:
        if feature['feature_name'] in names:
            candidate_rows.append({
                'split': group['split'],
                'pair': pair,
                **feature,
            })

candidate_table = pd.DataFrame(candidate_rows)
display(candidate_table[[
    'split', 'pair', 'feature_name', 'error_median_prediction_push',
    'paired_median_prediction_push_difference', 'paired_difference_positive_rate',
]].style.format({
    'error_median_prediction_push': '{:.4f}',
    'paired_median_prediction_push_difference': '{:.4f}',
    'paired_difference_positive_rate': '{:.0%}',
}, na_rep='—'))

assert attribution['technical_gates']['all_160_locked_targets_processed']
assert attribution['technical_gates']['all_80_error_control_pairs_processed']
assert attribution['technical_gates']['all_pair_contrasts_finite']
assert attribution['technical_gates']['model_state_unchanged']
assert attribution['technical_gates']['all_feature_values_finite']""", "attribution-result"),
        markdown("""## 9. Community-risk trong FP và FN

Community-risk được tách riêng vì đây là feature train-only bổ sung sau message passing. Bảng dưới báo effect có dấu, IQR và tỷ lệ effect dương. Effect dương chỉ nói feature hỗ trợ phía dự đoán hiện tại trên phép thay giá trị tham chiếu này.""", "risk-heading"),
        code("""risk_table = error_feature_rows[[
    'split', 'cohort', 'community_risk_median_prediction_push',
    'community_risk_iqr_prediction_push', 'community_risk_positive_rate',
]].copy()
display(risk_table.style.format({
    'community_risk_median_prediction_push': '{:.4f}',
    'community_risk_positive_rate': '{:.1%}',
}))

print('Feature-mask nonzero-gradient rate:', attribution['diagnostics']['feature_nonzero_gradient_rate'])""", "risk-result"),
        markdown("""## 10. Độ nhạy của frozen TGAT

Bốn cấu hình chạy trên toàn bộ validation và test với cùng loader: FULL, thời gian hằng trong neighborhood, neighbor về baseline và tắt message passing. Bảng dùng AP và ROC-AUC để nhất quán với cách đánh giá model ở các sprint trước. Đây là ablation ở thời điểm inference, không phải retraining.""", "ablation-heading"),
        code("""ablation_rows = pd.DataFrame(ablation['split_results'])
full_reference = ablation_rows[ablation_rows.variant.eq('FULL')][
    ['split', 'average_precision', 'roc_auc']
].rename(columns={'average_precision': 'full_ap', 'roc_auc': 'full_roc_auc'})
ablation_view = ablation_rows.merge(full_reference, on='split')
ablation_view['delta_ap'] = ablation_view['average_precision'] - ablation_view['full_ap']
ablation_view['delta_roc_auc'] = ablation_view['roc_auc'] - ablation_view['full_roc_auc']
display(ablation_view[[
    'split', 'variant', 'average_precision', 'delta_ap', 'roc_auc', 'delta_roc_auc',
]].style.format({
    'average_precision': '{:.4f}', 'delta_ap': '{:+.4f}',
    'roc_auc': '{:.4f}', 'delta_roc_auc': '{:+.4f}',
}))
display(Image(filename=str(ablation_runner.FIGURE_PATH), width=1050))

display(pd.DataFrame([ablation['mp_off_root_path_check']]))
display(pd.DataFrame(ablation['technical_gates'].items(), columns=['Check', 'Passed']))
assert all(ablation['technical_gates'].values())""", "ablation-result"),
        markdown("""## 11. Một số error case có effect lớn

Các panel chọn một FP và một FN có tổng absolute feature effect lớn trên mỗi split. Chúng dùng để đọc hành vi của model theo từng trường hợp, không đại diện cho toàn cohort và không phải bằng chứng fraud ring.""", "case-heading"),
        code("""error_targets = [
    row for row in attribution['targets'] if row['cohort'] in {'FP', 'FN'}
]
case_rows = []
for split in ('validation', 'test'):
    for cohort in ('FP', 'FN'):
        candidates = [r for r in error_targets if r['split'] == split and r['cohort'] == cohort]
        chosen = max(candidates, key=lambda r: sum(r['feature_direct_absolute_effect']))
        case_rows.append({
            'split': split, 'cohort': cohort, 'node_id': chosen['node_id'],
            'direct_top5': chosen['direct_top5'],
            'gnnexplainer_top5': chosen['gnnexplainer_top5'],
            'feature_spearman': chosen['feature_spearman'],
        })
display(pd.DataFrame(case_rows))
display(Image(filename=str(attribution_runner.CASE_FIGURE), width=1100))""", "case-result"),
        markdown("""## 12. Kết luận

- Không có feature nào vừa hỗ trợ phía dự đoán sai, vừa hỗ trợ mạnh hơn control, rồi lặp lại kết quả đó trên cả validation và test. F16-missing chỉ đạt hai điều kiện ở FP trên test; community-risk chỉ đạt ở FN trên test.
- Community-risk có median effect nhỏ và đổi dấu theo cohort/split. Nó có thể hỗ trợ phía dự đoán của một số lỗi, nhưng không giải thích một mình toàn bộ FP/FN.
- T-CONST làm AP và ROC-AUC giảm trên cả hai split, cho thấy model sử dụng sự khác nhau về thời gian giữa các event.
- N-BASE làm cả hai metric giảm rõ hơn T-CONST; MP-OFF tạo mức giảm lớn nhất, với AP giảm khoảng 0,0142–0,0160 và ROC-AUC giảm khoảng 0,059 trên hai split. Unit check đồng thời xác nhận target core feature và community-risk vẫn thay đổi logit, còn thay neighbor không ảnh hưởng khi message passing đã tắt.

Các kết quả mô tả độ nhạy và cơ chế dự đoán của frozen TGAT trong bài toán full-history transductive node classification. Chúng không chứng minh đầu vào gây ra fraud hoặc gây ra lỗi trong thế giới thực.""", "final-conclusion"),
    ])
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK_PATH)


if __name__ == "__main__":
    build()
    print(NOTEBOOK_PATH)
