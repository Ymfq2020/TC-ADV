#!/usr/bin/env bash
# 一键复现第 4 章实验（ModelScope 单 A10 缩比例版）。
# 用法:
#   bash run_chapter4.sh check         # 检查环境
#   bash run_chapter4.sh main          # 主对比 5-seed
#   bash run_chapter4.sh ablation      # 消融 + γ 扫描 + 温度扫描
#   bash run_chapter4.sh robustness    # 噪声扫描 + 多步
#   bash run_chapter4.sh analysis      # 错误分类 / 长尾 / 复杂度
#   bash run_chapter4.sh figures       # 一键画图
#   bash run_chapter4.sh all           # 全部
#
# 环境覆盖:
#   ROOT_TCADV / ROOT_LMCA / VENV
#   ICEWS14_CONFIG / ICEWS18_CONFIG / GDELT_CONFIG

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_TCADV="${ROOT_TCADV:-${SCRIPT_DIR}}"
ROOT_LMCA="${ROOT_LMCA:-$(cd "${ROOT_TCADV}/.." && pwd)/LMCA-TIC}"
VENV="${VENV:-/mnt/workspace/venvs/tcadv}"

ICEWS14_CONFIG="${ICEWS14_CONFIG:-configs/experiments/a10_icews14.yaml}"
ICEWS18_CONFIG="${ICEWS18_CONFIG:-configs/experiments/a10_icews18.yaml}"
GDELT_CONFIG="${GDELT_CONFIG:-configs/experiments/a10_gdelt.yaml}"

SEEDS=(42 1337 2024 7 9)
GAMMAS=(0.1 0.3 0.5 0.7 0.9)
TEMPERATURES=(0.1 0.3 0.5 0.7 0.9)
SIGMAS=(0.0 0.5 1.0 2.0)

activate_venv() {
  if [[ -d "${VENV}" ]]; then
    # shellcheck disable=SC1090
    source "${VENV}/bin/activate"
  fi
}

check_env() {
  echo "==== GPU ===="
  nvidia-smi || true
  echo
  echo "==== PATHS ===="
  for path in "${ROOT_TCADV}" "${ROOT_LMCA}" "${ROOT_TCADV}/${ICEWS14_CONFIG}" \
              "${ROOT_TCADV}/${ICEWS18_CONFIG}" "${ROOT_TCADV}/${GDELT_CONFIG}"; do
    if [[ -e "${path}" ]]; then
      echo "  ok: ${path}"
    else
      echo "  MISSING: ${path}" >&2
    fi
  done
  activate_venv
  python -V
  python - <<'PY' || true
import torch
print("torch =", torch.__version__, "cuda =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu =", torch.cuda.get_device_name(0))
PY
}

main_comparison() {
  activate_venv
  cd "${ROOT_TCADV}"
  echo "==== Seed sweep on ICEWS18 ===="
  tc-adv seed-sweep --config "${ICEWS18_CONFIG}" --seeds "${SEEDS[@]}"
  echo "==== Seed sweep on GDELT ===="
  tc-adv seed-sweep --config "${GDELT_CONFIG}" --seeds "${SEEDS[@]}"
}

ablation_runs() {
  activate_venv
  cd "${ROOT_TCADV}"
  echo "==== Component ablations on ICEWS14 ===="
  tc-adv train --config configs/experiments/ablation_no_trm.yaml
  tc-adv train --config configs/experiments/ablation_no_ecm.yaml
  tc-adv train --config configs/experiments/ablation_static_margin.yaml
  echo "==== Gamma sweep on ICEWS14 ===="
  tc-adv gamma-sweep --config "${ICEWS14_CONFIG}" --gammas "${GAMMAS[@]}"
  echo "==== Temperature sweep on ICEWS14 ===="
  tc-adv temperature-sweep --config "${ICEWS14_CONFIG}" --temperatures "${TEMPERATURES[@]}"
}

robustness_runs() {
  activate_venv
  cd "${ROOT_TCADV}"
  echo "==== Multi-step on ICEWS14 + GDELT ===="
  tc-adv multistep-eval --config "${ICEWS14_CONFIG}" --max-steps 5
  tc-adv multistep-eval --config "${GDELT_CONFIG}"  --max-steps 5
  echo "==== Noise sweep on ICEWS18 + GDELT ===="
  tc-adv noise-sweep --config "${ICEWS18_CONFIG}" --sigmas "${SIGMAS[@]}"
  tc-adv noise-sweep --config "${GDELT_CONFIG}"   --sigmas "${SIGMAS[@]}"
}

analysis_runs() {
  activate_venv
  cd "${ROOT_TCADV}"
  echo "==== Error type classification on ICEWS14 ===="
  if [[ -f outputs/tcadv_a10_icews14/test_diagnostics.json ]]; then
    tc-adv error-types --diagnostics outputs/tcadv_a10_icews14/test_diagnostics.json \
      --threshold 0.7 --output outputs/error_types_icews14.json
  else
    echo "  test_diagnostics.json missing — run main_comparison first" >&2
  fi
  echo "==== Complexity benchmark on ICEWS18 ===="
  tc-adv complexity --config "${ICEWS18_CONFIG}" --output outputs/complexity_icews18.json
  echo "==== Long-tail analysis on ICEWS18 ===="
  if [[ -f outputs/tcadv_a10_icews18/test_predictions.jsonl ]]; then
    python -m tc_adv.experiments.longtail_analysis \
      --train-quadruples data/processed/icews18_a10/train.tsv \
      --predictions outputs/tcadv_a10_icews18/test_predictions.jsonl \
      --label "LMCA-TIC + TC-ADV" \
      --output outputs/longtail_summary.json
  else
    echo "  test_predictions.jsonl missing — run main_comparison first" >&2
  fi
}

figures() {
  activate_venv
  cd "${ROOT_TCADV}"
  python scripts/plot_chapter4_figures.py --output-dir outputs/figures
}

usage() {
  sed -n '2,18p' "$0"
}

cmd="${1:-usage}"
case "${cmd}" in
  check)        check_env ;;
  main)         main_comparison ;;
  ablation)     ablation_runs ;;
  robustness)   robustness_runs ;;
  analysis)     analysis_runs ;;
  figures)      figures ;;
  all)
    check_env
    main_comparison
    ablation_runs
    robustness_runs
    analysis_runs
    figures
    ;;
  *) usage; exit 1 ;;
esac
