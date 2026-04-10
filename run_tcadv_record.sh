#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_TCADV="${ROOT_TCADV:-${SCRIPT_DIR}}"
ROOT_LMCA="${ROOT_LMCA:-$(cd "${ROOT_TCADV}/.." && pwd)/LMCA-TIC}"
VENV="${VENV:-/mnt/workspace/venvs/tcadv}"
CONFIG="${CONFIG:-configs/experiments/icews14_record_qwen25_05b_a10.yaml}"
RUN_NAME="${RUN_NAME:-tcadv_icews14_record_qwen25_05b_a10}"
LOCAL_MODEL="${LOCAL_MODEL:-${ROOT_LMCA}/models/Qwen2.5-0.5B-Instruct}"
TRAIN_LOG="${TRAIN_LOG:-${ROOT_TCADV}/train_record.log}"

OUT_DIR="${ROOT_TCADV}/outputs/${RUN_NAME}"
LOG_DIR="${ROOT_TCADV}/logs/${RUN_NAME}"
CKPT_DIR="${ROOT_TCADV}/checkpoints/${RUN_NAME}"
PROC_DIR="${ROOT_TCADV}/data/processed/${RUN_NAME}"

activate_venv() {
  if [[ ! -d "${VENV}" ]]; then
    echo "虚拟环境不存在: ${VENV}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${VENV}/bin/activate"
}

check_env() {
  echo "==== GPU ===="
  nvidia-smi || true
  echo

  echo "==== PATH CHECK ===="
  [[ -d "${ROOT_TCADV}" ]] && echo "TC-ADV ok: ${ROOT_TCADV}" || { echo "缺少 ${ROOT_TCADV}" >&2; exit 1; }
  [[ -d "${ROOT_LMCA}" ]] && echo "LMCA-TIC ok: ${ROOT_LMCA}" || { echo "缺少 ${ROOT_LMCA}" >&2; exit 1; }
  [[ -f "${ROOT_TCADV}/${CONFIG}" ]] && echo "config ok: ${ROOT_TCADV}/${CONFIG}" || { echo "缺少配置 ${ROOT_TCADV}/${CONFIG}" >&2; exit 1; }
  [[ -d "${LOCAL_MODEL}" ]] && echo "local model ok: ${LOCAL_MODEL}" || { echo "本地模型目录缺失: ${LOCAL_MODEL}" >&2; exit 1; }

  echo
  echo "==== PYTHON CHECK ===="
  activate_venv
  python -V
  python - <<PY
import sys
import torch

print("python =", sys.executable)
print("torch =", torch.__version__)
print("cuda_available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu =", torch.cuda.get_device_name(0))
PY
}

install_editable() {
  activate_venv
  cd "${ROOT_LMCA}"
  python -m pip install -e .
  cd "${ROOT_TCADV}"
  python -m pip install -e .
}

clean_run() {
  rm -rf "${OUT_DIR}" "${LOG_DIR}" "${CKPT_DIR}" "${PROC_DIR}" "${TRAIN_LOG}"
  echo "已清理:"
  echo "  ${OUT_DIR}"
  echo "  ${LOG_DIR}"
  echo "  ${CKPT_DIR}"
  echo "  ${PROC_DIR}"
  echo "  ${TRAIN_LOG}"
}

train_fg() {
  activate_venv
  cd "${ROOT_TCADV}"
  tc-adv train --config "${CONFIG}"
}

train_bg() {
  activate_venv
  cd "${ROOT_TCADV}"
  nohup tc-adv train --config "${CONFIG}" > "${TRAIN_LOG}" 2>&1 &
  local pid=$!
  echo "后台训练已启动, PID=${pid}"
  echo "日志文件: ${TRAIN_LOG}"
}

eval_best() {
  activate_venv
  cd "${ROOT_TCADV}"
  tc-adv evaluate --config "${CONFIG}"
}

show_results() {
  echo "==== test_metrics.json ===="
  [[ -f "${OUT_DIR}/test_metrics.json" ]] && cat "${OUT_DIR}/test_metrics.json" || echo "未找到 ${OUT_DIR}/test_metrics.json"
  echo
  echo "==== valid_metrics.json ===="
  [[ -f "${OUT_DIR}/valid_metrics.json" ]] && cat "${OUT_DIR}/valid_metrics.json" || echo "未找到 ${OUT_DIR}/valid_metrics.json"
  echo
  echo "==== train log tail ===="
  [[ -f "${LOG_DIR}/tc_adv.log" ]] && tail -n 40 "${LOG_DIR}/tc_adv.log" || echo "未找到 ${LOG_DIR}/tc_adv.log"
  echo
  echo "==== background log tail ===="
  [[ -f "${TRAIN_LOG}" ]] && tail -n 40 "${TRAIN_LOG}" || echo "未找到 ${TRAIN_LOG}"
}

tail_train_log() {
  if [[ ! -f "${TRAIN_LOG}" ]]; then
    echo "未找到 ${TRAIN_LOG}" >&2
    exit 1
  fi
  tail -f "${TRAIN_LOG}"
}

watch_gpu() {
  watch -n 2 nvidia-smi
}

usage() {
  cat <<EOF
用法:
  bash run_tcadv_record.sh check
  bash run_tcadv_record.sh install
  bash run_tcadv_record.sh clean
  bash run_tcadv_record.sh train
  bash run_tcadv_record.sh train_bg
  bash run_tcadv_record.sh eval
  bash run_tcadv_record.sh results
  bash run_tcadv_record.sh tail_log
  bash run_tcadv_record.sh watch_gpu
  bash run_tcadv_record.sh all

可覆盖环境变量:
  ROOT_TCADV
  ROOT_LMCA
  VENV
  CONFIG
  RUN_NAME
  LOCAL_MODEL
  TRAIN_LOG

说明:
  check     检查 GPU / 路径 / 模型目录 / Python 环境
  install   安装两个仓库的 editable 包
  clean     清理录屏版旧输出
  train     前台训练
  train_bg  后台训练
  eval      用 best checkpoint 做显式评测
  results   打印关键结果文件
  tail_log  持续查看后台训练日志
  watch_gpu 持续查看 GPU
  all       install + clean + train + eval + results
EOF
}

cmd="${1:-usage}"

case "${cmd}" in
  check)
    check_env
    ;;
  install)
    install_editable
    ;;
  clean)
    clean_run
    ;;
  train)
    train_fg
    ;;
  train_bg)
    train_bg
    ;;
  eval)
    eval_best
    ;;
  results)
    show_results
    ;;
  tail_log)
    tail_train_log
    ;;
  watch_gpu)
    watch_gpu
    ;;
  all)
    check_env
    install_editable
    clean_run
    train_fg
    eval_best
    show_results
    ;;
  *)
    usage
    exit 1
    ;;
esac
