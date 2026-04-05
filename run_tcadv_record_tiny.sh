#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_TCADV="${ROOT_TCADV:-${SCRIPT_DIR}}"
ROOT_LMCA="${ROOT_LMCA:-$(cd "${ROOT_TCADV}/.." && pwd)/LMCA-TIC}"
VENV="${VENV:-/mnt/workspace/venvs/tcadv}"
CONFIG="${CONFIG:-configs/experiments/icews14_record_tiny_qwen25_05b_a10.yaml}"
RUN_NAME="${RUN_NAME:-tcadv_icews14_record_tiny_qwen25_05b_a10}"
LOCAL_MODEL="${LOCAL_MODEL:-${ROOT_LMCA}/models/Qwen2.5-0.5B-Instruct}"
TRAIN_LOG="${TRAIN_LOG:-${ROOT_TCADV}/train_record_tiny.log}"

OUT_DIR="${ROOT_TCADV}/outputs/${RUN_NAME}"
LOG_DIR="${ROOT_TCADV}/logs/${RUN_NAME}"
CKPT_DIR="${ROOT_TCADV}/checkpoints/${RUN_NAME}"
PROC_DIR="${ROOT_TCADV}/data/processed/${RUN_NAME}"
TINY_ROOT="${ROOT_TCADV}/data/local/icews14_record_tiny"

activate_venv() {
  if [[ ! -d "${VENV}" ]]; then
    echo "虚拟环境不存在: ${VENV}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${VENV}/bin/activate"
}

prepare_tiny() {
  activate_venv
  cd "${ROOT_TCADV}"
  python scripts/make_icews14_record_tiny.py
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
  activate_venv
  python -V
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
  echo "已清理 ${RUN_NAME} 相关旧输出"
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
  echo "==== log tail ===="
  [[ -f "${LOG_DIR}/tc_adv.log" ]] && tail -n 40 "${LOG_DIR}/tc_adv.log" || echo "未找到 ${LOG_DIR}/tc_adv.log"
  echo
  echo "==== tiny dataset stats ===="
  for split in train valid test; do
    file="${TINY_ROOT}/raw/${split}.txt"
    if [[ -f "${file}" ]]; then
      echo "${split}: $(wc -l < "${file}")"
    fi
  done
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
  bash run_tcadv_record_tiny.sh check
  bash run_tcadv_record_tiny.sh install
  bash run_tcadv_record_tiny.sh prepare
  bash run_tcadv_record_tiny.sh clean
  bash run_tcadv_record_tiny.sh train
  bash run_tcadv_record_tiny.sh train_bg
  bash run_tcadv_record_tiny.sh eval
  bash run_tcadv_record_tiny.sh results
  bash run_tcadv_record_tiny.sh tail_log
  bash run_tcadv_record_tiny.sh watch_gpu
  bash run_tcadv_record_tiny.sh all
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
  prepare)
    prepare_tiny
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
    prepare_tiny
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
