#!/usr/bin/env bash
set -euo pipefail

PY=python
SCRIPT=run_c2sni.py

# ========= 运行环境 =========
GPU=${GPU:-0}
SEED=${SEED:-0}
RESUME=${RESUME:-1}   # 1=若日志已含 "[Final] Overall accuracy" 则跳过该组合

export CUDA_VISIBLE_DEVICES=${GPU}
export PYTHONIOENCODING=utf-8

# ========= 超参网格（按你的要求）=========
NES_EDGE_BUDGETS=(2 4 6 8)
DATASETS=(cora citeseer chameleon pubmed)
VICTIMS=(gcn)
INJ_NODE_BUDGETS=(0.05)
STEPS=(40)
KHOPS=(4)
BBQS=(24)  # bb_queries

# ========= 与你常用命令一致的固定参数 =========
BASE_ARGS="--nes_hard_topk --lam_edgecls 0.8 --lam_class 1.2 --lam_in 0.08"

# ========= 小工具：格式化秒为 HH:MM:SS =========
fmt_hms () {
  local s=$1
  printf "%02d:%02d:%02d" $((s/3600)) $(((s%3600)/60)) $((s%60))
}

# ========= 解析日志抽取关键指标，写 CSV =========
parse_log () {
  local log="$1"
  local add_pre
  add_pre=$(grep -Eo '\[After Stage 3\].*\(\+[-0-9]+ edges\)' "$log" | sed -E 's/.*\(\+([-0-9]+) edges\).*/\1/' | tail -n1)
  [[ -z "${add_pre}" ]] && add_pre="NA"

  local delta_sparse
  delta_sparse=$(grep -Eo '\[Delta\].*:\s*[-0-9]+' "$log" | sed -E 's/.*:\s*([-0-9]+)/\1/' | tail -n1)
  [[ -z "${delta_sparse}" ]] && delta_sparse="NA"

  local final_acc
  final_acc=$(grep -Eo '\[Final\]\s*Test accuracy.*:\s*[0-9.]+%' "$log" | sed -E 's/.*:\s*([0-9.]+)%.*/\1/' | tail -n1)
  [[ -z "${final_acc}" ]] && final_acc="NA"

  local overall_acc
  overall_acc=$(grep -Eo '\[Final\]\s*Overall accuracy.*:\s*[0-9.]+%' "$log" | sed -E 's/.*:\s*([0-9.]+)%.*/\1/' | tail -n1)
  [[ -z "${overall_acc}" ]] && overall_acc="NA"

  local anchors
  anchors=$(grep -Eo 'Picked\s+[0-9]+\s+anchors' "$log" | sed -E 's/.*Picked\s+([0-9]+)\s+anchors.*/\1/' | tail -n1)
  [[ -z "${anchors}" ]] && anchors="NA"

  local deg_inj
  deg_inj=$(grep -Eo 'deg per injected = [0-9]+' "$log" | sed -E 's/.*=\s*([0-9]+)/\1/' | tail -n1)
  [[ -z "${deg_inj}" ]] && deg_inj="NA"

  echo "${add_pre},${delta_sparse},${final_acc},${overall_acc},${anchors},${deg_inj}"
}

TS=$(date +"%Y%m%d-%H%M%S")

for ds in "${DATASETS[@]}"; do
  OUTROOT="results_0.05_gcn_Budget/${ds}/${TS}"
  LOGDIR="${OUTROOT}/logs"
  mkdir -p "${LOGDIR}"

  FULL_TXT="${OUTROOT}/results.txt"
  CSV="${OUTROOT}/results_summary.csv"

  echo "dataset,victim,nes_edge_budget,inj_node_budget,steps,khop_edge,bb_queries,gpu,seed,edges_before_sparse_added,edges_after_sparse_delta,final_test_acc,overall_acc,anchors,deg_per_inj,logfile,status,duration_sec,duration_hms" > "${CSV}"

  {
    echo "=============== Ablation Grid (dataset=${ds}) ==============="
    echo "Time: ${TS}"
    echo "Output dir: ${OUTROOT}"
    echo "GPU: ${GPU}  SEED: ${SEED}  RESUME: ${RESUME}"
    echo "-------------------------------------------------------------"
  } | tee -a "${FULL_TXT}"

  TOTAL_DS=$(( ${#VICTIMS[@]} * ${#NES_EDGE_BUDGETS[@]} * ${#INJ_NODE_BUDGETS[@]} * ${#STEPS[@]} * ${#KHOPS[@]} * ${#BBQS[@]} ))
  idx=0

  for victim in "${VICTIMS[@]}"; do
    for neb in "${NES_EDGE_BUDGETS[@]}"; do
      for injnb in "${INJ_NODE_BUDGETS[@]}"; do
        for st in "${STEPS[@]}"; do
          for kh in "${KHOPS[@]}"; do
            for qb in "${BBQS[@]}"; do
              idx=$((idx+1))
              tag="m${victim}_neb${neb}_injnb${injnb}_st${st}_kh${kh}_qb${qb}_g${GPU}_s${SEED}"
              log="${LOGDIR}/${tag}.log"

              if [[ "${RESUME}" -eq 1 ]] && [[ -f "${log}" ]] && grep -q "\[Final\] Overall accuracy" "${log}"; then
                echo "[SKIP ${idx}/${TOTAL_DS}] ${ds} | ${tag} (completed log found)" | tee -a "${FULL_TXT}"
                if ! grep -q "${tag}\.log" "${CSV}" 2>/dev/null; then
                  PRE_DELTA_ACC=$(parse_log "${log}")
                  IFS=',' read -r add_pre delta_sparse final_acc overall_acc anchors deg_inj <<< "${PRE_DELTA_ACC}"
                  echo "${ds},${victim},${neb},${injnb},${st},${kh},${qb},${GPU},${SEED},${add_pre},${delta_sparse},${final_acc},${overall_acc},${anchors},${deg_inj},${tag}.log,SKIP,0,00:00:00" >> "${CSV}"
                fi
                continue
              fi

              {
                echo "================================================================"
                echo "[${idx}/${TOTAL_DS}] ${ds} | ${tag}"
                echo "CMD: ${PY} -u ${SCRIPT} --dataset ${ds} --inj_node_budget ${injnb} --victim ${victim} --gpu ${GPU} --steps ${st} --bb_queries ${qb} ${BASE_ARGS} --nes_edge_budget ${neb} --khop_edge ${kh} --seed ${SEED}"
                echo "----------------------------------------------------------------"
              } | tee -a "${FULL_TXT}"

              start_ts=$(date +%s)
              STATUS="OK"
              if ! stdbuf -oL -eL ${PY} -u ${SCRIPT} \
                  --dataset "${ds}" \
                  --inj_node_budget "${injnb}" \
                  --victim "${victim}" \
                  --gpu "${GPU}" \
                  --steps "${st}" \
                  --bb_queries "${qb}" \
                  ${BASE_ARGS} \
                  --nes_edge_budget "${neb}" \
                  --khop_edge "${kh}" \
                  --seed "${SEED}" \
                  2>&1 | tee -a "${log}" | tee -a "${FULL_TXT}"; then
                STATUS="ERR"
                echo "[ERROR] Run failed: ${ds} | ${tag}" | tee -a "${FULL_TXT}"
              fi
              end_ts=$(date +%s)
              dur=$((end_ts - start_ts))
              dur_hms=$(fmt_hms "${dur}")

              echo "[TIME] ${ds} | ${tag} | elapsed ${dur_hms} (${dur} sec)" | tee -a "${FULL_TXT}" | tee -a "${log}"

              PRE_DELTA_ACC=$(parse_log "${log}")
              IFS=',' read -r add_pre delta_sparse final_acc overall_acc anchors deg_inj <<< "${PRE_DELTA_ACC}"
              echo "${ds},${victim},${neb},${injnb},${st},${kh},${qb},${GPU},${SEED},${add_pre},${delta_sparse},${final_acc},${overall_acc},${anchors},${deg_inj},${tag}.log,${STATUS},${dur},${dur_hms}" >> "${CSV}"
              echo >> "${FULL_TXT}"
            done
          done
        done
      done
    done
  done

  echo
  echo "[${ds}] DONE"
  echo "  全量日志: ${FULL_TXT}"
  echo "  精简汇总: ${CSV}"
  echo "  单次日志: ${LOGDIR}"
  echo
done

echo "[ALL DATASETS DONE]"
