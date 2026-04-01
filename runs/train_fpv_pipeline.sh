#!/usr/bin/env bash
# =============================================================================
# train_fpv_pipeline.sh  –  R2-Dreamer FPV Training Pipeline
# =============================================================================

set -euo pipefail

exec > >(tee -i train_drone_script.log) 2>&1

MODEL_SIZE="size100M"
LOGDIR_BASE="./logdir"
MIXED_PRECISION="fp16"

STEPS_PHASE1=20000
STEPS_PHASE2=5000
STEPS_PHASE3=50000

NUM_GPUS=1

log() { echo "[$(date '+%H:%M:%S')] $*"; }

init_env() {
    if [[ ! -d "venv" ]]; then
        log "Kein venv gefunden. Erstelle neues Virtual Environment..."
        python3 -m venv venv
    fi
    # shellcheck source=/dev/null
    source venv/bin/activate

    if [[ -f "requirements.txt" ]]; then
        if [[ ! -f ".last_install" || "requirements.txt" -nt ".last_install" ]]; then
            log "Installiere/Aktualisiere Requirements (das kann einen Moment dauern)..."
            pip install --upgrade pip
            pip install -r requirements.txt
            touch .last_install
            log "Installation abgeschlossen."
        else
            log "Requirements sind auf dem neuesten Stand."
        fi
    else
        log "WARNUNG: Keine requirements.txt gefunden!"
    fi

    NUM_GPUS=$(python3 -c "import torch; n=torch.cuda.device_count(); print(n if n > 0 else 1)" 2>/dev/null || echo 1)
    log "Gefundene GPUs: $NUM_GPUS"
}

get_checkpoint() {
    local logdir="$1"
    local ckpt_path="${logdir}/latest.pt"
    [[ -f "$ckpt_path" ]] && echo "$ckpt_path"
}

acc_launch() {
    accelerate launch \
        --num_processes "$NUM_GPUS" \
        --mixed_precision "$MIXED_PRECISION" \
        --dynamo_backend "no" \
        "$@"
}

run_phase1() {
    log "=== PHASE 1: World Model ==="
    local logdir="${LOGDIR_BASE}/phase1"
    mkdir -p "$logdir"
    local ckpt
    ckpt=$(get_checkpoint "$logdir" || true)
    local args=()
    [[ -n "$ckpt" ]] && args=("checkpoint=$ckpt")

    acc_launch train.py phase=1 logdir="$logdir" model="$MODEL_SIZE" \
        trainer.steps="$STEPS_PHASE1" use_depth=True "${args[@]}" "${EXTRA_ARGS[@]}"
}

run_phase2() {
    log "=== PHASE 2: BC + Safety ==="
    local logdir="${LOGDIR_BASE}/phase2"
    mkdir -p "$logdir"
    local ckpt
    ckpt=$(get_checkpoint "$logdir" || true)
    local args=()
    if [[ -n "$ckpt" ]]; then
        args=("checkpoint=$ckpt")
    else
        local p1_ckpt
        p1_ckpt=$(get_checkpoint "${LOGDIR_BASE}/phase1" || true)
        [[ -z "$p1_ckpt" ]] && { log "FEHLER: Phase 1 fehlt!"; exit 1; }
        args=("checkpoint=$p1_ckpt")
    fi

    acc_launch train.py phase=2 logdir="$logdir" model="$MODEL_SIZE" \
        trainer.steps="$STEPS_PHASE2" use_depth=True "${args[@]}" "${EXTRA_ARGS[@]}"
}

run_phase3() {
    log "=== PHASE 3: Online RL ==="
    local logdir="${LOGDIR_BASE}/phase3"
    mkdir -p "$logdir"
    local colosseum="false"
    if python3 -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1', 41451)); s.close()" 2>/dev/null; then
        colosseum="true"
        log "Simulator erkannt."
    fi

    local ckpt
    ckpt=$(get_checkpoint "$logdir" || true)
    local args=()
    if [[ -n "$ckpt" ]]; then
        args=("checkpoint=$ckpt")
    else
        local p2_ckpt
        p2_ckpt=$(get_checkpoint "${LOGDIR_BASE}/phase2" || true)
        [[ -z "$p2_ckpt" ]] && { log "FEHLER: Phase 2 fehlt!"; exit 1; }
        args=("checkpoint=$p2_ckpt")
    fi

    acc_launch train.py phase=3 logdir="$logdir" model="$MODEL_SIZE" \
        trainer.steps="$STEPS_PHASE3" use_depth=True env.colosseum.enabled="$colosseum" \
        "${args[@]}" "${EXTRA_ARGS[@]}"
}

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 [phase1|phase2|phase3|all] [Extra Args]"
    exit 1
fi

PHASE="$1"
shift
EXTRA_ARGS=("$@")

init_env

case "$PHASE" in
    phase1) run_phase1 ;;
    phase2) run_phase2 ;;
    phase3) run_phase3 ;;
    all) run_phase1; run_phase2; run_phase3 ;;
    *) echo "Unbekannt: $PHASE"; exit 1 ;;
esac

log "Pipeline fertig."
