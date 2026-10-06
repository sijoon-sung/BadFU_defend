#!/bin/bash
# 사용자의 exp_lora_badfu_grid.py 그리드가 끝나고 RAM 이 넉넉해지면
# regen_and_detect.sh 를 자동 실행한다. (폴링, GPU/RAM 거의 안 씀)
cd "$(dirname "$0")"
ROOT="$(pwd)"
WLOG="$ROOT/logs/watch.log"
mkdir -p "$ROOT/logs"
echo "==== watcher start $(date) ====" | tee -a "$WLOG"

grid_count() {
  powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*exp_lora_badfu_grid*' }).Count" 2>/dev/null | tr -d '[:space:]'
}
free_gb() {
  powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)" 2>/dev/null | tr -d '[:space:]'
}

MAX_WAIT=$((12*3600))   # 최대 12시간
WAITED=0
while true; do
  GC=$(grid_count); FG=$(free_gb)
  echo "$(date +%H:%M:%S) grid=$GC free_gb=$FG waited=${WAITED}s" | tee -a "$WLOG"
  # 그리드가 사라지고(0) RAM 4GB 이상이면 실행
  if [ "${GC:-1}" = "0" ]; then
    # RAM 비교(소수점 → 정수 비교 위해 awk)
    if awk "BEGIN{exit !(${FG:-0} >= 4.0)}"; then
      echo "$(date +%H:%M:%S) grid done & RAM ok -> launching run" | tee -a "$WLOG"
      break
    fi
  fi
  if [ "$WAITED" -ge "$MAX_WAIT" ]; then
    echo "$(date +%H:%M:%S) MAX_WAIT reached, launching anyway" | tee -a "$WLOG"
    break
  fi
  sleep 120; WAITED=$((WAITED+120))
done

bash "$ROOT/regen_and_detect.sh"
echo "==== watcher done $(date) ====" | tee -a "$WLOG"
