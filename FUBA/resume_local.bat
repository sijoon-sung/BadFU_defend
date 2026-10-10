@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
REM =====================================================================
REM  중단된 로컬(GPU) 실험을 이어서 돌린다. 끝난 단계(출력 JSON 있음)는 건너뛴다.
REM  Ctrl+C 로 멈추고, 다시 실행하면 그 자리부터 이어진다.
REM
REM  전제: conda 환경 %ENV% (torch+CUDA, timm, mpi4py, impi-rt), Git Bash 설치,
REM        BadFU 궤적 ..\BadFU_src\BadFU-main\logs\traj_full\round_0..14.pt (없으면 [1][2] 건너뜀)
REM  실행: FUBA 폴더에서 resume_local.bat   (더블클릭도 됨)
REM        명령만 보고 싶으면:  set DRY=1 ^& resume_local.bat
REM  로그: logs\*.log,  결과: logs\release_map, logs\progressive_fu, logs\bench
REM =====================================================================
set ENV=C:\Users\sijoo\.conda\envs\pytorch
set PATH=%ENV%;%ENV%\Scripts;%ENV%\Library\bin;C:\Program Files\Git\bin;%PATH%
set PYTHONIOENCODING=utf-8
set TEMP=%LOCALAPPDATA%\Temp
cd /d "%~dp0"
set BADFU=%~dp0..\BadFU_src\BadFU-main
for %%d in (logs logs\release_map logs\progressive_fu logs\bench checkpoints) do if not exist "%%d" mkdir "%%d"
set PY=python
set SH=bash
if defined DRY (set PY=echo python& set SH=echo bash)

set COUNTS=10545 9647 9641 9649 9668 10545
set BF=--fmt badfu --traj_dir "%BADFU%\logs\traj_full" --data_root "%BADFU%\data" --badfu_record "%BADFU%\record\badnet_dataset" --rounds 15 --requester 5 --attackers 0 --target 0 --probe 500 --counts %COUNTS%

echo === [1/5] BadFU: 증류 보정 v2 (op kd, lr 0.001, 모든 탐색)
if not exist "%BADFU%\logs\traj_full\round_14.pt" (
  echo   BadFU 궤적 없음 - [1][2] 건너뜀. 먼저 LOCAL_SMOKE.md 의 Part B 로 궤적을 만들 것
  goto :badfu_done
)
if exist logs\release_map\loc_badfu_kd2.json (echo   결과 있음 - 건너뜀) else (
  %PY% release_map.py --name loc_badfu_kd2 --ref population --op kd --rhos 0.0 0.2 0.4 0.6 --delta_mode all %BF% > logs\release_map_loc_badfu_kd2.log 2>&1
  echo   exit !errorlevel!
)
echo === [2/5] BadFU: 패치 열거 (grid, 작은 rho)
if exist logs\release_map\loc_badfu_grid.json (echo   결과 있음 - 건너뜀) else (
  %PY% release_map.py --name loc_badfu_grid --ref population --rhos 0.0 0.1 0.2 0.3 --cap 0.15 --delta_mode grid %BF% > logs\release_map_loc_badfu_grid.log 2>&1
  echo   exit !errorlevel!
)
:badfu_done

echo === [3/5] FUBA: 시드 1, 2 궤적 생성 + 진행형 FU + 빼 보기
call :fuba_seed 1
call :fuba_seed 2

echo === [4/5] 비용 벤치 (LeNet, K=8 20 50 100)
if exist logs\bench\release_map_fuba.json (echo   결과 있음 - 건너뜀) else (
  %PY% bench_release_map.py --fmt fuba --Ks 8 20 50 100 --rounds 4 > logs\bench_fuba.log 2>&1
  echo   exit !errorlevel!
)
echo === [5/5] 비용 벤치 (ResNet-18 합성, K=6 20 50)
if exist logs\bench\release_map_badfu_synth.json (echo   결과 있음 - 건너뜀) else (
  %PY% bench_release_map.py --fmt badfu_synth --Ks 6 20 50 --rounds 3 --configs cheap clean_only > logs\bench_badfu.log 2>&1
  echo   exit !errorlevel!
)
echo === 끝. 결과 올리기:
echo   git add FUBA/logs/release_map FUBA/logs/progressive_fu FUBA/logs/bench ^&^& git commit -m "local results" ^&^& git push
if not defined DRY pause
exit /b 0

REM ---------------------------------------------------------------------
REM  :fuba_seed N  →  loc_fuba_sN (seed 522+N): 궤적(8라운드) 없으면 생성, 이어서 두 분석
REM ---------------------------------------------------------------------
:fuba_seed
set N=loc_fuba_s%1
set /a SEED=522+%1
set CNT=0
for %%f in (checkpoints\local_net_model_%N%_round_*.pkl) do set /a CNT+=1
if !CNT! LSS 8 (
  echo   [%N%] 궤적 !CNT!/8 - 생성 - FUBA mpiexec, seed !SEED!
  set NB_CLIENTS=8
  set ROUNDS=8
  set WARMUP=3
  set EXTRA_ARGS=--split_seed !SEED!
  %SH% run_real_fuba.sh %N% !SEED! > logs\pipeline_%N%_attack.log 2>&1
  echo   [%N%] attack exit !errorlevel!
) else (echo   [%N%] 궤적 있음)
set CNT=0
for %%f in (checkpoints\local_net_model_%N%_round_*.pkl) do set /a CNT+=1
if !CNT! LSS 8 if not defined DRY (
  echo   [%N%] 궤적 생성 실패 - logs\pipeline_%N%_attack.log 확인
  exit /b 1
)
if exist logs\progressive_fu\%N%.json (echo   [%N%] 진행형 FU 결과 있음) else (
  %PY% progressive_fu.py --fmt fuba --name %N% --K 8 --rounds 8 --seed !SEED! --split_seed !SEED! --requesters 4 7 5 --out logs\progressive_fu\%N%.json > logs\progressive_fu_%N%.log 2>&1
  echo   [%N%] progressive exit !errorlevel!
)
if exist logs\release_map\%N%.json (echo   [%N%] 빼 보기 결과 있음) else (
  %PY% release_map.py --fmt fuba --name %N% --ref population --K 8 --rounds 8 --seed !SEED! --requester 4 --out logs\release_map\%N%.json > logs\release_map_%N%.log 2>&1
  echo   [%N%] release_map exit !errorlevel!
)
exit /b 0
