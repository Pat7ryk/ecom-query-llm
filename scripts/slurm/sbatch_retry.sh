# 被 chain.sh、run_d2.sh、run_d_rest.sh source。
# sbatch_retry <sbatch 参数...>：提交作业并打印作业号。碰到每人同时提交上限（QOSMaxSubmitJobPerUserLimit，
# 例如用户自己另外提交了作业，占了第 2 个名额）就每 2 分钟重试一次；其他错误照常失败（见 DEVLOG P20）。
sbatch_retry() {
  local err out
  err=$(mktemp)
  until out=$(sbatch "$@" 2>"$err"); do
    if ! grep -q QOSMaxSubmitJobPerUserLimit "$err"; then cat "$err" >&2; rm -f "$err"; return 1; fi
    sleep 120
  done
  rm -f "$err"
  echo "$out"
}
