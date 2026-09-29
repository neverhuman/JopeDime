#!/usr/bin/env bash
# Parse settings as data, never source/eval the env file.
load_probe_env() {
  local env_file="$1" key value line
  [[ -f "$env_file" ]] || { echo 'Required env file missing' >&2; return 1; }
  [[ "$(stat -c '%a' "$env_file")" == 600 ]] || { echo 'Env file must have mode 0600' >&2; return 1; }
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    [[ "$line" == *=* ]] || { echo 'Invalid env syntax' >&2; return 1; }
    key="${line%%=*}"; value="${line#*=}"
    case "$key" in
      AWS_PROFILE|AWS_REGION|AWS_ACCOUNT_ID|FUNCTION_NAME|RUN_DIR) ;;
      *) echo 'Unknown or credential-bearing env field refused' >&2; return 1 ;;
    esac
    [[ "$value" =~ ^[a-zA-Z0-9_./:-]+$ ]] || { echo 'Invalid env value' >&2; return 1; }
    [[ ! -v "SEEN_${key}" ]] || { echo 'Duplicate env field' >&2; return 1; }
    printf -v "$key" '%s' "$value"
    printf -v "SEEN_${key}" '%s' 1
  done < "$env_file"
  for key in AWS_PROFILE AWS_REGION AWS_ACCOUNT_ID FUNCTION_NAME RUN_DIR; do
    [[ -v "SEEN_${key}" ]] || { echo 'Required env setting absent' >&2; return 1; }
  done
  [[ "$AWS_REGION" == us-east-2 && "$AWS_ACCOUNT_ID" == 714044927654 ]] || return 1
  [[ "$FUNCTION_NAME" == jope-lambda-csv-20260928-rust ]] || return 1
  [[ "$RUN_DIR" == /home/ubuntu/scratch/jopedime-lambda-csv-20260928/rust-run || "$RUN_DIR" == /home/ubuntu/scratch/jopedime-lambda-csv-20260928/rust-run-repair ]] || return 1
  # Explicit profile selection must not be overridden by inherited static credentials.
  unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_SECURITY_TOKEN
  export AWS_PROFILE AWS_REGION AWS_DEFAULT_REGION="$AWS_REGION" AWS_PAGER=''
}
