#!/bin/bash
#
# Copyright (c) 2020, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

if [ "$0" != "/bin/bash" ]; then
    SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
fi

################################################################################
# Utility functions
################################################################################

c_str() {
    local old_color=39
    local old_attr=0
    local color=39
    local attr=0
    local text=""
    local no_change=0
    for i in "$@"; do
        case "$i" in
            r|R)
                color=31
                ;;
            g|G)
                color=32
                ;;
            y|Y)
                color=33
                ;;
            b|B)
                color=34
                ;;
            p|P)
                color=35
                ;;
            c|C)
                color=36
                ;;
            w|W)
                color=37
                ;;

            z|Z)
                color=0
                ;;
        esac
        case "$i" in
            l|L|R|G|Y|B|P|C|W)
                attr=1
                ;;
            n|N|r|g|y|b|p|c|w)
                attr=0
                ;;
            z|Z)
                attr=0
                ;;
            *)
                text="${text}$i"
        esac
        if [ ${old_color} -ne ${color} ] || [ ${old_attr} -ne ${attr} ]; then
            text="${text}\033[${attr};${color}m"
            old_color=$color
            old_attr=$attr
        fi
    done
    /bin/echo -en "$text"
}

c_echo() {
    local old_opt="$(shopt -op xtrace)" # save old xtrace option
    set +x # unset xtrace
    local text="$(c_str "$@")"
    /bin/echo -e "$text\033[0m"
    eval "${old_opt}" # restore old xtrace option
}


echo_err() {
    >&2 echo "$@"
}

c_echo_err() {
    >&2 c_echo "$@"
}

printf_err() {
    >&2 printf "$@"
}

get_item_ranges() {
  local indexes="$1"
  local list="$2"
  echo -n "$(echo "${list}" | xargs | cut -d " " -f "${indexes}")"
  return $?
}

get_unused_ports() {
    local num_of_ports=${1:-1}
    comm -23 \
    <(seq 49152 61000 | sort) \
    <(ss -tan | awk '{print $4}' | while read line; do echo ${line##*\:}; done | grep '[0-9]\{1,5\}' | sort -u) \
    | shuf | tail -n ${num_of_ports} # use tail instead head to avoid broken pipe in VSCode terminal
}

newline() {
    echo
}

info() {
    c_echo W "$(date -u '+%Y-%m-%d %H:%M:%S') [INFO] " Z "$@"
}

error() {
    echo R "$(date -u '+%Y-%m-%d %H:%M:%S') [ERROR] " Z "$@"
}

fatal() {
    echo R "$(date -u '+%Y-%m-%d %H:%M:%S') [FATAL] " Z "$@"
    echo
    if [ -n "${SCRIPT_DIR}" ]; then
        exit 1
    fi
}

run_command() {
  local status=0
  local cmd="$@"

  c_echo B "$(date -u '+%Y-%m-%d %H:%M:%S') \$ " G "${cmd}"

  [ "$(echo -n "$@")" = "" ] && return 1  # return 1 if there is no command available

  eval "$@"
  status=$?

  if [ -n "${cmd_result}" ]; then
    echo "${cmd_result}"
  fi
  unset IFS

  return $status
}

retry() {
  local retries=$1
  shift

  local count=0
  until run_command "$@"; do
    exit=$?
    wait=$((2 ** $count))
    count=$(($count + 1))
    if [ $count -lt $retries ]; then
      info "Retry $count/$retries. Exit code=$exit, Retrying in $wait seconds..."
      sleep $wait
    else
      fatal "Retry $count/$retries. Exit code=$exit, no more retries left."
      return 1
    fi
  done
  return 0
}

parse_args() {
    local OPTIND
    while getopts 'yh' option;
    do
        case "${option}" in
            # a)
            #     VALUE=${OPTARG}
            #     ;;
            y)
                ALWAYS_YES=true;
                ;;
            h)
                print_usage
                exit 1
                ;;
        esac
    done
    shift $((OPTIND-1))

    CMD="$1"
    shift

    ARGS=("$@")
}

print_usage() {
    set +x
    echo_err
    echo_err "USAGE: $0 [command] [arguments]..."
    echo_err ""
    echo_err "Global Arguments"
    echo_err
    echo_err "Command List"

    echo_err
    echo_err "Examples"
}

init_script() {
    TOP=$(git rev-parse --show-toplevel || pwd)
}

main() {
    parse_args "$@"
    local file_type
    case "$CMD" in
        list|ls)
            # list_testdata "${ARGS[@]}"
            ;;
        ''|main)
            print_usage
            ;;
        *)
            if type ${CMD} > /dev/null 2>&1; then
                init_script
                run_command "$CMD" "${ARGS[@]}"
            else
                print_usage
                exit 1
            fi
            ;;
    esac
}

if [ -n "${SCRIPT_DIR}" ]; then
    main "$@"
fi

# CLARA_VERSION=0.7.1-2008.4 ./serverctl get_latest_version_of recipes clara_bootstrap 2> /dev/null
