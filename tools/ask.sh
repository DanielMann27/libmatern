# ask.sh -- questions on a terminal for the benchmark scripts (POSIX sh, sourced:
# `. ./tools/ask.sh`).  A question is asked only when stdin and stdout are terminals
# and BENCH_UNATTENDED is unset; otherwise every question counts as answered no.

# ask_tty: true when a question can be asked.
ask_tty() {
    [ -z "${BENCH_UNATTENDED:-}" ] && [ -t 0 ] && [ -t 1 ]
}

# ask_yn <question>: 0 for yes, 1 for no (the default, also on end of input or
# without a terminal); another answer repeats the question, at most three times in all.
ask_yn() {
    ask_tty || return 1
    _ask_n=0
    while [ "$_ask_n" -lt 3 ]; do
        printf '%s [y/N] ' "$1"
        read -r _ask_a || { echo; return 1; }
        case "$_ask_a" in
            y|Y|yes|Yes|YES) return 0 ;;
            ""|n|N|no|No|NO) return 1 ;;
        esac
        echo "  answer y or n"
        _ask_n=$((_ask_n + 1))
    done
    return 1
}

# ask_box: stdin framed in a box of '#', 78 columns, with an empty line before and
# after.  Text is wrapped at 72 columns; ASCII only (the padding counts bytes).
ask_box() {
    _ask_rule='##############################################################################'
    echo
    echo "$_ask_rule"
    printf '#%76s#\n' ''
    fold -s -w 72 | while IFS= read -r _ask_l; do
        printf '#   %-72s #\n' "$_ask_l"
    done
    printf '#%76s#\n' ''
    echo "$_ask_rule"
    echo
}
