# GLR-GLALR-Parser-Generator

GLR(1) - GLALR(1) parser generator.

## `parser_generator_lib`
- `analyze_lexical(lexical_fsm, keywords, code) -> tokens_and_lexemes`
- `analyze_syntax(initial_nonterminal, lr_table, tokens_and_lexemes, tokens_to_skip) -> history`
- `lr_table` ← `ebnf_to_lr_table` = `convert_to_bnf` + `construct_first_sets` + `construct_lr_closure_table` + `construct_lr_table`

You can build lr table from the extended Backus-Naur form (`ebnf_to_lr_table`).

## C90 parser example

C90 syntax is from Kernighan, Brian; Ritchie, Dennis M. (March 1988). The C Programming Language (2nd ed.). Englewood Cliffs, NJ: Prentice Hall. ISBN 0-13-110362-8.

```commandline
python run_example.py
```

Or more customizable:

```commandline
python c90_parser.py --paths c90_examples/not_exist.c c90_examples/normal.c c90_examples/lexical_error.c c90_examples/syntax_error.c --glalr -v
```

Output:

```
C:\Users\quint\PycharmProjects\GLR-GLALR-Parser-Generator\c90_examples\not_exist.c [Errno 2] No such file or directory: 'C:\\Users\\quint\\PycharmProjects\\GLR-GLALR-Parser-Generator\\c90_examples\\not_exist.c'
====================================================
===================== normal.c =====================
Lexical Analysis: Success
Syntax Analysis: Success

====================================================
================= lexical_error.c ==================
Lexical Analysis: [code_startswith: 'привет' | symbol: 'п' | line: 5]
====================== Error =======================
====================================================

====================================================
================== syntax_error.c ==================
Lexical Analysis: Success
Syntax Analysis: [message: "Unexpected lexeme }. Expected any of: {';', ':', ',', ')', ']'}" | position: '30:0']
====================== Error =======================
====================================================
```

---

```
usage: c90_parser [-h] -p PATHS [PATHS ...] (--glr | --glalr) [--l-history]
                  [--p-history] [-v] [--debug-args]

C90 Parser

options:
  -h, --help            show this help message and exit
  -p PATHS [PATHS ...], --paths PATHS [PATHS ...]
                        paths to dirs/files [example: `/path/to/dir /path/to/file`]
  --glr                 use GLR(1) parser generator
  --glalr               use GLALR(1) parser generator
  --l-history           show lexical history
  --p-history           show syntax parsing history
  -v, --verbose         verbose mode
  --debug-args          debug mode
```

