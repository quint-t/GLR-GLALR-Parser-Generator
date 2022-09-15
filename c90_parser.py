import argparse
import os
import re
import string
import sys
import textwrap
import traceback
from typing import Union, Tuple, List, Set, Dict, Pattern, TypedDict

from parser_generator_lib import ULexicalError, USyntaxError
from parser_generator_lib import convert_to_lexical_fsm, analyze_lexical, analyze_syntax, ebnf_to_lr_table


class C90Language:
    """Класс языка C90: лексика и синтаксис"""

    # ========================================== Для лексического анализа

    # поля, описывающие шаблоны лексем токенов в виде регулярных выражений
    single_line_comment = re.compile(r'//.*')
    multi_line_comment = re.compile(r'/\*.*?\*/', re.DOTALL)
    character = re.compile(r"L?\'(?:[^\'\\\n]|\\['\"?aAbBfFnNrRtTvV]?|\\[0-7]{1,3}|\\[xX][0-9a-fA-F]+|\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})(?<!\\)\'",
                           re.DOTALL)
    string = re.compile(r'L?\".*?(?<!\\)\"', re.DOTALL)
    integer = re.compile(r"(?:[1-9][0-9]*|0[0-7]*|0X[0-9A-F]+|0B[01]+)U?L{,2}(?![0-9A-Z.])",
                         re.IGNORECASE)
    float = re.compile(r"(?:(?:\d*\.\d+|\d+\.)(?:E[+-]?\d+)?|\d+E[+-]?\d+)[FL]?(?![0-9A-Z.])",
                       re.IGNORECASE)
    space = re.compile(r"\s+")
    identifier = re.compile(r'[A-Z_][A-Z0-9_]*', re.IGNORECASE)
    preprocessor_directive = re.compile(r"(#[^\n]*(?:\\\n[^\n]*)*?[^\\](?=\n))", re.DOTALL)
    # обычно обрабатываются препроцессором до лексического анализа
    # будем пропускать директивы препроцессора

    keywords = {  # множество ключевых слов языка
        'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do', 'double', 'else', 'enum',
        'extern', 'float', 'for', 'goto', 'if', 'int', 'long', 'register', 'return', 'short', 'signed', 'sizeof',
        'static', 'struct', 'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile', 'while'
    }

    @staticmethod
    def get_lexical_fsm() -> Dict[str,
                                  Union[str,
                                        dict,
                                        Tuple[str, Pattern],
                                        List[Tuple[str, Pattern]]]]:
        """Возвращает лексику языка в виде конечного автомата"""
        # ----- ручное задание конечного автомата
        # fsm = {'!': {'': '<logical-not>', '=': '<not-equals>'},
        #        '"': ('<string>', C90Language.string),
        #        '#': ('<preprocessor-directive>', C90Language.preprocessor_directive),
        #        '%': {'': '<modulus>', '=': '<modulus-equals>'},
        #        '&': {'': '<binary-and>', '&': '<logical-and>', '=': '<binary-and-equals>'},
        #        "'": ('<character-constant>', C90Language.character),
        #        '(': '<open-bracket>', ')': '<close-bracket>',
        #        '*': {'': '<multiply-or-pointer>', '=': '<mul-equals>'},
        #        '+': {'': '<plus>', '+': '<increment>', '=': '<plus-equals>'},
        #        ',': '<comma>',
        #        '-': {'': '<minus>',
        #              '-': '<decrement>',
        #              '=': '<minus-equals>',
        #              '>': '<pointer-access>'},
        #        '.': {'': '<dot>', '.': {'.': '<ellipsis>'}},
        #        '/': {'': '<divide>',
        #              '*': ('<multi-line-comment>', C90Language.multi_line_comment),
        #              '/': ('<single-line-comment>', C90Language.single_line_comment),
        #              '=': '<div-equals>'},
        #        ':': '<colon>', ';': '<semicolon>',
        #        '<': {'': '<less-than>', '<': '<left-shift>', '=': '<less-than-or-equal-to>'},
        #        '=': {'': '<assignment>', '=': '<equal-to>'},
        #        '>': {'': '<greater-than>',
        #              '=': '<greater-than-or-equal-to>',
        #              '>': '<right-shift>'},
        #        'L': {'"': ('<string>', C90Language.string),
        #              "'": ('<character-constant>', C90Language.character)},
        #        '[': '<open-square-bracket>', ']': '<close-square-bracket>',
        #        '^': {'': '<binary-xor>', '=': '<binary-xor-equals>'},
        #        '{': '<open-curly-bracket>', '}': '<close-curly-bracket>',
        #        '|': {'': '<binary-or>', '=': '<binary-or-equals>', '|': '<logical-or>'},
        #        '~': '<tilde>'}
        # for digit in string.digits:
        #     fsm[digit] = [('<integer-constant>', C90Language.integer), ('<floating-constant>', C90Language.float)]
        # for letter in string.ascii_letters + '_':
        #     fsm[letter] = ('<identifier>', C90Language.identifier)
        # for space in string.whitespace:
        #     fsm[space] = ('<whitespace>', C90Language.space)
        # fsm['L'] = {'"': ('<string>', C90Language.string),
        #             "'": ('<character-constant>', C90Language.character),
        #             '': fsm['L']}
        # return fsm

        # ----- автоматическое задание конечного автомата
        # (результат работы аналогичен закомментированному ручному заданию конечного автомата)
        fsm = {
            ':': '<colon>', ';': '<semicolon>', ',': '<comma>', '~': '<tilde>',
            '(': '<open-bracket>', ')': '<close-bracket>',
            '[': '<open-square-bracket>', ']': '<close-square-bracket>',
            '{': '<open-curly-bracket>', '}': '<close-curly-bracket>',
            '>>': '<right-shift>', '>=': '<greater-than-or-equal-to>', '>': '<greater-than>',
            '<<': '<left-shift>', '<=': '<less-than-or-equal-to>', '<': '<less-than>',
            '--': '<decrement>', '-=': '<minus-equals>', '->': '<pointer-access>', '-': '<minus>',
            '++': '<increment>', '+=': '<plus-equals>', '+': '<plus>',
            '.': '<dot>', '...': '<ellipsis>',
            '&&': '<logical-and>', '&=': '<binary-and-equals>', '&': '<binary-and>',
            '||': '<logical-or>', '|=': '<binary-or-equals>', '|': '<binary-or>',
            '^=': '<binary-xor-equals>', '^': '<binary-xor>',
            '*=': '<mul-equals>', '*': '<multiply-or-pointer>',
            '/*': ('<multi-line-comment>', C90Language.multi_line_comment),
            '//': ('<single-line-comment>', C90Language.single_line_comment),
            '/=': '<div-equals>', '/': '<divide>',
            '%=': '<modulus-equals>', '%': '<modulus>',
            '==': '<equal-to>', '=': '<assignment>',
            '!=': '<not-equals>', '!': '<logical-not>',
            '#': ('<preprocessor-directive>', C90Language.preprocessor_directive),
            "'": ('<character-constant>', C90Language.character),
            '"': ('<string>', C90Language.string),
            "L'": ('<character-constant>', C90Language.character),
            'L"': ('<string>', C90Language.string)
        }
        for digit in string.digits:
            fsm[digit] = [('<integer-constant>', C90Language.integer), ('<floating-constant>', C90Language.float)]
        for letter in string.ascii_letters + '_':
            fsm[letter] = ('<identifier>', C90Language.identifier)
        for space in string.whitespace:
            fsm[space] = ('<whitespace>', C90Language.space)
        # ----- преобразование структуры в конечный автомат
        return convert_to_lexical_fsm(fsm)

    @staticmethod
    def get_keywords() -> Set[str]:
        """Возвращает множество ключевых слов языка"""
        return set(C90Language.keywords)

    # ========================================== Для синтаксического анализа

    tokens_to_skip = {'<multi-line-comment>', '<whitespace>',  # множество пропускаемых токенов
                      '<single-line-comment>', '<preprocessor-directive>'}

    @staticmethod
    def get_syntax_ebnf() -> (str, Dict[str, Union[str,
                                                   Set[Tuple[str, str]],
                                                   Tuple[Union[str, Set[Tuple[str, str]]]],
                                                   List[Union[str,
                                                              Set[Tuple[str, str]],
                                                              Tuple[Union[str, Set[Tuple[str, str]]]]]]]]):
        """
        Возвращает начальный нетерминал и синтаксис языка в виде расширенной формы Бэкуса-Наура:
            str -- обязательное вхождение (ровно 1 раз)
            {(str, ???)} -- повторение str:
                {(str, '?')} -- 0 или 1 раз
                {(str, '+')} -- от 1 раза и более
                {(str, '*')} -- от 0 раз и более
            (e1, e2, ..., en) -- конкатенация элементов (ei -- либо обязательное вхождение, либо повторение)
            [v1, v2, ..., vn] -- выбор одного из n вариантов (vi -- либо обязательное вхождение,
                                                              либо повторение, либо конкатенация)
        Синтаксис взят из приложения A13 в книге
        "Kernighan, Brian; Ritchie, Dennis M. (March 1988).
         The C Programming Language (2nd ed.). Englewood Cliffs, NJ: Prentice Hall. ISBN 0-13-110362-8."
        """
        initial_nonterminal = '<translation-unit>'
        syntax_ebnf = {
            '<translation-unit>': {('<external-declaration>', '+')},
            '<external-declaration>': ['<function-definition>', '<declaration>'],
            '<function-definition>': ({('<declaration-specifier>', '*')}, '<declarator>',
                                      {('<declaration>', '*')}, '<compound-statement>'),
            '<declaration>': ({('<declaration-specifier>', '+')}, {('<init-declarator-list>', '?')}, ';'),
            '<declaration-specifier>': ['<storage-class-specifier>', '<type-specifier>', '<type-qualifier>'],
            '<storage-class-specifier>': ['auto', 'register', 'static', 'extern', 'typedef'],
            '<type-specifier>': ['void', 'char', 'short', 'int', 'long', 'float', 'double', 'signed', 'unsigned',
                                 '<struct-or-union-specifier>', '<enum-specifier>', '<typedef-name>'],
            '<typedef-name>': '<identifier>',
            '<type-qualifier>': ['const', 'volatile'],
            '<struct-or-union-specifier>': [
                ('<struct-or-union>', {('<identifier>', '?')}, '{', {('<struct-declaration>', '+')}, '}'),
                ('<struct-or-union>', '<identifier>')
            ],
            '<struct-or-union>': ['struct', 'union'],
            '<init-declarator-list>': ['<init-declarator>',
                                       ('<init-declarator-list>', ',', '<init-declarator>')],
            '<init-declarator>': ['<declarator>',
                                  ('<declarator>', '=', '<initializer>')],
            '<struct-declaration>': ({('<specifier-qualifier>', '+')}, '<struct-declarator-list>', ';'),
            '<specifier-qualifier>': ['<type-specifier>', '<type-qualifier>'],
            '<struct-declarator-list>': ['<struct-declarator>',
                                         ('<struct-declarator-list>', ',', '<struct-declarator>')],
            '<struct-declarator>': ['<declarator>',
                                    ({('<declarator>', '?')}, ':', '<constant-expression>')],
            '<enum-specifier>': [('enum', {('<identifier>', '?')}, '{', '<enumerator-list>', '}'),
                                 ('enum', '<identifier>')],
            '<enumerator-list>': ['<enumerator>',
                                  ('<enumerator-list>', ',', '<enumerator>')],
            '<enumerator>': ['<identifier>',
                             ('<identifier>', '=', '<constant-expression>')],
            '<declarator>': ({('<pointer>', '?')}, '<direct-declarator>'),
            '<direct-declarator>': ['<identifier>',
                                    ('(', '<declarator>', ')'),
                                    ('<direct-declarator>', '[', {('<constant-expression>', '?')}, ']'),
                                    ('<direct-declarator>', '(', '<parameter-type-list>', ')'),
                                    ('<direct-declarator>', '(', {('<identifier>', '*')}, ')')],
            '<pointer>': ('*', {('<type-qualifier>', '*')}, {('<pointer>', '?')}),
            '<parameter-type-list>': ['<parameter-list>',
                                      ('<parameter-list>', ',', '...')],
            '<parameter-list>': ['<parameter-declaration>',
                                 ('<parameter-list>', ',', '<parameter-declaration>')],
            '<parameter-declaration>': [
                ({('<declaration-specifier>', '+')}, '<declarator>'),
                ({('<declaration-specifier>', '+')}, {('<abstract-declarator>', '?')}),
            ],
            '<initializer>': ['<assignment-expression>',
                              ('{', '<initializer-list>', {(',', '?')}, '}')],
            '<initializer-list>': ['<initializer>',
                                   ('<initializer-list>', ',', '<initializer>')],
            '<type-name>': ({('<specifier-qualifier>', '+')}, {('<abstract-declarator>', '?')}),
            '<abstract-declarator>': ['<pointer>',
                                      ({('<pointer>', '?')}, '<direct-abstract-declarator>')],
            '<direct-abstract-declarator>': [('(', '<abstract-declarator>', ')'),
                                             ({('<direct-abstract-declarator>', '?')},
                                              '[', {('<constant-expression>', '?')}, ']'),
                                             ({('<direct-abstract-declarator>', '?')},
                                              '(', {('<parameter-type-list>', '?')}, ')')],
            '<statement>': ['<labeled-statement>', '<expression-statement>', '<compound-statement>',
                            '<selection-statement>', '<iteration-statement>', '<jump-statement>'],
            '<labeled-statement>': [('<identifier>', ':', '<statement>'),
                                    ('case', '<constant-expression>', ':', '<statement>'),
                                    ('default', ':', '<statement>')],
            '<expression-statement>': ({('<expression>', '?')}, ';'),
            '<compound-statement>': ('{', {('<declaration>', '*')}, {('<statement>', '*')}, '}'),
            '<selection-statement>': [('if', '(', '<expression>', ')', '<statement>'),
                                      ('if', '(', '<expression>', ')', '<statement>', 'else', '<statement>'),
                                      ('switch', '(', '<expression>', ')', '<statement>')],
            '<iteration-statement>': [('while', '(', '<expression>', ')', '<statement>'),
                                      ('do', '<statement>', 'while', '(', '<expression>', ')', ';'),
                                      ('for', '(', {('<expression>', '?')}, ';', {('<expression>', '?')}, ';',
                                       {('<expression>', '?')}, ')', '<statement>')],
            '<jump-statement>': [('goto', '<identifier>', ';'),
                                 ('continue', ';'),
                                 ('break', ';'),
                                 ('return', {('<expression>', '?')}, ';')],
            '<expression>': ['<assignment-expression>',
                             ('<expression>', ',', '<assignment-expression>')],
            '<assignment-expression>': ['<conditional-expression>',
                                        ('<unary-expression>', '<assignment-operator>', '<assignment-expression>')],
            '<assignment-operator>': ['=', '*=', '/=', '%=', '+=', '-=', '<<=', '>>=', '&=', '^=', '|='],
            '<conditional-expression>': ['<logical-or-expression>',
                                         ('<logical-or-expression>',
                                          '?', '<expression>',
                                          ':', '<conditional-expression>')],
            '<constant-expression>': '<conditional-expression>',
            '<logical-or-expression>': ['<logical-and-expression>',
                                        ('<logical-or-expression>', '||', '<logical-and-expression>')],
            '<logical-and-expression>': ['<inclusive-or-expression>',
                                         ('<logical-and-expression>', '&&', '<inclusive-or-expression>')],
            '<inclusive-or-expression>': ['<exclusive-or-expression>',
                                          ('<inclusive-or-expression>', '|', '<exclusive-or-expression>')],
            '<exclusive-or-expression>': ['<and-expression>',
                                          ('<exclusive-or-expression>', '^', '<and-expression>')],
            '<and-expression>': ['<equality-expression>',
                                 ('<and-expression>', '&', '<equality-expression>')],
            '<equality-expression>': ['<relational-expression>',
                                      ('<equality-expression>', '==', '<relational-expression>'),
                                      ('<equality-expression>', '!=', '<relational-expression>')],
            '<relational-expression>': ['<shift-expression>',
                                        ('<relational-expression>', '<', '<shift-expression>'),
                                        ('<relational-expression>', '>', '<shift-expression>'),
                                        ('<relational-expression>', '<=', '<shift-expression>'),
                                        ('<relational-expression>', '>=', '<shift-expression>')],
            '<shift-expression>': ['<additive-expression>',
                                   ('<shift-expression>', '<<', '<additive-expression>'),
                                   ('<shift-expression>', '>>', '<additive-expression>')],
            '<additive-expression>': ['<multiplicative-expression>',
                                      ('<additive-expression>', '+', '<multiplicative-expression>'),
                                      ('<additive-expression>', '-', '<multiplicative-expression>')],
            '<multiplicative-expression>': ['<cast-expression>',
                                            ('<multiplicative-expression>', '*', '<cast-expression>'),
                                            ('<multiplicative-expression>', '/', '<cast-expression>'),
                                            ('<multiplicative-expression>', '%', '<cast-expression>')],
            '<cast-expression>': ['<unary-expression>', ('(', '<type-name>', ')', '<cast-expression>')],
            '<unary-expression>': ['<postfix-expression>',
                                   ('++', '<unary-expression>'),
                                   ('--', '<unary-expression>'),
                                   ('<unary-operator>', '<cast-expression>'),
                                   ('sizeof', '<unary-expression>'),
                                   ('sizeof', '(', '<type-name>', ')')],
            '<unary-operator>': ['&', '*', '+', '-', '~', '!'],
            '<postfix-expression>': ['<primary-expression>',
                                     ('<postfix-expression>', '[', '<expression>', ']'),
                                     ('<postfix-expression>', '(', '<expression>', ')'),
                                     ('<postfix-expression>', '.', '<identifier>'),
                                     ('<postfix-expression>', '->', '<identifier>'),
                                     ('<postfix-expression>', '++'),
                                     ('<postfix-expression>', '--')],
            '<primary-expression>': ['<identifier>', '<constant>', '<string>', ('(', '<expression>', ')')],
            '<constant>': ['<integer-constant>', '<character-constant>', '<floating-constant>'],
        }
        return initial_nonterminal, syntax_ebnf


# ========================================== MAIN ==========================================


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='c90_parser', description="C90 Parser",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-p", "--paths", type=str, default=[], required=True, nargs='+',
                        help="paths to dirs/files [example: `/path/to/dir /path/to/file`]")
    x_group_2 = parser.add_mutually_exclusive_group(required=True)
    x_group_2.add_argument("--glr", default=False, action='store_true', help="use GLR(1) parser generator")
    x_group_2.add_argument("--glalr", default=False, action='store_true', help="use GLALR(1) parser generator")
    parser.add_argument('--l-history', action='store_true', required=False, help='show lexical history')
    parser.add_argument('--p-history', action='store_true', required=False, help='show syntax parsing history')
    parser.add_argument('-v', '--verbose', action='store_true', required=False, help='verbose mode')
    parser.add_argument('--debug-args', action='store_true', required=False, help='debug mode')
    return parser


class ParserArguments(TypedDict):
    paths: List[str]
    glr: bool
    glalr: bool
    l_history: bool
    p_history: bool
    verbose: bool
    debug_args: bool


def parse_arguments(parser: argparse.ArgumentParser, *, verbose_stderr: bool) -> Union[int, ParserArguments]:
    try:
        args = parser.parse_args()
    except:
        return 1
    files = []
    for path in args.paths:
        path = os.path.normpath(path)
        if os.path.isdir(path):
            for dirpath, dirnames, filenames in os.walk(top=path, topdown=True):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    try:
                        with open(file_path, 'r') as fp:
                            fp.seek(0, 2)
                            symbols = fp.tell()
                        if symbols > 0:
                            files.append(file_path)
                        elif verbose_stderr:
                            print(f'File {file_path} is empty, skipped', file=sys.stderr)
                    except:
                        if verbose_stderr:
                            print(f'File {file_path} cannot be open for reading, skipped', file=sys.stderr)
        elif os.path.isfile(path):
            files.append(path)
        elif verbose_stderr:
            print(f'Object {path} is not dir or file, skipped', file=sys.stderr)

    glr = bool(args.glr)
    glalr = bool(args.glalr)
    l_history = bool(args.l_history)
    p_history = bool(args.p_history)
    verbose = bool(args.verbose)
    debug_args = bool(args.debug_args)
    return ParserArguments(paths=files, glr=glr, glalr=glalr, l_history=l_history, p_history=p_history,
                           verbose=verbose, debug_args=debug_args)


def print_lexical_history(tokens_and_lexemes: List[Tuple[str, str]]):
    maximum_left_indent = max(map(len, re.findall(r'\'([\w\-<>]+)\'', str([x[0] for x in tokens_and_lexemes]))))
    format_function = ('{:>' + str(maximum_left_indent) + '} |\'{}\'|').format  # функция для форматирования
    print(*(format_function(token, repr(lexeme)[1:-1]) for token, lexeme in tokens_and_lexemes),
          sep='\n', end='')


def print_parsing_history(history: List[Tuple[str, str, str, str]]):
    """Выводит историю синтаксического анализа"""

    def format_function(i: int, format_strings: List[str],
                        s1: str, s2: str, s3: str, s4: str, wrapper: textwrap.TextWrapper):
        """Форматирует записи истории синтаксического анализа"""
        if s2 != 'skip':
            xs = [x if x == s1 else wrapper.fill(x) for x in (s1, s2, s3, s4) if x]
        else:
            xs = [s1 + f', {s2}']
        return format_strings[len(xs) - 1].format(i, *xs)

    max_n = len(str(len(history)))
    wrapper = textwrap.TextWrapper(width=200, initial_indent=' ' * (max_n + 2),
                                   subsequent_indent=' ' * (max_n + 4), break_on_hyphens=False)
    max_n_s = str(max_n)
    format_strings = ['{:>' + max_n_s + '}. {}' + '\n{}' * i for i in range(4)]
    print(*(format_function(i, format_strings, s1, s2, s3, s4, wrapper)
            for i, (s1, s2, s3, s4) in enumerate(history)), sep='\n', end='')


def main(dict_args):
    def print_center(*, message: str, width: int, is_start=False, end='\n'):
        """Печатает текст в центре согласно общей ширине width"""
        print('=' * width * bool(is_start),
              message.center(width, '='),
              '=' * width * (not bool(is_start)), sep='\n', end=end)

    if isinstance(dict_args, int):
        sys.exit(dict_args)
    files = dict_args['paths']
    parser = 'glalr' if dict_args['glalr'] else 'glr'
    l_history = dict_args['l_history']
    p_history = dict_args['p_history']
    verbose = dict_args['verbose']
    debug_args = dict_args['debug_args']
    if debug_args:
        print(f'parser: {parser}\n'
              f'files: ' + " ".join(f"\"{file}\"" for file in files) + '\n' +
              f'verbose: {verbose}\n'
              f'debug_args: {debug_args}',
              file=sys.stderr)
        sys.exit(0)

    # инициализация
    lexical_fsm = C90Language.get_lexical_fsm()  # лексика языка в виде конечного автомата
    keywords = C90Language.get_keywords()  # множество ключевых слов языка
    initial_nonterminal, syntax_ebnf = C90Language.get_syntax_ebnf()  # синтаксис языка в расширенной форме Бэкуса-Наура
    tokens_to_skip = C90Language.tokens_to_skip  # множество пропускаемых токенов
    maximum_left_indent = max(map(len, re.findall(r'\'([\w\-<>]+)\'', str(lexical_fsm))))  # макс. длина токена
    lr_table = None

    # лексический и синтаксический анализ .c и .h файлов
    c_filename_regex = re.compile(r'^.*\.[ch]$', re.MULTILINE)
    common_prefix = os.path.commonprefix([os.path.dirname(file) for file in files])
    for file in files:
        try:
            if c_filename_regex.match(file):
                with open(file, 'r', encoding='utf-8') as fp:
                    file_content = fp.read()
                    print_center(message=f' {file[len(common_prefix) + 1:]} ', width=maximum_left_indent * 2,
                                 is_start=True, end='')

                    # лексический анализ
                    try:
                        tokens_and_lexemes = analyze_lexical(lexical_fsm, keywords, file_content)
                        print('Lexical Analysis: Success')
                        if l_history:
                            print_lexical_history(tokens_and_lexemes)
                            print(end='\n\n')
                    except ULexicalError as e:
                        print(f'Lexical Analysis: {e}', end='')
                        print_center(message=' Error ', width=maximum_left_indent * 2, is_start=False, end='\n\n')
                        continue
                    except Exception:
                        print(traceback.format_exc(), end='')
                        print_center(message=' Error ', width=maximum_left_indent * 2, is_start=False, end='\n\n')
                        continue

                    # преобразование названий некоторых токенов в лексемы: '<plus>' в '+' и др.
                    tokens_and_lexemes = [(token if any(x.isalnum() or x == '_' for x in lexeme)
                                                    or token in tokens_to_skip else lexeme,
                                           lexeme)
                                          for token, lexeme in tokens_and_lexemes]

                    # синтаксический анализ (работает только по токенам; лексемы используются в информации об ошибке)
                    try:
                        if lr_table is None:  # построение таблицы по первой необходимости
                            initial_nonterminal, lr_table = ebnf_to_lr_table(initial_nonterminal, syntax_ebnf,
                                                                             'lalr' in parser)
                        history = analyze_syntax(initial_nonterminal, lr_table, tokens_and_lexemes, tokens_to_skip)
                        print('Syntax Analysis: Success')
                        if p_history:
                            print_parsing_history(history)
                            print(end='\n\n')
                        else:
                            print()
                    except USyntaxError as e:
                        print(f'Syntax Analysis: {e}', end='')
                        print_center(message=' Error ', width=maximum_left_indent * 2, is_start=False, end='\n\n')
                        continue
                    except Exception:
                        print(traceback.format_exc(), end='')
                        print_center(message=' Error ', width=maximum_left_indent * 2, is_start=False, end='\n\n')
                        continue
        except Exception as e:
            if verbose:
                print(file, e, file=sys.stderr)


if __name__ == "__main__":
    main(parse_arguments(get_parser(), verbose_stderr=True))
