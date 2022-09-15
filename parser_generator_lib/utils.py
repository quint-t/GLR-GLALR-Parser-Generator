import collections
import re
from enum import Enum
from itertools import permutations
from typing import Union, Set, Dict, Tuple, List  # , Deque

from .exceptions import USyntaxEBNFError


class FillType(Enum):
    """
    ONE_PER_LINE: <syntax> ::= 1
                             | 2
                             | 3
    ALL_PER_LINE: <syntax> ::= 1 | 2 | 3 ... \\ без перевода строк
    ALL_PER_LINE_NL: <syntax> ::= 1 | 2 | 3 ...
                                | 999 | 1000 ... \\ с переводом строк если length > max_line_length
    OPTIMAL_PER_LINE: <syntax> ::= 1 | 2 2 | 3 3 3
                                 | 4 4 4 ... \\ минимум строк с максимальным заполнением до max_line_length
    """
    ONE_PER_LINE = 0
    ALL_PER_LINE = 1
    ALL_PER_LINE_NL = 2
    OPTIMAL_PER_LINE = 3


def check_ebnf(initial_nonterminal: str,
               syntax_ebnf: Dict[str, Union[str,
                                            Set[Tuple[str, str]],
                                            Tuple[Union[str, Set[Tuple[str, str]]]],
                                            List[Union[str,
                                                       Set[Tuple[str, str]],
                                                       Tuple[Union[str, Set[Tuple[str, str]]]]]]]]) -> None:
    """
    Проверяет syntax_ebnf -- синтаксис языка в расширенной форме Бэкуса-Наура
    :param initial_nonterminal: начальный нетерминал
    :param syntax_ebnf: синтаксис языка в расширенной форме Бэкуса-Наура
    """
    q = collections.deque([initial_nonterminal])  # очередь
    new_initial_nonterminal = initial_nonterminal + "'"  # новый начальный нетерминал
    while new_initial_nonterminal in syntax_ebnf:  # должен иметь новое уникальное название
        new_initial_nonterminal += "'"
    seen = set()  # множество уже обработанных нетерминалов
    while q:
        nonterminal = q.popleft()  # достаем первый нетерминал из очереди
        if nonterminal not in syntax_ebnf:
            raise USyntaxEBNFError(f'Nonterminal {repr(nonterminal)} missing in syntax')
        rules = syntax_ebnf[nonterminal]
        if not isinstance(nonterminal, str):  # нетерминал должен быть строкой
            raise USyntaxEBNFError(f'Invalid type: {nonterminal} [{type(nonterminal)}] (must be str)')
        if not isinstance(rules, (str, set, tuple, list)):  # правило может быть только str / set / tuple / list
            raise USyntaxEBNFError(f'Invalid type: {rules} [{type(rules)}] (must be str / set / tuple / list)')
        if not isinstance(rules, list):  # если одно правило, то преобразование в список
            rules = [rules]
        for rule in rules:  # для каждого правила из старого списка правил
            if not isinstance(rule, (str, set, tuple)):  # правило должно быть только str / set / tuple
                raise USyntaxEBNFError(f'Invalid type: {rule} [{type(rule)}] (must be str / set / tuple)')
            elif not isinstance(rule, tuple):  # если не конкатенация, то преобразование в конкатенацию
                rule = (rule,)
            for subrule in rule:  # для каждой части правила
                if not isinstance(subrule, (str, set)):  # часть правила должна быть str / set
                    raise USyntaxEBNFError(f'Invalid type: {subrule} [{type(subrule)}] (must be str / set)')
                if isinstance(subrule, set):  # если часть правила -- повторение
                    if len(subrule) != 1:  # длина set должна быть == 1
                        raise USyntaxEBNFError(f'Number of tuples in set must be == 1, got: {subrule}')
                    str_tuple = next(iter(subrule))
                    # set должен содержать один tuple с 2 строками
                    if not isinstance(str_tuple, tuple) or len(str_tuple) != 2 \
                            or not all(isinstance(s, str) for s in str_tuple):
                        raise USyntaxEBNFError(f'Set must contains one tuple with two strings, got: {str_tuple}')
                    s, times = str_tuple
                    if not s:  # если s пуст (epsilon)
                        raise USyntaxEBNFError(f'An empty character is forbidden to repeat, got: {rule}')
                    if times not in '?+*':
                        raise USyntaxEBNFError(f'Second value must be in "?+*", got: {times}')
                    if s in syntax_ebnf and s not in seen:  # если s есть в синтаксисе и еще не был обработан
                        q.append(s)  # добавляем в очередь
                        seen.add(s)  # добавляем в множество уже обработанных нетерминалов
                elif subrule or len(rule) == 1:  # если непустая часть правила или epsilon-правило
                    # если subrule есть в синтаксисе и еще не был обработан
                    if subrule in syntax_ebnf and subrule not in seen:
                        q.append(subrule)  # добавляем в очередь
                        seen.add(subrule)  # добавляем в множество уже обработанных нетерминалов


def ebnf_to_string(initial_nonterminal: str,
                   syntax_ebnf: Dict[str, Union[str,
                                                Set[Tuple[str, str]],
                                                Tuple[Union[str, Set[Tuple[str, str]]]],
                                                List[Union[str,
                                                           Set[Tuple[str, str]],
                                                           Tuple[Union[str, Set[Tuple[str, str]]]]]]]],
                   extern_tokens: Union[List[str], Set[str], Tuple[str]] = None,
                   extern_token_regexp: re.Pattern = None,
                   spaces: int = None,
                   max_line_length: int = 100,
                   fill_type: FillType = FillType.ALL_PER_LINE_NL) -> str:
    """
    Преобразует syntax_ebnf -- синтаксис языка в расширенной форме Бэкуса-Наура -- в строковый формат
    :param initial_nonterminal: начальный нетерминал
    :param syntax_ebnf: синтаксис языка в расширенной форме Бэкуса-Наура
    :param extern_tokens: внешние токены (не описанные в расширенной форме Бэкуса-Наура)
    :param extern_token_regexp: регулярное выражение токенов (например, r"^<.+>$")
    :param spaces: число пробелов после перевода строки (если None, то выравнивается по знаку =)
    :param max_line_length: максимальная длина строки, описывающей нетерминал, для случая описания в виде одной строки
    :param fill_type: тип заполнения
    """
    check_ebnf(initial_nonterminal, syntax_ebnf)
    all_tokens = set(syntax_ebnf.keys()).union(extern_tokens if extern_tokens is not None else set())

    def is_token(token):
        nonlocal extern_token_regexp, all_tokens
        if isinstance(extern_token_regexp, re.Pattern):
            return extern_token_regexp.match(token)
        return token in all_tokens

    strings = []
    seen = set()
    q = collections.deque()
    q.append(initial_nonterminal)
    while q:
        nonterminal = q.popleft()  # достаем первый нетерминал из очереди
        parts = [f"{nonterminal} ::= "]
        sign_index = parts[0].rfind('=') if spaces is None else spaces
        rules = syntax_ebnf[nonterminal]
        if not isinstance(rules, list):  # если одно правило, то преобразование в список
            rules = [rules]
        for rule in rules:  # для каждого правила из старого списка правил
            if not isinstance(rule, tuple):  # если не конкатенация, то преобразование в конкатенацию
                rule = (rule,)
            rule_strings = []
            for subrule in rule:  # для каждой части правила
                if isinstance(subrule, set):  # если часть правила -- повторение
                    str_tuple = next(iter(subrule))
                    s, times = str_tuple
                    rule_strings.append('{' + (s if is_token(s) else repr(s)) + '}' + times)
                    if s in syntax_ebnf and s not in seen:  # если s есть в синтаксисе и еще не был обработан
                        q.append(s)  # добавляем в очередь
                        seen.add(s)  # добавляем в множество уже обработанных нетерминалов
                elif subrule or len(rule) == 1:  # если непустая часть правила или epsilon-правило
                    rule_strings.append(subrule if is_token(subrule) else repr(subrule))
                    # если subrule есть в синтаксисе и еще не был обработан
                    if subrule in syntax_ebnf and subrule not in seen:
                        q.append(subrule)  # добавляем в очередь
                        seen.add(subrule)  # добавляем в множество уже обработанных нетерминалов
            parts.append(' '.join(rule_strings))
        if len(parts) > 1:
            rule = ''
            if fill_type is FillType.ONE_PER_LINE:
                rule = parts[0] + parts[1] + ''.join('\n' + ' ' * sign_index + '| ' + part for part in parts[2:])
            elif fill_type is FillType.ALL_PER_LINE:
                rule = parts[0] + ' | '.join(parts[1:])
            elif fill_type is FillType.ALL_PER_LINE_NL:
                parts = [parts[0] + parts[1], *parts[2:]]
                new_parts = [parts[0]]
                for i in range(1, len(parts)):
                    part = parts[i]
                    if len(new_parts[-1]) + len(part) <= max_line_length:
                        new_parts[-1] += ' | ' + part
                    else:
                        new_parts.append(' ' * sign_index + '| ' + part)
                rule = '\n'.join(new_parts)
            elif fill_type is FillType.OPTIMAL_PER_LINE:
                optimal_n = float('inf')
                optimal_parts = []
                parts_wo_first = parts[1:]
                for p in permutations(parts_wo_first):
                    tmp = [len(parts[0]) + len(p[0])]
                    for i in range(1, len(parts_wo_first)):
                        part_length = len(p[i])
                        if tmp[-1] + part_length <= max_line_length:
                            # length(' | ') == 3
                            tmp[-1] += 3 + part_length
                        else:
                            # length(' ' * sign_index + '| ' + part) == sign_index + 2 + part_length
                            tmp.append(sign_index + 2 + part_length)
                    if optimal_n > len(tmp):
                        optimal_n = len(tmp)
                        optimal_parts = p
                        if optimal_n <= 2:
                            break
                tmp = [len(parts[0]) + len(optimal_parts[0])]
                new_parts = [parts[0] + optimal_parts[0]]
                for i in range(1, len(optimal_parts)):
                    part_length = len(optimal_parts[i])
                    if tmp[-1] + part_length <= max_line_length:
                        # length(' | ') == 3
                        tmp[-1] += 3 + part_length
                        new_parts[-1] += ' | ' + optimal_parts[i]
                    else:
                        # length(' ' * sign_index + '| ' + part) == sign_index + 2 + part_length
                        tmp.append(sign_index + 2 + part_length)
                        new_parts.append(' ' * sign_index + '| ' + optimal_parts[i])
                rule = '\n'.join(new_parts)
            strings.append(rule)
    return '\n'.join(strings)
