import collections
import re
from typing import Union, Dict, Tuple, Pattern, List, Set, Deque

from .exceptions import ULexicalError, ULexicalFSMError


def convert_to_lexical_fsm(fsm: Dict[str, Union[str, Tuple[str, Pattern], List[Tuple[str, Pattern]]]]) -> Dict[
    str, Union[dict, str, Tuple[str, Pattern], List[Tuple[str, Pattern]]]]:
    # ----- алгоритм преобразования структуры в конечный автомат
    # -1- удаляем вложенность
    q: Deque = collections.deque([('', fsm)])  # очередь: (lexeme, token)
    fsm = dict()  # новый конечный автомат
    while q:  # пока очередь не пуста
        lexeme, token = q.popleft()  # получаем из очереди очередной узел
        if isinstance(token, dict):  # если token -- это промежуточный подузел
            for lex, tok in token.items():  # проходим по каждому подузлу промежуточного подузла
                q.append((lexeme + lex, tok))  # добавляем в очередь, пример: ('-' + '=', '<minus-equals>')
        else:  # если token -- это конечный подузел
            if fsm.get(lexeme, token) != token:  # если уже есть запись по fsm[lexeme], не равная token
                raise ULexicalFSMError(f'Invalid: {fsm[lexeme]} != {token} [{fsm}]')
            fsm[lexeme] = token  # если записи по fsm[lexeme] нет или есть, но равна token
    # -2- формируем вложенность
    for lexeme, token in list(fsm.items()):  # проходим по каждому подузлу
        lexeme_len = len(lexeme)  # длина лексемы (например, len('-=') -> 2)
        if lexeme_len > 1:  # если длина лексемы больше 1
            current = fsm  # current указывает на текущий узел
            i = 0  # индекс текущего символа в lexeme
            while i < lexeme_len:  # пока индекс < длины лексемы
                ch = lexeme[i]  # получаем i символ
                if i == lexeme_len - 1:  # если i == последнему индексу
                    # если уже есть запись по current[ch], не равная token
                    if current.get(ch, token) != token:
                        raise ULexicalFSMError(f'Invalid: {current[ch]} != {token} [{current}]')
                    current[ch] = token  # если записи по current[ch] нет или есть, но равна token
                elif ch in current:  # если i < последний индекс и символ лексемы есть в current
                    if isinstance(current[ch], dict):  # если current[ch] -- это промежуточный узел
                        current = current[ch]  # переходим в него
                    else:  # иначе
                        i += 1  # переходим к следующему символу
                        if i == lexeme_len - 1:  # если символов не осталось, записываем как str/tuple
                            current[ch] = {lexeme[i]: token, '': current[ch]}
                        else:  # иначе записываем как dict и переходим в него
                            d = dict()
                            current[ch] = {lexeme[i]: d, '': current[ch]}
                            current = d
                else:  # если i < последний индекс и символ лексемы отсутствует в current
                    if i >= lexeme_len - 1:  # если символов не осталось, записываем как str/tuple
                        current[ch] = token
                    else:  # иначе записываем как dict и переходим в него
                        d = dict()
                        current[ch] = d
                        current = d
                i += 1  # переходим к следующему символу
            fsm.pop(lexeme)  # удаляем соответствие lexeme (длина > 1) -> token
    return fsm


def analyze_lexical(lexical_fsm: Dict[str,
                                      Union[str,
                                            dict,
                                            Tuple[str, Pattern],
                                            List[Tuple[str, Pattern]]]],
                    keywords: Union[List[str], Set[str]],
                    code: str) -> List[Tuple[str, str]]:
    """
    Лексер анализирует код согласно лексике языка и возвращает список токенов и лексем, либо выбрасывает исключение
    :param lexical_fsm: лексика языка в виде конечного автомата
    :param keywords: список/множество ключевых слов языка
    :param code: код для лексического анализа
    :return: список токенов и лексем; например, [..., ('<identifier>', 'Animal'), ...]
    """
    # проверка lexical_fsm -- лексики языка в виде конечного автомата
    q = collections.deque(lexical_fsm.items())  # очередь для проверки
    while q:
        tuple_item = q.popleft()
        if not isinstance(tuple_item, tuple):  # tuple ключ:значение
            raise ULexicalFSMError(f"Invalid type: {tuple_item} [{type(tuple_item)}] (must be tuple)")
        key, value = tuple_item
        if not isinstance(key, str):  # ключ должен быть строкой
            raise ULexicalFSMError(f"Invalid type: {repr(key)} [{type(key)}] (must be str)")
        if len(key) > 1:  # длина ключа должна быть <= 1
            raise ULexicalFSMError(f"Length of string must be <= 1, got {repr(key)} with len = {len(key)}")
        if not isinstance(value, (str, tuple, list, dict)):  # значение может быть только str / tuple / list / dict
            raise ULexicalFSMError(f"Invalid type: {value} [{type(value)}] (must be str / tuple / list / dict)")
        if isinstance(value, (tuple, list)):  # если значение tuple / list
            tuples = [value] if isinstance(value, tuple) else value  # создаем list
            if len(tuples) == 0:  # длина списка должна быть > 0
                raise ULexicalFSMError(f"Number of values in list must be >= 1, got: {tuples}")
            for tuple_item in tuples:  # для каждого кортежа в списке
                if len(tuple_item) != 2:  # длина кортежа должна быть == 2
                    raise ULexicalFSMError(f'Number of values in tuple must be 2, got: {tuple_item}')
                key, value = tuple_item
                if not isinstance(key, str):  # ключ должен быть str
                    raise ULexicalFSMError(f"Invalid type: {repr(key)} [{type(key)}] (must be str)")
                if not isinstance(value, Pattern):  # значение должно быть Pattern [re.compile(...)]
                    raise ULexicalFSMError(f"Invalid type: {repr(value)} [{type(value)}] (must be Pattern)")
        elif isinstance(value, dict):  # в случае словаря -- добавляем его элементы в очередь
            q.extend(value.items())

    # лексический анализ
    tokens_and_lexemes = []  # список токенов и лексем
    current_state = lexical_fsm  # текущее состояние
    p, i, n = 0, 0, len(code)  # предыдущий индекс, текущий индекс и длина code
    token, lexeme = '', ''  # токен и лексема
    line = 1  # текущий номер строки кода
    while i < n:  # пока текущий индекс < длины code
        r = current_state.get(code[i])  # переходим к следующему состоянию конечного автомата
        if r is None:  # если следующего состояния нет, то смотрим состояние "по умолчанию"
            r = current_state.get('')  # пробуем перейти в него
            i -= 1  # был считан один лишний символ (по индексу i), поэтому отнимаем 1
        if r is None:  # если нужной ветки нет, то получена неверная лексема
            i += 1  # увеличиваем на 1, чтобы получить индекс неверной лексемы
            prev_n = code[:i].rfind('\n') + 1  # начало строки
            next_n = code[i:].find('\n') + i  # конец строки
            raise ULexicalError(code_startswith=code[prev_n if prev_n != -1 else i:next_n if next_n != -1 else i + 1],
                                symbol=code[i], line=line)
        if isinstance(r, str):  # если символ-атом
            i += 1  # переход к следующему символу
            token = r
            lexeme = repr(code[p:i])[1:-1]
            p = i  # обновляем предыдущий индекс
            line += code[p:i].count('\n')  # обновляем номер текущей строки
        elif isinstance(r, (tuple, list)):  # если регулярное выражение
            if not isinstance(r, list):
                r = [r]
            for attempt, t in enumerate(r, 1):  # r заведомо не пустой
                token, token_regexp = t
                match = re.match(token_regexp, code[p:])  # поиск по регулярному выражению
                if not match and attempt == len(r):  # если неудачный поиск на последней попытке
                    pos = code[p:].find('\n')
                    startswith = code[p:(p + pos + 1 if pos != -1 else None)]
                    raise ULexicalError(code_startswith=startswith, line=line, regexp=token_regexp)
                elif not match:  # если неудачный поиск на текущей попытке, то переход к следующей
                    continue
                match_len = match.span(0)[1]  # если поиск удачный, получаем число найденных символов
                i = p + match_len  # обновляем текущую позицию
                lexeme = code[p:i]
                p = i  # обновляем предыдущий индекс
                if lexeme in keywords:  # если лексема есть в множестве ключевых слов языка
                    token = lexeme
                line += lexeme.count('\n')  # обновляем номер текущей строки
                break
        else:  # если несколько веток (dict)
            i += 1  # один символ был считан, поэтому добавляем 1
            current_state = r  # переходим к следующему состоянию
            continue
        tokens_and_lexemes.append((token, lexeme))  # добавляем токен и лексему
        current_state = lexical_fsm  # текущее состояние -- начальное состояние
    return tokens_and_lexemes
