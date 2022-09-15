import collections
import pprint
from typing import Set, Dict, Tuple, List, Union, Any, Deque

from .exceptions import USyntaxError, USyntaxEBNFError
from .utils import check_ebnf


class Rule:
    """
    Класс "Правило"
    Пример: S -> F F
    self.nonterminal = "S"
    self.rule_tuple = ("F", "F")
    """

    def __init__(self,
                 nonterminal: str,
                 rule: Tuple[str],
                 syntax_bnf: Dict[str, List[Tuple[str]]] = None,
                 first_sets: Dict[str, Set[str]] = None):
        """
        Конструктор
        :param syntax_bnf: синтаксис языка в обычной (не расширенной) форме Бэкуса-Наура
        :param first_sets: словарь, нетерминал -> множество терминалов,
                                    которые могут появиться первыми при полном выводе
        :param nonterminal: нетерминал
        :param rule: правило
        """
        self.nonterminal = nonterminal  # ссылка (str -- non immutable)
        self.rule_tuple = rule  # ссылка (tuple -- non immutable)
        self.syntax_bnf = syntax_bnf if isinstance(syntax_bnf, dict) else {}  # ссылка (предполагается неизменяемость)
        self.first_sets = first_sets if isinstance(first_sets, dict) else {}  # ссылка (предполагается неизменяемость)

    def get_sequence_firsts(self, sequence: Tuple[str]) -> Set[str]:
        """
        Возвращает множество терминалов, которые могут появиться первыми при полном выводе sequence
        :param sequence: упорядоченный список терминалов и нетерминалов
        :return: множество терминалов, которые могут появиться первыми при полном выводе sequence
        """
        sequence_firsts = set()
        epsilon_in_symbol_firsts = True
        for symbol in sequence:
            epsilon_in_symbol_firsts = False
            if symbol not in self.syntax_bnf:  # если symbol нетерминал
                sequence_firsts.add(symbol)
                break
            for first in self.first_sets[symbol]:
                if first != '':
                    sequence_firsts.add(first)
                else:
                    epsilon_in_symbol_firsts = True
            epsilon_in_symbol_firsts |= not bool(self.first_sets[symbol])
            if not epsilon_in_symbol_firsts:
                break
        if epsilon_in_symbol_firsts:
            sequence_firsts.add('')
        return sequence_firsts

    def __eq__(self, other: 'Rule') -> bool:
        """
        Проверяет на равенство правила self и other
        :param other: правило для сравнения
        :return: True, если правила одинаковы по нетерминалу и содержанию, иначе False
        """
        return self.nonterminal == other.nonterminal and self.rule_tuple == other.rule_tuple

    def __hash__(self) -> int:
        """
        Хэш-функция опирается только на терминал и правило
        Запрещается объединение правил разных синтаксисов и разных словарей множеств FIRST
        :return: значение хэш-функции в виде целого числа
        """
        return hash((self.nonterminal, self.rule_tuple))

    def __str__(self) -> str:
        """Возвращает строковое представление объекта"""
        return f"{type(self).__name__}({self.nonterminal} -> {' '.join(x if x else repr('') for x in self.rule_tuple)})"

    def __repr__(self) -> str:
        """Возвращает строковое представление объекта"""
        return f"{type(self).__name__}({repr(self.nonterminal)}, ({', '.join(repr(x if x else '') for x in self.rule_tuple)},))"


class AbstractItem:
    """
    Абстрактный класс "Продукция состояния"
    """

    def __init__(self, rule: Rule, dot_index: int, lookaheads: set = None):
        """
        Конструктор
        :param rule: правило
        :param dot_index: индекс текущей позиции в правиле
        """
        if type(self) is AbstractItem:
            raise NotImplementedError(f'{type(self).__name__} is an abstract class and cannot be instantiated directly')
        self.rule = rule  # ссылка
        self.dot_index = dot_index
        self.lookaheads = lookaheads if isinstance(lookaheads, set) else set()
        self.generate_item = AbstractItem  # атрибут должен быть переопределен в дочернем классе

    def new_items_from_symbol_after_dot(self) -> Set['AbstractItem']:
        """
        Возвращает новые элементы по текущей позиции (в правиле)
        :return: множество новых элементов по текущей позиции (в правиле)
        """
        r = set()
        if self.dot_index < len(self.rule.rule_tuple):
            nonterminal = self.rule.rule_tuple[self.dot_index]
            for rule in self.rule.syntax_bnf.get(nonterminal, []):
                self.generate_item(rule=Rule(nonterminal, rule, self.rule.syntax_bnf, self.rule.first_sets),
                                   dot_index=0).add_unique_to(r)
        return r

    def new_item_after_shift(self) -> 'AbstractItem':
        """
        Возвращает новый элемент после текущей позиции в правиле
        :return: новый элемент после текущей позиции в правиле
        """
        if self.dot_index < len(self.rule.rule_tuple) and self.rule.rule_tuple[self.dot_index] != '':
            return self.generate_item(rule=self.rule, dot_index=self.dot_index + 1)

    def add_unique_to(self, items: Set['AbstractItem']) -> bool:
        """
        Добавляет текущий элемент в множество items
        :param items: множество элементов
        :return: True, если элемент был добавлен в items, иначе False
        """
        previous_length = len(items)
        items.add(self)
        return previous_length != len(items)

    def __eq__(self, other: 'AbstractItem') -> bool:
        """
        Проверяет на равенство элементы self и other
        :param other: элемент для сравнения
        :return: True, если элементы одинаковы по правилу и текущей позиции, иначе False
        """
        return self.rule == other.rule and self.dot_index == other.dot_index

    def __hash__(self) -> int:
        """
        Хэш-функция опирается только на правило и текущую позицию
        Запрещается объединение элементов с одинаковыми правилами и позициями, но с разными lookaheads
        :return: значение хэш-функции в виде целого числа
        """
        return hash((self.rule, self.dot_index))

    def __str__(self) -> str:
        """Возвращает строковое представление объекта"""
        return f"{type(self).__name__}(rule={self.rule}, dot_index={self.dot_index}, lookaheads={self.lookaheads})"

    def __repr__(self) -> str:
        """Возвращает строковое представление объекта"""
        return f"{type(self).__name__}(rule={repr(self.rule)}, " \
               f"dot_index={repr(self.dot_index)}, lookaheads={repr(self.lookaheads)})"


class LR1Item(AbstractItem):
    """
    Класс "LR(1) продукция состояния"
    """

    def __init__(self, rule: Rule, dot_index: int):
        """
        Конструктор
        :param rule: правило
        :param dot_index: индекс текущей позиции в правиле
        """
        super().__init__(rule, dot_index)
        self.generate_item = LR1Item
        if self.dot_index == 0:
            self.lookaheads.add('$')

    def new_items_from_symbol_after_dot(self) -> Set['AbstractItem']:  # overrides method in AbstractItem
        """
        Возвращает новые элементы по текущей позиции (в правиле)
        :return: множество новых элементов по текущей позиции (в правиле)
        """
        r = super().new_items_from_symbol_after_dot()
        if not r:
            return r
        new_lookaheads = set()
        firsts_after_symbol_after_dot = self.rule.get_sequence_firsts(self.rule.rule_tuple[self.dot_index + 1:])
        if '' in firsts_after_symbol_after_dot:
            firsts_after_symbol_after_dot.remove('')
            new_lookaheads.update(self.lookaheads)
        new_lookaheads.update(firsts_after_symbol_after_dot)
        for x in r:
            x.lookaheads = new_lookaheads.copy()
        return r

    def new_item_after_shift(self) -> 'AbstractItem':  # overrides method in AbstractItem
        """
        Возвращает новый элемент после текущей позиции в правиле
        :return: новый элемент после текущей позиции в правиле
        """
        r = super().new_item_after_shift()
        if r is not None:
            r.lookaheads = self.lookaheads.copy()
        return r

    def add_unique_to(self, items: Set['AbstractItem']) -> bool:  # overrides method in AbstractItem
        """
        Добавляет текущий элемент в список/множество items
        :param items: список/множество элементов
        :return: True, если элемент был добавлен в items, иначе False
        """
        for item in items:
            if super().__eq__(item):
                previous_length = len(item.lookaheads)
                item.lookaheads.update(self.lookaheads)
                return previous_length != len(item.lookaheads)
        items.add(self)
        return True

    def __eq__(self, other: 'LR1Item') -> bool:  # overrides method in AbstractItem
        """
        Проверяет на равенство элементы self и other
        :param other: элемент для сравнения
        :return: True, если элементы одинаковы по правилу и текущей позиции, иначе False
        """
        return self.rule == other.rule and self.dot_index == other.dot_index and self.lookaheads == other.lookaheads

    def __hash__(self) -> int:  # overrides method in AbstractItem
        """
        Хэш-функция опирается только на правило и текущую позицию
        Запрещается объединение элементов с одинаковыми правилами и позициями, но с разными lookaheads
        :return: значение хэш-функции в виде целого числа
        """
        return hash((self.rule, self.dot_index, tuple(sorted(self.lookaheads))))


class LALR1Item(LR1Item):
    """
    Класс "LALR(1) продукция состояния"
    """

    def __init__(self, rule: Rule, dot_index: int):
        """
        Конструктор
        :param rule: правило
        :param dot_index: индекс текущей позиции в правиле
        """
        super().__init__(rule, dot_index)
        self.generate_item = LALR1Item

    def __eq__(self, other: 'LALR1Item') -> bool:  # overrides method in LR1Item
        return AbstractItem.__eq__(self, other)

    def __hash__(self) -> int:  # overrides method in LR1Item
        return AbstractItem.__hash__(self)


class State:
    """
    Класс "Состояние"
    """

    def __init__(self, index: int, items: Set[AbstractItem],
                 keys: set = None, closure: set = None, gotos: set = None):
        """
        Конструктор
        :param index: индекс состояния
        :param items: продукции состояния
        """
        self.index = index
        self.items = items  # ссылка
        self.closure = items.copy()  # копия
        self.gotos = dict()
        self.keys = set()

    def __eq__(self, other: 'State') -> bool:
        """
        Проверяет на равенство состояния self и other
        :param other: состояние для сравнения
        :return: True, если состояния одинаковы по продукциям состояния, иначе False
        """
        return self.items == other.items

    def __str__(self, width=80) -> str:
        """Возвращает строковое представление объекта"""
        items = pprint.pformat(self.items, width=width)
        closure = pprint.pformat(self.closure, width=width)
        gotos = pprint.pformat(self.gotos, width=width)
        return f"{type(self).__name__}(index={self.index}, items={items},\n" \
               f"keys={self.keys},\nclosure={closure},\ngotos={gotos})"

    def __repr__(self, width=80) -> str:
        """Возвращает строковое представление объекта"""
        items = pprint.pformat(self.items, width=width)
        closure = pprint.pformat(self.closure, width=width)
        gotos = pprint.pformat(self.gotos, width=width)
        return f"{type(self).__name__}(index={repr(self.index)}, items={repr(items)},\n" \
               f"keys={repr(self.keys)},\nclosure={repr(closure)},\ngotos={repr(gotos)})"


def construct_first_sets(syntax_bnf: Dict[str, List[Tuple[str]]]) -> Dict[str, Set[str]]:
    """
    Конструирует словарь множеств FIRST
    :param syntax_bnf: синтаксис языка в обычной (не расширенной) форме Бэкуса-Наура
    :return: словарь множеств FIRST
    """
    first_sets = {k: set() for k in syntax_bnf}
    changed = True
    while changed:
        changed = False
        for nonterminal, rules in syntax_bnf.items():
            for rule in rules:
                for symbol in rule:
                    if symbol in syntax_bnf:
                        have_epsilon = False
                        for first_terminal in first_sets[symbol]:
                            have_epsilon |= (first_terminal == '')
                            if first_terminal not in first_sets[nonterminal]:
                                first_sets[nonterminal].add(first_terminal)
                                changed = True
                        if not have_epsilon:
                            break
                    else:
                        if symbol not in first_sets[nonterminal]:
                            first_sets[nonterminal].add(symbol)
                            changed = True
                        if symbol != '':
                            break
    return first_sets


def construct_lr_closure_table(initial_nonterminal: str,
                               syntax_bnf: Dict[str, List[Tuple[str]]],
                               first_sets: Dict[str, Set[str]],
                               lalr: bool) -> List[State]:
    """
    Конструирует таблицу LR closure
    :param initial_nonterminal: начальный нетерминал
    :param syntax_bnf: синтаксис языка в обычной (не расширенной) форме Бэкуса-Наура
    :param first_sets: словарь множеств FIRST
    :param lalr: использовать GLR(1) [False] или GLALR(1) [True] парсер;
                 в случае GLR: число состояний, затрат по времени и памяти будет больше,
                               но при этом бОльшая распознавательная способность
                 в случае GLALR: число состояний, затрат по времени и памяти будет меньше,
                               но при этом меньшая распознавательная способность
                 большинство реально используемых языков программирования имеют GLALR(1)-грамматики
    :return: таблица LR closure
    """
    generate_item = LALR1Item if lalr else LR1Item
    lr_closure_table = [State(index=0, items={generate_item(rule=Rule(
        initial_nonterminal, syntax_bnf[initial_nonterminal][0], syntax_bnf, first_sets
    ), dot_index=0)})]
    i = 0
    while i < len(lr_closure_table):
        state = lr_closure_table[i]
        update_closure(state)
        if add_gotos(state, lr_closure_table):
            i = 0
        else:
            i += 1
    return lr_closure_table


def update_closure(state: State) -> None:
    """
    Дополняет closure состояния state
    :param state: состояние как минимум с одним closure, на основе которого происходит дополнение
    """
    hq = collections.deque(state.closure)
    while hq:
        for x in hq.popleft().new_items_from_symbol_after_dot():
            if x.add_unique_to(state.closure):
                hq.append(x)


def add_gotos(state: State, lr_closure_table: List[State]) -> bool:
    """
    Добавляет новые состояния и переходы в соответствии с state
    :param state: состояние с заполненным closure
    :param lr_closure_table: таблица LR closure
    :return: True, если нужно вернуться к первому состоянию в lr_closure_table, иначе False
    """
    lookaheads_propagated = False
    new_states = dict()
    for item in state.closure:
        new_item = item.new_item_after_shift()
        if new_item is not None:
            symbol_after_dot = item.rule.rule_tuple[item.dot_index]
            state.keys.add(symbol_after_dot)
            new_item.add_unique_to(new_states.setdefault(symbol_after_dot, set()))
    for key in state.keys:
        new_state = State(len(lr_closure_table), set(new_states[key]))
        target_state_index = next((i for i, prev_state in enumerate(lr_closure_table)
                                   if prev_state == new_state), -1)
        if target_state_index == -1:
            lr_closure_table.append(new_state)
            target_state_index = new_state.index
        else:
            for item in new_state.items:
                lookaheads_propagated |= item.add_unique_to(lr_closure_table[target_state_index].items)
        state.gotos.setdefault(key, set()).add(target_state_index)
    return lookaheads_propagated


def construct_lr_table(lr_closure_table: List[State]) -> Dict[Tuple[int, str], Set[Union[int, Rule]]]:
    """
    Конструирует таблицу LR на основе таблицы LR closure
    :param lr_closure_table: таблица LR closure
    :return: таблица LR
    """
    n_states = []
    lr_table = dict()
    for state in lr_closure_table:
        n_state = len(n_states)
        n_states.append(n_state)
        for key in state.keys:
            lr_table[(n_state, key)] = state.gotos[key]
        for item in state.closure:
            if item.dot_index == len(item.rule.rule_tuple) or item.rule.rule_tuple[0] == '':
                for lookahead in item.lookaheads:
                    lr_table.setdefault((n_state, lookahead), set()).add(item.rule)
    return lr_table


def convert_to_bnf(initial_nonterminal: str,
                   syntax_ebnf: Dict[str, Union[str,
                                                Set[Tuple[str, str]],
                                                Tuple[Union[str, Set[Tuple[str, str]]]],
                                                List[Union[str,
                                                           Set[Tuple[str, str]],
                                                           Tuple[Union[str, Set[Tuple[str, str]]]]]]]]) -> Tuple[str, Dict[str, List[Tuple[str]]]]:
    """
    Проверяет syntax_ebnf -- синтаксис языка в расширенной форме Бэкуса-Наура,
    а также преобразовывает его в синтаксис языка в обычной форме Бэкуса-Наура
    :param initial_nonterminal: начальный нетерминал
    :param syntax_ebnf: синтаксис языка в расширенной форме Бэкуса-Наура
    :return: синтаксис языка в обычной форме Бэкуса-Наура
    """

    check_ebnf(initial_nonterminal, syntax_ebnf)

    def gen_rules(s: str, new_s: str, times: str) -> List[Tuple[str]]:
        """Генерирует правила для нового состояния-повторения"""
        rules = []
        if times != '+':  # times == '*' или times == '?'
            rules.append(('',))
        if times != '*':  # times == '+' или times == '?'
            rules.append((s,))
        if times != '?':  # times == '*' или times == '+'
            rules.append((new_s, s))
        return rules

    q = collections.deque([initial_nonterminal])  # очередь
    new_initial_nonterminal = initial_nonterminal + "'"  # новый начальный нетерминал
    while new_initial_nonterminal in syntax_ebnf:  # должен иметь новое уникальное название
        new_initial_nonterminal += "'"
    syntax_bnf = {new_initial_nonterminal: [(initial_nonterminal,)]}  # синтаксис в виде обычной формы Бэкуса-Наура
    seen = set()  # множество уже обработанных нетерминалов
    while q:
        nonterminal = q.popleft()  # достаем первый нетерминал из очереди
        rules = syntax_ebnf[nonterminal]
        if not isinstance(rules, list):  # если одно правило, то преобразование в список
            rules = [rules]
        new_rules = list()  # список новых правил
        for rule in rules:  # для каждого правила из старого списка правил
            if not isinstance(rule, tuple):  # если не конкатенация, то преобразование в конкатенацию
                rule = (rule,)
            new_rule = []  # новое правило для добавления в список новых правил
            for subrule in rule:  # для каждой части правила
                if isinstance(subrule, set):  # если часть правила -- повторение
                    str_tuple = next(iter(subrule))
                    # set должен содержать один tuple с 2 строками
                    s, times = str_tuple
                    new_s = s + times  # новое название для нетерминала-повторения
                    while new_s in syntax_bnf and syntax_bnf[new_s] != gen_rules(s, new_s, times):
                        new_s += times
                    syntax_bnf[new_s] = gen_rules(s, new_s, times)  # правила для нового нетерминала
                    new_rule.append(new_s)  # добавление нетерминала в новое правило
                    if s in syntax_ebnf and s not in seen:  # если s есть в синтаксисе и еще не был обработан
                        q.append(s)  # добавляем в очередь
                        seen.add(s)  # добавляем в множество уже обработанных нетерминалов
                elif subrule or len(rule) == 1:  # если непустая часть правила или epsilon-правило
                    new_rule.append(subrule)  # добавляем часть правила
                    # если subrule есть в синтаксисе и еще не был обработан
                    if subrule in syntax_ebnf and subrule not in seen:
                        q.append(subrule)  # добавляем в очередь
                        seen.add(subrule)  # добавляем в множество уже обработанных нетерминалов
            new_rules.append(tuple(new_rule))  # добавляем новое правило в список
        syntax_bnf[nonterminal] = new_rules  # сопоставляем нетерминалу список правил
    initial_nonterminal = new_initial_nonterminal  # заменяем начальный нетерминал
    return initial_nonterminal, syntax_bnf


def ebnf_to_lr_table(initial_nonterminal: str,
                     syntax_ebnf: Dict[str, Union[str,
                                                  Set[Tuple[str, str]],
                                                  Tuple[Union[str, Set[Tuple[str, str]]]],
                                                  List[Union[str,
                                                             Set[Tuple[str, str]],
                                                             Tuple[Union[str, Set[Tuple[str, str]]]]]]]],
                     lalr: bool) -> Tuple[str, Dict[Tuple[int, str], Set[Union[int, Rule]]]]:
    """
    Генерирует GLR(1) или GLALR(1) парсер (LR-Table) по синтаксису языка в расширенной форме Бэкуса-Наура (EBNF)
    :param initial_nonterminal: начальный нетерминал
    :param syntax_ebnf: синтаксис языка в расширенной форме Бэкуса-Наура
    :param lalr: использовать GLR(1) [False] или GLALR(1) [True] парсер;
                 в случае GLR: число состояний, затрат по времени и памяти будет больше,
                               но при этом бОльшая распознавательная способность + подсказки по ошибкам синтаксиса
                 в случае GLALR: число состояний, затрат по времени и памяти будет меньше,
                               но при этом меньшая распознавательная способность и наличие некорректных подсказок
                 (большинство реально используемых языков программирования имеют GLALR(1)-грамматики)
    :return: новый начальный нетерминал, таблица LR
    """

    # проверка syntax_ebnf -- синтаксиса языка в расширенной форме Бэкуса-Наура
    # а также преобразование его в синтаксис языка в обычной форме
    initial_nonterminal, syntax_bnf = convert_to_bnf(initial_nonterminal, syntax_ebnf)
    # конструируем словарь множеств first_sets
    first_sets = construct_first_sets(syntax_bnf)
    # конструируем таблицу LR closure
    lr_closure_table = construct_lr_closure_table(initial_nonterminal, syntax_bnf, first_sets, lalr=lalr)
    # конструируем таблицу LR на основе таблицы LR closure
    lr_table = construct_lr_table(lr_closure_table)
    return initial_nonterminal, lr_table


def analyze_syntax(initial_nonterminal: str,
                   lr_table: Dict[Tuple[int, str], Set[Union[int, Rule]]],
                   tokens_and_lexemes: List[Tuple[str, str]],
                   tokens_to_skip: Set[str]) -> List[Tuple[str, str, str, str]]:
    """
    Анализирует токены согласно сгенерированному парсеру (initial_nonterminal, lr_table)
    :param initial_nonterminal: начальный нетерминал
    :param lr_table: LR таблица
    :param tokens_and_lexemes: список токенов и лексем; например, [..., ('<identifier>', 'Animal'), ...]
    :param tokens_to_skip: множество пропускаемых токенов
    :return: история синтаксического анализа
    """
    if '$' in tokens_to_skip:
        raise USyntaxEBNFError(f'Dollar symbol must not be in skipped tokens, got: {tokens_to_skip}')

    def to_readable_format(xs: Union[List[Any], Tuple[Any]]):
        """Преобразует входной список xs в строку, разделенную пробелами"""
        return ' '.join(repr(x) for x in xs) if xs else "''"

    def format_state_and_token(state_and_token: Tuple[str, str]):
        """Форматирует текущее состояние и токен для записи истории синтаксического анализа"""
        return f'state={repr(state_and_token[0])}, token={repr(state_and_token[1])}'

    def update_error(state: int, error_row: int, error_col: int, description: str):
        """Регистрирует последнюю ошибку"""
        nonlocal last_error_row, last_error_col, last_error_string
        if error_row > last_error_row or error_row == last_error_row and error_col >= last_error_col:
            last_error_row = error_row
            last_error_col = error_col
            expected_lexemes = set()
            for n, lookahead in lr_table.keys():
                if n == state:
                    expected_lexemes.add(lookahead)
            last_error_string = f'{description}.' * bool(description)
            last_error_string += f' Expected any of: {expected_lexemes}' * bool(expected_lexemes)

    # определение типа синтаксиса: неоднозначный (Ambiguous) / однозначный (Unambiguous)
    syntax_rules = len(lr_table.keys())  # число всех правил
    ambiguous_syntax_rules = sum(len(v) > 1 for k, v in lr_table.items())  # число неоднозначных правил
    main_history = []  # история синтаксического анализа
    if ambiguous_syntax_rules:
        main_history.append(('Ambiguous syntax ' +
                             f'({ambiguous_syntax_rules} ambiguous rules / {syntax_rules} rules)',
                             '', '', ''))
    else:
        main_history.append((f'Unambiguous syntax ({syntax_rules} rules)', '', '', ''))
    # проверка последовательности токенов на соответствие синтаксису языка (с механизмом разветвления стеков)
    stacks: Deque[Tuple[Any, Any, int, int, int]] = collections.deque([(main_history, [0], 0, 1, 1)])  # очередь стеков
    status = False  # статус проверки
    last_error_row = 0  # последняя позиция ошибки (строка)
    last_error_col = 0  # последняя позиция ошибки (столбец)
    last_error_string = ''  # описание последней ошибки
    while stacks:  # пока есть стеки
        history, stack, token_index, error_row, error_col = stacks.popleft()  # достаем первый стек из очереди
        if not stack:  # если стек пуст, регистрируем ошибку (такая ошибка маловероятна)
            update_error(-1, error_row, error_col, 'Parse stack is empty')
            continue
        # если все токены обработаны
        if token_index >= len(tokens_and_lexemes):
            token, lexeme = ('$', '')  # последний токен и пустая лексема
            state_and_token = (stack[-1], token)
            if state_and_token not in lr_table:  # если завершения нет
                update_error(stack[-1], error_row, error_col, f'Unexpected end')  # регистрация ошибки
                continue
            elif any(r.nonterminal == initial_nonterminal for r in lr_table[state_and_token]
                     if isinstance(r, Rule)):  # если есть завершение
                status = True  # анализ завершен
                main_history = history
                break  # выход из цикла с успешным анализом
            # если завершение требует reduce
        else:  # если не все токены обработаны
            token, lexeme = tokens_and_lexemes[token_index]  # получаем очередной токен и лексему
            state_and_token = (stack[-1], token)  # составляем условие перехода
        new_error_row, new_error_col = error_row, error_col
        if '\n' in lexeme:  # создаем новые текущие позиции
            new_error_row += lexeme.count('\n')
            new_error_col = len(lexeme) - lexeme.rfind('\n') - 1
        else:
            new_error_col += len(lexeme)
        if token in tokens_to_skip:  # если токен в списке пропускаемых токенов, то пропускаем его
            stacks.append((history + [(format_state_and_token(state_and_token),
                                       'skip', '',
                                       to_readable_format(stack))],
                           stack, token_index + 1, new_error_row, new_error_col))
            continue
        if state_and_token not in lr_table:  # если условие перехода не найдено, то
            state_and_token = (stack[-1], '')  # попытка epsilon-правила
            if state_and_token not in lr_table:  # если и epsilon-правила не было, тогда регистрация ошибки
                update_error(stack[-1], error_row, error_col, f'Unexpected lexeme {lexeme}')
                continue
        for action in lr_table[state_and_token]:  # для каждого возможного перехода
            if isinstance(action, int):  # если новое состояние
                new_stack = stack + [token, action]  # добавляем токен и состояние в стек
                stacks.append((history + [(format_state_and_token(state_and_token),
                                           f'shift {repr(token)} {action}', '',
                                           to_readable_format(new_stack))],
                               new_stack, token_index + 1, new_error_row, new_error_col))
            elif isinstance(action, Rule):  # если нужно применить правило
                nonterminal = action.nonterminal
                rule = action.rule_tuple
                removed = 0  # число последних элементов stack для удаления
                stack_pos = len(stack) - 2  # stack[-1] -- int, поэтому от len(stack) - 1 отнимаем 1
                rule_pos = len(rule) - 1
                while stack_pos >= 1 and rule_pos >= 0:  # проверка правила с учетом epsilon-переходов
                    if stack[stack_pos] != rule[rule_pos]:
                        if rule[rule_pos] == '':
                            rule_pos -= 1
                        else:
                            break
                    else:
                        removed += 2
                        rule_pos -= 1
                        stack_pos -= 2
                popped = stack[len(stack) - removed:]
                new_stack = stack[:len(stack) - removed] + [nonterminal]
                if (new_stack[-2], new_stack[-1]) not in lr_table:  # если условие перехода не найдено, то ошибка
                    update_error(new_stack[-2], error_row, error_col, f'Unexpected lexeme {lexeme}')
                    continue
                for x in lr_table[(new_stack[-2], new_stack[-1])]:  # для каждого возможного перехода
                    next_stack = new_stack + [x]  # формируем стек и добавляем его в очередь стеков
                    stacks.append((history +
                                   [(format_state_and_token(state_and_token),
                                     f'reduce {to_readable_format(popped)} -> {repr(next_stack[-2])}',
                                     f'push {next_stack[-1]} by ({new_stack[-2]}, {repr(new_stack[-1])})',
                                     to_readable_format(next_stack))],
                                   next_stack, token_index, error_row, error_col))
    if not status:
        raise USyntaxError(message=f"{last_error_string}",
                           position=f'{last_error_row}:{last_error_col}')
    return main_history
