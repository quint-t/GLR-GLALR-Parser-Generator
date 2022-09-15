class ULexicalError(ValueError):
    """Исключение возникает из-за лексической ошибки"""

    def __init__(self, **kwargs):
        self.params = {k: repr(v) if isinstance(v, str) else v for k, v in kwargs.items()
                       if isinstance(v, (int, float)) or v}
        super().__init__()

    def __str__(self):
        return '[' + ' | '.join(f"{name}: {p}" for name, p in self.params.items()) + ']'


class ULexicalFSMError(TypeError):
    """Исключение возникает из-за ошибки в описании лексики языка в виде конечного автомата"""
    pass  # не требует реализации; функциональность = TypeError


class USyntaxError(ValueError):
    """Исключение возникает из-за синтаксической ошибки"""

    def __init__(self, **kwargs):
        self.params = {k: repr(v) if isinstance(v, str) else v for k, v in kwargs.items()
                       if isinstance(v, (int, float)) or v}
        super().__init__()

    def __str__(self):
        return '[' + ' | '.join(f"{name}: {p}" for name, p in self.params.items()) + ']'


class USyntaxEBNFError(TypeError):
    """Исключение возникает из-за ошибки в описании синтаксиса языка в виде расширенной формы Бэкуса-Наура"""
    pass  # не требует реализации; функциональность = TypeError
