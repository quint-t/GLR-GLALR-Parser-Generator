import c90_parser
import os

directory = os.path.dirname(__file__)
not_exist_file = os.path.join(directory, 'c90_examples', 'not_exist.c')
normal_file = os.path.join(directory, 'c90_examples', 'normal.c')
lexical_error_file = os.path.join(directory, 'c90_examples', 'lexical_error.c')
syntax_error_file = os.path.join(directory, 'c90_examples', 'syntax_error.c')

arguments = c90_parser.ParserArguments(paths=[not_exist_file, normal_file, lexical_error_file, syntax_error_file],
                                       glr=False,
                                       glalr=True,
                                       l_history=False,
                                       p_history=False,
                                       verbose=True,
                                       debug_args=False)
c90_parser.main(arguments)
