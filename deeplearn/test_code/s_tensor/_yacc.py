"""This file contains the parser rules.

The function yacc.parse, which this function makes available, returns a parse
tree.  The parse tree is a set of nested lists containing ints, floats,
strings, Symbols, etc.
"""
import ply.yacc as yacc
import sys
from _lexer import lexer, tokens
from _types import _symbol, _keyword, _pair, _empty_list


def p_program(t):
    """sexps : sexp sexps"""
    t[0] = [t[1]] + (t[2] if t[2] is not None else [])


def p_sexp_atomic(t):
    '''sexp : KEYWORD
            | INT
            | FLOAT
            | STRING
            | SYMBOL
            '''
    t[0] = t[1]


def p_sexp(t):
    '''sexp : LBRACKET sexps RBRACKET
            '''
    print(str(t[2]))
    t[0] = t[2]


def p_sexp_list(t):
    '''sexp : LPAREN sexps RPAREN'''
    l = _empty_list()
    elements = [x for x in (t[2] if t[2] is not None else [])]
    while len(elements) > 0:
        l = _pair(elements.pop(), l)
    t[0] = l


def p_sexp_dict(t):
    '''sexp : LBRACE sexps RBRACE'''
    l = {}
    assert len(t[2]) % 2 == 0
    keys = [t[2][2 * i] for i in range(len(t[2]) // 2)]
    values = [t[2][2 * i + 1] for i in range(len(t[2]) // 2)]
    t[0] = {k: v for k, v in zip(keys, values)}


def p_sexps(t):
    'sexps : empty'
    t[0] = None


def p_empty(t):
    'empty :'
    pass


def p_error(t):
    raise SyntaxError("invalid syntax")


parser = yacc.yacc()


if __name__ == '__main__':
    #lexer = lex.lex()
    x = open('ch5.scm').read()
    z = """(define (compile-and-go expression)
      (let ((instructions
             (assemble (statements
                        (compile expression 'val 'return))
                       eceval)))
        (set! the-global-environment (setup-environment))
        (set-register-contents! eceval 'val instructions)
        (set-register-contents! eceval 'flag true)
        (start eceval)))"""

    z = "{define: foo bar: 4} [1 2 3 4 5 4] (define: foo bar: 4)"
    y = parser.parse(x, debug=True, lexer=lexer)
    print(y)  # y._car, y._cdr._car, y._cdr._cdr._car)
