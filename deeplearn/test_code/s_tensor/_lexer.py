"""This file contains the lexer rules and the list of valid tokens."""
import ply.lex as lex
import sys
import re

tokens = (
    'INT',
    'FLOAT',
    'STRING',
    'LPAREN',
    'RPAREN',
    'LBRACE',
    'RBRACE',
    'LBRACKET',
    'RBRACKET',
    'SYMBOL',
    'KEYWORD'
)

# These are regular expression rules for simple tokens.
t_LBRACE = r'\{'
t_RBRACE = r'\}'
t_LBRACKET = r'\['
t_RBRACKET = r'\]'
t_LPAREN = r'\('
t_RPAREN = r'\)'

# Read in a float.  This rule has to be done before the int rule.


def t_FLOAT(t):
    r'-?\d+\.\d*(e-?\d+)?'
    t.value = float(t.value)
    return t

# Read in an int.


def t_INT(t):
    r'-?\d+'
    t.value = int(t.value)
    return t

# Read in a string, as in C.  The following backslash sequences have their
# usual special meaning: \", \\, \n, and \t.


def t_STRING(t):
    r'\"([^\\"]|(\\.))*\"'
    escaped = 0
    str = t.value[1:-1]
    new_str = ""
    for i in range(0, len(str)):
        c = str[i]
        if escaped:
            if c == "n":
                c = "\n"
            elif c == "t":
                c = "\t"
            new_str += c
            escaped = 0
        else:
            if c == "\\":
                escaped = 1
            else:
                new_str += c
    t.value = new_str
    return t

# Ignore comments.


def t_comment(t):
    r'[;][^\n]*'
    pass

# Track line numbers.


def t_newline(t):
    r'\n+'
    # print(t.value)
    t.lexer.lineno += len(t.value)


def t_KEYWORD(t):
    r'[^0-9(){}\[\]][^(){}\[\]\ \t\n]*:'
    return t

# Read in a symbol.  This rule must be practically last since there are so few
# rules concerning what constitutes a symbol.


def t_SYMBOL(t):
    r'[^0-9(){}\[\]][^(){}\[\]\ \t\n]*'
    return t


# These are the things that should be ignored.
t_ignore = ' \t'

# Handle errors.


def t_error(t):
    raise SyntaxError("syntax error on line %d near '%s'" %
                      (t.lineno, t.value))


# Build the lexer.
# lex.lex()
lexer = lex.lex()


if __name__ == '__main__':
    #lexer = lex.lex()
    x = open('ch5.scm').read()  # "(define foo bar)"
    lexer.input(x)
    # Tokenize
    while True:
        tok = lexer.token()
        if not tok:
            break      # No more input
        print(tok)
