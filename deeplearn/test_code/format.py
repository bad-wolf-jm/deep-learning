#import zmq
import argparse
#import pymysql
#import numpy as np


def table_cell(str_, alignment=None, size=None):
    size = 'width: {size}%'.format(size=size) if size is not None else ''

    if alignment is not None:
        return "<td style='text-align: {alignment}; {size}'>{elem}</td>".format(elem=str_, alignment=alignment, size=size)
    else:
        return "<td style='{size}'>{elem}</td>".format(elem=str_)


def table_header_cell(str_, alignment=None, size=None):
    size = 'width: {size}%'.format(size=size) if size is not None else ''
    if alignment is not None:
        return "<th style='text-align: {alignment}; {size}'>{elem}</th>".format(elem=str_, alignment=alignment, size=size)
    else:
        return "<th style='{size}'>{elem}</th>".format(elem=str_)


def table_row(list_of_cells, color=None, size=None):
    r = "".join(list_of_cells)
    color = color if color is not None else 'rgba(0, 0, 0, 0);'
    return "<tr style='background-color: {color};'>{elem}</tr>".format(color=color, elem=r)


def list_to_row(obj_list, alignments, sizes=None, color=None):
    r = [table_cell(x, y, s) for x, y, s in zip(obj_list, alignments, sizes)]
    return table_row(r, color)


def list_to_header_row(obj_list, alignments, sizes):
    r = [table_header_cell(x, y, s) for x, y, s in zip(obj_list, alignments, sizes)]
    return table_row(r, color=None)


def format_table(list_of_rows, titles, alignments, sizes=None, row_colors=None, print_header=True):
    header = ''
    if print_header:
        header = list_to_header_row(titles, alignments, sizes)
    content = []
    list_of_rows = list(list_of_rows)
    row_colors = row_colors if row_colors is not None else [None] * len(list_of_rows)
    for row, color in zip(list_of_rows, row_colors):
        content.append(list_to_row(row, alignments, sizes, color))
    table_body = "\n".join(content)
    table_def = '<table  style="width:100%">{header}{table_body}</table>'
    table_def = table_def.format(header=header, table_body=table_body)
    return table_def


def format_confusion_matrix(label_dict, true_labels, predicted_labels):
    matrix = {}
    for t_l, p_l in zip(true_labels, predicted_labels):
        if (t_l, p_l) not in matrix:
            matrix[(t_l, p_l)] = 0
        matrix[(t_l, p_l)] += 1
    rows = []
    col_head = ['<td style="border: .5px solid black;"></td>']
    for r_label in label_dict:
        row_data = '<td style="border: .5px solid black;"><b>{label}</b></td>'.format(label=label_dict[r_label])
        cells = [row_data]
        col_header = '<td style="border: .5px solid black; text-align:center"><b>{label}</b></td>'.format(label=label_dict[r_label])
        col_head.append(col_header)
        for c_label in label_dict:
            num = matrix.get((r_label, c_label), 0)
            cell = '<td style="border: .5px solid black; text-align:center">{label}</td>'.format(label=num)
            cells.append(cell)
        row_data = ''.join(cells)
        row_data = '<tr>{row_data}</tr>'.format(row_data=row_data)
        rows.append(row_data)
    table_header = ''.join(col_head)
    table_body = "".join(rows)
    table_body = '<table style="width:100%; border: 3px solid black;">{table_header}{table_body}</table>'.format(table_header=table_header, table_body=table_body)
    return table_body


"""
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

output_ = open('foo.html', 'w')

code = open('format.py').read()
css = HtmlFormatter().get_style_defs('.source')
output_.write('<style>{s}</style>'.format(s=css)) #open('style.css').read()))
output_.write(highlight(code, PythonLexer(), HtmlFormatter(linenos=False, cssclass='source')))
"""
