import re

Categories = {
    'RA': 'Refereed Article',
    'CA': 'Conference Abstract',
    'BC': 'Refereed Book Chapter',
    'CP': 'Refereed Conference Proceedings'
}


def make_cell(text, size=''):
    """
        wrap text in a makecell
    """
    # split text by commas:
    text = ''.join([x + ',\\\\' for x in text.split(',')])
    text = text[:-3]
    text = "{" + size + " \\makecell{ " + text + "} }"
    return text


def tex_escape(text):
    """
        :param text: a plain text message
        :return: the message escaped to appear correctly in LaTeX
    """
    conv = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
        'Ω': r'$\Omega$',
        'δ': r'$\delta$',
        '’': r"'",
        '‐': r'--',
        '“': r'``',
        '”': r"''",
        'é': r'\'e'
    }

    text = str(text)
    regex = re.compile('|'.join(re.escape(key) for key in sorted(conv.keys(), key=lambda item: - len(item))))  # NOQA
    result = regex.sub(lambda match: conv[match.group()], text)
    if result == 'nan':
        result = ''
    return result


def str_join(df, sep, *cols):
    from functools import reduce
    return reduce(lambda x, y: x.astype(str).str.cat(y.astype(str), sep=sep),
                  [df[col] for col in cols])


def colonify(string):
    if string:
        return ": " + string
    else:
        return ""
