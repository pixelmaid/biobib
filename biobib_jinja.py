from jinja2 import Environment, FileSystemLoader
from tables import Courses
import time
from filters import *

path = 'tex/'

env = Environment(
    block_start_string='\BLOCK{',
    block_end_string='}',
    variable_start_string='\VAR{',
    variable_end_string='}',
    comment_start_string='\#{',
    comment_end_string='}',
    line_statement_prefix='%%',
    line_comment_prefix='%#',
    trim_blocks=True,
    autoescape=False,
    loader=FileSystemLoader('templates'))

env.filters['colonify'] = colonify
env.filters['str_join'] = str_join
env.filters['tex_escape'] = tex_escape
env.filters['make_cell'] = make_cell
env.filters['author_format'] = author_format


# Make the Courses Table using a Jinja2 Template:
from tables import Courses  # NOQA
courses_file = 'CV/Teaching-Table.csv'
courses = Courses(csv_file=courses_file, name='Courses')

template = env.get_template('Courses.template')

rendered_tex = template.render(
    created=time.strftime("%Y-%m-%d %H:%M"),
    courses=courses.df.to_dict('records')
)

output_file = 'test.tex'
with open(output_file, "w") as f:
    print(rendered_tex, file=f)
