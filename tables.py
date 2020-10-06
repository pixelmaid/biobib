import pandas as pd
import numpy as np
import time
import re
from jinja2 import Environment, FileSystemLoader

latex_env = Environment(
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

latex_env.filters['colonify'] = colonify
latex_env.filters['str_join'] = str_join
latex_env.filters['tex_escape'] = tex_escape
latex_env.filters['make_cell'] = make_cell


class Table:

    def __init__(self, name=None, csv_file=None, env=latex_env,
            template_file=None, filters=None):
        self.name = name
        self.df = pd.read_csv(csv_file)
        self.columns = ""
        self.type = "longtable"
        self.env = env
        if filters:
            for item in filters:
                self.env.filters[item] = filters[item]
        self.template = self.env.get_template(template_file)
        # self.df = self.clean_df()

    def write_template(self):
        return NotImplementedError

    def clean_df(self):
        return self.df

    def clean_cumulative(self, df):
        if self.cumulative is False:
            df = df[df.Eval == 1]
        return df

    def render_template(self):
        rendered_tex = self.template.render(
            created=time.strftime("%Y-%m-%d %H:%M"),
            items=list(self.df.to_dict('records'))
        )
        return rendered_tex

    def write_template(self, path=None):
        content = self.render_template()
        if path:
            file = path + self.name + '.tex'
        else:
            file = self.name + 'tex'

        with open(file, "w") as f:
            print(content, file=f)


class Service(Table):

    def __init__(
            self,
            name='Service',
            csv_file=None,
            category='P',
            cumulative=False,
            template_file='biobib/Service.template'):
        super(Service, self).__init__(
            name=name, csv_file=csv_file, template_file=template_file)
        self.cumulative = cumulative
        self.category = category
        self.df = self.clean_df()
        
    def render_template(self):
        rendered_tex = self.template.render(
            created=time.strftime("%Y-%m-%d %H:%M"),
            items=self.df.to_dict('records')
        )
        return rendered_tex

    def clean_df(self):
        df = self.df
        # Step 1: drop any service from before this eval period
        df = self.clean_cumulative(df)
        # Step 2: filter to only the current category (or don't!)
        if self.category:
            df = df[df.Type == self.category]
        df = df.sort_values(by=['Year'], ascending=[True])
        return df


class ProfessionalService(Service):

    def __init__(
            self,
            name='ProfessionalService',
            csv_file=None,
            category='P',
            cumulative=False):
        super(ProfessionalService, self).__init__(
            name=name, csv_file=csv_file,
            category=category, cumulative=cumulative)
        self.category = category


class UniversityService(Service):

    def __init__(
            self,
            name='UniversityService',
            csv_file=None,
            category='U',
            cumulative=False):
        super(UniversityService, self).__init__(
            name=name, csv_file=csv_file,
            category=category, cumulative=cumulative)
        self.category = category


class DepartmentalService(Service):

    def __init__(
            self,
            name='DepartmentalService',
            csv_file=None,
            category='D',
            cumulative=False):
        super(DepartmentalService, self).__init__(
            name=name, csv_file=csv_file,
            category=category, cumulative=cumulative)
        self.category = category


class Publications(Table):

    def __init__(
            self,
            name='Publications',
            csv_file=None,
            category='P',
            template_file='biobib/Publications.template'):
        self.filters = {
            'make_row': self.make_row,
            'doi': self.doi,
            'href': self.href
        }
        super(Publications, self).__init__(
            name=name, csv_file=csv_file, template_file=template_file, filters=self.filters)
        self.category = category
        self.cumulative = True  # Always provide complete publication list
        self.df = self.clean_df()
        
    def clean_df(self):
        df = self.df
        df = self.clean_cumulative(df)
        # Step 1: drop any papers not published
        df = df[df.S == self.category]
        # Step 2: Concatenate authors into a single list, making sure to drop
        # empty author columns
        df['authors'] = list(
            pd.Series(df[['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15']]  # NOQA
                      .fillna('').values.tolist())
            .apply(lambda x: [i for i in x if i != ''])
            .apply(lambda x: ', '.join(x))
        )
        df['editors'] = list(
            pd.Series(df[['E1', 'E2', 'E3', 'E4']]  # NOQA
                      .fillna('').values.tolist())
            .apply(lambda x: [i for i in x if i != ''])
            .apply(lambda x: ', '.join(x))
        )
        # Step 3: Cast DOI as a string and remove nan
        df.loc[df['DOI'] == 'nan', 'DOI'] = np.nan

        # Step 4: Cast Pages as a string and remove nan
        df.loc[df['PAGES'] == 'nan', 'PAGES'] = np.nan

        # Step 5: Cast Volume as a string and remove nan
        df.loc[df['VOL'] == 'nan', 'VOL'] = np.nan

        return df

    def category_lookup(self, category):
        Categories = {
            'RA': 'Refereed Article',
            'CD': 'Conference Demonstration',
            'CA': 'Conference Abstract',
            'BC': 'Refereed Book Chapter',
            'CP': 'Refereed Conference Proceedings',
            'MA': 'Magazine Article'
        }
        return Categories[category]

    def doi(self, this_doi):
        if this_doi is np.NaN:
            return ""
        else:
            return "doi:{doi}.".format(doi=this_doi)

    def href(self, this_href):
        if this_href is np.NaN:
            return ""
        else:
            return "\\href{{{href}}}{{[pdf]}}".format(href=this_href)

    def make_row(self, row):
        if row['Type'] == 'RA' or row['Type'] == 'CA' or row['Type'] == 'MA':
            return self.make_article(row)
        elif row['Type'] == 'CD':
            return self.make_demonstration(row)
        elif row['Type'] == 'BC':
            return self.make_chapter(row)
        elif row['Type'] == 'CP':
            return self.make_chapter(row)

    def make_article(self, this_row):
        row = ""
        row += "{code} & {year} & {{\\bf {title}}}, {authors}. {href} & \\emph{{ {publisher} }} {volume}{pages}. {doi}  & {category}".format(  # NOQA
            code=tex_escape(this_row['NUM']),
            year=tex_escape(str(round(this_row['YEAR']))),
            title=tex_escape(this_row['TITLE']),
            authors=tex_escape(this_row['authors']),
            #doi=self.doi(this_row['DOI']),
            href=self.href(this_row['Link']),
            volume=tex_escape(this_row['VOL']),
            pages=colonify(tex_escape(this_row['PAGES'])),
            publisher=tex_escape(this_row['PUBLISHER']),
            category=tex_escape(self.category_lookup(this_row['Type']))
        )
        row += "\\\\"
        return row

    def make_demonstration(self, this_row):
        row = ""
        row += "{code} & {year} & {{\\bf {title}}}, {authors}. {href} & \\emph{{ {publisher} }} {volume}{pages}. {doi}  & {category}".format(  # NOQA
            code=tex_escape(this_row['NUM']),
            year=tex_escape(str(this_row['YEAR'])),
            title=tex_escape(this_row['TITLE']),
            authors=tex_escape(this_row['authors']),
            doi=self.doi(this_row['DOI']),
            href=self.href(this_row['Link']),
            volume=tex_escape(this_row['VOL']),
            pages=colonify(tex_escape(this_row['PAGES'])),
            publisher=tex_escape(this_row['PUBLISHER']),
            category=tex_escape(self.category_lookup(this_row['Type']))
        )
        row += "\\\\"
        return row

    def get_editors(self, editors):
        if editors:
            return editors + " (eds.)."
        else:
            return ""

    def make_chapter(self, this_row):
        row = ""
        row += "{code} & {year} & {{\\bf {title}}}, {authors} & {editors} \\emph{{ {book} }}. {publisher} & {category}".format(  # NOQA
            code=tex_escape(this_row['NUM']),
            year=tex_escape(str(this_row['YEAR'])),
            title=tex_escape(this_row['TITLE']),
            authors=tex_escape(this_row['authors']),
            editors=tex_escape(self.get_editors(this_row['editors'])),
            book=tex_escape(this_row['Book Title']),
            publisher=tex_escape(this_row['PUBLISHER']),
            category=tex_escape(self.category_lookup(this_row['Type']))
        )
        row += "\\\\"
        return row


class Exhibitions(Table):

    def __init__(
            self,
            name='Exhibitions',
            csv_file=None,
            category='P',
            template_file='biobib/Exhibitions.template'):
        self.filters = {
            'make_row': self.make_row
        }
        super(Exhibitions, self).__init__(
            name=name, csv_file=csv_file, template_file=template_file, filters=self.filters)
        self.category = category
        self.cumulative = True  # Always provide complete publication list
        self.df = self.clean_df()
        
    def clean_df(self):
        df = self.df
        df = self.clean_cumulative(df)
        # Step 1: drop any papers not published
        df = df[df.S == self.category]
        # Step 2: Concatenate authors into a single list, making sure to drop
        # empty author columns
        df['authors'] = list(
            pd.Series(df[['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15']]  # NOQA
                      .fillna('').values.tolist())
            .apply(lambda x: [i for i in x if i != ''])
            .apply(lambda x: ', '.join(x))
        )
        return df

    def category_lookup(self, category):
        Categories = {
            'AE': 'Artwork and Exhibition',
            'AP': 'Artwork and Performance',
            'A': 'Animation'
        }
        return Categories[category]


    def make_row(self, row):
        if row['Type'] == 'AE' or row['Type'] == 'AP' or row['Type'] == 'A':
            return self.make_article(row)

    def make_article(self, this_row):
        row = ""
        row += "{code} & {year} & {{\\bf {title}}}, {authors}. & \\emph{{ {publisher} }}.  & {category}".format(  # NOQA
            code=tex_escape(this_row['NUM']),
            year=tex_escape(str(round(this_row['YEAR']))),
            title=tex_escape(this_row['TITLE']),
            authors=tex_escape(this_row['authors']),
            #doi=self.doi(this_row['DOI']),
            publisher=tex_escape(this_row['PUBLISHER']),
            category=tex_escape(self.category_lookup(this_row['Type']))
        )
        row += "\\\\"
        return row

class InPress(Publications):

    def __init__(self, name='InPress',
            csv_file=None,  category='A',
            template_file='biobib/InPressPublications.template'):
        super(InPress, self).__init__(
            name=name, csv_file=csv_file,
            template_file=template_file, category=category)
        self.category = category

class Submitted(Publications):

    def __init__(self, name='Submitted', csv_file=None,  category='R',
            template_file='biobib/InPressPublications.template'):
        super(Submitted, self).__init__(
            name=name, csv_file=csv_file,
            template_file=template_file, category=category)
        self.category = category

class Courses(Table):

    def __init__(self, name='Courses', csv_file=None, cumulative=False,
            template_file='biobib/Courses.template'):
        super(Courses, self).__init__(
            name=name, csv_file=csv_file, template_file=template_file)
        self.cumulative = cumulative
        self.df = self.clean_df()
        
    def render_template(self):
        rendered_tex = self.template.render(
            created=time.strftime("%Y-%m-%d %H:%M"),
            courses=self.df.to_dict('records')
        )
        return rendered_tex

    def clean_df(self):
        df = self.df
        # Step 1: drop any service from before this eval period
        df = self.clean_cumulative(df)
        df = df.sort_values(['Year', 'Q', 'Title'], ascending=[True, True, True])  # NOQA
        return df

class MESM(Table):

    def __init__(self, name='MESMProject', csv_file=None, cumulative=False,
            template_file='biobib/MESMProjects.template'):
        super(MESM, self).__init__(
            name=name, csv_file=csv_file, template_file=template_file)
        self.cumulative = cumulative
        self.df = self.clean_df()

class Undergrads(Table):

    def __init__(self, name='Undergradautes', csv_file=None, cumulative=False):
        super(Undergrads, self).__init__(
            name=name, csv_file=csv_file, template_file=template_file)
        self.cumulative = cumulative
        self.df = self.clean_df()


class Visitors(Table):

    def __init__(self, name='Visitors', csv_file=None, cumulative=False):
        super(Visitors, self).__init__(
            name=name, csv_file=csv_file, template_file=template_file)
        self.cumulative = cumulative
        self.df = self.clean_df()


class GraduateAdvising(Table):

    def __init__(self, name='GraduateAdvising', csv_file=None, cumulative=False,   # NOQA
                template_file='biobib/GradAdvising.template'):
        super(GraduateAdvising, self).__init__(
            name=name, csv_file=csv_file, template_file=template_file)
        self.cumulative = cumulative
        self.df = self.clean_df()
        
    def clean_df(self):
        df = self.df
        # Step 1: drop any committee work from prior evaluation
        df = self.clean_cumulative(df)
        df = df.sort_values(
            by=['Year', 'Role', 'Student'], ascending=[True, True, True])
        return df


class PostdoctoralAdvising(Table):

    def __init__(self, name='PostdoctoralAdvising', csv_file=None, cumulative=False,  # NOQA
            template_file='biobib/PostdoctoralAdvising.template'): 
        super(PostdoctoralAdvising, self).__init__(
            name=name, csv_file=csv_file, template_file=template_file)
        self.cumulative = cumulative
        self.df = self.clean_df()

    def clean_df(self, cumulative=False):
        df = self.df
        # Step 1: drop any advising work from prior evaluation
        df = self.clean_cumulative(df)
        return df


class Lectures(Table):

    def __init__(self, name='Lectures', csv_file=None, cumulative=False,
            template_file='biobib/Lectures.template'):
        super(Lectures, self).__init__(
            name=name,
            csv_file=csv_file,
            template_file=template_file)
        self.cumulative = cumulative
        self.df = self.clean_df()

    def clean_df(self, cumulative=False):
        df = self.df
        df = self.clean_cumulative(df)
        df = df.sort_values(by=['Year', 'Month'])
        return df


class Proceedings(Lectures):

    def __init__(self, name='Proceedings', csv_file=None, cumulative=False,
            template_file='biobib/Proceedings.template'):
        super(Lectures, self).__init__(name=name, csv_file=csv_file, template_file=template_file)
        self.cumulative = cumulative
        self.df = self.clean_df(cumulative=cumulative)

    def clean_df(self, cumulative=False):
        df = self.df
        df = self.clean_cumulative(df)
        # df['Topic'] = df['Title'] + ". " + df['Authors']
        df = df.sort_values(by=['Year'], ascending=[True])
        df.Year = df.Year.astype(int)
        return df


class Funding(Table):

    def __init__(self, name='Funding', csv_file=None, cumulative=False,
            template_file='biobib/Funding.template'):
        self.filters = {
            'make_years': self.make_years
        }
        super(Funding, self).__init__(
            name=name, csv_file=csv_file,
            template_file=template_file, filters=self.filters)
        self.cumulative = cumulative
        self.df = self.clean_df()
        
    def clean_df(self):
        df = self.df
        df = self.clean_cumulative(df)
        # Replace NaN with a 'nan' string for checking later
        df['Start Date'].fillna('nan', inplace=True)
        df['End Date'].fillna('nan', inplace=True)
        df = df.sort_values(by=['Start Year', 'Amount'],
                            ascending=[True, False])
        return df

    def make_years(self, row):
        if row['Start Date'] != 'nan' and row['End Date'] != 'nan':
            return "{start}-{end}".format(
                start=row['Start Date'],
                end=row['End Date'])
        else:
            return "{start}-{end}".format(
                start=row['Start Year'],
                end=row['End Year'])


class Reviews(Table):

    def __init__(self, name='Reviews', csv_file=None, cumulative=False,
            template_file='biobib/Reviews.template'):
        self.filters = {
            'add_count': self.add_count
        }
        super(Reviews, self).__init__(
            name=name, csv_file=csv_file,
            template_file=template_file, filters=self.filters)
        self.cumulative = cumulative
        self.df = self.clean_df()
        
    def clean_df(self):
        df = self.df
        df = self.clean_cumulative(df)
        df = df.sort_values(by=['Year', 'Role'], ascending=[True, False])
        df = (df.groupby(['Year', 'Role', 'Journal or Agency'])
              .size()
              .to_frame(name='count')
              .reset_index()
              )
        return df

    def add_count(self, count):
        if count > 1:
            return "({count})".format(count=count)
        else:
            return ""
