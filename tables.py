import pandas as pd
import numpy as np
import time
import re

Categories = {
    'RA': 'Refereed Article',
    'CA': 'Conference Abstract'
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
    regex = re.compile('|'.join(re.escape(key)
                                for key in sorted(conv.keys(), key=lambda item: - len(item))))
    result = regex.sub(lambda match: conv[match.group()], text)
    if result == 'nan':
        result = ''
    return result


def str_join(df, sep, *cols):
    from functools import reduce
    return reduce(lambda x, y: x.astype(str).str.cat(y.astype(str), sep=sep),
                  [df[col] for col in cols])


class Table:

    def __init__(self, name=None, csv_file=None):
        self.name = name
        self.df = pd.read_csv(csv_file)
        # self.df = self.clean_df()

    def clean_df(self):
        return self.df

    def make_header(self):
        line = ""
        line += "\\tablehead{"
        line += ' & '.join(self.header_columns)
        line += "\\\\ \\hline}\n"
        return line

    def make_rows(self):
        rows = []
        if not self.df.empty:
            for index, row in self.df.iterrows():
                rows.append(self.make_row(row))
            return '\n'.join(rows)
        else:
            return ''

    def make_row(self, row=None):
        """ This function is specific to each Table """
        raise NotImplementedError

    def begin_super_tabular(self):
        """ This function is specific to each Table, but we have a default """
        return "\\begin{supertabuler}\n"

    def end_super_tabular(self):
        return "\\end{supertabular}\n"

    def make_table(self):
        table = ""
        table += self.make_header()
        table += self.begin_super_tabular()
        table += self.make_rows()
        table += "\n"
        table += self.end_super_tabular()
        return table

    def write_table(self, path=None):
        if path:
            file = path + self.name + '.tex'
        else:
            file = self.name + '.tex'
        f = open(file, 'w')
        if self.description:
            f.write("% {description}\n".format(description=self.description))
        f.write("% Created on {created}\n".format(
            created=time.strftime("%Y-%m-%d %H:%M")))
        f.write("\n")
        f.write(self.make_table())
        f.write("\n")
        f.close


class Service(Table):

    def __init__(
            self,
            name='Service',
            csv_file=None,
            category='P',
            cumulative=False):
        super(Service, self).__init__(name=name, csv_file=csv_file)
        self.header_columns = [
            'Year', 'Service'
        ]
        self.description = "UC Bio-bib Sevice Table"
        self.cumulative = cumulative
        self.category = category
        self.df = self.clean_df()

    def clean_df(self):
        df = self.df
        # Step 1: drop any service from before this eval period
        if self.cumulative is False:
            df = df[df.Eval == 1]
        # Step 2: filter to only the current category
        df = df[df.Type == self.category]
        df = df.sort_values(by=['Year'], ascending=[True])
        return df

    def begin_super_tabular(self):
        return "\\begin{supertabular}{lp{15cm}} \n"

    def make_row(self, this_row):
        row = ""
        row += "{year} & {role}, {committee} ".format(  # NOQA
                            year=tex_escape(this_row['Year']),
                            role=tex_escape(str(this_row['Role'])),
                            committee=tex_escape(str(this_row['Committee']))
                        )
        row += "\\\\"
        return row


class ProfessionalService(Service):

    def __init__(
            self,
            name='ProfessionalService',
            csv_file=None,
            category='P',
            cumulative=False):
        super(ProfessionalService, self).__init__(
            name=name, csv_file=csv_file, category=category, cumulative=cumulative)
        self.description = "UC Bio-bib Professional Service Table"
        self.category = category


class UniversityService(Service):

    def __init__(
            self,
            name='UniversityService',
            csv_file=None,
            category='U',
            cumulative=False):
        super(UniversityService, self).__init__(
            name=name, csv_file=csv_file, category=category, cumulative=cumulative)
        self.description = "UC Bio-bib University Service Table"
        self.category = category


class DepartmentalService(Service):

    def __init__(
            self,
            name='DepartmentalService',
            csv_file=None,
            category='D',
            cumulative=False):
        super(DepartmentalService, self).__init__(
            name=name, csv_file=csv_file, category=category, cumulative=cumulative)
        self.description = "UC Bio-bib Departmental Service Table"
        self.category = category


class Publications(Table):

    def __init__(self, name='Publications', csv_file=None,  category='P'):
        super(Publications, self).__init__(name=name, csv_file=csv_file)
        self.header_columns = ['\\#', 'Year',
                               'Title and Authors', 'Publisher', 'Category']
        self.description = "UC Bio-bib Publication Table"
        self.category = category
        self.cumulative = True  # Always provide complete publication list
        self.df = self.clean_df()

    def clean_df(self):
        df = self.df
        # Step 1: drop any papers not published
        df = df[df.S == self.category]
        # Step 2: Concatenate authors into a single list, making sure to drop
        # empty author columns
        df['authors'] = list(
            pd.Series(df[['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11']]  # NOQA
                      .fillna('').values.tolist())
            .apply(lambda x: [i for i in x if i != ''])
            .apply(lambda x: ', '.join(x))
        )
        # Step 3: Cast DOI as a string and remove nan
        df.loc[df['DOI'] == 'nan', 'DOI'] = np.nan
        return df

    def href(self, this_doi):
        if this_doi is np.NaN:
            return ""
        else:
            return "[\\url{{ https://doi.org/{doi} }}]".format(doi=this_doi)

    def begin_super_tabular(self):
        return "\\begin{supertabular}{lcp{8.25cm}p{4.75cm}p{1.75cm}}\n"

    def make_rows(self):
        rows = []
        old = self.df[self.df['New?'] != 'Y']
        new = self.df[self.df['New?'] == 'Y']
        for index, row in old.iterrows():
            rows.append(self.make_row(row))
        rows[-1] += '\\hline'
        rows[-1] += '\\hline'
        for index, row in new.iterrows():
            rows.append(self.make_row(row))
        return '\n'.join(rows)

    def make_row(self, this_row):
        row = ""
        row += "{code} & {year} & {{\\bf {title}}}, {authors} {href} & \\emph{{ {journal} }} & {category}".format(
            code=tex_escape(this_row['NUM']),
            year=tex_escape(str(this_row['YEAR'])),
            title=tex_escape(this_row['TITLE']),
            authors=tex_escape(this_row['authors']),
            href=self.href(this_row['DOI']),
            journal=tex_escape(this_row['JOURNAL']),
            category=tex_escape(Categories[this_row['Type']])
        )
        row += "\\\\"
        return row


class InPress(Publications):

    def __init__(self, name='InPress', csv_file=None,  category='A'):
        super(InPress, self).__init__(
            name=name, csv_file=csv_file, category=category)
        self.description = "UC Bio-bib Publications In Press Table"
        self.category = category

    def make_rows(self):
        rows = []
        for index, row in self.df.iterrows():
            rows.append(self.make_row(row))
        return '\n'.join(rows)


class Submitted(Publications):

    def __init__(self, name='Submitted', csv_file=None,  category='R'):
        super(Submitted, self).__init__(
            name=name, csv_file=csv_file, category=category)
        self.description = "UC Bio-bib Publications Submitted Table"
        self.category = category

    def make_rows(self):
        rows = []
        for index, row in self.df.iterrows():
            rows.append(self.make_row(row))
        return '\n'.join(rows)


class Courses(Table):

    def __init__(self, name='Courses', csv_file=None, cumulative=False):
        super(Courses, self).__init__(name=name, csv_file=csv_file)
        self.header_columns = ['Qtr', 'Course', 'Class Type',
                               'Units', 'Hrs/Wk', 'Enrollment', 'Evals Available']
        self.description = "UC Bio-bib Catalog Courses Table"
        self.cumulative = cumulative
        self.df = self.clean_df()

    def begin_super_tabular(self):
        # \begin{supertabular}{lp{6.5cm}lrrrc}
        return "\\begin{supertabular}{lp{6.5cm}lrrrc}\n"

    def make_row(self, this_row):
        row = ""
        row += "{quarter} & {course}, {title} & {type} & {units} & {hrs_per_week} & {enrollment} & {evals} ".format(  # NOQA
                            quarter=tex_escape(this_row['QYR']),
                            course=tex_escape(str(this_row['Course'])),
                            title=tex_escape(this_row['Title']),
                            type=tex_escape(this_row['Class Type']),
                            units=tex_escape(this_row['Units']),
                            hrs_per_week=tex_escape(
                                this_row['Hours per Week']),
                            enrollment=tex_escape(this_row['Enrollment']),
                            evals=tex_escape(this_row['Evals'])
                        )
        row += "\\\\"
        return row


class MESM(Table):

    def __init__(self, name='MESMProject', csv_file=None, cumulative=False):
        super(MESM, self).__init__(name=name, csv_file=csv_file)
        self.header_columns = ['Year', 'Project Title', 'Students',
                               'Q3', 'Q4', 'Q5', 'Q7']
        self.description = "UC Bio-bib MESM Projects Table"
        self.cumulative = cumulative
        self.df = self.clean_df()

    def begin_super_tabular(self):
        # \begin{supertabular}{lp{6.5cm}lrrrc}
        return "\\begin{supertabular}{lp{2.5cm}p{3cm}p{2cm}p{2cm}p{2cm}p{2cm}}\n"

    def make_row(self, this_row):
        row = ""
        row += "{year} & {title} & {students} & {Q3} & {Q4} & {Q5} & {Q7} ".format(  # NOQA
                            year=tex_escape(this_row['Year']),
                            title=tex_escape(str(this_row['Project Title'])),
                            students=tex_escape(this_row['Students']),
                            # students=make_cell(tex_escape(this_row['Students']), size="\\small"),  # NOQA
                            Q3=tex_escape(this_row['Q3']),
                            Q4=tex_escape(this_row['Q4']),
                            Q5=tex_escape(this_row['Q5']),
                            Q7=tex_escape(this_row['Q7'])
                            # Q3=make_cell(tex_escape(this_row['Q3']), size="\\footnotesize"),  # NOQA
                            # Q4=make_cell(tex_escape(this_row['Q4']), size="\\footnotesize"),  # NOQA
                            # Q5=make_cell(tex_escape(this_row['Q5']), size="\\footnotesize"),  # NOQA
                            # Q7=make_cell(tex_escape(this_row['Q7']), size="\\footnotesize"),  # NOQA
                        )
        row += "\\\\"
        return row


class GraduateAdvising(Table):

    def __init__(self, name='GraduateAdvising', csv_file=None):
        super(GraduateAdvising, self).__init__(name=name, csv_file=csv_file)
        self.header_columns = ['Student', 'Year',
                               'Instituion', 'Chair/Member', 'Current Employment']
        self.description = "UC Bio-bib Catalog GraduateAdvising Table"
        self.df = self.clean_df()

    def clean_df(self):
        df = self.df
        # Step 1: drop any committee work from prior evaluation
        df = df[df.Eval == 1]
        df = df.sort_values(
            by=['Year', 'Role', 'Student'], ascending=[True, True, True])
        return df

    def begin_super_tabular(self):
        return "\\begin{supertabular}{lp{1.5cm} p{3.5cm}p{2cm}p{4.5cm}}\n"

    def make_row(self, this_row):
        row = ""
        row += "{student} & {year} & {institution}, {department} & {role} & {title} - {organization}".format(  # NOQA
                            student=tex_escape(this_row['Student']),
                            year=tex_escape(str(this_row['Year'])),
                            institution=tex_escape(this_row['Institution']),
                            department=tex_escape(this_row['Department']),
                            role=tex_escape(this_row['Role']),
                            title=tex_escape(this_row['Title']),
                            organization=tex_escape(this_row['Organization'])
                        )
        row += "\\\\"
        return row


class PostdoctoralAdvising(Table):

    def __init__(self, name='PostdoctoralAdvising', csv_file=None):
        super(PostdoctoralAdvising, self).__init__(
            name=name, csv_file=csv_file)
        self.header_columns = [
            'Postdoctoral Researcher', 'Years', 'Affiliation', 'Current Employment']
        self.description = "UC Bio-bib Catalog Postdoctoral Advising Table"
        self.df = self.clean_df()

    def clean_df(self, cumulative=False):
        df = self.df
        # Step 1: drop any advising work from prior evaluation
        if cumulative is False:
            df = df[df.Eval == 1]
        return df

    def begin_super_tabular(self):
        return "\\begin{supertabular}{lp{1.5cm} p{3.5cm}p{4.5cm}}\n"

    def make_row(self, this_row):
        row = ""
        row += "{researcher} & {start_year} - {end_year} & {institution}, {department} &  {title} - {organization}".format(  # NOQA
                            researcher=tex_escape(this_row['Postdoc']),
                            start_year=tex_escape(str(this_row['Start Year'])),
                            end_year=tex_escape(str(this_row['End Year'])),
                            institution=tex_escape(this_row['Institution']),
                            department=tex_escape(this_row['Department']),
                            title=tex_escape(this_row['Title']),
                            organization=tex_escape(this_row['Organization'])
                        )
        row += "\\\\"
        return row


class Lectures(Table):

    def __init__(self, name='Lectures', csv_file=None, cumulative=False):
        super(Lectures, self).__init__(name=name, csv_file=csv_file)
        self.header_columns = ['Month/Year', 'Topic', 'Place/Conference']
        self.description = "UC Bio-bib Lectures Table"
        self.proceedings_file = 'CV/Conference Abstracts-Table.csv'
        self.cumulative = cumulative
        self.conferences = Proceedings(
            csv_file=self.proceedings_file,
            name='Proceedings',
            cumulative=self.cumulative)
        self.df = self.clean_df()

    def clean_df(self, cumulative=False):
        df = self.df
        df2 = self.conferences.df
        if cumulative is False:
            df = df[df.Eval == 1]
        df2['Topic'] = df2['Title'] + ". " + df2['Authors']
        df2['Place'] = df2['Conference']
        df = df.append(df2)
        df = df.sort_values(by=['Year', 'Month'])
        return df

    def begin_super_tabular(self):
        # \begin{supertabular}{lp{6.5cm}lrrrc}
        return "\\begin{supertabular}{lp{10.0cm}p{4.5cm}}\n"

    def make_row(self, this_row):
        row = ""
        row += "{month}/{year} & {topic} & {place} ".format(  # NOQA
                            month=tex_escape(this_row['Month']),
                            year=tex_escape(str(this_row['Year'])),
                            topic=tex_escape(this_row['Topic']),
                            place=tex_escape(this_row['Place'])
                        )
        row += "\\\\"
        return row


class Proceedings(Table):

    def __init__(self, name='Proceedings', csv_file=None, cumulative=True):
        super(Proceedings, self).__init__(name=name, csv_file=csv_file)
        self.header_columns = ['\\#', 'Year', 'Authors', 'Title', 'Venue']
        self.description = "UC Bio-bib Proceedings Table"
        self.cumulative = cumulative
        self.df = self.clean_df()

    def clean_df(self):
        df = self.df
        if self.cumulative is False:
            df = df[df.Eval == 1]
        df = df.sort_values(by=['Year'], ascending=[True])
        df.Year = df.Year.astype(int)
        return df

    def make_rows(self):
        rows = []
        df = self.df
        old = df[df['Eval'] == 0]
        new = df[df['Eval'] == 1]
        for index, row in old.iterrows():
            rows.append(self.make_row(row))
        rows[-1] += '\\hline'
        rows[-1] += '\\hline'
        for index, row in new.iterrows():
            rows.append(self.make_row(row))
        return '\n'.join(rows)

    def begin_super_tabular(self):
        return "\\begin{supertabular}{llp{6cm}p{6cm}p{3cm}}\n"

    def make_row(self, this_row):
        row = ""
        row += "{code} & {year} & {authors} & {title} & {venue}".format(
            code=tex_escape(this_row['CODE']),
            year=tex_escape(this_row['Year']),
            authors=tex_escape(this_row['Authors']),
            title=tex_escape(this_row['Title']),
            venue=tex_escape(this_row['Conference'])
        )
        row += "\\\\"
        return row


class Funding(Table):

    def __init__(self, name='Funding', csv_file=None, cumulative=False):
        super(Funding, self).__init__(name=name, csv_file=csv_file)
        self.header_columns = ['Year', 'Source', 'Title', 'Amount', 'Role']
        self.description = "UC Bio-bib Funding Table"
        self.cumulative = cumulative
        self.df = self.clean_df()

    def clean_df(self):
        df = self.df
        if self.cumulative is False:
            df = df[df.Eval == 1]
        df = df.sort_values(by=['Start Year', 'Amount'],
                            ascending=[True, False])
        return df

    def begin_super_tabular(self):
        # \begin{supertabular}{lp{6.5cm}lrrrc}
        return "\\begin{supertabular}{lp{3.5cm}p{7cm}ll}\n"

    def make_row(self, this_row):
        row = ""
        row += "{year} & {source} & {title} & {amount} & {role} ".format(  # NOQA
                            year=tex_escape(this_row['Start Year']),
                            source=tex_escape(str(this_row['Source'])),
                            title=tex_escape(this_row['Title']),
                            amount=tex_escape(this_row['Amount']),
                            role=tex_escape(this_row['Role'])
                        )
        row += "\\\\"
        return row


class Reviews(Table):

    def __init__(self, name='Reviews', csv_file=None, cumulative=False):
        super(Reviews, self).__init__(name=name, csv_file=csv_file)
        self.header_columns = ['Year', 'Activity', 'Journal/Agency']
        self.description = "UC Bio-bib Reveiwer Activity Table"
        self.cumulative = cumulative
        self.df = self.clean_df()

    def clean_df(self):
        df = self.df
        if self.cumulative is False:
            df = df[df.Eval == 1]
        df = df.sort_values(by=['Year', 'Role'], ascending=[True, False])
        df = (df.groupby(['Year', 'Role', 'Journal or Agency'])
              .size()
              .to_frame(name='count')
              .reset_index()
              )
        return df

    def begin_super_tabular(self):
        # \begin{supertabular}{lp{6.5cm}lrrrc}
        return "\\begin{supertabular}{llp{12cm}}\n"

    def add_count(self, this_count=1):
        if this_count > 1:
            return tex_escape("({count})".format(count=this_count))
        else:
            return ""

    def make_row(self, this_row):
        row = ""
        row += "{year} & {role} & {org}  {count} ".format(  # NOQA
                            year=tex_escape(this_row['Year']),
                            role=tex_escape(str(this_row['Role'])),
                            org=tex_escape(this_row['Journal or Agency']),
                            count=self.add_count(this_row['count'])
                        )
        row += "\\\\"
        return row
