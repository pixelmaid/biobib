import pandas as pd
import numpy as np
import time
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


def colonify(string):
    if string:
        return ": " + string
    else:
        return ""


class Table:

    def __init__(self, name=None, csv_file=None):
        self.name = name
        self.df = pd.read_csv(csv_file)
        self.columns = ""
        self.type = "longtable"
        # self.df = self.clean_df()

    def clean_df(self):
        return self.df

    def make_header(self):
        line = ""
        # line += "\\tablehead{"
        line += ' & '.join(self.header_columns) + "\\\\\n"
        line += "\\hline \n"
        line += "\\endhead \n"
        return line

    def make_rows(self):
        rows = []
        if not self.df.empty:
            for index, row in self.df.iterrows():
                rows.append(self.make_row(row))
            return '\n'.join(rows)
        else:
            return ''

    def begin_table(self):
        if self.type is "super_tabular":
            return self.begin_super_tabular()
        elif self.type is "longtable":
            return self.begin_longtable()

    def end_table(self):
        if self.type is "super_tabular":
            return self.end_super_tabular()
        elif self.type is "longtable":
            return self.end_longtable()

    def make_row(self, row=None):
        """ This function is specific to each Table """
        raise NotImplementedError

    def begin_super_tabular(self):
        """ This function is generalized to use columns defined in each table """  # NOQA
        return "\\begin{supertabular}" + self.columns + "\n"

    def end_super_tabular(self):
        return "\\end{supertabular}\n"

    def begin_longtable(self):
        """ This function is generalized to use columns defined in each table """  # NOQA
        return "\\begin{longtable}" + self.columns + "\n"

    def end_longtable(self):
        return "\\end{longtable}\n"

    def make_table(self):
        table = ""
        table += self.begin_table()
        table += self.make_header()
        # table += self.begin_super_tabular()
        # table += self.begin_longtable()
        table += self.make_rows()
        table += "\n"
        # table += self.end_super_tabular()
        # table += self.end_longtable()
        table += self.end_table()
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
        self.type = "longtable"
        self.columns = "{lp{15cm}}"
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

    # def begin_super_tabular(self):
    #    return "\\begin{supertabular}{lp{15cm}} \n"

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
            name=name, csv_file=csv_file,
            category=category, cumulative=cumulative)
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
            name=name, csv_file=csv_file,
            category=category, cumulative=cumulative)
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
            name=name, csv_file=csv_file,
            category=category, cumulative=cumulative)
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
        self.type = "longtable"
        self.columns = "{lcp{7.75cm}>{\\raggedright}p{5.25cm}p{1.75cm}}"

    def clean_df(self):
        df = self.df
        # Step 1: drop any papers not published
        df = df[df.S == self.category]
        # Step 2: Concatenate authors into a single list, making sure to drop
        # empty author columns
        df['authors'] = list(
            pd.Series(df[['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13']]  # NOQA
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

    def make_rows(self):
        rows = []
        old = self.df[self.df['New?'] != 'Y']
        new = self.df[self.df['New?'] == 'Y']
        for index, row in old.iterrows():
            rows.append(self.make_row(row))
        rows[-1] += '\\hline'
        rows[-1] += '\\hline'
        rows[-1] += "   &   & {\\bf Since Appointment:} &    &   \\\\"
        for index, row in new.iterrows():
            rows.append(self.make_row(row))
        return '\n'.join(rows)

    def make_row(self, row):
        if row['Type'] == 'RA':
            return self.make_article(row)
        elif row['Type'] == 'BC':
            return self.make_chapter(row)
        elif row['Type'] == 'CP':
            return self.make_chapter(row)

    def make_article(self, this_row):
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
            category=tex_escape(Categories[this_row['Type']])
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
                               'Units', 'Hrs/Wk', 'Enrollment',
                               'ESCI/Written Evals Avail.']
        self.description = "UC Bio-bib Catalog Courses Table"
        self.cumulative = cumulative
        self.df = self.clean_df()
        self.type = "longtable"
        self.columns = "{lp{6.5cm}p{1cm}rrrp{2cm}}"

    def clean_df(self):
        df = self.df
        df = df.sort_values(['Year', 'Q', 'Title'], ascending=[True, True, True])  # NOQA
        return df

    # def begin_super_tabular(self):
    #     # \begin{supertabular}{lp{6.5cm}lrrrc}
    #     return "\\begin{supertabular}{lp{6.5cm}p{1cm}rrrp{2cm}}\n"

    def make_row(self, this_row):
        row = ""
        row += "{quarter} & {course}, {title} & {type} & {units} & {hrs_per_week} & {enrollment} & {esci}/{evals} ".format(  # NOQA
                            quarter=tex_escape(this_row['QYR']),
                            course=tex_escape(str(this_row['Course'])),
                            title=tex_escape(this_row['Title']),
                            type=tex_escape(this_row['Class Type']),
                            units=tex_escape(this_row['Units']),
                            hrs_per_week=tex_escape(
                                this_row['Hours per Week']),
                            enrollment=tex_escape(this_row['Enrollment']),
                            esci=tex_escape(this_row['ESCI']),
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
        self.columns = "{p{1cm}p{2.5cm}p{3cm}p{2cm}p{2cm}p{2cm}p{2cm}}"

    # def begin_super_tabular(self):
    #     # \begin{supertabular}{lp{6.5cm}lrrrc}
    #     return "\\begin{supertabular}{p{1cm}p{2.5cm}p{3cm}p{2cm}p{2cm}p{2cm}p{2cm}}\n"

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
        self.columns = "{lp{1.5cm} p{4.5cm}p{2cm}p{4cm}}"

    def clean_df(self):
        df = self.df
        # Step 1: drop any committee work from prior evaluation
        df = df[df.Eval == 1]
        df = df.sort_values(
            by=['Year', 'Role', 'Student'], ascending=[True, True, True])
        return df

    # def begin_super_tabular(self):
    #     return "\\begin{supertabular}{lp{1.5cm} p{4.5cm}p{2cm}p{4cm}}\n"

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
            'Postdoctoral Researcher', 'Years',
            'Affiliation', 'Current Employment']
        self.description = "UC Bio-bib Catalog Postdoctoral Advising Table"
        self.df = self.clean_df()
        self.columns = "{lp{1.5cm} p{3.5cm}p{4.5cm}}"

    def clean_df(self, cumulative=False):
        df = self.df
        # Step 1: drop any advising work from prior evaluation
        if cumulative is False:
            df = df[df.Eval == 1]
        return df

    # def begin_super_tabular(self):
    #     return "\\begin{supertabular}{lp{1.5cm} p{3.5cm}p{4.5cm}}\n"

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
        self.header_columns = ['Month/Year', 'Title', 'Meeting/Place']
        self.description = "UC Bio-bib Lectures Table"
        self.cumulative = cumulative
        self.df = self.clean_df()
        self.columns = "{lp{10.0cm}p{4.5cm}}"

    def clean_df(self, cumulative=False):
        df = self.df
        if cumulative is False:
            df = df[df.Eval == 1]
        df = df.sort_values(by=['Year', 'Month'])
        return df

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


class Proceedings(Lectures):

    def __init__(self, name='Proceedings', csv_file=None, cumulative=False):
        super(Lectures, self).__init__(name=name, csv_file=csv_file)
        self.header_columns = ['Month/Year', 'Title', 'Meeting/Place']
        self.description = "UC Bio-bib Proceedings Table"
        self.cumulative = cumulative
        self.df = self.clean_df(cumulative=cumulative)
        self.columns = "{lp{10.0cm}p{4.5cm}}"

    def clean_df(self, cumulative=False):
        df = self.df
        if cumulative is False:
            df = df[df.Eval == 1]
        # df['Topic'] = df['Title'] + ". " + df['Authors']
        df = df.sort_values(by=['Year'], ascending=[True])
        df.Year = df.Year.astype(int)
        return df

    def invited(self, this_row):
        if this_row['Invited'] == 1:
            return "(INVITED)"
        else:
            return ""

    def make_row(self, this_row):
        row = ""
        row += " {month}/{year} & {{\\bf {title}}}. {authors} {invited} & {venue}".format(  # NOQA
            year=tex_escape(this_row['Year']),
            month=tex_escape(this_row['Month']),
            invited=tex_escape(self.invited(this_row)),
            title=tex_escape(this_row['Title']),
            authors=tex_escape(this_row['Authors']),
            venue=tex_escape(this_row['Conference'])
        )
        row += "\\\\"
        return row


class Funding(Table):

    def __init__(self, name='Funding', csv_file=None, cumulative=False):
        super(Funding, self).__init__(name=name, csv_file=csv_file)
        self.header_columns = [
            'Year', 'Source', 'Title',
            'Role', 'Amount', 'Personal Share', 'New/Cont.'
        ]
        self.description = "UC Bio-bib Funding Table"
        self.cumulative = cumulative
        self.df = self.clean_df()
        self.columns = "{p{1.75cm}>{\\raggedright}p{2.75cm}p{5.5cm}p{1cm}p{1.25cm}p{1.25cm}p{1cm}}"  # NOQA

    def clean_df(self):
        df = self.df
        if self.cumulative is False:
            df = df[df.Eval == 1]
        # Replace NaN with a 'nan' string for checking later
        df['Start Date'].fillna('nan', inplace=True)
        df['End Date'].fillna('nan', inplace=True)
        # df.loc[df['Start Date'] == 'nan', 'Start Date'] = None
        # df.loc[df['End Date'] == 'nan', 'End Date'] = None
        df = df.sort_values(by=['Start Year', 'Amount'],
                            ascending=[True, False])
        return df

    # def begin_super_tabular(self):
    #     # \begin{supertabular}{lp{6.5cm}lrrrc}
    #     return "\\begin{supertabular}{lp{3.5cm}p{7cm}ll}\n"

    def make_role(self, row):
        if row['Pooled Funds']:
            return row['Role'] + " (pooled funds)"
        else:
            return row['Role']

    def make_years(self, row):
        if row['Start Date'] != 'nan' and row['End Date'] != 'nan':
            return "{start}-{end}".format(
                start=row['Start Date'],
                end=row['End Date'])
        else:
            return "{start}-{end}".format(
                start=row['Start Year'],
                end=row['End Year'])

    def make_row(self, this_row):
        row = ""
        row += "{year} & {source} & {title} & {role} & {amount} & {share} & {type}".format(  # NOQA
                            year=tex_escape(self.make_years(this_row)),
                            source=tex_escape(str(this_row['Source'])),
                            title=tex_escape(this_row['Title']),
                            amount=tex_escape(this_row['Amount']),
                            share=tex_escape(this_row['Personal Share']),
                            role=tex_escape(self.make_role(this_row)),
                            type=tex_escape(this_row['Type'])
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
        self.columns = "{llp{12cm}}"

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

    # def begin_super_tabular(self):
    #     # \begin{supertabular}{lp{6.5cm}lrrrc}
    #     return "\\begin{supertabular}{llp{12cm}}\n"

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
