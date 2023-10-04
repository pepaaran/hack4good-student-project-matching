# This is the stripped down file to run the student-project matching of Hack4Good

import pandas as pd
import numpy as np
from mip import Model, MINIMIZE, BINARY, CONTINUOUS, xsum
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='H4G matching')

parser.add_argument('-M', '--max_team_size', default=5, type=int,
                    help='maximum number of students per team')

parser.add_argument('-m', '--min_team_size', default=3, type=int,
                    help='minimum number of students per team')

parser.add_argument('-d', '--ds_skill_q', default=0.6, type=float,
                    help='best data science skills considered for students above this quantile')

parser.add_argument('-p', '--pm_skill_q', default=0.6, type=float,
                    help='best project management skills considered for students above this quantile')

args = parser.parse_args()

print("Starting student-project matching.\n")

#### PARAMETER DEFINITION ####

# team size bounds
MAX_TEAM_SIZE = args.max_team_size
MIN_TEAM_SIZE = args.min_team_size

# define skill quantiles as threshold, because scores are normalised
DS_SKILLED_QUANTILE = args.ds_skill_q
PM_SKILLED_QUANTILE = args.pm_skill_q

# NOTE: Project names must coincide with the column names in the matching_input.xlsx table
PROJECTS = ['OECD', 'Impact Initiatives', 'WWF', 'BASE', 'IFPRI', 'CSS', 'HRW']
print("Projects:", PROJECTS, "\n")

print("Student scores are taken from matching_input.csv and the assigned projects are saved as matching_output.csv\n")

#### DATA PREPARATION ####

# read data
students = pd.read_csv('matching_input.csv')

# define skill threshold
DS_SKILLED_THRESHOLD = np.quantile(students['ds_skill'], q = DS_SKILLED_QUANTILE)
PM_SKILLED_THRESHOLD = np.quantile(students['pm_skill'], q = PM_SKILLED_QUANTILE)

# decide whether students are data science / project management skilled (based on thresholds)
students['data_science_skilled'] = 0
students.loc[students['ds_skill'] >= DS_SKILLED_THRESHOLD, 'data_science_skilled'] = 1
students['project_management_skilled'] = 0
students.loc[students['pm_skill'] >= PM_SKILLED_THRESHOLD, 'project_management_skilled'] = 1

# create column is_male
students['is_male'] = 0
students.loc[students['gender'] == 'Male', 'is_male'] = 1

# calculate number of students
number_students = students.shape[0]

#### SET UP MODEL ####

m = Model(name="matching_ip", sense=MINIMIZE)

#### DECLARE DECISION VARIABLES ####

# set up assignment variables, where 
# x[i][p] = 1 if student i is assigned to project p
# x[i][p] = 0 if student i is not assigned to project p
x = {}
for i in range(number_students):
    x[i] = {p: m.add_var(f'x_{i}_{p}', var_type=BINARY) for p in PROJECTS}

# set up diversity coefficient for each project
gender_diversity = {project: m.add_var(f'gd_{project}', lb=0, var_type=CONTINUOUS) for project in PROJECTS}

#### IMPOSE STRICT REQUIREMENTS ####

# every student is assigned to precisely one project
for i in range(number_students):
    m += xsum(x[i][project] for project in PROJECTS) == 1

for project in PROJECTS:
    # each project gets assigned at least MIN_TEAM_SIZE students
    m += xsum(x[i][project] for i in range(number_students)) >= MIN_TEAM_SIZE

    # each project gets assigned at most MAX_TEAM_SIZE students
    m += xsum(x[i][project] for i in range(number_students)) <= MAX_TEAM_SIZE

    # each project must have at least one data science skilled student
    m += xsum(students.loc[i, 'data_science_skilled'] * x[i][project] for i in range(number_students)) >= 1

    # each project must have at least one project management skilled student
    m += xsum(students.loc[i, 'project_management_skilled'] * x[i][project] for i in range(number_students)) >= 1

#### IMPOSE OBJECTIVE FUNCTION ####

# this part of the objective function makes sure students get assigned to a project they like
priority_objective_function = xsum(
    students.iloc[i][project]*x[i][project] for i in range(number_students) for project in PROJECTS
)

# calculate the gender diversity coefficient for every project
for project in PROJECTS:
    number_of_males = xsum(x[i][project] * students.loc[i, 'is_male'] for i in range(number_students))
    number_of_females = xsum(x[i][project] * (1-students.loc[i, 'is_male']) for i in range(number_students))
    gender_diversity[project] >=  number_of_males - number_of_females
    gender_diversity[project] >=  - number_of_males + number_of_females


# this part of the objective function makes sure that we favour gender-diverse teams
gender_diversity_objective_function = xsum(
    gender_diversity[project] for project in PROJECTS
)


m += priority_objective_function + gender_diversity_objective_function

#### SOLVE THE MIP ####

m.optimize()

#### WRITE OUTPUT ####

# write assigned project in a new column in the dataframe
students['assigned_project'] = ''
for i in range(number_students):
    for project in PROJECTS:
        if x[i][project].x == 1:
            students.loc[i, 'assigned_project'] = project

# export to Excel
students.to_csv(f'matching_output_min{MIN_TEAM_SIZE}_max{MAX_TEAM_SIZE}.csv')