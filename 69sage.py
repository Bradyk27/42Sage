# %%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import re
import seaborn as sns
import numpy as np
import math
from collections import Counter
import matplotlib.ticker as ticker
from datetime import date
import os
import shutil
import warnings
import argparse
warnings.filterwarnings("ignore")

SMALL_SIZE = 8
MEDIUM_SIZE = 15
BIGGER_SIZE = 25
COLOR = '#AAAAAA'
palette = 'winter'

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'black', 'axes.edgecolor':'darkgrey', 'grid.color':'black'})
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR
plt.rc("axes.spines", top=False, right=False)
rcParams.update({'figure.autolayout': True})

top_dir = "results/"+str(date.today())
lr_dir = top_dir + "/LR"
lg_dir = top_dir + "/LG"
rc_dir = top_dir + "/RC"


argParser = argparse.ArgumentParser()
argParser.add_argument("-t", "--time", help="time limit", default=35)
argParser.add_argument("-m", "--missed", help="missed questions", default=2)
args = argParser.parse_args()
time_limit = float(args.time)
missed_questions = float(args.missed)

# %%
if os.path.exists(top_dir):
    shutil.rmtree(top_dir)

os.mkdir(top_dir)
os.mkdir(lr_dir)
os.mkdir(rc_dir)
os.mkdir(lg_dir)

# %%
def to_minute_fraction(x):
    l = x.split(':')
    return int(l[0]) + (int(l[1]) / 60)

def adapted_moving_average(x, i):
    if i == 0:
        return x
    return df['minute_per_correct_question'][0:i].mean()

# %% [markdown]
# # Logic Games

# %%
df = pd.read_excel('data/scores_times_lg.xlsx')
df['Time'] = df['Time'].apply(lambda x: to_minute_fraction(x))
df['Test Number'] = df.index+1
df['minute_per_correct_question'] = df['Time'] / (df['Total Questions']-df['Questions Wrong'])
df['minute_per_correct_question_ra'] = df.apply(lambda row: adapted_moving_average(row['minute_per_correct_question'], row.name), axis=1)
df['minute_per_correct_question_ra_5'] = df['minute_per_correct_question'].rolling(5, min_periods=1).mean()

# %%
plt.figure(figsize=(10,6))
norm = plt.Normalize(df['Test Number'].min(), df['Test Number'].max())
sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
sm.set_array([])

ax = sns.scatterplot(x='Questions Wrong', y='Time', data=df, palette=palette, hue='Test Number', s=200, linewidth=0)
ax.get_legend().remove()
cbar = ax.figure.colorbar(sm, ticks=[df['Test Number'].min(),df['Test Number'].max()])

plt.axhline(time_limit, ls='--', color='green')
plt.axvline(missed_questions, ls='--', color='red')

plt.title('Logic Games, Time vs. Questions Wrong', fontdict={'fontsize': BIGGER_SIZE})
plt.savefig(lg_dir + "/time_v_qw.png")

# %%
perfect_score = time_limit / (df['Total Questions'].mean())

plt.figure(figsize=(10,6))
sns.lineplot(data=df, x=df['Test Number'], y='minute_per_correct_question', marker='o')
plt.xticks(list(range(0, df['Test Number'].max(), 5))[1:])
plt.ylabel('MPCQ', labelpad=20)
plt.xlabel('Test Number', labelpad=20)
plt.axhline(perfect_score, ls='--', color='green')
plt.title('Logic Games, Minutes Per Correct Question', fontdict={'fontsize': BIGGER_SIZE})
plt.savefig(lg_dir + "/mpcq.png")

# %%
plt.figure(figsize=(10,6))
sns.lineplot(data=df, x=df['Test Number'], y='minute_per_correct_question_ra', marker='o')
plt.xticks(list(range(0, df['Test Number'].max(), 5))[1:])
plt.ylabel('MPCQ', labelpad=20)
plt.xlabel('Test Number', labelpad=20)
plt.ylim(1, 5)
plt.axhline(perfect_score, ls='--', color='green')
plt.title('Logic Games, Minutes Per Correct Question RA', fontdict={'fontsize': BIGGER_SIZE})
plt.savefig(lg_dir + "/mpcq_ra.png")

# %%
plt.figure(figsize=(10,6))
ax = sns.lineplot(data=df, x=df['Test Number'], y='minute_per_correct_question_ra_5', marker='o')
plt.xticks(list(range(0, df['Test Number'].max(), 5))[1:])
plt.ylabel('MPCQ', labelpad=20)
plt.xlabel('Test Number', labelpad=20)
plt.ylim(1, 5)
plt.axhline(perfect_score, ls='--', color='green')
plt.title('Logic Games, Minutes Per Correct Question RA 5', fontdict={'fontsize': BIGGER_SIZE})
plt.savefig(lg_dir + "/mpcq_ra5.png")

# %% [markdown]
# # Logical Reasoning

# %%
df = pd.read_excel('data/scores_times_lr.xlsx', sheet_name='Scores')
df['Time'] = df['Time'].apply(lambda x: to_minute_fraction(x))
df['Test Number'] = df.index+1
df['minute_per_correct_question'] = df['Time'] / (df['Total Questions']-df['Questions Wrong'])
df['minute_per_correct_question_ra'] = df.apply(lambda row: adapted_moving_average(row['minute_per_correct_question'], row.name), axis=1)
df['minute_per_correct_question_ra_5'] = df['minute_per_correct_question'].rolling(5, min_periods=1).mean()
df

# %%
plt.figure(figsize=(10,6))
norm = plt.Normalize(df['Test Number'].min(), df['Test Number'].max())
sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
sm.set_array([])

ax = sns.scatterplot(x='Questions Wrong', y='Time', data=df, palette=palette, hue='Test Number', s=200, linewidth=0)
ax.get_legend().remove()
cbar = ax.figure.colorbar(sm, ticks=[df['Test Number'].min(),df['Test Number'].max()])

plt.axhline(time_limit, ls='--', color='green')
plt.axvline(missed_questions, ls='--', color='red')

plt.title('Logical Reasoning, Time vs. Questions Wrong', fontdict={'fontsize': BIGGER_SIZE})

plt.savefig(lr_dir + "/time_v_qw.png")

# %%
perfect_score = time_limit / (df['Total Questions'].mean())

plt.figure(figsize=(10,6))
ax = sns.lineplot(data=df, x=df['Test Number'], y='minute_per_correct_question', marker='o')
plt.xticks(list(range(0, df['Test Number'].max(), 5))[1:])
plt.ylabel('MPCQ', labelpad=20)
plt.xlabel('Test Number', labelpad=20)
plt.axhline(perfect_score, ls='--', color='green')
plt.title('Logical Reasoning, Minutes Per Correct Question', fontdict={'fontsize': BIGGER_SIZE})
plt.savefig(lr_dir + "/mpcq.png")

# %%
plt.figure(figsize=(10,6))
ax = sns.lineplot(data=df, x=df['Test Number'], y='minute_per_correct_question_ra', marker='o')
plt.xticks(list(range(0, df['Test Number'].max(), 5))[1:])
plt.ylabel('MPCQ', labelpad=20)
plt.xlabel('Test Number', labelpad=20)
plt.axhline(perfect_score, ls='--', color='green')
plt.title('Logical Reasoning, Minutes Per Correct Question RA', fontdict={'fontsize': BIGGER_SIZE})
plt.savefig(lr_dir + "/mpcq_ra.png")

# %%
plt.figure(figsize=(10,6))
ax = sns.lineplot(data=df, x=df['Test Number'], y='minute_per_correct_question_ra_5', marker='o')
plt.xticks(list(range(0, df['Test Number'].max(), 5))[1:])
plt.ylabel('MPCQ', labelpad=20)
plt.xlabel('Test Number', labelpad=20)
plt.axhline(perfect_score, ls='--', color='green')
plt.title('Logical Reasoning, Minutes Per Correct Question RA 5', fontdict={'fontsize': BIGGER_SIZE})
plt.savefig(lr_dir + "/mpcq_ra5.png")

# %%
key = pd.read_excel('data/scores_times_lr.xlsx', sheet_name='Key')
key_dict = {}
for i, r in key.iterrows():
    key_dict[r['Key']] = r['Question Type']

missed_q = []
for v in df['Missed Question Type']:
    for q in v.split(','):
        missed_q.append(key_dict[q.replace(' ','')])

missed_q_df = pd.DataFrame(missed_q).rename(columns={0:'Question Type'})

# %%
plt.figure(figsize=(10,6))
ax = sns.countplot(x=missed_q_df['Question Type'], palette=palette)
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.ylabel('Number Wrong', labelpad=20)
plt.xlabel('')
plt.xticks(rotation=90)
plt.title('Logical Reasoning, Missed Question Type', fontdict={'fontsize': BIGGER_SIZE})
plt.savefig(lr_dir + "/missed_q.png")

# %% [markdown]
# # Reading Comprehension

# %%
df = pd.read_excel('data/scores_times_rc.xlsx', sheet_name='Scores')
df['Time'] = df['Time'].apply(lambda x: to_minute_fraction(x))
df['Test Number'] = df.index+1
df['minute_per_correct_question'] = df['Time'] / (df['Total Questions']-df['Questions Wrong'])
df['minute_per_correct_question_ra'] = df.apply(lambda row: adapted_moving_average(row['minute_per_correct_question'], row.name), axis=1)
df['minute_per_correct_question_ra_5'] = df['minute_per_correct_question'].rolling(5, min_periods=1).mean()
df

# %%
plt.figure(figsize=(10,6))
norm = plt.Normalize(df['Test Number'].min(), df['Test Number'].max())
sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
sm.set_array([])

ax = sns.scatterplot(x='Questions Wrong', y='Time', data=df, palette=palette, hue='Test Number', s=200, linewidth=0)
ax.get_legend().remove()
cbar = ax.figure.colorbar(sm, ticks=[df['Test Number'].min(),df['Test Number'].max()])

plt.axhline(time_limit, ls='--', color='green')
plt.axvline(missed_questions, ls='--', color='red')

plt.title('Reading Comprehension, Time vs. Questions Wrong', fontdict={'fontsize': BIGGER_SIZE})

plt.savefig(rc_dir + "/time_v_qw.png")

# %%
perfect_score = time_limit / (df['Total Questions'].mean())

plt.figure(figsize=(10,6))
ax = sns.lineplot(data=df, x=df['Test Number'], y='minute_per_correct_question', marker='o')
plt.xticks(list(range(0, df['Test Number'].max(), 5))[1:])
plt.ylabel('MPCQ', labelpad=20)
plt.xlabel('Test Number', labelpad=20)
plt.axhline(perfect_score, ls='--', color='green')
plt.title('Reading Comprehension, Minutes Per Correct Question', fontdict={'fontsize': BIGGER_SIZE})
plt.savefig(rc_dir + "/mpcq.png")

# %%
plt.figure(figsize=(10,6))
ax = sns.lineplot(data=df, x=df['Test Number'], y='minute_per_correct_question_ra', marker='o')
plt.xticks(list(range(0, df['Test Number'].max(), 5))[1:])
plt.ylabel('MPCQ', labelpad=20)
plt.xlabel('Test Number', labelpad=20)
plt.axhline(perfect_score, ls='--', color='green')
plt.title('Reading Comprehension, Minutes Per Correct Question RA', fontdict={'fontsize': BIGGER_SIZE})
plt.savefig(rc_dir + "/mpcq_ra.png")

# %%
plt.figure(figsize=(10,6))
ax = sns.lineplot(data=df, x=df['Test Number'], y='minute_per_correct_question_ra_5', marker='o')
plt.xticks(list(range(0, df['Test Number'].max(), 5))[1:])
plt.ylabel('MPCQ', labelpad=20)
plt.xlabel('Test Number', labelpad=20)
plt.axhline(perfect_score, ls='--', color='green')
plt.title('Reading Comprehension, Minutes Per Correct Question RA 5', fontdict={'fontsize': BIGGER_SIZE})
plt.savefig(rc_dir + "/mpcq_ra5.png")

# %%
key = pd.read_excel('data/scores_times_rc.xlsx', sheet_name='Key')
key_dict = {}
for i, r in key.iterrows():
    key_dict[r['Key']] = r['Question Type']

missed_q = []
for v in df['Missed Question Type']:
    if pd.notna(v):
        for q in v.split(','):
         missed_q.append(key_dict[q])

missed_q_df = pd.DataFrame(missed_q).rename(columns={0:'Question Type'})

# %%
plt.figure(figsize=(10,6))
ax = sns.countplot(x=missed_q_df['Question Type'], palette=palette)
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.ylabel('Number Wrong', labelpad=20)
plt.xlabel('')
plt.xticks(rotation=90)
plt.title('Reading Comprehension, Missed Question Type', fontdict={'fontsize': BIGGER_SIZE})
plt.savefig(rc_dir + "/missed_q.png")

# %% [markdown]
# # Total Score

# %%
df = pd.read_excel('data/lsat_scores.xlsx')
df = df.reset_index().rename(columns={'Test Number': 'PT Number', 'index':'Test Number'})
df['Test Number'] = df['Test Number'] + 1
df

# %%
school_df = pd.read_excel('data/schools.xlsx')
rcParams.update({'figure.autolayout': False})
# %%
y_min = df['Score'].mean() - 5
if school_df['LSAT Score'].min() < df['Score'].mean():
    y_min = school_df['LSAT Score'].min() - 5
if y_min < 130:
    y_min = 130

y_max = df['Score'].mean() - 5
if school_df['LSAT Score'].max() > df['Score'].mean():
    y_max = school_df['LSAT Score'].max() + 5
if y_max > 180:
    y_max = 180

all_schools = school_df.drop(columns='School').values.tolist()

fig_x = 20
fig_y = 11.3
dpi = 120
line_thickness = 1.75
dash_length = 12
dash_blank = 5

# Create base figure
fig, ax = plt.subplots(figsize=(fig_x,fig_y), dpi=dpi)
sns.lineplot(data=df, x=df['Test Number'], y='Score', marker='o', color='white')
plt.xticks(df['Test Number'])
plt.ylabel('Score', labelpad=20)
plt.xlabel('Test Number', labelpad=20)
plt.ylim(y_min, y_max)
plt.legend(labels=['LSAT Score'])
plt.title('Law Schools', fontdict={'fontsize': BIGGER_SIZE})

# Add transparency and adjusted coordinates
x = .75
t = 1
for s in all_schools:
    inc = .99 / len(all_schools)
    if len(s) < 4:
        s.append(t)
    t -= inc

    x0, y0 = ax.transData.transform((x, s[0]+.1))
    s.append(x0)
    s.append(y0)
    x += (len(df) - 1) / len(all_schools)


# Count duplicate entries to make duplicate lines
all_scores = [s[0] for s in all_schools]
occurences = Counter(all_scores)
dups = {}
for k,v in occurences.items():
    if v > 1:
        all_i = []
        for i, value in enumerate(all_scores):
            if value == k:
                all_i.append([all_schools[i][2], all_schools[i][3]])
        dups[k] = all_i


# Add lines

# Add solid color lines
for s in all_schools:
    if s[0] not in dups.keys():
        line = plt.axhline(s[0], color=s[2], alpha=s[3], linewidth=line_thickness)
        line.set_dashes([dash_length, dash_blank])

# Add multicolor lines using the dups dictionary
for k in dups.keys():
    dup_l = dups[k]
    decrement = dash_length / len(dup_l)
    color = dash_length
    blank = dash_blank
    for h in dup_l:
        line = plt.axhline(k, color=h[0], alpha=h[1], linewidth=line_thickness)
        line.set_dashes([color, blank])
        color -= decrement
        blank += decrement

# Add solid line
plt.axhline(df['Score'].mean(), ls='solid', color='white', alpha=1, linewidth=line_thickness+1)

# Create subplot for each image and add to figure
subplots = [None] * len(all_schools)
for s in all_schools:
    im = plt.imread('images/' + s[1])
    subplots[all_schools.index(s)] = fig.add_axes([(s[-2]/ (fig_x * dpi)), (s[-1]/ (fig_y * dpi)), 0.04, 0.04], anchor='NE')
    subplots[all_schools.index(s)].imshow(im, alpha=s[3])
    subplots[all_schools.index(s)].axis('off')

fig.savefig(top_dir + "/lsat_scores.png")

# %%



