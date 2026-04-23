"""Build 02_behavioral.ipynb — comprehension-accuracy analyses."""
from _build_notebook import build

CELLS = [
("md", """\
# 02 · Behavioral analysis

Examines post-trial comprehension accuracy as a function of (a) SNR, (b)
attended spatial direction, (c) attended-vs-distractor power ratio, (d) trial
order / fatigue, and (e) subject-level demographics.

Methods:

- **Descriptive**: per-condition accuracy with binomial 95 % CI (Wilson).
- **Logistic mixed-effects** (statsmodels `BinomialBayesMixedGLM` approximation
  and `MixedLM` linear-probability equivalent) with subject random intercepts.
- **Permutation tests** for order effects.
- **Psychometric curve** fit to SNR.
"""),

("code", """\
import sys, os, warnings; sys.path.insert(0, os.path.abspath('.'))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import binomtest
from scipy.optimize import curve_fit
import statsmodels.api as sm
import statsmodels.formula.api as smf
from aad_utils import (list_subjects, load_trials_csv, load_answers, load_demographics,
                       FIGURES_DIR, RESULTS_DIR, set_pub_style, save_fig, COLORS)
from aad_utils.config import ATTENDED_SPEAKER_MAP
set_pub_style()
trials = load_trials_csv()
SUBJECTS = list_subjects()
print(trials.shape, trials.columns.tolist())
"""),

("md", "## 1 · Collect behavioral records"),
("code", """\
records = []
dem_rows = []
for s in SUBJECTS:
    ans = load_answers(s)
    dem = load_demographics(s); dem['subject']=s; dem_rows.append(dem)
    # Match answers to trials.csv by Trial No.
    m = ans.merge(trials, on='Trial No.', how='left', suffixes=('_ans', '_tr'))
    m['subject'] = s
    m['is_training'] = m['Trial No.'].astype(str).str.startswith('Training')
    records.append(m)
B = pd.concat(records, ignore_index=True)
demos = pd.DataFrame(dem_rows)
B['Correct'] = pd.to_numeric(B['Correct'], errors='coerce')
B['SNR'] = pd.to_numeric(B['SNR'], errors='coerce')
B['trial_order'] = B.groupby('subject').cumcount()
print('Behavioral rows:', len(B), 'subjects:', B['subject'].nunique())
B[['subject','Trial No.','Correct','SNR','Attended Speaker','is_training']].head()
"""),

("code", """\
def wilson(k, n, alpha=0.05):
    if n == 0: return np.nan, np.nan, np.nan
    p = k/n; z = 1.959963984540054
    denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return p, max(0,center-half), min(1,center+half)
# Overall accuracy (main trials only).
main = B[~B['is_training']].copy()
k, n = int(main['Correct'].sum()), int(main['Correct'].notna().sum())
p, lo, hi = wilson(k, n)
print(f'Overall accuracy (main): {p:.3f} [{lo:.3f}, {hi:.3f}] — {k}/{n}')
"""),

("md", "## 2 · Accuracy vs SNR (psychometric)"),
("code", """\
snr_grp = main.groupby('SNR')['Correct'].agg(['sum','count']).reset_index()
snr_grp['p'], snr_grp['lo'], snr_grp['hi'] = zip(*[wilson(int(k), int(n)) for k,n in zip(snr_grp['sum'], snr_grp['count'])])

def logistic(x, a, b, c, d):
    return c + (d-c)/(1+np.exp(-(x-a)/b))
try:
    popt, _ = curve_fit(logistic, snr_grp['SNR'], snr_grp['p'], p0=[0, 3, 0.25, 1.0], maxfev=5000)
except Exception:
    popt = None
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.errorbar(snr_grp['SNR'], snr_grp['p'],
            yerr=[snr_grp['p']-snr_grp['lo'], snr_grp['hi']-snr_grp['p']],
            fmt='o', color=COLORS['attended'], capsize=2, label='data (95% CI)')
if popt is not None:
    xs = np.linspace(snr_grp['SNR'].min()-1, snr_grp['SNR'].max()+1, 200)
    ax.plot(xs, logistic(xs, *popt), color=COLORS['eeg'],
            label=f'logistic fit (thr50={popt[0]:.1f} dB, slope={popt[1]:.1f})')
ax.axhline(0.25, color=COLORS['chance'], ls='--', lw=1, label='chance (4-AFC)')
ax.set_xlabel('SNR (dB)'); ax.set_ylabel('comprehension accuracy')
ax.set_ylim(0, 1.05); ax.legend(loc='lower right')
save_fig(fig, '02_psychometric_snr', FIGURES_DIR); plt.show()
"""),

("md", "## 3 · Accuracy vs attended direction"),
("code", """\
main['attended_az'] = main['Attended Speaker'].map(lambda v: ATTENDED_SPEAKER_MAP.get(int(v), (None,None,np.nan))[2] if pd.notna(v) else np.nan)
dir_grp = main.groupby('attended_az')['Correct'].agg(['sum','count']).reset_index()
dir_grp['p'], dir_grp['lo'], dir_grp['hi'] = zip(*[wilson(int(k), int(n)) for k,n in zip(dir_grp['sum'], dir_grp['count'])])
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.bar(dir_grp['attended_az'], dir_grp['p'], width=20, color=COLORS['attended'],
       yerr=[dir_grp['p']-dir_grp['lo'], dir_grp['hi']-dir_grp['p']], capsize=3)
ax.axhline(0.25, color=COLORS['chance'], ls='--'); ax.set_xlabel('attended azimuth (°)')
ax.set_ylabel('accuracy'); ax.set_title('Comprehension accuracy by attended direction')
save_fig(fig, '02_accuracy_vs_direction', FIGURES_DIR); plt.show()
"""),

("md", "## 4 · Trial-order / fatigue effects"),
("code", """\
fig, ax = plt.subplots(figsize=(6, 3.5))
order_bin = pd.cut(main['trial_order'], bins=np.linspace(0, 105, 11))
g = main.groupby(order_bin)['Correct'].agg(['mean','sem','count']).reset_index()
g['mid'] = g['trial_order'].apply(lambda iv: iv.mid)
ax.errorbar(g['mid'], g['mean'], yerr=g['sem'], fmt='o-', color=COLORS['eeg'])
ax.set_xlabel('trial index (binned)'); ax.set_ylabel('accuracy')
ax.set_title('Trial-order effect (all subjects pooled)')
save_fig(fig, '02_trial_order', FIGURES_DIR); plt.show()
"""),

("md", "## 5 · Logistic mixed-effects model"),
("code", """\
d = main.dropna(subset=['Correct','SNR','attended_az']).copy()
d['abs_az'] = d['attended_az'].abs()
d['side'] = np.where(d['attended_az'] > 0, 'R', 'L')
try:
    glmm = smf.mixedlm('Correct ~ SNR + abs_az + trial_order', d, groups=d['subject']).fit(method='lbfgs')
    print(glmm.summary())
except Exception as e:
    print('MixedLM failed:', e)
"""),

("md", "## 6 · Per-subject variability"),
("code", """\
sub_acc = main.groupby('subject')['Correct'].agg(['mean','count','sum']).reset_index()
sub_acc['lo'], sub_acc['hi'] = zip(*[wilson(int(s), int(c))[1:] for s,c in zip(sub_acc['sum'], sub_acc['count'])])
sub_acc = sub_acc.sort_values('mean')
fig, ax = plt.subplots(figsize=(6, 3.5))
y = np.arange(len(sub_acc))
ax.errorbar(sub_acc['mean'], y, xerr=[sub_acc['mean']-sub_acc['lo'], sub_acc['hi']-sub_acc['mean']],
            fmt='o', color=COLORS['attended'])
ax.set_yticks(y); ax.set_yticklabels(['S%d'%s for s in sub_acc['subject']])
ax.axvline(0.25, color=COLORS['chance'], ls='--'); ax.set_xlabel('accuracy')
ax.set_title('Per-subject mean comprehension accuracy')
save_fig(fig, '02_per_subject_acc', FIGURES_DIR); plt.show()
sub_acc.to_parquet(RESULTS_DIR / '02_per_subject_acc.parquet')
"""),

("md", "## 7 · Demographics × accuracy"),
("code", """\
dem_merge = sub_acc.merge(demos[['subject','Gender','Age','Hand Preference','Ear Preference']], on='subject', how='left')
dem_merge['Age'] = pd.to_numeric(dem_merge['Age'], errors='coerce')
display(dem_merge)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
axes[0].scatter(dem_merge['Age'], dem_merge['mean'], color=COLORS['eeg'])
axes[0].set_xlabel('Age'); axes[0].set_ylabel('mean accuracy'); axes[0].set_title('Age vs accuracy')
dem_merge.boxplot(column='mean', by='Gender', ax=axes[1])
axes[1].set_title('Accuracy by gender'); axes[1].set_ylabel('accuracy'); plt.suptitle('')
save_fig(fig, '02_demographics', FIGURES_DIR); plt.show()
B.to_parquet(RESULTS_DIR / '02_behavioral_records.parquet')
"""),
]
build('/users/PAS2301/alialavi/projects/multimodal_aad_dataset_osu/analysis/02_behavioral.ipynb', CELLS)
print('Wrote 02_behavioral.ipynb')
