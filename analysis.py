"""Statistics and figures for the paired experiments.

For each scenario and Adaptive EKF variant, every run i gives
    Dᵢ = RMSE(adaptive, i) − RMSE(EKF, i)
(negative: the adaptive filter did better). The primary test, fixed in
advance, is a one-sided paired t-test of H₀: E[D] = 0 against H₁: E[D] < 0.
Alongside it: a 95% confidence interval for E[D], the effect size d_z, a
Shapiro–Wilk check of the t-test's normality assumption and a Wilcoxon
signed-rank test that does not need it. p-values are Holm-adjusted across
all the tests in a table, so testing many scenarios does not inflate the
chance of a false "better".
"""

import math
from pathlib import Path

from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from scipy import stats

from localization import ADAPTIVE, COVERAGE_LIMIT, NAMES
from scenarios import BY_KEY

ALPHA = .05
SHAPIRO_MAX = 5000  # Shapiro–Wilk p-values are inaccurate for larger samples


def paired_test(adaptive, standard):
    """Compare paired RMSEs (arrays). Returns a dict of statistics; see the module docstring."""
    adaptive, standard = np.asarray(adaptive, float), np.asarray(standard, float)
    d = adaptive - standard
    n = len(d)
    mean, sd = d.mean(), d.std(ddof=1) if n > 1 else math.nan
    result = {'n': n, 'ekf_mean': standard.mean(), 'adaptive_mean': adaptive.mean(), 'mean_d': mean, 'sd_d': sd,
              'ci_low': math.nan, 'ci_high': math.nan, 't': math.nan, 'p': math.nan, 'd_z': math.nan,
              'shapiro_p': math.nan, 'wilcoxon_p': math.nan, 'change': mean/standard.mean()}
    if n < 2 or sd == 0:
        return result
    half = stats.t.ppf(.975, n - 1) * sd/math.sqrt(n)
    test = stats.ttest_rel(adaptive, standard, alternative='less')
    result.update(ci_low=mean - half, ci_high=mean + half, t=test.statistic, p=test.pvalue, d_z=mean/sd)
    if n >= 3:
        result['shapiro_p'] = stats.shapiro(d).pvalue
    if np.any(d != 0):
        result['wilcoxon_p'] = stats.wilcoxon(d, alternative='less').pvalue
    return result


def holm(p_values):
    """Holm–Bonferroni adjusted p-values (NaNs are left out of the correction)."""
    p = np.asarray(p_values, float)
    adjusted = np.full_like(p, math.nan)
    valid = np.flatnonzero(~np.isnan(p))
    order = valid[np.argsort(p[valid])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(order) - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def verdict(row):
    """Plain-language result of one comparison."""
    if math.isnan(row['p_holm']):
        return 'Not enough runs'
    percent = abs(row['change'])*100
    if row['p_holm'] < ALPHA and row['mean_d'] < 0:
        return f'Better: {percent:.0f}% lower error'
    if row['ci_low'] > 0:
        return f'Worse: {percent:.0f}% higher error'
    return 'No clear difference'


def summary(results):
    """One row per scenario and adaptive variant, with the paired test against the EKF."""
    rows = []
    scenarios = list(dict.fromkeys(results['scenario']))
    for variant in ADAPTIVE:
        for key in scenarios:
            runs = results[results['scenario'] == key]
            row = {'scenario': key, 'scenario_name': BY_KEY[key].name, 'variant': variant,
                   'variant_name': NAMES[variant], 'routes': ', '.join(dict.fromkeys(runs['route']))}
            row.update(paired_test(runs[f'{variant}_rmse'], runs['ekf_rmse']))
            row['ekf_coverage'] = runs['ekf_coverage'].mean()
            row['adaptive_coverage'] = runs[f'{variant}_coverage'].mean()
            rows.append(row)
    table = pd.DataFrame(rows)
    table['p_holm'] = holm(table['p'])
    table['verdict'] = table.apply(verdict, axis=1)
    return table


def headline(table, variant='aekf'):
    """A sentence or two summarising the comparisons for one variant."""
    rows = table[table['variant'] == variant]
    better = rows[rows['verdict'].str.startswith('Better')]['scenario_name'].tolist()
    worse = rows[rows['verdict'].str.startswith('Worse')]['scenario_name'].tolist()
    runs = int(rows['n'].max()) if len(rows) else 0
    parts = [f'{NAMES[variant]} vs. EKF, {runs} paired runs per scenario.']
    parts.append(f"Significantly better in: {', '.join(better)}." if better else 'Not significantly better in any scenario.')
    if worse:
        parts.append(f"Worse in: {', '.join(worse)}.")
    return ' '.join(parts)


# Figures. Each draws onto a matplotlib Figure in the given theme (ui.THEMES).

def style(axes, theme):
    axes.set_facecolor(theme['surface'])
    axes.tick_params(colors=theme['muted'], length=0, labelsize=8)
    axes.grid(color=theme['grid'], linewidth=.8, axis='y')
    axes.set_axisbelow(True)
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.xaxis.label.set_color(theme['muted'])
    axes.yaxis.label.set_color(theme['muted'])
    axes.title.set_color(theme['text'])


def short(key):
    """Scenario label for an axis, on two lines."""
    return BY_KEY[key].short.replace(' ', '\n', 1)


def errors_figure(figure, results, variant, theme):
    """Box plots of RMSE per scenario: EKF beside the adaptive variant."""
    figure.clear()
    figure.set_facecolor(theme['surface'])
    axes = figure.add_subplot()
    style(axes, theme)
    scenarios = list(dict.fromkeys(results['scenario']))
    for offset, name in ((-.18, 'ekf'), (.18, variant)):
        data = [results.loc[results['scenario'] == key, f'{name}_rmse'] for key in scenarios]
        boxes = axes.boxplot(data, positions=np.arange(len(scenarios)) + offset, widths=.3, patch_artist=True,
                             showfliers=True, medianprops={'color': theme['text']},
                             flierprops={'markersize': 3, 'markeredgecolor': theme[name]},
                             whiskerprops={'color': theme['muted']}, capprops={'color': theme['muted']})
        for box in boxes['boxes']:
            box.set(facecolor=theme[name], edgecolor=theme[name], alpha=.75)
        axes.plot([], [], color=theme[name], linewidth=6, alpha=.75, label=NAMES[name])
    axes.set_xticks(range(len(scenarios)), [short(key) for key in scenarios])
    axes.set_yscale('log')
    axes.set_ylabel('Position RMSE per run (m, log scale)')
    axes.set_title('Error of each run', fontsize=10, loc='left')
    axes.legend(frameon=False, labelcolor=theme['text'], fontsize=8, loc='upper left')
    figure.tight_layout()


def difference_figure(figure, table, variant, theme):
    """Mean paired difference D with its 95% confidence interval, per scenario."""
    figure.clear()
    figure.set_facecolor(theme['surface'])
    axes = figure.add_subplot()
    style(axes, theme)
    axes.grid(color=theme['grid'], linewidth=.8, axis='x')
    rows = table[table['variant'] == variant].reset_index(drop=True)
    ys = np.arange(len(rows))[::-1]
    for y, (_, row) in zip(ys, rows.iterrows()):
        color = theme[variant] if row['verdict'].startswith('Better') else \
            theme['danger'] if row['verdict'].startswith('Worse') else theme['muted']
        axes.errorbar(row['mean_d']*1000, y, xerr=[[(row['mean_d'] - row['ci_low'])*1000],
                                                  [(row['ci_high'] - row['mean_d'])*1000]],
                      fmt='o', color=color, capsize=4, markersize=6, linewidth=2)
    axes.axvline(0, color=theme['text'], linewidth=1)
    axes.set_yticks(ys, [BY_KEY[key].short for key in rows['scenario']])
    axes.set_xlabel('Adaptive − EKF RMSE (mm): mean and 95% confidence interval')
    axes.set_title(f'{NAMES[variant]} vs. EKF (left of 0 = adaptive better)', fontsize=10, loc='left')
    figure.tight_layout()


def calibration_figure(figure, table, theme):
    """How often the true position lies inside each filter's 95% ellipse (ideal: 95%)."""
    figure.clear()
    figure.set_facecolor(theme['surface'])
    axes = figure.add_subplot()
    style(axes, theme)
    scenarios = list(dict.fromkeys(table['scenario']))
    names = ['ekf', *ADAPTIVE]
    width = .8/len(names)
    for i, name in enumerate(names):
        if name == 'ekf':
            values = [table[(table['scenario'] == key)]['ekf_coverage'].iloc[0] for key in scenarios]
        else:
            values = [table[(table['scenario'] == key) & (table['variant'] == name)]['adaptive_coverage'].iloc[0]
                      for key in scenarios]
        axes.bar(np.arange(len(scenarios)) + (i - (len(names) - 1)/2)*width, np.array(values)*100, width,
                 color=theme[name], label=NAMES[name])
    axes.axhline(95, color=theme['text'], linewidth=1, linestyle=(0, (4, 2)))
    axes.text(len(scenarios) - .5, 96, 'ideal 95%', color=theme['text'], fontsize=8, ha='right', va='bottom')
    axes.set_xticks(range(len(scenarios)), [short(key) for key in scenarios])
    axes.set_ylim(0, 105)
    axes.set_ylabel('Time truth is inside the 95% ellipse (%)')
    axes.set_title('Is each filter honest about its uncertainty?', fontsize=10, loc='left')
    axes.legend(frameon=False, labelcolor=theme['text'], fontsize=8, loc='lower right', ncol=3,
                bbox_to_anchor=(1, 1.02), borderaxespad=0)
    figure.tight_layout()


def noise_figure(figure, fix_errors, gyro_errors, fix_sigma, gyro_sigma, theme):
    """Histograms of the simulated sensor errors against the N(0, σ²) they should follow."""
    figure.clear()
    figure.set_facecolor(theme['surface'])
    for axes, errors, sigma, unit, title in zip(
            figure.subplots(1, 2), (np.ravel(fix_errors), np.asarray(gyro_errors)), (fix_sigma, gyro_sigma),
            ('m', 'rad/s'), ('Position fix error', 'Gyro error')):
        style(axes, theme)
        axes.hist(errors, bins=24, density=True, color=theme['fix'], alpha=.8, label='Simulated errors')
        xs = np.linspace(-4*sigma, 4*sigma, 200)
        axes.plot(xs, stats.norm.pdf(xs, 0, sigma), color=theme['true'], linewidth=2, label=f'N(0, {sigma:g}²)')
        # Every k-th error keeps the test within its valid sample size.
        normality = stats.shapiro(errors[::math.ceil(len(errors)/SHAPIRO_MAX)]).pvalue
        axes.set_title(f'{title}: mean {errors.mean():+.4f}, σ̂ {errors.std(ddof=1):.4f} {unit}\n'
                       f'n = {len(errors)}, Shapiro–Wilk p = {normality:.2f}', fontsize=9, loc='left')
        axes.set_xlabel(f'Error ({unit})')
        axes.set_ylabel('Probability density')
        axes.legend(frameon=False, labelcolor=theme['text'], fontsize=8)
    figure.tight_layout()


def export(results, folder, fix_errors, gyro_errors, fix_sigma, gyro_sigma):
    """Save the raw results, the summary table and every figure (light theme) to folder."""
    from ui import THEMES
    theme = THEMES['light']
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    table = summary(results)
    paths = [folder / 'runs.csv', folder / 'summary.csv']
    results.to_csv(paths[0], index=False)
    table.to_csv(paths[1], index=False)
    figures = [('calibration', lambda f: calibration_figure(f, table, theme)),
               ('noise_check', lambda f: noise_figure(f, fix_errors, gyro_errors, fix_sigma, gyro_sigma, theme))]
    for variant in ADAPTIVE:
        figures += [(f'errors_{variant}', lambda f, v=variant: errors_figure(f, results, v, theme)),
                    (f'difference_{variant}', lambda f, v=variant: difference_figure(f, table, v, theme))]
    for name, draw in figures:
        figure = Figure(figsize=(8, 4.2), dpi=150)
        draw(figure)
        paths.append(folder / f'{name}.png')
        figure.savefig(paths[-1], facecolor=theme['surface'])
    return paths
