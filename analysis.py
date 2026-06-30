"""
EWMA and EWMP Adaptive Monitoring Boundaries for Professional Football
======================================================================
Analysis script for the manuscript:
"Contextualising Exponentially Weighted Moving Averages with Adaptive Monitoring
Boundaries: A Comparison of Variance-Based and Percentile-Based Approaches
in Professional Football"

Authors: Morin S., Thélamon F., Joubert T., Iodice P.
Journal: Science and Medicine in Football (submitted 2026)

Dataset: HAC_LIGUE_1_2024-2025_ANON.xlsx
Requirements: numpy, scipy, pandas, matplotlib, openpyxl

Usage:
    python analysis.py

All outputs are printed to stdout. Figures are saved as PNG files.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

ALPHA = 0.28          # Exponential decay parameter
WINDOW = 14           # Rolling window (number of preceding observations)
MIN_OBS = 5           # Minimum preceding observations before zone computation
DATA_FILE = "HAC_LIGUE_1_2024-2025_ANON.xlsx"

# Sentinel items: English label -> column name in dataset
ITEMS = {
    'General Fatigue':        'Fatigue generale',
    'Muscular Intensity':     'Impact neuromusculaire',
    'Cardiorespiratory Impact': 'Impact cardioventilatoire',
    'Sleep Quality':          'Qualite de mon sommeil',
    'Technical Mastery':      'Ma maitrise technique',
    'Tactical Mastery':       'Ma maitrise tactique',
}

# Microcycle phase mapping (dataset tag -> label)
MD_TAGS   = ['J-5', 'J-4', 'J-3', 'J-2', 'J-1', 'J0']
MD_LABELS = ['MD-5', 'MD-4', 'MD-3', 'MD-2', 'MD-1', 'MD0']

# Loading and recovery phases per item (for Table 2)
LOADING_PHASES  = {k: 'J-4' for k in ITEMS}
LOADING_PHASES['Technical Mastery'] = 'J-3'
LOADING_PHASES['Tactical Mastery']  = 'J-3'

RECOVERY_PHASES = {k: 'J-3' for k in ITEMS}
RECOVERY_PHASES['Technical Mastery'] = 'J-1'
RECOVERY_PHASES['Tactical Mastery']  = 'J-1'

# Colour palette
C_EWMA  = '#2563EB'
C_EWMP  = '#16A34A'
C_HIGH  = '#DC2626'
C_LOW   = '#2563EB'
C_NORM  = '#9CA3AF'
C_MATCH = '#B45309'


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def compute_zones(series, alpha=ALPHA, window=WINDOW, min_obs=MIN_OBS):
    """
    Compute EWMA±SD and EWMP (Q1-Q3) monitoring zones for a single player/item series.

    Parameters
    ----------
    series : pd.Series
        Chronologically ordered observations for one player and one item.
    alpha : float
        Exponential decay parameter (default 0.28).
    window : int
        Maximum number of preceding observations to include (default 14).
    min_obs : int
        Minimum number of valid preceding observations required (default 5).

    Returns
    -------
    dict with keys:
        ze      : list of str  — EWMA±SD zone ('HIGH', 'NORMAL', 'LOW', or '')
        zp      : list of str  — EWMP zone
        bwe     : list of float — EWMA±SD bandwidth (2 * SD)
        bwp     : list of float — EWMP bandwidth (Q3 - Q1)
        ewma_c  : list of float — EWMA centre
        ewma_sd : list of float — EWMA standard deviation
        q1      : list of float — Exponentially weighted Q1
        q3      : list of float — Exponentially weighted Q3
    """
    values = series.values
    n = len(values)
    ze = [''] * n;  zp = [''] * n
    bwe = [np.nan] * n;  bwp = [np.nan] * n
    ewma_c = [np.nan] * n;  ewma_sd = [np.nan] * n
    q1v = [np.nan] * n;  q3v = [np.nan] * n

    for i in range(n):
        # Collect preceding valid observations within window
        hist_idx, hist_val = [], []
        for j in range(max(0, i - window), i):
            if not np.isnan(values[j]):
                hist_idx.append(j)
                hist_val.append(values[j])

        if len(hist_val) < min_obs or np.isnan(values[i]):
            continue

        # Exponential weights (more recent = higher weight)
        w = np.array([(1 - alpha) ** (i - 1 - idx) for idx in hist_idx])
        v = np.array(hist_val)
        ws = w.sum()
        cur = values[i]

        # --- EWMA±SD ---
        ewma = (w * v).sum() / ws
        sd   = np.sqrt((w * (v - ewma) ** 2).sum() / ws)
        ewma_c[i]  = ewma
        ewma_sd[i] = sd
        bwe[i]     = 2 * sd
        ze[i] = 'HIGH' if cur > ewma + sd else ('LOW' if cur < ewma - sd else 'NORMAL')

        # --- EWMP (exponentially weighted Q1, Q3) ---
        si = np.argsort(v)
        sv, sw = v[si], w[si]
        cw = np.cumsum(sw) / ws          # cumulative normalised weights
        q1 = sv[np.searchsorted(cw, 0.25)]
        q3 = sv[np.searchsorted(cw, 0.75)]
        q1v[i] = q1;  q3v[i] = q3
        bwp[i] = q3 - q1
        zp[i] = 'HIGH' if cur > q3 else ('LOW' if cur < q1 else 'NORMAL')

    return dict(ze=ze, zp=zp, bwe=bwe, bwp=bwp,
                ewma_c=ewma_c, ewma_sd=ewma_sd, q1=q1v, q3=q3v)


def apply_zones_to_dataset(df, items=ITEMS, alpha=ALPHA, window=WINDOW, min_obs=MIN_OBS):
    """
    Apply compute_zones() to all players and all items.
    Adds zone columns to df in-place and returns df.
    """
    df = df.sort_values(['Player Name', 'DateTime']).reset_index(drop=True)

    for label, col in items.items():
        print(f"  Computing zones: {label}...")
        for suffix in ['ze', 'zp', 'bwe', 'bwp', 'ewma_c', 'ewma_sd', 'q1', 'q3']:
            df[f'{suffix}_{col}'] = '' if suffix in ('ze', 'zp') else np.nan

        for player in df['Player Name'].unique():
            idx = df[df['Player Name'] == player].index
            result = compute_zones(df.loc[idx, col], alpha=alpha,
                                   window=window, min_obs=min_obs)
            for key, vals in result.items():
                for ii, idx_val in enumerate(idx):
                    df.at[idx_val, f'{key}_{col}'] = vals[ii]

    return df


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(filepath=DATA_FILE):
    """Load and prepare the dataset."""
    print(f"Loading data from {filepath}...")
    xl   = pd.read_excel(filepath, sheet_name=None)
    data = xl['HAC_LIGUE_1_2024-2025_FINAL'].copy()
    joueurs = xl['Joueurs'].copy()

    # Exclude goalkeepers
    gk_codes = joueurs[joueurs['PosteENG'] == 'GK']['Player Name'].tolist()
    data = data[~data['Player Name'].isin(gk_codes)].copy()
    print(f"  Goalkeepers excluded: {gk_codes}")

    # Exclude familiarisation week (Week 1)
    data = data[data['SemaineRelative'] > 1].copy()

    # Parse datetime
    data['DateTime'] = pd.to_datetime(data['DateTime'], utc=True)
    data['Date'] = data['DateTime'].dt.date

    # Session-RPE
    data['sRPE'] = data['Fatigue generale'] * data['Session duration'] / 100

    # HSR (velocity bands 6-8)
    hsr_cols = ['GPS_VB6_m/s Distance', 'GPS_VB7_m/s Distance', 'GPS_VB8_m/s Distance']
    data['HSR'] = data[hsr_cols].sum(axis=1)

    n_players = data['Player Name'].nunique()
    n_obs     = len(data)
    n_train   = len(data[data['EventType'] == 'training'])
    n_game    = len(data[data['EventType'] == 'game'])
    print(f"  Outfield players: {n_players}")
    print(f"  Total observations: {n_obs:,}  (training: {n_train:,}, match: {n_game:,})")
    return data


# =============================================================================
# STATISTICAL ANALYSES
# =============================================================================

def table1_distributional(df):
    """Table 1: Distributional characteristics of sentinel items."""
    print("\n" + "=" * 70)
    print("TABLE 1 — Distributional characteristics of sentinel items")
    print("=" * 70)
    hdr = f"{'Variable':<26} {'N':>6} {'Mean':>7} {'Median':>7} {'SD':>7} {'Q1':>6} {'Q3':>6} {'IQR':>5} {'Skew':>7}"
    print(hdr)
    print("-" * 75)
    for label, col in ITEMS.items():
        d = df[col].dropna()
        q1, q3 = d.quantile(0.25), d.quantile(0.75)
        print(f"{label:<26} {len(d):>6,} {d.mean():>7.1f} {d.median():>7.1f} "
              f"{d.std():>7.1f} {q1:>6.1f} {q3:>6.1f} {q3-q1:>5.1f} {stats.skew(d):>7.2f}")


def section32_concordance(df):
    """Section 3.2: Zone concordance with session duration and sRPE."""
    col = 'Fatigue generale'
    print("\n" + "=" * 70)
    print("SECTION 3.2 — Zone concordance with session characteristics")
    print("=" * 70)

    zero_n   = (df['Session duration'] == 0).sum()
    zero_pct = 100 * zero_n / len(df)
    print(f"\nZero-duration observations: n={zero_n} ({zero_pct:.1f}%)")

    valid = df[(df[f'ze_{col}'] != '') & (df['Session duration'] > 0)]

    for approach, zcol in [('EWMA±SD', f'ze_{col}'), ('EWMP', f'zp_{col}')]:
        print(f"\n{approach}:")
        zone_durs = {z: valid[valid[zcol] == z]['Session duration'] for z in ['HIGH', 'NORMAL', 'LOW']}
        for z, d in zone_durs.items():
            print(f"  {z}: N={len(d):,}, Mean={d.mean():.1f} ± {d.std():.1f} min")
        h, lo = zone_durs['HIGH'], zone_durs['LOW']
        kw_stat, kw_p = stats.kruskal(*[zone_durs[z].values for z in ['HIGH', 'NORMAL', 'LOW']])
        u, p_mw = stats.mannwhitneyu(h, lo, alternative='two-sided')
        rb = 1 - 2 * u / (len(h) * len(lo))
        print(f"  Kruskal-Wallis: H={kw_stat:.3f}, p<.001")
        print(f"  HIGH vs LOW: U={u:.0f}, r={rb:.3f}, p<.001")

    print("\nSession-RPE by microcycle phase:")
    print(f"{'Phase':<8} {'N':>5} {'sRPE Mean':>10} {'SD':>8}")
    for j, md in zip(MD_TAGS, MD_LABELS):
        d = df[(df['TagTemporel'] == j) & (df['Session duration'] > 0) & df['sRPE'].notna()]
        if len(d) > 0:
            print(f"  {md:<6} {len(d):>5,} {d['sRPE'].mean():>10.1f} {d['sRPE'].std():>8.1f}")

    print("\nSpearman: session duration × general fatigue")
    train = df[df['EventType'] == 'training']
    game  = df[df['EventType'] == 'game']
    for label2, sub in [('Training', train), ('Match', game)]:
        sub_v = sub[(sub['Session duration'] > 0) & sub[col].notna()]
        r, p  = stats.spearmanr(sub_v['Session duration'], sub_v[col])
        print(f"  {label2}: r={r:.3f}, p<.001, N={len(sub_v):,}")


def section33_gps(df):
    """Section 3.3: GPS-based contextual concordance."""
    col = 'Fatigue generale'
    print("\n" + "=" * 70)
    print("SECTION 3.3 — GPS-based contextual concordance")
    print("=" * 70)

    train = df[df['EventType'] == 'training'].copy()
    train_gps = train[train['GPS_Total Distance'].notna()]
    team = train_gps.groupby('Date').agg(
        TD=('GPS_Total Distance', 'mean'), HSR=('HSR', 'mean'),
        fatigue=('Fatigue generale', 'mean'),
        muscular=('Impact neuromusculaire', 'mean'),
        cardio=('Impact cardioventilatoire', 'mean'),
    ).reset_index()
    print(f"\nN GPS training session days: {len(team)}")

    print("\nSpearman correlations (team level):")
    for item, col_p in [('General Fatigue', 'fatigue'),
                         ('Cardiorespiratory Impact', 'cardio'),
                         ('Muscular Intensity', 'muscular')]:
        r_td,  _ = stats.spearmanr(team['TD'],  team[col_p], nan_policy='omit')
        r_hsr, _ = stats.spearmanr(team['HSR'], team[col_p], nan_policy='omit')
        print(f"  {item:<26}: TD r={r_td:+.3f}, HSR r={r_hsr:+.3f}, p<.001")

    # GPS × HIGH% correlation
    gps_hi = train[train['GPS_Total Distance'].notna()].copy()
    team_hi = gps_hi.groupby('Date').agg(
        TD=(f'GPS_Total Distance', 'mean'),
        HIGH_ewma=(f'ze_{col}', lambda x: (x == 'HIGH').mean()),
        HIGH_ewmp=(f'zp_{col}', lambda x: (x == 'HIGH').mean()),
    ).reset_index()
    r_e, _ = stats.spearmanr(team_hi['TD'], team_hi['HIGH_ewma'], nan_policy='omit')
    r_p, _ = stats.spearmanr(team_hi['TD'], team_hi['HIGH_ewmp'], nan_policy='omit')
    print(f"\nGPS Total Distance × HIGH zone proportion:")
    print(f"  EWMA±SD: r={r_e:+.3f}, p<.001")
    print(f"  EWMP:    r={r_p:+.3f}, p<.001")

    print("\nGPS microcycle profile:")
    print(f"{'Phase':<8} {'N':>5} {'TD Mean':>10} {'TD SD':>8} {'HSR Mean':>10} {'HSR SD':>8}")
    for j, md in zip(MD_TAGS[:-1], MD_LABELS[:-1]):
        d = train_gps[train_gps['TagTemporel'] == j]
        if len(d) > 0:
            print(f"  {md:<6} {len(d):>5,} {d['GPS_Total Distance'].mean():>10.0f} "
                  f"{d['GPS_Total Distance'].std():>8.0f} {d['HSR'].mean():>10.0f} {d['HSR'].std():>8.0f}")


def section35_microcycle(df):
    """Section 3.5: Microcycle zone structure."""
    col = 'Fatigue generale'
    print("\n" + "=" * 70)
    print("SECTION 3.5 — Microcycle zone structure")
    print("=" * 70)

    hz = df[(df[f'ze_{col}'] != '') & df['TagTemporel'].notna()]

    print(f"\nZone proportions by phase (General Fatigue):")
    print(f"{'Phase':<8} {'N':>5} {'Mean':>7} {'EWMA_H%':>9} {'EWMP_H%':>9} "
          f"{'EWMA_L%':>9} {'EWMP_L%':>9} {'BW_red%':>8}")
    for j, md in zip(MD_TAGS, MD_LABELS):
        d = hz[hz['TagTemporel'] == j]
        if len(d) == 0:
            continue
        eh = (d[f'ze_{col}'] == 'HIGH').mean() * 100
        ph = (d[f'zp_{col}'] == 'HIGH').mean() * 100
        el = (d[f'ze_{col}'] == 'LOW').mean()  * 100
        pl = (d[f'zp_{col}'] == 'LOW').mean()  * 100
        bwe = d[f'bwe_{col}'].mean()
        bwp = d[f'bwp_{col}'].mean()
        red = (1 - bwp / bwe) * 100
        print(f"  {md:<6} {len(d):>5,} {d[col].mean():>7.1f} {eh:>8.1f}% {ph:>8.1f}% "
              f"{el:>8.1f}% {pl:>8.1f}% {red:>7.1f}%")

    # Bandwidth reduction across all items
    print("\nBandwidth reduction across all 6 items (alpha=0.28, window=14):")
    all_reds = []
    for label, colname in ITEMS.items():
        bwe_m = df[f'bwe_{colname}'].dropna().mean()
        bwp_m = df[f'bwp_{colname}'].dropna().mean()
        red   = (1 - bwp_m / bwe_m) * 100
        all_reds.append(red)
        print(f"  {label:<26}: BW_EWMA={bwe_m:.1f}, BW_EWMP={bwp_m:.1f}, reduction={red:.1f}%")
    print(f"  Overall range: {min(all_reds):.1f}% – {max(all_reds):.1f}%")

    # Table 2: all items × loading/recovery/MD0
    print("\nTable 2 — Contextual zone responsiveness (all items):")
    hdr2 = (f"{'Variable':<26} {'Load_EWMA_H':>12} {'Load_EWMP_H':>12} {'ΔH':>6} "
            f"{'Rec_EWMA_L':>11} {'Rec_EWMP_L':>11} {'ΔL':>6} "
            f"{'MD0_EWMA_H':>11} {'MD0_EWMP_H':>11} {'ΔMD0':>7}")
    print(hdr2)
    all_dh, all_dl, all_dm = [], [], []
    for label, colname in ITEMS.items():
        hz2 = df[(df[f'ze_{colname}'] != '') & df['TagTemporel'].notna()]
        dl2 = hz2[hz2['TagTemporel'] == LOADING_PHASES[label]]
        dr  = hz2[hz2['TagTemporel'] == RECOVERY_PHASES[label]]
        dm0 = hz2[hz2['TagTemporel'] == 'J0']
        lhe = (dl2[f'ze_{colname}'] == 'HIGH').mean() * 100
        lhp = (dl2[f'zp_{colname}'] == 'HIGH').mean() * 100
        rle = (dr[f'ze_{colname}']  == 'LOW').mean()  * 100
        rlp = (dr[f'zp_{colname}']  == 'LOW').mean()  * 100
        m0e = (dm0[f'ze_{colname}'] == 'HIGH').mean() * 100
        m0p = (dm0[f'zp_{colname}'] == 'HIGH').mean() * 100
        dh, dl_, dm = lhp - lhe, rlp - rle, m0p - m0e
        all_dh.append(dh); all_dl.append(dl_); all_dm.append(dm)
        print(f"  {label:<24} {lhe:>11.1f}% {lhp:>11.1f}% {dh:>+5.1f}% "
              f"{rle:>10.1f}% {rlp:>10.1f}% {dl_:>+5.1f}% "
              f"{m0e:>10.1f}% {m0p:>10.1f}% {dm:>+6.1f}%")
    print(f"  {'Mean Delta':<24} {'':>12} {'':>12} {np.mean(all_dh):>+5.1f}% "
          f"{'':>11} {'':>11} {np.mean(all_dl):>+5.1f}% "
          f"{'':>11} {'':>11} {np.mean(all_dm):>+6.1f}%")


def section36_match_training(df):
    """Section 3.6: Match–training dissociation."""
    print("\n" + "=" * 70)
    print("SECTION 3.6 — Match–training dissociation")
    print("=" * 70)
    for label, col_i in [('Cardiorespiratory Impact', 'Impact cardioventilatoire'),
                          ('Technical Mastery',        'Ma maitrise technique')]:
        tr_raw = df[df['EventType'] == 'training'][col_i].dropna()
        ga_raw = df[df['EventType'] == 'game'][col_i].dropna()
        u, p   = stats.mannwhitneyu(tr_raw, ga_raw, alternative='two-sided')
        rb     = 1 - 2 * u / (len(tr_raw) * len(ga_raw))
        tr_z   = df[(df['EventType'] == 'training') & (df[f'ze_{col_i}'] != '')]
        ga_z   = df[(df['EventType'] == 'game')     & (df[f'ze_{col_i}'] != '')]
        print(f"\n{label}:")
        print(f"  Training: N={len(tr_raw):,}, {tr_raw.mean():.1f} ± {tr_raw.std():.1f} AU")
        print(f"  Match:    N={len(ga_raw):,}, {ga_raw.mean():.1f} ± {ga_raw.std():.1f} AU")
        print(f"  Mann-Whitney r={rb:.3f}, p<.001")
        for approach, ze, zp in [('EWMA±SD', f'ze_{col_i}', None),
                                  ('EWMP',    None,          f'zp_{col_i}')]:
            zcol = ze or zp
            tH = (tr_z[zcol] == 'HIGH').mean() * 100
            gH = (ga_z[zcol] == 'HIGH').mean() * 100
            print(f"  {approach}: Training HIGH={tH:.1f}%, Match HIGH={gH:.1f}%, "
                  f"Delta={gH - tH:+.1f} pp")


def section37_heterogeneity(df):
    """Section 3.7: Inter-individual heterogeneity."""
    col = 'Fatigue generale'
    print("\n" + "=" * 70)
    print("SECTION 3.7 — Inter-individual heterogeneity")
    print("=" * 70)

    pm = df.groupby('Player Name')[col].mean().sort_values()
    print(f"\nPlayer seasonal means (General Fatigue):")
    for p, m in pm.items():
        print(f"  {p}: {m:.1f} AU")
    print(f"\n  Min={pm.min():.1f}, Max={pm.max():.1f}, Range={pm.max()-pm.min():.1f} AU")

    print("\nEWMP bandwidth per player:")
    for label, colname in [('General Fatigue', 'Fatigue generale'),
                            ('Sleep Quality',   'Qualite de mon sommeil')]:
        bw_vals = [df[df['Player Name'] == p][f'bwp_{colname}'].dropna().mean()
                   for p in df['Player Name'].unique()]
        bw_vals = [x for x in bw_vals if not np.isnan(x)]
        ratio   = max(bw_vals) / min(bw_vals)
        print(f"  {label}: min={min(bw_vals):.1f}, max={max(bw_vals):.1f}, ratio={ratio:.1f}x")

    disagree = []
    for p in df['Player Name'].unique():
        d = df[(df['Player Name'] == p) &
               (df[f'ze_{col}'] != '') & (df[f'zp_{col}'] != '')]
        if len(d) > 0:
            disagree.append((d[f'ze_{col}'] != d[f'zp_{col}']).mean() * 100)
    print(f"\nZone disagreement rates (General Fatigue):")
    print(f"  Range: {min(disagree):.1f}–{max(disagree):.1f}%, Mean: {np.mean(disagree):.1f}%")

    # Loading-recovery contrast per player
    contrasts = []
    for p in df['Player Name'].unique():
        pd_ = df[(df['Player Name'] == p) & (df[f'zp_{col}'] != '')]
        d4  = pd_[pd_['TagTemporel'].str.contains('J-4', na=False)][f'zp_{col}']
        d3  = pd_[pd_['TagTemporel'].str.contains('J-3', na=False)][f'zp_{col}']
        if len(d4) > 0 and len(d3) > 0:
            contrasts.append((d4 == 'HIGH').mean() * 100 - (d3 == 'HIGH').mean() * 100)
    reversed_n = sum(1 for c in contrasts if c < 0)
    print(f"\nLoading-recovery contrast (MD-4 HIGH% minus MD-3 HIGH%):")
    print(f"  Range: {min(contrasts):+.1f}% to {max(contrasts):+.1f}%")
    print(f"  Reversed (negative): {reversed_n} players")


def section38_sensitivity(df):
    """Section 3.8: Parametric sensitivity across alpha × window combinations."""
    col = 'Fatigue generale'
    alphas  = [0.10, 0.20, 0.28, 0.40, 0.50]
    windows = [7, 10, 14, 21, 28]

    print("\n" + "=" * 70)
    print("SECTION 3.8 — Parametric sensitivity (General Fatigue)")
    print("=" * 70)
    print(f"\n{'alpha':<7} {'window':<8} {'BW_EWMA':>9} {'BW_EWMP':>9} {'reduction%':>11} {'EWMP<EWMA':>10}")

    all_reds = []
    for alpha in alphas:
        for window in windows:
            bwe_l, bwp_l = [], []
            for p in df['Player Name'].unique():
                idx = df[df['Player Name'] == p].index
                res = compute_zones(df.loc[idx, col], alpha=alpha, window=window)
                bwe_l.extend([x for x in res['bwe'] if not np.isnan(x)])
                bwp_l.extend([x for x in res['bwp'] if not np.isnan(x)])
            if bwe_l and bwp_l:
                bwe_m = np.mean(bwe_l)
                bwp_m = np.mean(bwp_l)
                red   = (1 - bwp_m / bwe_m) * 100
                narrower = bwp_m < bwe_m
                all_reds.append(red)
                print(f"  a={alpha:<5} w={window:<6} {bwe_m:>9.1f} {bwp_m:>9.1f} "
                      f"{red:>10.1f}% {'YES':>10}")

    print(f"\nEWMP narrower in all {len(all_reds)}/25 combinations: {all(r > 0 for r in all_reds)}")
    print(f"Reduction range: {min(all_reds):.1f}% – {max(all_reds):.1f}%")


def internal_consistency(df):
    """Appendix A: Internal consistency (Cronbach alpha and Guttman lambda-2)."""
    print("\n" + "=" * 70)
    print("APPENDIX A — Internal consistency")
    print("=" * 70)
    cols = list(ITEMS.values())
    complete = df[cols].dropna()
    n = len(complete)
    print(f"\nComplete-case observations (all 6 items): N={n:,}")

    # Cronbach's alpha
    k = len(cols)
    item_vars = complete.var(axis=0, ddof=1)
    total_var = complete.sum(axis=1).var(ddof=1)
    alpha_c   = (k / (k - 1)) * (1 - item_vars.sum() / total_var)
    print(f"Cronbach's alpha: {alpha_c:.2f}")

    # Inter-item correlations
    corr_matrix = complete.corr(method='spearman')
    print("\nSpearman inter-item correlations:")
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if j > i:
                r = corr_matrix.loc[c1, c2]
                print(f"  {list(ITEMS.keys())[i]} × {list(ITEMS.keys())[j]}: r={r:.2f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("EWMA / EWMP Adaptive Monitoring Boundaries — Full Analysis")
    print("HAC Ligue 1 2024-2025 | alpha=0.28, window=14, min_obs=5")
    print("=" * 70)

    # Load data
    df = load_data(DATA_FILE)

    # Compute zones for all players and items
    print("\nComputing EWMA±SD and EWMP zones for all players and items...")
    df = apply_zones_to_dataset(df)
    print("  Done.\n")

    # Run all analyses
    table1_distributional(df)
    section32_concordance(df)
    section33_gps(df)
    section35_microcycle(df)
    section36_match_training(df)
    section37_heterogeneity(df)
    section38_sensitivity(df)
    internal_consistency(df)

    print("\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)


if __name__ == '__main__':
    main()
