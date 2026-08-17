from io         import StringIO
from pathlib    import Path
from typing     import Dict, List, Optional, Tuple

import concurrent.futures   as fut
import matplotlib.pyplot    as plt
import numpy                as np
import pandas               as pd
import requests             as req
import statsmodels.api      as sm

QUICK_MODE: bool = False

CATEGORY_THRESHOLD  : int = 5
TIME_DIFFERENCE     : int = 9
TOTAL_THRESHOLD     : int = 40

WEEKDAY_MAP: Dict[str, str] = {
    'Monday'    : 'Mon',
    'Tuesday'   : 'Tue',
    'Wednesday' : 'Wed',
    'Thursday'  : 'Thu',
    'Friday'    : 'Fri',
    'Saturday'  : 'Sat',
    'Sunday'    : 'Sun'
}

ZONE_MAP: Dict[str, str] = {
    'Pre-NA'    : '<NA',
    'Post-NA'   : 'NA>',
    'NA-Asia'   : 'NA-AS',
    'Pre-Asia'  : '<AS',
    'Post-Asia' : 'AS>',
    'Asia-EU'   : 'AS-EU',
    'Pre-EU'    : '<EU',
    'Post-EU'   : 'EU>',
    'EU-NA'     : 'EU-NA'
}

BASE_URL            : str = "https://docs.google.com/spreadsheets/d/1Fm6pMyXv7qhOQkLah4yX9HNow4WaDR4HJuAVMukQl34/export?format=csv&gid="
NAME_URL            : str = "https://docs.google.com/spreadsheets/d/10YBcZP_l5Tjf1MOiWeBlLg-ATuAWXgTPsj7bW79bU30/export?format=csv&gid=1934025140"
WATCHED_THRESHOLD   : str = '2602160000'

SIGMOIDS: Dict[str, List[Tuple[str, float, float]]] = {
    'Watched': [
        ('Elo Gain/Loss',       5.00, 20),
        ('Guess Rate',          1.00, 20),
        ('Usefulness',          1.00, 15),
        ('Solos',               1.00, 5),
        ('Sevens',              -1.0, 5), 
        ('Contribution Rate',   1.00, 5),
        ('Corrects',            1.00, 5), 
        ('Win Rate',            1.00, 5),
        ('Onlist Guess Rate',   1.00, 10),
        ('Offlist Guess Rate',  5.00, 10)
    ],

    'Usual': [
        ('Elo Gain/Loss',       10.0, 20),
        ('Guess Rate',          1.00, 20),
        ('Usefulness',          5.00, 20),
        ('Solos',               1.00, 10),
        ('Sevens',              -1.0, 10),
        ('Corrects',            1.00, 10),
        ('Win Rate',            1.00, 10)
    ]
}

BOXPLOT_STYLE: dict = dict(
    patch_artist = True,
    showmeans    = False,
    medianprops  = dict(color = 'black'),
    whiskerprops = dict(color = 'black'),
    capprops     = dict(color = 'black'),
    boxprops     = dict(color = 'black', facecolor = 'white'),
    meanprops    = dict(marker = 'o', markeredgecolor = 'black', markerfacecolor = 'black'),
    flierprops   = dict(marker = 'o', markeredgecolor = 'black', markerfacecolor = 'white')
)

def fetch_csv(url: str) -> pd.DataFrame:
    response = req.get(url)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))

def format_timestamp(ts) -> str:
    try                             : return pd.to_datetime(ts).strftime('%y%m%d%H%M')
    except (ValueError, TypeError)  : return str(ts)

def assign_zones(df: pd.DataFrame) -> np.ndarray:
    hours = df['dt'].dt.hour + df['dt'].dt.minute / 60.0

    conditions = [
        (hours >= 0.5)  & (hours < 2.5),
        (hours >= 2.5)  & (hours < 4.5),
        (hours >= 4.5)  & (hours < 7.5),
        (hours >= 7.5)  & (hours < 9.5),
        (hours >= 9.5)  & (hours < 11.5),
        (hours >= 11.5) & (hours < 14.5),
        (hours >= 14.5) & (hours < 17.5),
        (hours >= 17.5) & (hours < 21.5),
    ]

    choices = ["Asia-EU", "Pre-EU", "Post-EU", "EU-NA", "Pre-NA", "Post-NA", "NA-Asia", "Pre-Asia"]
    return np.select(conditions, choices, default="Post-Asia")

def style_axis(ax: plt.Axes, title: str, low: float, high: float, is_large: bool = False) -> None:
    ax.spines[['left', 'right']].set_visible(False)

    title_size = 30.0 if is_large else 20
    label_size = 22.5 if is_large else 15

    ax.set_title    (title, fontsize = title_size, pad = 20)
    ax.yaxis.grid   (True, linestyle = '-', color = 'black')
    ax.set_ylim     (high, low) if 'Average Over-8' in title else ax.set_ylim(low, high)
    ax.set_yticks   (np.linspace(low, high, 5))
    ax.tick_params  (axis = 'both', length = 0, labelsize = label_size, pad = 10)

def _compute_lowess_smooth(x_numeric: np.ndarray, y_series: pd.Series) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    y_numeric   = y_series.to_numpy(dtype=float)
    valid_mask  = ~np.isnan(y_numeric) & ~np.isnan(x_numeric)
    x_clean     = x_numeric[valid_mask]
    y_clean     = y_numeric[valid_mask]

    if len(x_clean) > 5:
        lowess_fit = sm.nonparametric.lowess(y_clean, x_clean, frac=0.25)
        return lowess_fit[:, 1], valid_mask

    return None, None

def get_limits(series: pd.Series, metric: str, mode: Optional[str]) -> Tuple[float, float]:
    full_range_metrics = {'Contribution Rate', 'Win Rate', 'List Guess Rate', 'Rating'}

    if metric in full_range_metrics : return 0, 100
    if metric == 'Average Over-8'   : return 2, 6

    s_min, s_max = series.min(), series.max()
    if pd.isna(s_min) or pd.isna(s_max): return 0, 100

    if metric == 'Elo Gain/Loss':
        abs_max = np.ceil(max(abs(s_min), abs(s_max)))
        if abs_max % 2 != 0: abs_max += 1
        return -abs_max, abs_max

    over_8_metrics      = {'Solos', 'Threes', 'Sevens', 'Over-8s'}
    small_usual_metrics = {'Elo', 'Usefulness'}

    if metric in over_8_metrics or (metric in small_usual_metrics and mode == 'Usual'):
        low     = np.floor  (s_min) if metric == 'Elo' else max(0, np.floor(s_min))
        high    = np.ceil   (s_max)   
        rem     = int(high - low) % 4

        if rem != 0: high += (4 - rem)
        return low, high

    low     = int(max(0, np.floor   (s_min / 5) * 5))
    high    = int(np.ceil           (s_max / 5) * 5)

    while (high - low) % 4 != 0:
        if low > 0:
            low -= 5
            if (high - low) % 4 == 0: break

        high += 5

    return low, high

def _prepare_scatter_metrics(df: pd.DataFrame, metrics: List[Tuple[str, float, float]], x_numeric: np.ndarray) -> List[Tuple[str, float, float]]:
    groups = {
        'Combined Guess Rate'   : ({'Opening Guess Rate', 'Ending Guess Rate', 'Insert Guess Rate'},    0, 100),
        'List Guess Rate'       : ({'Onlist Guess Rate', 'Offlist Guess Rate'},                         0, 100),
        'Over-8s'               : ({'Solos', 'Sevens', 'Threes'},                                       0, 4)
    }

    seen_groups         = set()
    filtered_metrics    = []

    for name, low, high in metrics:
        matched_group = None

        for g_name, (members, _, _) in groups.items():
            if name in members:
                matched_group = g_name
                break

        if matched_group:
            if matched_group not in seen_groups:
                members, def_low, def_high  = groups[matched_group]
                available                   = [m for m in members if m in df.columns]
                mins, maxs                  = [], []

                for m in available:
                    y_smooth, _ = _compute_lowess_smooth(x_numeric, df[m])

                    if y_smooth is not None:
                        mins.append(y_smooth.min())
                        maxs.append(y_smooth.max())

                if mins and maxs    : low, high = get_limits(pd.Series([min(mins), max(maxs)]), matched_group, None)
                else                : low, high = def_low, def_high

                filtered_metrics.append((matched_group, low, high))
                seen_groups.add(matched_group)

        else: filtered_metrics.append((name, low, high))

    return filtered_metrics

def create_plots(df: pd.DataFrame, metrics: List[Tuple[str, float, float]], file_path: Path, plot_type: str, group_col: Optional[str] = None) -> None:
    if plot_type == 'scatter':
        if      'dt'        in df.columns   : x_visual = df['dt']
        elif    'Timestamp' in df.columns   : x_visual = pd.to_datetime(df['Timestamp'].astype(str), format = '%y%m%d%H%M', errors = 'coerce')
        else                                : x_visual = pd.Series(np.arange(len(df)), index = df.index)

        is_datetime = isinstance(x_visual.dtype, pd.DatetimeTZDtype) or np.issubdtype(x_visual.dtype, np.datetime64)
        x_numeric   = x_visual.astype('int64') // 10 ** 9 if is_datetime else np.arange(len(df))

    else: x_numeric = None

    metrics_to_process  = _prepare_scatter_metrics(df, metrics, x_numeric) if plot_type == 'scatter' else metrics
    watched_metrics     = {'List Guess Rate', 'Onlist Guess Rate', 'Offlist Guess Rate', 'Rig Rate', 'Rig Rate by Day', 'Rig Rate by Zone'}
    is_watched          = any(m[0] in watched_metrics for m in metrics_to_process)
    rows, cols          = 4, (5 if is_watched and plot_type == 'boxplot' else 4)
    fig                 = plt.figure(figsize = (cols * 10, rows * 10))
    gs                  = fig.add_gridspec(rows, cols)
    plot_positions      = []

    if is_watched:
        plot_positions.append(gs[0 : 2, 0 : 2])

        for r in range(rows):
            for c in range(cols):
                if r < 2 and c < 2: continue
                plot_positions.append(gs[r, c])

    else:
        if plot_type == 'scatter':
            plot_positions.append(gs[0 : 2, 0 : 3])

            for r in range(rows):
                for c in range(cols):
                    if r < 2 and c < 3: continue
                    plot_positions.append(gs[r, c])

        else:
            plot_positions.append(gs[0 : 2, 0])
            plot_positions.append(gs[0 : 2, 1])

            for r in range(rows):
                for c in range(cols):
                    if c < 2 and r < 2: continue
                    plot_positions.append(gs[r, c])

    if plot_type == 'boxplot':
        group_order     = list(WEEKDAY_MAP.keys()) if group_col == 'Day' else list(ZONE_MAP.keys())
        valid_groups    = [g for g in group_order if len(df[df[group_col] == g]) >= CATEGORY_THRESHOLD]
        labels          = [WEEKDAY_MAP.get(g, g) if group_col == 'Day' else ZONE_MAP.get(g, g) for g in valid_groups]

    for i, (col, low, high) in enumerate(metrics_to_process):
        if i >= len(plot_positions): break

        pos         = plot_positions[i]
        is_large    = (pos.rowspan.stop - pos.rowspan.start > 1) or (pos.colspan.stop - pos.colspan.start > 1)
        ax          = fig.add_subplot(pos)

        if plot_type == 'scatter':
            if col in {'Combined Guess Rate', 'List Guess Rate', 'Over-8s'}:
                sub_styles = {
                    'Combined Guess Rate': {
                        'Opening Guess Rate'    : {'color': 'black', 'label': 'OP',     'linestyle': '-'},
                        'Ending Guess Rate'     : {'color': 'black', 'label': 'ED',     'linestyle': '--'},
                        'Insert Guess Rate'     : {'color': 'black', 'label': 'IN',     'linestyle': ':'}
                    },

                    'List Guess Rate': {
                        'Onlist Guess Rate'     : {'color': 'black', 'label': 'On',     'linestyle': '-'},
                        'Offlist Guess Rate'    : {'color': 'black', 'label': 'Off',    'linestyle': '--'}
                    },

                    'Over-8s': {
                        'Solos'                 : {'color': 'black', 'label': 'Solos',  'linestyle': '-'},
                        'Threes'                : {'color': 'black', 'label': 'Threes', 'linestyle': '--'},
                        'Sevens'                : {'color': 'black', 'label': 'Sevens', 'linestyle': ':'}
                    }
                }[col]

                ax2 = None
                if col == 'List Guess Rate' and 'Offlist Guess Rate' in df.columns: ax2 = ax.twinx()

                for g_col, style in sub_styles.items():
                    if g_col not in df.columns: continue
                    y_smooth, mask = _compute_lowess_smooth(x_numeric, df[g_col])

                    if y_smooth is not None:
                        x_chunk = x_visual[mask]

                        if col == 'List Guess Rate' and g_col == 'Offlist Guess Rate' and ax2 is not None   : ax2   .plot(x_chunk, y_smooth, linewidth = 3, zorder = 3, **style)
                        else                                                                                : ax    .plot(x_chunk, y_smooth, linewidth = 3, zorder = 3, **style)

                if col == 'List Guess Rate':
                    if 'Onlist Guess Rate' in df.columns:
                        y_smooth_on, _  = _compute_lowess_smooth(x_numeric, df['Onlist Guess Rate'])
                        s_min           = y_smooth_on.min() if (y_smooth_on is not None and len(y_smooth_on) > 0) else df['Onlist Guess Rate'].min()
                        s_max           = y_smooth_on.max() if (y_smooth_on is not None and len(y_smooth_on) > 0) else df['Onlist Guess Rate'].max()

                        if pd.isna(s_min) or pd.isna(s_max): low_on, high_on = 0, 100

                        else:
                            low_on  = int(max(0, np.floor(s_min / 5) * 5))
                            high_on = int(np.ceil(s_max / 5) * 5)

                            while (high_on - low_on) % 20 != 0:
                                if low_on > 0:
                                    low_on -= 5
                                    if (high_on - low_on) % 20 == 0: break

                                high_on += 5

                            if high_on > 100:
                                low_on  -= high_on - 100
                                high_on -= high_on - 100

                    else: low_on, high_on = 0, 100

                    if 'Offlist Guess Rate' in df.columns:
                        y_smooth_off, _ = _compute_lowess_smooth(x_numeric, df['Offlist Guess Rate'])
                        s_min           = y_smooth_off.min() if (y_smooth_off is not None and len(y_smooth_off) > 0) else df['Offlist Guess Rate'].min()
                        s_max           = y_smooth_off.max() if (y_smooth_off is not None and len(y_smooth_off) > 0) else df['Offlist Guess Rate'].max()

                        if pd.isna(s_min) or pd.isna(s_max): low_off, high_off = 0, 100

                        else:
                            low_off     = int(max(0, np.floor(s_min / 5) * 5))
                            high_off    = int(np.ceil(s_max / 5) * 5)

                            while (high_off - low_off) % 20 != 0:
                                if low_off > 0:
                                    low_off -= 5
                                    if (high_off - low_off) % 20 == 0: break

                                high_off += 5

                    else: low_off, high_off = 0, 100

                    style_axis(ax, col, low_on, high_on, is_large = is_large)
                    if ax2 is not None: style_axis(ax2, "", low_off, high_off, is_large = is_large)

                    lines1, labels1 = ax    .get_legend_handles_labels()
                    lines2, labels2 = ax2   .get_legend_handles_labels() if ax2 is not None else ([], [])

                    ax.legend(lines1 + lines2, labels1 + labels2, fontsize = 14 if is_large else 12, frameon = True, facecolor = 'white', edgecolor = 'none')

                else:
                    style_axis(ax, 'Type Guess Rates' if col == 'Combined Guess Rate' else col, low, high, is_large = is_large)
                    ax.legend(fontsize = 14 if is_large else 12, frameon = True, facecolor = 'white', edgecolor = 'none')

                ax.set_xticks([])

            else:
                if col not in df.columns: continue
                ax.scatter(x_visual, df[col], color = 'white', zorder = 2, edgecolor = 'black')
                y_smooth, mask = _compute_lowess_smooth(x_numeric, df[col])

                if y_smooth is not None:
                    x_chunk = x_visual[mask]
                    ax.plot(x_chunk, y_smooth, color = 'black', zorder = 3)

                style_axis(ax, col, low, high, is_large = is_large)
                ax.set_xticks([])

        else:
            if col not in df.columns: continue
            data_to_plot = [df[df[group_col] == g][col].dropna() for g in valid_groups]
            ax.boxplot(data_to_plot, tick_labels = labels, **BOXPLOT_STYLE)

            means       = [d.mean() for d in data_to_plot]
            x_positions = range(1, len(data_to_plot) + 1)

            ax.plot(x_positions, means, color = 'black', linestyle = '-', zorder = 3)
            style_axis(ax, f"{col} by {group_col}", low, high, is_large = is_large)

    plt.subplots_adjust (hspace = 0.25, wspace = 0.25)
    plt.savefig         (file_path, bbox_inches = 'tight', dpi = 100, pad_inches = 0.5)
    plt.close           (fig)

def extract_and_prepare_data() -> Tuple[pd.DataFrame, dict]:
    print("[?] Processing alias data")

    names_df        = fetch_csv(NAME_URL)[['Player ID', 'Player Name']].rename(columns = {'Player ID': 'ID', 'Player Name': 'Name'})
    categories_data = {}
    categories      = {'Usual': {'detail': '0', 'record': '286606464'}, 'Watched': {'detail': '2040874005', 'record': '985078008'}}

    col_map = {
        'Player name'       : 'Name',
        'Rank'              : 'Elo',
        'Guess rate'        : 'Guess Rate',
        'Usefulness'        : 'Usefulness',
        'erigs'             : 'Solos',
        '# 3/8s or below'   : 'Threes',
        '7/8s'              : 'Sevens',
        'avg/8'             : 'Average Over-8',
        'OP guess rate'     : 'Opening Guess Rate',
        'ED guess rate'     : 'Ending Guess Rate',
        'IN guess rate'     : 'Insert Guess Rate',
        'Lives taken'       : 'Points',
        'Lives saved'       : 'Blocks',
        'Total hit'         : 'Corrects'
    }

    def fetch_single_category(cat: str, gids: dict) -> Tuple[str, dict]:
        print(f"[?] Processing {cat} data")

        detail_raw              = fetch_csv(BASE_URL + gids['detail'])
        detail_df               = detail_raw[['Player Name', 'Player Elo', '# Tours played in 2026']].copy()
        detail_df.columns       = ['Name', 'Elo', 'Tour Count']
        detail_df               = detail_df[detail_df['Tour Count'] >= TOTAL_THRESHOLD].sort_values(by = 'Elo', ascending = False)

        record_raw              = fetch_csv(BASE_URL + gids['record'])
        record_raw              = record_raw[record_raw['Timestamp'].notna()].copy()
        record_raw['Timestamp'] = record_raw['Timestamp'].apply(format_timestamp)

        w, d, l                 = record_raw['WIN'].fillna(0), record_raw['TIE'].fillna(0), record_raw['LOSE'].fillna(0)
        record_raw['Win Rate']  = 100 * (w + d / 2) / (w + l + d)

        current_col_map = col_map.copy()
        if cat == 'Watched': current_col_map.update({'Onlist': 'Onlist Guess Rate', 'Offlist': 'Offlist Guess Rate', 'Rig %': 'Rig Rate'})

        record_df = record_raw[['Timestamp'] + list(current_col_map.keys()) + ['Win Rate']].rename(columns = current_col_map)

        return cat, {'detail': detail_df, 'record': record_df}

    with fut.ThreadPoolExecutor() as executor:
        futures = [executor.submit(fetch_single_category, cat, gids) for cat, gids in categories.items()]

        for future in fut.as_completed(futures):
            cat, data               = future.result()
            categories_data[cat]    = data

    return names_df, categories_data

def process_and_output_player(category_name: str, prefix: str, df: pd.DataFrame, sigmoids: list) -> None:
    df['dt']            = pd.to_datetime(df['Timestamp'].astype(str), format = '%y%m%d%H%M') + pd.Timedelta(hours = TIME_DIFFERENCE)
    df['Day']           = df['dt'].dt.day_name()
    df['Zone']          = assign_zones(df)

    metrics             = [m[0] for m in sigmoids]
    sensitivities       = np.array([m[1] for m in sigmoids])
    weights             = np.array([m[2] for m in sigmoids])
    rolling_averages    = df[metrics].rolling(window = 10, min_periods = 1).mean().to_numpy()
    current_values      = df[metrics].to_numpy()
    exponent            = -sensitivities * (current_values - rolling_averages)
    clipped_exponent    = np.clip(exponent, -10, 10)
    sigmoid_matrix      = weights / (1.0 + np.exp(clipped_exponent))
    df['Rating']        = np.round(np.sum(sigmoid_matrix, axis = 1), 2)

    cols_order          = ['Day', 'Zone'] + [c for c in df.columns if c not in {'Day', 'Zone', 'dt'}]
    df_out              = df[cols_order].set_index('Timestamp')    
    output_dir          = Path(category_name) / prefix

    output_dir.mkdir(parents = True, exist_ok = True)
    df_out.to_csv(output_dir / f'{prefix}.csv', float_format = '%.2f')

    df_plot = df.reset_index(drop = True)

    if category_name == 'Watched':
        mask_before_threshold = df_plot['Timestamp'].astype(str) < WATCHED_THRESHOLD

        for col in ['Elo', 'Elo Gain/Loss']:
            if col in df_plot.columns: df_plot.loc[mask_before_threshold, col] = np.nan

    skips               = {'Day', 'Zone', 'Timestamp', 'dt'}
    plot_candidates     = [c for c in df.columns if c not in skips]
    scatter_metrics     = [(col, *get_limits(df_plot[col], col, category_name)) for col in plot_candidates]
    boxplot_metrics     = [m for m in scatter_metrics if m[0] != 'Elo']

    create_plots(df_plot, scatter_metrics, output_dir / f'{prefix}-General.png',    'scatter')
    create_plots(df_plot, boxplot_metrics, output_dir / f'{prefix}-Day.png',        'boxplot',  'Day')
    create_plots(df_plot, boxplot_metrics, output_dir / f'{prefix}-Zone.png',       'boxplot',  'Zone')

    print(f'[✓] Processed {output_dir}')

def worker_task(mode: str, prefix: str, names: pd.DataFrame, cats: dict) -> None:
    record_df           = cats[mode]['record']
    lookup              = names[names['Name'].str.lower() == prefix.lower()]
    associated_names    = [prefix] if lookup.empty else names[names['ID'] == lookup['ID'].iloc[0]]['Name'].tolist()
    player_data         = record_df[record_df['Name'].isin(associated_names)].copy()

    if player_data.empty or len(player_data) < TOTAL_THRESHOLD: return
    incremental_metrics = ['Solos', 'Threes', 'Sevens', 'Corrects']
    for col in incremental_metrics: player_data[col] = pd.to_numeric(player_data[col], errors = 'coerce').fillna(0).astype(int)

    detail_df           = cats[mode]['detail']
    current_elo_series  = detail_df[detail_df['Name'].isin(associated_names)]['Elo']
    current_elo         = current_elo_series.iloc[0] if not current_elo_series.empty else np.nan
    next_elo            = player_data['Elo'].shift(-1)

    if pd.notna(current_elo) and len(next_elo) > 0: next_elo.iloc[-1] = current_elo

    player_data['Elo Gain/Loss']        = (next_elo - player_data['Elo']).round(2)
    player_data['Contribution Rate']    = 100 * (player_data['Points'] + player_data['Blocks']) / player_data['Corrects']

    if mode == 'Watched':
        mask = player_data['Timestamp'].astype(str) < WATCHED_THRESHOLD

        for col in ['Elo', 'Elo Gain/Loss', 'Usefulness']:
            if col in player_data.columns:
                player_data[col]            =   pd.to_numeric(player_data[col], errors='coerce')
                player_data.loc[mask, col]  *=  10

    output_columns = [
        'Timestamp',
        'Elo',
        'Elo Gain/Loss',
        'Guess Rate',
        'Usefulness',
        'Solos',
        'Threes',
        'Sevens',
        'Average Over-8',
        'Opening Guess Rate',
        'Ending Guess Rate',
        'Insert Guess Rate',
        'Contribution Rate',
        'Corrects',
        'Win Rate'
    ]

    if mode == 'Watched': output_columns.extend(['Onlist Guess Rate', 'Offlist Guess Rate', 'Rig Rate'])
    player_data = player_data[[col for col in output_columns if col in player_data.columns]]
    process_and_output_player(mode, prefix, player_data, SIGMOIDS[mode])

if __name__ == '__main__':    
    print("[?] Processing input data")
    names, cats = extract_and_prepare_data()

    if QUICK_MODE:
        print("[?] Processing in Quick Mode")
        targets = [('Watched', 'hakohoka'), ('Usual', 'florenz')]

    else:
        print("[?] Processing in Sweep Mode")
        targets = []

        for mode in ['Watched', 'Usual']:
            detail_df                   = cats[mode]['detail'].copy()
            detail_df['Name_lower']     = detail_df['Name'].str.lower()
            names_lower                 = names.copy()
            names_lower['Name_lower']   = names_lower['Name'].str.lower()
            name_to_id                  = pd.merge(detail_df[['Name_lower']], names_lower, on = 'Name_lower', how = 'left')
            unique_ids                  = name_to_id['ID'].dropna().unique()

            for player_id in unique_ids:
                associated_names = names[names['ID'] == player_id]['Name'].tolist()
                if associated_names: targets.append((mode, associated_names[0]))

    print("[?] Delegating to ProcessPoolExecutor")

    with fut.ProcessPoolExecutor() as executor:
        futures = [executor.submit(worker_task, mode, prefix, names, cats) for mode, prefix in targets]
        fut.wait(futures)

    print("[✓] Processed all data")