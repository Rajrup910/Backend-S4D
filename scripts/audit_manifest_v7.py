"""Audit script for ml/data/manifest.csv and related datasets for V7 Branch B specification.
"""
import pandas as pd
import numpy as np

def run_audit():
    df = pd.read_csv('ml/data/manifest.csv')
    total_images = len(df)
    unique_lesions = df['lesion_id'].nunique()
    
    print(f"=== MANIFEST.CSV OVERVIEW ===")
    print(f"Total image rows: {total_images}")
    print(f"Unique lesion IDs: {unique_lesions}")
    print(f"Images per lesion mean: {total_images / unique_lesions:.3f}")
    
    # Missingness definitions
    age_nan = df['age'].isna()
    sex_unk = df['sex'].isin(['unknown', 'Unknown', np.nan]) | df['sex'].isna()
    loc_unk = df['localization'].isin(['unknown', 'Unknown', np.nan]) | df['localization'].isna()
    
    print("\n=== COLUMN MISSINGNESS (IMAGE LEVEL, N=10015) ===")
    print(f"Age NaN: {age_nan.sum()} ({age_nan.mean()*100:.3f}%)")
    print(f"Sex Unknown/NaN: {sex_unk.sum()} ({sex_unk.mean()*100:.3f}%)")
    print(f"Localization Unknown/NaN: {loc_unk.sum()} ({loc_unk.mean()*100:.3f}%)")
    
    any_missing = age_nan | sex_unk | loc_unk
    all_missing = age_nan & sex_unk & loc_unk
    age_and_sex_missing = age_nan & sex_unk
    print(f"Any metadata missing: {any_missing.sum()} ({any_missing.mean()*100:.3f}%)")
    print(f"Age AND Sex missing: {age_and_sex_missing.sum()} ({age_and_sex_missing.mean()*100:.3f}%)")
    print(f"All 3 missing: {all_missing.sum()} ({all_missing.mean()*100:.3f}%)")
    print(f"Complete cases (Age present, Sex known, Loc known): {(~any_missing).sum()} ({(~any_missing).mean()*100:.3f}%)")
    
    # Lesion-level audit (aggregating by lesion_id)
    print("\n=== LESION-LEVEL DEDUPLICATED AUDIT (N=7470) ===")
    lesion_df = df.groupby('lesion_id').first()
    l_age_nan = lesion_df['age'].isna()
    l_sex_unk = lesion_df['sex'].isin(['unknown', 'Unknown', np.nan]) | lesion_df['sex'].isna()
    l_loc_unk = lesion_df['localization'].isin(['unknown', 'Unknown', np.nan]) | lesion_df['localization'].isna()
    l_any_missing = l_age_nan | l_sex_unk | l_loc_unk
    l_all_missing = l_age_nan & l_sex_unk & l_loc_unk
    
    print(f"Unique lesions: {len(lesion_df)}")
    print(f"Lesions with Age NaN: {l_age_nan.sum()} ({l_age_nan.mean()*100:.3f}%)")
    print(f"Lesions with Sex Unknown: {l_sex_unk.sum()} ({l_sex_unk.mean()*100:.3f}%)")
    print(f"Lesions with Loc Unknown: {l_loc_unk.sum()} ({l_loc_unk.mean()*100:.3f}%)")
    print(f"Lesions with Any missing: {l_any_missing.sum()} ({l_any_missing.mean()*100:.3f}%)")
    print(f"Lesions with All 3 missing: {l_all_missing.sum()} ({l_all_missing.mean()*100:.3f}%)")
    
    # Are metadata consistent across multiple images of same lesion?
    print("\n=== LESION-LEVEL METADATA CONSISTENCY ===")
    consistency_age = df.groupby('lesion_id')['age'].nunique(dropna=False)
    consistency_sex = df.groupby('lesion_id')['sex'].nunique()
    consistency_loc = df.groupby('lesion_id')['localization'].nunique()
    print(f"Lesions with conflicting age: {(consistency_age > 1).sum()}")
    print(f"Lesions with conflicting sex: {(consistency_sex > 1).sum()}")
    print(f"Lesions with conflicting localization: {(consistency_loc > 1).sum()}")
    
    # Diagnosis breakdown
    print("\n=== MISSINGNESS BY DIAGNOSTIC CLASS (IMAGE LEVEL) ===")
    dx_classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    rows = []
    for c in dx_classes:
        sub = df[df['dx'] == c]
        n_c = len(sub)
        n_age_nan = sub['age'].isna().sum()
        n_sex_unk = sub['sex'].isin(['unknown']).sum()
        n_loc_unk = sub['localization'].isin(['unknown']).sum()
        n_any = (sub['age'].isna() | (sub['sex'] == 'unknown') | (sub['localization'] == 'unknown')).sum()
        rows.append({
            'dx': c,
            'N': n_c,
            'pct_of_total': n_c / total_images * 100,
            'age_nan': n_age_nan,
            'age_nan_pct': n_age_nan / n_c * 100,
            'sex_unk': n_sex_unk,
            'sex_unk_pct': n_sex_unk / n_c * 100,
            'loc_unk': n_loc_unk,
            'loc_unk_pct': n_loc_unk / n_c * 100,
            'any_missing': n_any,
            'any_missing_pct': n_any / n_c * 100
        })
    dx_stats_df = pd.DataFrame(rows)
    print(dx_stats_df.to_string(index=False))
    
    # Same breakdown at lesion level
    print("\n=== MISSINGNESS BY DIAGNOSTIC CLASS (LESION LEVEL) ===")
    rows_l = []
    for c in dx_classes:
        sub = lesion_df[lesion_df['dx'] == c]
        n_c = len(sub)
        n_age_nan = sub['age'].isna().sum()
        n_sex_unk = sub['sex'].isin(['unknown']).sum()
        n_loc_unk = sub['localization'].isin(['unknown']).sum()
        n_any = (sub['age'].isna() | (sub['sex'] == 'unknown') | (sub['localization'] == 'unknown')).sum()
        rows_l.append({
            'dx': c,
            'N_lesions': n_c,
            'pct_of_lesions': n_c / len(lesion_df) * 100,
            'age_nan': n_age_nan,
            'age_nan_pct': n_age_nan / n_c * 100,
            'sex_unk': n_sex_unk,
            'sex_unk_pct': n_sex_unk / n_c * 100,
            'loc_unk': n_loc_unk,
            'loc_unk_pct': n_loc_unk / n_c * 100,
            'any_missing': n_any,
            'any_missing_pct': n_any / n_c * 100
        })
    dx_stats_l_df = pd.DataFrame(rows_l)
    print(dx_stats_l_df.to_string(index=False))
    
    # Missingness by dx_type (ground truth validation type)
    print("\n=== MISSINGNESS BY DX_TYPE (IMAGE LEVEL) ===")
    rows_dxtype = []
    for dt in df['dx_type'].unique():
        sub = df[df['dx_type'] == dt]
        n_dt = len(sub)
        n_age_nan = sub['age'].isna().sum()
        n_sex_unk = sub['sex'].isin(['unknown']).sum()
        n_loc_unk = sub['localization'].isin(['unknown']).sum()
        n_any = (sub['age'].isna() | (sub['sex'] == 'unknown') | (sub['localization'] == 'unknown')).sum()
        rows_dxtype.append({
            'dx_type': dt,
            'N': n_dt,
            'pct': n_dt / total_images * 100,
            'age_nan': n_age_nan,
            'age_nan_pct': n_age_nan / n_dt * 100,
            'sex_unk': n_sex_unk,
            'sex_unk_pct': n_sex_unk / n_dt * 100,
            'loc_unk': n_loc_unk,
            'loc_unk_pct': n_loc_unk / n_dt * 100,
            'any_missing': n_any,
            'any_missing_pct': n_any / n_dt * 100
        })
    print(pd.DataFrame(rows_dxtype).to_string(index=False))

    # Age distribution statistics
    print("\n=== AGE DISTRIBUTION BY CLASS ===")
    age_rows = []
    for c in dx_classes:
        ages = df[df['dx'] == c]['age'].dropna()
        age_rows.append({
            'dx': c,
            'N_valid': len(ages),
            'mean': ages.mean(),
            'std': ages.std(),
            'min': ages.min(),
            'q25': ages.quantile(0.25),
            'median': ages.median(),
            'q75': ages.quantile(0.75),
            'max': ages.max()
        })
    all_ages = df['age'].dropna()
    age_rows.append({
        'dx': 'ALL',
        'N_valid': len(all_ages),
        'mean': all_ages.mean(),
        'std': all_ages.std(),
        'min': all_ages.min(),
        'q25': all_ages.quantile(0.25),
        'median': all_ages.median(),
        'q75': all_ages.quantile(0.75),
        'max': all_ages.max()
    })
    print(pd.DataFrame(age_rows).to_string(index=False))

    # Age stratification breakdown (<20, 20-29, 30-39, 40-59, 60+)
    print("\n=== AGE STRATIFICATION BREAKDOWN (5 BANDS) ===")
    def get_age_band(age):
        if pd.isna(age):
            return 'missing'
        if age < 20:
            return '<20'
        elif age < 30:
            return '20-29'
        elif age < 40:
            return '30-39'
        elif age < 60:
            return '40-59'
        else:
            return '60+'

    df['age_band_5'] = df['age'].apply(get_age_band)
    band_order = ['<20', '20-29', '30-39', '40-59', '60+', 'missing']
    
    ct = pd.crosstab(df['age_band_5'], df['dx']).reindex(band_order)
    ct['TOTAL'] = ct.sum(axis=1)
    # Add escalating class count (mel, bcc, akiec)
    ct['ESCALATING (mel+bcc+akiec)'] = ct['mel'] + ct['bcc'] + ct['akiec']
    ct['ESCALATING_PREVALENCE_%'] = ct['ESCALATING (mel+bcc+akiec)'] / ct['TOTAL'] * 100
    ct['MELANOMA_PREVALENCE_%'] = ct['mel'] / ct['TOTAL'] * 100
    print(ct.to_string())

if __name__ == '__main__':
    run_audit()
