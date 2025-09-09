import pandas as pd
import matplotlib.pyplot as plt

RETENTION_CSV = "/home/s3986160/master-thesis/Retention/new definition/retention_AND.csv"
USERS_CSV = "/home/s3986160/master-thesis/Active users/sarafu_users_20210615.csv"

retention_df = pd.read_csv(RETENTION_CSV)
users_df = pd.read_csv(USERS_CSV, low_memory=False)
users_df.rename(columns={'xDAI_blockchain_address': 'address'}, inplace=True)

def analyze_demographics(retention_df, users_df, threshold, features):
    col_ret = f'retained_{threshold}'
    col_not = f'not_retained_{threshold}'

    retained_all = retention_df[retention_df[col_ret] == 1]
    not_retained_all = retention_df[retention_df[col_not] == 1]

    print(f"\nTOTAL retained users at {threshold}%: {len(retained_all)}")
    print(f"TOTAL not retained users at {threshold}%: {len(not_retained_all)}")

    retained = pd.merge(retained_all, users_df, on='address', how='left')
    not_retained = pd.merge(not_retained_all, users_df, on='address', how='left')

    results = {}

    for feature in features:
        n_missing_ret = retained[feature].isna().sum()
        n_missing_not = not_retained[feature].isna().sum()

        print(f"\n--- {feature.upper()} ---")
        print(f"Retained: {len(retained)} users, {n_missing_ret} missing {feature}")
        print(f"Not Retained: {len(not_retained)} users, {n_missing_not} missing {feature}")

        retained_valid = retained[retained[feature].notna()]
        not_retained_valid = not_retained[not_retained[feature].notna()]

        ret_counts = retained_valid[feature].value_counts(dropna=False).sort_index()
        not_counts = not_retained_valid[feature].value_counts(dropna=False).sort_index()

        all_labels = sorted(set(ret_counts.index).union(set(not_counts.index)))
        ret_counts = ret_counts.reindex(all_labels, fill_value=0)
        not_counts = not_counts.reindex(all_labels, fill_value=0)

        print("\nRetained:")
        print(ret_counts)
        print("\nNot Retained:")
        print(not_counts)

        labels = [str(label) for label in all_labels]
        x = range(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x, ret_counts.values, width=width, label='Ret', color='mediumseagreen')
        ax.bar([p + width for p in x], not_counts.values, width=width, label='Not Ret', color='salmon')
        ax.set_xticks([p + width / 2 for p in x])
        ax.set_xticklabels(labels, rotation=45)
        ax.set_title(f"{feature.capitalize()} @ {threshold}%")
        ax.set_ylabel("Count")
        ax.legend()
        plt.tight_layout()

        filename = f"{feature}_{threshold}.png"
        plt.savefig(filename)
        plt.close()

        results[feature] = pd.DataFrame({
            'Retained': ret_counts,
            'Not Retained': not_counts
        })

    return results

if __name__ == "__main__":
    threshold = 25
    features = ['business_type'] 

    summary_tables = analyze_demographics(retention_df, users_df, threshold, features)

    print("\n Done. Summary tables printed. Plots saved in the current folder.")
