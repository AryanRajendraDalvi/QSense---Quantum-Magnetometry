"""Parse benchmark JSON and print the pre-registered results table."""
import json
import sys

GAMMA = 0.1759  # rad/us per uT

def get(data, method, scenario, k):
    for r in data:
        if r['method'] == method and r['scenario'] == scenario and r['k'] == k:
            return r
    return None

def row(r, label=None):
    if r is None:
        return f"  {'N/A':<28} {'---':<12} {'---':<8} {'---':<8} {'---'}"
    name = label or r['method']
    short = {'bayes_assumed': 'B1', 'bayes_nuisance': 'B2',
             'bayes_nuisance_shape': 'B3', 'nn': 'NN'}
    m = short.get(name, name)
    w95 = r.get('width95_uT', r.get('mean_width95_uT', None))
    w95_str = f"{w95:.4f}" if w95 else "N/A"
    return (f"  {m:<4} RMSE={r['rmse_uT']:.4f} uT  "
            f"cov68={r['cov68']:.3f}  cov95={r['cov95']:.3f}  "
            f"width95={w95_str}")

def section(data, title, scenarios, k):
    print(f"\n{'='*68}")
    print(f" {title}  (k={k})")
    print('='*68)
    methods = ['bayes_assumed', 'bayes_nuisance', 'bayes_nuisance_shape', 'nn']
    for sc in scenarios:
        print(f"\n  Scenario: {sc}")
        for m in methods:
            r = get(data, m, sc, k)
            print(row(r, m))

def main(path='results/final_benchmark.json'):
    data = json.load(open(path))

    print('\n' + '='*68)
    print(' PRE-REGISTERED WIN CRITERIA: cov95 and RMSE on drift')
    print('='*68)
    print(' Criteria: NN wins if cov95 in [0.93, 0.97] AND RMSE <= 1.1x B3 (at k=16)')
    print(' Gray zone: B3 cov95=0.901 at drift 3 rad (not cleanly < 0.90)')
    print()
    for drift_sc, label in [('drift_2rad_fast', 'Drift 2 rad'), ('drift_3rad_fast', 'Drift 3 rad')]:
        for k_val in [4, 16]:
            print(f"\n  {label}, k={k_val}:")
            b3 = get(data, 'bayes_nuisance_shape', drift_sc, k_val)
            nn = get(data, 'nn', drift_sc, k_val)
            b3_rmse = b3['rmse_uT'] if b3 else None
            nn_rmse = nn['rmse_uT'] if nn else None
            for m in ['bayes_assumed', 'bayes_nuisance', 'bayes_nuisance_shape', 'nn']:
                r = get(data, m, drift_sc, k_val)
                print(row(r, m))
            if k_val == 16 and b3_rmse and nn_rmse:
                ratio = nn_rmse / b3_rmse
                nn_cov95 = nn['cov95'] if nn else None
                win_rmse = ratio <= 1.1
                win_cov = nn_cov95 and 0.93 <= nn_cov95 <= 0.97
                print(f"\n    -> NN/B3 RMSE ratio = {ratio:.3f}  ({'PASS' if win_rmse else 'FAIL'} <= 1.10)")
                if nn_cov95:
                    print(f"    -> NN cov95 = {nn_cov95:.3f}  ({'PASS' if win_cov else 'FAIL'} in [0.93, 0.97])")

    section(data, 'MATCHED NOISE', ['matched'], 16)
    section(data, 'SHAPE MISMATCH', ['shape_p2', 'shape_p3'], 16)
    section(data, 'COMBINED CONDITIONS', ['shape_p2_t2_half', 'shape_p2_t2_double',
                                           'shape_p3_t2_half', 'shape_p3_t2_double'], 16)
    section(data, 'DRIFT SWEEP', ['drift_1rad_fast', 'drift_2rad_fast', 'drift_3rad_fast'], 16)
    section(data, 'VISIBILITY MISMATCH', ['visibility_half'], 16)

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'results/final_benchmark.json'
    main(path)
