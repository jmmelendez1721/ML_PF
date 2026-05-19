import numpy as np
from scipy import stats

def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)

    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j

    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2



def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m

    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])

    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n

    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m

    sx = np.cov(v01)
    sy = np.cov(v10)

    delongcov = sx / m + sy / n

    return aucs, delongcov



def calc_pvalue(aucs, sigma):
    aucs = np.asarray(aucs).reshape(-1)

    # validar tamaño mínimo
    if aucs.size < 2:
        return 1.0  # no hay comparación válida

    l = np.array([[1, -1]])

    denom = np.dot(np.dot(l, sigma), l.T)

    # evitar división por cero o negativos numéricos
    denom = np.asarray(denom).squeeze()
    denom = np.maximum(denom, 1e-12)

    z = np.abs(np.diff(aucs)) / np.sqrt(denom)

    pvalue = 2 * (1 - stats.norm.cdf(z))

    pvalue = np.asarray(pvalue).reshape(-1)

    return float(pvalue[0])



def delong_roc_test(y_true, pred_one, pred_two):
    order = np.argsort(-y_true)
    label_1_count = int(np.sum(y_true))

    predictions_sorted_transposed = np.vstack(
        (pred_one, pred_two)
    )[:, order]

    aucs, delongcov = fast_delong(
        predictions_sorted_transposed,
        label_1_count
    )

    return calc_pvalue(aucs, delongcov)