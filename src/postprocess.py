

def postprocess(y_pred, threshold=0.5, min_counts=[800, 1000, 1800, 5000]):
    n_class = y_pred.shape[-1]
    if n_class = 