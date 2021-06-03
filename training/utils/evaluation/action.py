def performance(output, target):
    dummy, pred_class = output.max(1)
    return pred_class.eq(target).sum().float() / pred_class.size(0)


def final_preds(output, center, scale, res):
    return output
