from codes.metric.trackable_metric import TrackableMetric

def get_metric_dict(time_span=100):

    metric_dict = {
        "val_loss": TrackableMetric(name = "val_loss", default_value=1e6, time_span=time_span, mode="min"),
        "val_acc": TrackableMetric(name = "val_acc", default_value=0, time_span=time_span, mode='max')
    }
    return metric_dict
