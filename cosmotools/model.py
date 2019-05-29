

def psd_mean(*args,**kwargs):
    s = psd(*args, **kwargs)
    return (np.mean(s[0], axis=0), *s[1:])


def cosmo_metric_list(recompute_real=False):
    metric_list = []
    metric_list.append(StatisticalMetricLim(Statistic(mass_hist, name='mass_histogram', group='cosmology'), log=True, recompute_real=recompute_real, stype=3))
    metric_list.append(StatisticalMetricLim(Statistic(peak_hist, name='peak_histogram', group='cosmology'), log=True, recompute_real=recompute_real, stype=3))
    metric_list.append(StatisticalMetric(Statistic(psd_mean, name='psd', group='cosmology'), log=True, recompute_real=recompute_real, stype=3))
    # metric_list.append(MetricSum(metric_list[:3], name ='global_score', group='cosmology', recompute_real=recompute_real, stype=0))

    metric_list = [MetricSum(metric_list, name ='global_score', group='cosmology', recompute_real=recompute_real, stype=0)]

    # TODO: wasserstein

    return metric_list


def global_score(recompute_real=False):
    return cosmo_metric_list(recompute_real)[-1]



class CosmoWGAN(WGAN):
    def default_params(self):
        d_params = deepcopy(super().default_params())
        d_params['cosmology'] = dict()
        d_params['cosmology']['forward_map'] = None
        d_params['cosmology']['backward_map'] = None
        return d_params

    def _build_stat_summary(self):
        super()._build_stat_summary()
        self._cosmo_metric_list = ganlist.cosmo_metric_list()
        for met in self._cosmo_metric_list:
            met.add_summary(collections="model")

    def preprocess_summaries(self, X_real, **kwargs):
        super().preprocess_summaries(X_real, **kwargs)
        if self.params['cosmology']['backward_map']:
            X_real = self.params['cosmology']['backward_map'](X_real)
        for met in self._cosmo_metric_list:
            met.preprocess(X_real, **kwargs)

    def compute_summaries(self, X_real, X_fake, feed_dict={}):
        feed_dict = super().compute_summaries(X_real, X_fake, feed_dict)
        if self.params['cosmology']['backward_map']:
            if X_real is not None:
                X_real = self.params['cosmology']['backward_map'](X_real)
            X_fake = self.params['cosmology']['backward_map'](X_fake)
        for met in self._cosmo_metric_list:
            feed_dict = met.compute_summary(X_fake, X_real, feed_dict)
        return feed_dict