import io
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use('Agg')

# Inspired by Andres


class PlotSummary(object):
    def __init__(self, name, cat, collections=None):
        self._name = name
        self._placeholder = tf.placeholder(tf.uint8, (None, None, None, None))
        self._summary = tf.summary.image(
            cat + '/' + name, self._placeholder, collections=collections)
        self._image = None

    def produceSummaryToWrite(self, session, *args, **kwargs):
        self.plot(*args, **kwargs)
        self._fill_from_figure()
        decoded_image = session.run(self._image)
        feed_dict = {self._placeholder: decoded_image}
        return session.run(self._summary, feed_dict=feed_dict)

    def plot(self):
        plt.Figure()

    def _fill_from_figure(self):
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        self._image = image


class PlotSummaryLog(PlotSummary):
    def plot(self, x, real, fake):
        super().plot()
        ax = plt.gca()
        ax.set_xscale("log")
        ax.set_yscale("log")
        linestyle = {
            "linewidth": 1,
            "markeredgewidth": 0,
            "markersize": 3,
            "marker": "o",
            "linestyle": "-"
        }
        ax.plot(x, real, label="Fake", color='r', **linestyle)
        ax.plot(x, fake, label="Real", color='b', **linestyle)

        # ax.set_ylim(bottom=0.1)
        ax.title.set_text(self._name + "\n")
        ax.title.set_fontsize(11)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend()
        # self._fill_from_figure()