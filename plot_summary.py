import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import tensorflow as tf


# Inspired by Andre


class PlotSummary(object):
    def __init__(self, name):
        self._name = name
        self._placeholder = tf.placeholder(tf.uint8, (None, None, None, None))
        self._summary = tf.summary.image(name, self._placeholder)
        self._image = None

    def produceSummaryToWrite(self, session):
        decoded_image = session.run(self._image)
        feed_dict = {self._placeholder: decoded_image}
        return session.run(self._summary, feed_dict=feed_dict)

    def plot_psd(self, x, real, fake):
        plt.Figure()
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
        ax.plot(x, real, label="Fake $\mathcal{F}(X))^2$", color='r', **linestyle)
        ax.plot(x, fake, label="Real $\mathcal{F}(X))^2$", color='b', **linestyle)

        # ax.set_ylim(bottom=0.1)
        ax.title.set_text("Power Spectrum\n")
        ax.title.set_fontsize(11)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        self._image = image