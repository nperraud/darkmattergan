import io
import tensorflow as tf
import numpy as np
import matplotlib
import socket
if 'nid' in socket.gethostname() or 'lo-' in socket.gethostname():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Inspired by Andres


class PlotSummary(object):
    def __init__(self, name, cat, collections=None):
        self._name = name
        self._cat = cat
        self._collections = collections
        self._plot_str = tf.placeholder(tf.string)
        self._build_graph()

    def _build_graph(self):
        '''
        Build the tf graph for creating plot summary
        '''
        image = tf.image.decode_png(self._plot_str, channels=4)
        image = tf.expand_dims(image, 0)

        self._summary = tf.summary.image(
            self._cat + '/' + self._name, image, collections=self._collections)

    def produceSummaryToWrite(self, session, *args, **kwargs):
        self.plot(*args, **kwargs)
        feed_dict = {self._plot_str : self._get_plot_str()}
        return session.run(self._summary, feed_dict=feed_dict)

    def plot(self):
        plt.Figure()
        # ax = plt.gca()
        # N = 10
        # x = np.linspace(0, N - 1, N)
        # y = np.random.rand(N)
        # ax.plot(x, y, label="Random data", color='r')
        # ax.title.set_text(self._name + "\n")
        # ax.title.set_fontsize(11)
        # ax.tick_params(axis='both', which='major', labelsize=10)
        # ax.legend()

    def _get_plot_str(self):
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf.getvalue()


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
        ax.plot(x, real, label="Real", color='r', **linestyle)
        ax.plot(x, fake, label="Fake", color='b', **linestyle)

        # ax.set_ylim(bottom=0.1)
        ax.title.set_text(self._name + "\n")
        ax.title.set_fontsize(11)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend()
        # self._fill_from_figure()


if __name__ == '__main__':
    print('Testing the plot module')
    obj = PlotSummary('Objname', 'ObjCat')
    with tf.Session() as sess:
        test = obj.produceSummaryToWrite(sess)
    print('Test 1 done!')
    N = 10
    x = np.linspace(1, N, N)
    y1 = np.random.rand(N)
    y2 = np.random.rand(N)
    obj = PlotSummaryLog('Objname', 'ObjCat')
    with tf.Session() as sess:
        test = obj.produceSummaryToWrite(sess, x, y1, y2)
    print('Test 2 done!')
