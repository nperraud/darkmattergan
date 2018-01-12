# Simple Command Line Argument parsing relying on the TensorFlow flags module.
#
# Network parameters should not be defined here, instead define them as in the example in the Parameters
# directory and reference the definition in a command line path which you can define here.

import tensorflow.app.flags as flags

flags.DEFINE_string("discriminator_parameter_path", None, "Path to parameters for a discriminator")
flags.DEFINE_string("generator_parameter_path", None, "Path to parameters for a generator")
flags.DEFINE_string("optimizer_parameter_path", None, "Path to parameters for an optimizer")
flags.DEFINE_string("general_parameter_path", None, "Path to general parameters")
