# import tensorflow as tf

# from tensorflow.python.ops import math_ops


# def build_optmizer(params, has_encoder=False):

#     if params['gen_optimizer'] == "adam":
#         optimizer_G = tf.train.AdamOptimizer(learning_rate=params['gen_learning_rate'], beta1=params['beta1'],
#                                              beta2=params['beta2'], epsilon=params['epsilon'])
#         if has_encoder:
#             optimizer_E = tf.train.AdamOptimizer(learning_rate=params['enc_learning_rate'], beta1=params['beta1'],
#                                              beta2=params['beta2'], epsilon=params['epsilon'])
#         else:
#             optimizer_E = None
#     elif params['gen_optimizer'] == "rmsprop":
#         optimizer_G = tf.train.RMSPropOptimizer(learning_rate=params['gen_learning_rate'])
#         if has_encoder:
#             optimizer_E = tf.train.RMSPropOptimizer(learning_rate=params['enc_learning_rate'])
#         else:
#             optimizer_E = None
#     elif params['gen_optimizer'] == "sgd":
#         optimizer_G = tf.train.GradientDescentOptimizer(learning_rate=params['gen_learning_rate'])
#         if has_encoder:
#             optimizer_E = tf.train.GradientDescentOptimizer(learning_rate=params['enc_learning_rate'])
#         else:
#             optimizer_E = None       
#     else:
#         raise Exception(" [!] Choose optimizer between [adam,rmsprop,sgd]")

#     if params['disc_optimizer'] == "adam":

#         optimizer_D = tf.train.AdamOptimizer(learning_rate=params['disc_learning_rate'], beta1=params['beta1'],
#                                              beta2=params['beta2'], epsilon=params['epsilon'])
#     elif params['disc_optimizer'] == "rmsprop":
#         optimizer_D = tf.train.RMSPropOptimizer(learning_rate=params['disc_learning_rate'])
#     elif params['disc_optimizer'] == "sgd":
#         optimizer_D = tf.train.GradientDescentOptimizer(learning_rate=params['disc_learning_rate'])
#     else:
#         raise Exception(" [!] Choose optimizer between [adam,rmsprop]")

#     return optimizer_D, optimizer_G, optimizer_E


# def buid_opt_summaries(optimizer_D, grads_and_vars_d, optimizer_G, grads_and_vars_g, optimizer_E, params):

#     grad_norms_d = [tf.sqrt(tf.nn.l2_loss(g[0]) * 2) for g in grads_and_vars_d]
#     grad_norm_d = [tf.reduce_sum(grads) for grads in grad_norms_d]
#     final_grad_d = tf.reduce_sum(grad_norm_d)
#     tf.summary.scalar("Disc/Gradient_Norm", final_grad_d, collections=["Training"])

#     grad_norms_g = [tf.sqrt(tf.nn.l2_loss(g[0]) * 2) for g in grads_and_vars_g]
#     grad_norm_g = [tf.reduce_sum(grads) for grads in grad_norms_g]
#     final_grad_g = tf.reduce_sum(grad_norm_g)
#     tf.summary.scalar("Gen/Gradient_Norm", final_grad_g, collections=["Training"])

#     if params['gen_optimizer'] == "adam":
#         optim_learning_rate_G = tf.log(get_lr_ADAM(optimizer_G, params['gen_learning_rate']))
#         tf.summary.scalar('Gen/Log_of_ADAM_learning_rate', optim_learning_rate_G, collections=["Training"])

#         if optimizer_E is not None:
#             optim_learning_rate_E = tf.log(get_lr_ADAM(optimizer_E, params['enc_learning_rate']))
#             tf.summary.scalar('Gen/Log_of_ADAM_learning_rate', optim_learning_rate_E, collections=["Training"])

#     if params['disc_optimizer'] == "adam":   
#         optim_learning_rate_D = tf.log(get_lr_ADAM(optimizer_D, params['disc_learning_rate']))
#         tf.summary.scalar('Disc/Log_of_ADAM_learning_rate', optim_learning_rate_D, collections=["Training"])

   


# def get_lr_ADAM(optimizer, learning_rate):
#     beta1_power, beta2_power = optimizer._get_beta_accumulators()
#     optim_learning_rate = (learning_rate * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

#     return optim_learning_rate