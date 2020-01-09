from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
import tensorflow as tf
#import tensorflow as tf
import numpy

__name__ = "rprop"
"""
Both of these implementations are fixed versions of code found on
https://stackoverflow.com/questions/43768411/implementing-the-rprop-algorithm-in-keras/45849212#45849212
So credits go to the stackoverflow community and the specific members that authored the questions and answers.

"""
'''
  ### Write a customized optimizer.
  If you intend to create your own optimization algorithm, simply inherit from
  this class and override the following methods:

    - resource_apply_dense (update variable given gradient tensor is dense)
    - resource_apply_sparse (update variable given gradient tensor is sparse)
    - create_slots (if your optimizer algorithm requires additional variables)
    - get_config (serialization of the optimizer, include all hyper parameters)

'''


class RProp2(optimizer_v2.OptimizerV2):
    def __init__(self, init_alpha=1e-3, scale_up=1.2, scale_down=0.5, min_alpha=1e-6, max_alpha=50., **kwargs):
        super(RProp2, self).__init__("rprop",**kwargs)
        self._set_hyper('init_alpha', init_alpha)
        self._set_hyper('scale_up', scale_up)
        self._set_hyper('scale_down', scale_down)
        self._set_hyper('min_alpha', min_alpha)
        self._set_hyper('max_alpha', max_alpha)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'old_grad')
        for var in var_list:
            self.add_slot(var, 'old_weight_delta')
        for var in var_list:
            self.add_slot(var, 'alpha')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        self.local_step = math_ops.cast(tf.identity(self.iterations) + 1, var_dtype)
        init_alpha_t = array_ops.identity(self._get_hyper('init_alpha', var_dtype))
        scale_up_t = array_ops.identity(self._get_hyper('scale_up', var_dtype))
        scale_down_t = array_ops.identity(self._get_hyper('scale_down', var_dtype))
        min_alpha_t = array_ops.identity(self._get_hyper('min_alpha', var_dtype))
        max_alpha_t = array_ops.identity(self._get_hyper('max_alpha', var_dtype))

        apply_state[(var_device, var_dtype)].update(dict(
            init_alpha_t = init_alpha_t,
            scale_up_t = scale_up_t,
            scale_down_t = scale_down_t,
            min_alpha_t = min_alpha_t,
            max_alpha_t = max_alpha_t
        ))

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))
        old_grad = self.get_slot(var, "old_grad")
        old_weight_delta = self.get_slot(var, "old_weight_delta")
        alpha = self.get_slot(var, "alpha")

        gradsign = K.less(grad*old_grad,0)

        # equation 4
        scale_down = coefficients["scale_down_t"]
        min_alpha = coefficients["min_alpha_t"]
        alpha_t = state_ops.\
            assign(alpha,
                   K.switch(gradsign,
                            K.minimum(alpha * coefficients["scale_up_t"],
                                      coefficients["max_alpha_t"]),
                            K.switch(gradsign,
                                     K.maximum(alpha * scale_down, min_alpha),
                                     alpha)),
                   use_locking=self._use_locking)

        # equation 5

        new_tmp_delta_t = K.switch(K.greater(grad,0),
                                   -alpha_t,
                                   K.switch(K.less(grad,0),
                                            alpha_t,
                                            K.zeros_like(alpha_t)))

        # equation 7
        old_weight_delta_t = state_ops.assign(old_weight_delta, K.switch(gradsign, -old_weight_delta, alpha_t))

        # equation 6
        var_new = math_ops.add(var, old_weight_delta_t)
        var_update = state_ops.assign(var, var_new)
        old_grad_t = state_ops.assign(old_grad, grad)
        return control_flow_ops.group(*[var_update, old_grad_t, old_weight_delta_t, alpha_t])
        

    def _resource_apply_sparse(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))
        old_grad = self.get_slot(var, "old_grad")
        old_weight_delta = self.get_slot(var, "old_weight_delta")
        alpha = self.get_slot(var, "alpha")

        gradsign = K.less(grad*old_grad,0)

        # equation 4
        scale_down = coefficients["scale_down"]
        min_alpha = coefficients["min_alpha"]
        alpha_t = state_ops.\
            assign(alpha,
                   K.switch(gradsign,
                            K.minimum(alpha * coefficients["scale_up"],
                                      coefficients["max_alpha"]),
                            K.switch(gradsign,
                                     K.maximum(alpha * scale_down, min_alpha),
                                     alpha)),
                   use_locking=self._use_locking)

        # equation 5

        new_tmp_delta_t = K.switch(K.greater(grad,0),
                                   -alpha_t,
                                   K.switch(K.less(grad,0),
                                            alpha_t,
                                            K.zeros_like(alpha_t)))

        # equation 7
        old_weight_delta_t = state_ops.assign(old_weight_delta, K.switch(gradsign, -old_weight_delta, new_tmp_delta_t))

        # equation 6
        var_new = math_ops.add(var, old_weight_delta_t)
        var_update = state_ops.assign(var, var_new)
        old_grad_t = state_ops.assign(old_grad, grad)
        return control_flow_ops.group(*[var_update, old_grad_t, old_weight_delta_t, alpha_t])

    def get_config(self):
        config = {
            'init_alpha': float(K.get_value(self.init_alpha)),
            'scale_up': float(K.get_value(self.scale_up)),
            'scale_down': float(K.get_value(self.scale_down)),
            'min_alpha': float(K.get_value(self.min_alpha)),
            'max_alpha': float(K.get_value(self.max_alpha)),
        }
        base_config = super(RProp, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RProp(Optimizer):
    def __init__(self, init_alpha=1e-3, scale_up=1.2, scale_down=0.5, min_alpha=1e-6, max_alpha=50., **kwargs):
        super(RProp, self).__init__("rprop",**kwargs)
        self.init_alpha = K.variable(init_alpha, name='init_alpha')
        self.scale_up = K.variable(scale_up, name='scale_up')
        self.scale_down = K.variable(scale_down, name='scale_down')
        self.min_alpha = K.variable(min_alpha, name='min_alpha')
        self.max_alpha = K.variable(max_alpha, name='max_alpha')

    def get_updates(self, params, loss):
        grads = self.get_gradients(loss, params)
        shapes = [K.shape(p) for p in params]
        alphas = [K.variable(K.ones(shape) * self.init_alpha) for shape in shapes]
        old_grads = [K.zeros(shape) for shape in shapes]
        prev_weight_deltas = [K.zeros(shape) for shape in shapes]
        self.weights = alphas + old_grads
        self.updates = []

        for param, grad, old_grad, prev_weight_delta, alpha in zip(params, grads,
                                                                   old_grads, prev_weight_deltas,
                                                                   alphas):
            # equation 4
            new_alpha = K.switch(
                K.greater(grad * old_grad, 0),
                K.minimum(alpha * self.scale_up, self.max_alpha),
                K.switch(K.less(grad * old_grad, 0), K.maximum(alpha * self.scale_down, self.min_alpha), alpha)
            )

            # equation 5
            new_delta = K.switch(K.greater(grad, 0),
                                 -new_alpha,
                                 K.switch(K.less(grad, 0),
                                          new_alpha,
                                          K.zeros_like(new_alpha)))

            # equation 7
            weight_delta = K.switch(K.less(grad*old_grad, 0), -prev_weight_delta, new_delta)

            # equation 6
            new_param = param + weight_delta

            # reset gradient_{t-1} to 0 if gradient sign changed (so that we do
            # not "double punish", see paragraph after equation 7)
            grad = K.switch(K.less(grad*old_grad, 0), K.zeros_like(grad), grad)


            # Apply constraints
            #if param in constraints:
            #    c = constraints[param]
            #    new_param = c(new_param)

            self.updates.append(K.update(param, new_param))
            self.updates.append(K.update(alpha, new_alpha))
            self.updates.append(K.update(old_grad, grad))
            self.updates.append(K.update(prev_weight_delta, weight_delta))

        return self.updates

    def get_config(self):
        config = {
            'init_alpha': float(K.get_value(self.init_alpha)),
            'scale_up': float(K.get_value(self.scale_up)),
            'scale_down': float(K.get_value(self.scale_down)),
            'min_alpha': float(K.get_value(self.min_alpha)),
            'max_alpha': float(K.get_value(self.max_alpha)),
        }
        base_config = super(RProp, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class iRprop_(optimizer_v2.OptimizerV2):
    def __init__(self, init_alpha=0.01, scale_up=1.2, scale_down=0.5, min_alpha=0.00001, max_alpha=50., **kwargs):
        super(iRprop_, self).__init__(**kwargs)
        self.init_alpha = K.variable(init_alpha, name='init_alpha')
        self.scale_up = K.variable(scale_up, name='scale_up')
        self.scale_down = K.variable(scale_down, name='scale_down')
        self.min_alpha = K.variable(min_alpha, name='min_alpha')
        self.max_alpha = K.variable(max_alpha, name='max_alpha')

    def get_updates(self, params, loss):
        grads = self.get_gradients(loss, params)
        shapes = [K.get_variable_shape(p) for p in params]
        alphas = [K.variable(K.ones(shape) * self.init_alpha) for shape in shapes]
        old_grads = [K.zeros(shape) for shape in shapes]
        self.weights = alphas + old_grads
        self.updates = []

        for p, grad, old_grad, alpha in zip(params, grads, old_grads, alphas):
            grad = K.sign(grad)
            new_alpha = K.switch(
                K.greater(grad * old_grad, 0),
                K.minimum(alpha * self.scale_up, self.max_alpha),
                K.switch(K.less(grad * old_grad, 0),K.maximum(alpha * self.scale_down, self.min_alpha),alpha)
            )

            grad = K.switch(K.less(grad * old_grad, 0),K.zeros_like(grad),grad)
            new_p = p - grad * new_alpha

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            self.updates.append(K.update(p, new_p))
            self.updates.append(K.update(alpha, new_alpha))
            self.updates.append(K.update(old_grad, grad))

        return self.updates

    def get_config(self):
        config = {
        'init_alpha': float(K.get_value(self.init_alpha)),
        'scale_up': float(K.get_value(self.scale_up)),
        'scale_down': float(K.get_value(self.scale_down)),
        'min_alpha': float(K.get_value(self.min_alpha)),
        'max_alpha': float(K.get_value(self.max_alpha)),
        }
        base_config = super(iRprop_, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
