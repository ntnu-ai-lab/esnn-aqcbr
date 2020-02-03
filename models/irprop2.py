from keras.backend import dtype
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras import backend as K
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
import tensorflow as tf

__name__ = "rprop"

from tensorflow_core.python.ops import init_ops

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


class iRProp2notrack(optimizer_v2.OptimizerV2):
    def __init__(self,
                 init_alpha=1e-1,
                 scale_up=1.2,
                 scale_down=0.5,
                 min_alpha=1e-6,
                 max_alpha=50.,
                 **kwargs):
        super(iRProp2notrack, self).__init__("irprop-", **kwargs)
        self.init_alpha = init_alpha
        self._set_hyper('init_alpha', init_alpha)
        self._set_hyper('scale_up', scale_up)
        self._set_hyper('scale_down', scale_down)
        self._set_hyper('min_alpha', min_alpha)
        self._set_hyper('max_alpha', max_alpha)


    def _create_slots(self, var_list):
        for var in var_list:
            dtype = var.dtype.base_dtype
            init = init_ops.constant_initializer(
                self.init_alpha, dtype=dtype)
            self.add_slot(var, 'alpha', init)
        for var in var_list:
            self.add_slot(var, 'old_grad')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(iRProp2notrack, self)._prepare_local(var_device, var_dtype, apply_state)
        self.local_step = math_ops.cast(
            tf.identity(self.iterations) + 1, var_dtype)
        init_alpha_t = array_ops.identity(
            self._get_hyper('init_alpha', var_dtype))
        scale_up_t = array_ops.identity(self._get_hyper('scale_up', var_dtype))
        scale_down_t = array_ops.identity(
            self._get_hyper('scale_down', var_dtype))
        min_alpha_t = array_ops.identity(
            self._get_hyper('min_alpha', var_dtype))
        max_alpha_t = array_ops.identity(
            self._get_hyper('max_alpha', var_dtype))
        alpha_t = array_ops.identity(self._get_hyper("init_alpha", var_dtype))
        apply_state[(var_device, var_dtype)].update(dict(
            init_alpha_t=init_alpha_t,
            scale_up_t=scale_up_t,
            scale_down_t=scale_down_t,
            min_alpha_t=min_alpha_t,
            max_alpha_t=max_alpha_t,
            alpha=alpha_t))

    def _resource_apply_dense(self, grad, var, apply_state=None):
        return self._resource_apply_sparse(grad, var, apply_state)

    def _resource_apply_sparse(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))
        alpha = self.get_slot(var, "alpha")
        old_grad = self.get_slot(var, "old_grad")

        #gra dsign = tf.sign(grad * old_grad)

        # gradneg = K.less(grad * old_grad, 0)

        # equation 4
        scale_down = coefficients["scale_down_t"]
        scale_up = coefficients["scale_up_t"]
        min_alpha = coefficients["min_alpha_t"]
        max_alpha = coefficients["max_alpha_t"]

        alpha_t = state_ops.\
            assign(alpha, K.switch(K.greater(grad * old_grad, 0),
                                   K.minimum(alpha * scale_up,
                                             max_alpha),
                                   K.maximum(alpha * scale_down,
                                             min_alpha)),
                   use_locking=self._use_locking)
        old_grad_t = state_ops.assign(old_grad, K.switch(K.greater(grad * old_grad, 0), grad, tf.zeros_like(grad)))

        tf.print("updating weight with ",alpha_t)

        # equation 6
        # var_new = math_ops.add(var, old_weight_delta_t)
        var_update = state_ops.assign_sub(var, tf.sign(old_grad_t) * alpha_t,
                                          use_locking=self._use_locking)
        #old_grad_t = state_ops.assign(grad, grad)

        return control_flow_ops.group(
            *[var_update, alpha_t, old_grad_t])

    def get_config(self):
        config = {
            'init_alpha': float(K.get_value(self.init_alpha)),
            'scale_up': float(K.get_value(self.scale_up)),
            'scale_down': float(K.get_value(self.scale_down)),
            'min_alpha': float(K.get_value(self.min_alpha)),
            'max_alpha': float(K.get_value(self.max_alpha)),
        }
        base_config = super(iRProp2notrack, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
