from keras.legacy import interfaces
import keras.backend as K
from keras.optimizers import Optimizer

class LR_SGD(Optimizer):
    """Stochastic gradient descent optimizer.

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
        multipliers: a dict of multipliers, key is layer, value is
        multiplier on learning rate for each layer
        **kwargs: Keyword arguments. Allowed to be one of "clipnorm"
        or "clipvalue". "clipnorm" (float) clips gradients by norm; 
        "clipvalue" (float) clips gradients by value.
    
    update rule for parameter w with gradient g when motemum is 0:
    w = w - learning rate * g

    update rule when momentum is larger than 0:
    velocity = momentum * velocity - learning_rate * g
    w = w + velocity

    when nesterov = True, the rule becomes:
    velocity = momentum * velocity - learning_rate * g
    w = w + momentum * velocity - learning_rate * g
    """

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False,multipliers=None,**kwargs):
        super(LR_SGD, self).__init__(**kwargs) 
        #.name_scope pushes a name scope so all operations added within it has a prefix
        #for example, all iteration variables will have iterations as a prefix to the name
        with K.name_scope(self.__class__.__name__): 
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov
        self.lr_multipliers = multipliers

    #I'm sorry but idk what the interface is
    #I assume this is to get information from the model but idk
    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        #get the gradients of the model 
        grads = self.get_gradients(loss, params)
        #self.update_add updates the value of self.iterations by adding 1 to it
        #self.updates is a list? that holds iterations + 1?
        self.updates = [K.update_add(self.iterations, 1)]

        #if decay is positive, update learning rate with the equation 
        #learning rate =  learning rate * (1/ (1 + decay * iterations))
        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        # momentum

        #shapes is a list of shapes of the params: that has param items; 
        #   each item is the shape of param p
        shapes = [K.int_shape(p) for p in params]
        #moments is a list of zeros with param items
        moments = [K.zeros(shape) for shape in shapes]
        #weights is a list of param items, each item is a iteration?
        self.weights = [self.iterations] + moments
        #iterate through each param, gradient, moment  (zip makes an iterator that aggregates
        #  elements from each of the iterables)
        for p, g, m in zip(params, grads, moments):
            #create a list of multiplier keys if the key is params.name
            # I DON'T KNOW WHAT .name IS
            matched_layer = [x for x in self.lr_multipliers.keys() if x in p.name]
            #If there is a matched_layer is not empty, then change learning rate to be the formula 
            # new learning rate = learning rate * multiplier[first existing multiplier key]
            if matched_layer:
                new_lr = lr * self.lr_multipliers[matched_layer[0]]
            else:
                new_lr = lr

            #calculate new velocity for SGD depending on momentum, the current moment,
            #  new leaening rate, and gradient
            v = self.momentum * m - new_lr * g  # velocity
            #inncrement moment with the new velocity and append it to the list of updates
            self.updates.append(K.update(m, v))

            #if there is nestervov, follow Nesterov's accelerated gradient descent, otherwise, 
            # calculate params with momentum based gradient descent 
            if self.nesterov:
                new_p = p + self.momentum * v - new_lr * g
            else:
                new_p = p + v

            #getattr() returns value of named attribute of object. 
            #if there is something under the name 'constraint', then change value of new_p
            # IDK WHAT .constraint() IS
            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            #update the value of p by iterating new_p
            #then append p to updates
            self.updates.append(K.update(p, new_p))

        #return updates, which should now inclue iterations? and updated moments, 
        # velocity, params for each p, g, m tuple
        return self.updates

    #create a dictionary mapping from old configuraed items to newly configerated items
    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(LR_SGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items())) 