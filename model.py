import numpy as np
import tensorflow as tf
import time

class PhysicsInformedNN:
    """ Physics-Informed Neural Network for Navier-Stokes equations """
    def __init__(self, layers, lb, ub):
        self.lb = lb
        self.ub = ub
        self.layers = layers
        
        # Weight initialization
        self.weights, self.biases = self.initialize_NN(layers)
        
        # Define advection parameter (fixed)
        self.lambda_1 = tf.constant([1.0], dtype=tf.float32)
        
        # Define viscosity parameter (variable/trainable) 
        self.lambda_2 = tf.Variable([0.01], dtype=tf.float32)
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # Loss history tracking
        self.loss_history = {'total': [], 'data': [], 'phys': []}

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def reset_optimizer(self):
        """ Reset optimizer for different training stages """
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        print("Optimizer successfully reset.")

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        # Input normalization
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
        
    def net_NS(self, x, y, t):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        
        # Triple Nesting (Nested Gradients)
        with tf.GradientTape(persistent=True) as t3:
            t3.watch([x, y, t])
            with tf.GradientTape(persistent=True) as t2:
                t2.watch([x, y, t])
                with tf.GradientTape(persistent=True) as t1:
                    t1.watch([x, y, t])
                    X = tf.concat([x, y, t], 1)
                    psi_p = self.neural_net(X, self.weights, self.biases)
                    psi = psi_p[:, 0:1] # Stream function
                    p = psi_p[:, 1:2] # Pressure field

                # Compute velocities from stream function
                u = t1.gradient(psi, y)
                v = -t1.gradient(psi, x)
                del t1
                
            # First-order derivatives
            u_t = t2.gradient(u, t); u_x = t2.gradient(u, x); u_y = t2.gradient(u, y)
            v_t = t2.gradient(v, t); v_x = t2.gradient(v, x); v_y = t2.gradient(v, y)
            p_x = t2.gradient(p, x); p_y = t2.gradient(p, y)
            del t2

        # Second-order derivatives
        u_xx = t3.gradient(u_x, x); u_yy = t3.gradient(u_y, y)
        v_xx = t3.gradient(v_x, x); v_yy = t3.gradient(v_y, y)
        del t3
        
        # Navier-Stokes residuals
        f_u = u_t + lambda_1*(u*u_x + v*u_y) + p_x - lambda_2*(u_xx + u_yy) 
        f_v = v_t + lambda_1*(u*v_x + v*v_y) + p_y - lambda_2*(v_xx + v_yy)
        
        return u, v, p, f_u, f_v
    
    def train_step(self, x, y, t, u_meas, v_meas, train_params=True, physics_weight=1.0):
        with tf.GradientTape() as tape:
            u_pred, v_pred, p_pred, f_u_pred, f_v_pred = self.net_NS(x, y, t)
            
            # Data Loss
            loss_data = tf.reduce_mean(tf.square(u_meas - u_pred)) + \
                        tf.reduce_mean(tf.square(v_meas - v_pred))
            
            # Physics Loss
            loss_physics = tf.reduce_mean(tf.square(f_u_pred)) + \
                           tf.reduce_mean(tf.square(f_v_pred))
            
            loss = loss_data + (physics_weight * loss_physics)
        
        trainable_vars = self.weights + self.biases
        
        if train_params:
            trainable_vars += [self.lambda_2] 
            
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        return loss, loss_data, loss_physics

    def train(self, x, y, t, u, v, epochs, train_params=True, physics_weight=1.0):
        # Local compilation to create a fresh graph (solves optimizer issues)
        @tf.function
        def compiled_step(x, y, t, u, v):
            return self.train_step(x, y, t, u, v, train_params, physics_weight)

        start_time = time.time()
        for it in range(epochs):
            loss_t, loss_d, loss_p = compiled_step(x, y, t, u, v)

            self.loss_history['total'].append(loss_t.numpy())
            self.loss_history['data'].append(loss_d.numpy())
            self.loss_history['phys'].append(loss_p.numpy())
            
            if it % 100 == 0:
                elapsed = time.time() - start_time

                l1 = self.lambda_1.numpy()[0]
                l2 = self.lambda_2.numpy()[0]
                
                mode = "PARAMS:ON" if train_params else "PARAMS:OFF"
                phys = "PHYS:ON" if physics_weight > 0 else "PHYS:OFF"
                
                print(f'It: {it} [{mode}|{phys}], Loss: {loss_t:.3e}, l1: {l1:.4f}, l2: {l2:.5f}, Time: {elapsed:.2f}s')
                start_time = time.time()

    pass

class GeostrophicPINN:
    """ PINN for Geostrophic flows including Coriolis effects """
    def __init__(self, layers, lb, ub):
        self.lb = lb
        self.ub = ub
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # Parameters: Advection (fixed), Viscosity (trainable), Coriolis (trainable)
        self.lambda_1 = tf.constant([0.0], dtype=tf.float32)
        self.lambda_2 = tf.Variable([0.01], dtype=tf.float32)
        self.lambda_3 = tf.Variable([0.5], dtype=tf.float32)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        self.loss_history = {'total': [], 'data': [], 'phys': []}

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def reset_optimizer(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        print("Optimizer successfully reset.")

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
        
    def net_NS(self, x, y, t):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        lambda_3 = self.lambda_3 
        
        with tf.GradientTape(persistent=True) as t3:
            t3.watch([x, y, t])
            with tf.GradientTape(persistent=True) as t2:
                t2.watch([x, y, t])
                with tf.GradientTape(persistent=True) as t1:
                    t1.watch([x, y, t])
                    X = tf.concat([x, y, t], 1)
                    psi_p = self.neural_net(X, self.weights, self.biases)
                    psi = psi_p[:, 0:1]
                    p = psi_p[:, 1:2]
                
                u = t1.gradient(psi, y)
                v = -t1.gradient(psi, x)
                del t1

            u_t = t2.gradient(u, t); u_x = t2.gradient(u, x); u_y = t2.gradient(u, y)
            v_t = t2.gradient(v, t); v_x = t2.gradient(v, x); v_y = t2.gradient(v, y)
            p_x = t2.gradient(p, x); p_y = t2.gradient(p, y)
            del t2

        u_xx = t3.gradient(u_x, x); u_yy = t3.gradient(u_y, y)
        v_xx = t3.gradient(v_x, x); v_yy = t3.gradient(v_y, y)
        del t3

        # Momentum equations (Navier-Stokes + Coriolis terms)
        f_u = u_t + lambda_1*(u*u_x + v*u_y) - lambda_3*v + p_x - lambda_2*(u_xx + u_yy) 
        f_v = v_t + lambda_1*(u*v_x + v*v_y) + lambda_3*u + p_y - lambda_2*(v_xx + v_yy)
        
        return u, v, p, f_u, f_v    

    def train_step(self, x, y, t, u_meas, v_meas, p_meas, train_params=True, physics_weight=1.0):
        with tf.GradientTape() as tape:
            u_pred, v_pred, p_pred, f_u_pred, f_v_pred = self.net_NS(x, y, t)

            # Loss Data (Velocity and Pressure)
            loss_data = tf.reduce_mean(tf.square(u_meas - u_pred)) + \
                        tf.reduce_mean(tf.square(v_meas - v_pred)) + \
                        tf.reduce_mean(tf.square(p_meas - p_pred)) 

            # Loss Physics (Navier-Stokes residuals)
            loss_physics = tf.reduce_mean(tf.square(f_u_pred)) + \
                           tf.reduce_mean(tf.square(f_v_pred))
            
            loss = loss_data + (physics_weight * loss_physics)
        
        trainable_vars = self.weights + self.biases
        if train_params:
            trainable_vars += [self.lambda_2, self.lambda_3] 
            
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        return loss, loss_data, loss_physics

    def train(self, x, y, t, u, v, p, epochs, train_params=True, physics_weight=1.0):
        @tf.function
        def compiled_step(x, y, t, u, v, p):
            return self.train_step(x, y, t, u, v, p, train_params, physics_weight)

        start_time = time.time()
        for it in range(epochs):
            loss_t, loss_d, loss_p = compiled_step(x, y, t, u, v, p)
            
            # Store history
            self.loss_history['total'].append(loss_t.numpy())
            self.loss_history['data'].append(loss_d.numpy())
            self.loss_history['phys'].append(loss_p.numpy())
            
            if it % 100 == 0:
                elapsed = time.time() - start_time
                l1 = self.lambda_1.numpy()[0]
                l2 = self.lambda_2.numpy()[0]
                l3 = self.lambda_3.numpy()[0]
                mode = "PARAMS:ON" if train_params else "PARAMS:OFF"
                phys = "PHYS:ON" if physics_weight > 0 else "PHYS:OFF"
                
                print(f'It: {it} [{mode}|{phys}], Loss: {loss_t:.3e}, l1: {l1:.1f}, l2(Visc): {l2:.5f}, l3(Cor): {l3:.4f}, Time: {elapsed:.2f}s')
                start_time = time.time()
