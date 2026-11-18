"""Extracted official iMF methods for fixed-input parity (MIT, Yiyang Lu).

Source: imeanflow-main/imf.py SHA256
8203e692a21f6e61e2b18d24579f85b3f04e6608a9fb419972e44718cebeb2f5.
Method bodies unchanged except noise and dropout uniform draws are injected.
Image rank 4 is deliberately retained as the independent reference.
"""
import jax
import jax.numpy as jnp


class OfficialCoreReference:
    def __init__(self, net, fixed, norm_p, norm_eps):
        self.net = net
        self.fixed = fixed
        self.dtype = jnp.float32
        self.num_classes = 1
        self.class_dropout_prob = 0.5
        self.norm_p = norm_p
        self.norm_eps = norm_eps

    def sample_tr(self, bz):
        return self.fixed["t"], self.fixed["r"], self.fixed["fm"]

    def sample_cfg_interval(self, bz, fm_mask):
        return self.fixed["t_min"], self.fixed["t_max"]

    def sample_cfg_scale(self, bz):
        return self.fixed["omega"]

    def u_fn(self, x, t, h, omega, t_min, t_max, y):
        """
        Compute the predicted u component from the model.
        By default, we use auxiliary v-head to predict v component as well.

        Args:
            x: Noisy image at time t.
            t: Current time step.
            h: Time difference t - r.
            omega: CFG scale.
            t_min, t_max: Guidance interval.
            y: Class labels.
        Returns: (u, v)
            u: Predicted u (average velocity field).
            v: Predicted v (instantaneous velocity field).
        """
        bz = x.shape[0]
        return self.net(
            x,
            t.reshape(bz),
            h.reshape(bz),
            omega.reshape(bz),
            t_min.reshape(bz),
            t_max.reshape(bz),
            y,
        )

    def v_cond_fn(self, x, t, omega, y):
        """
        Compute the predicted v component conditioned on class labels.

        Args:
            x: Noisy image at time t.
            t: Current time step.
            omega: CFG scale.
            y: Class labels.
        
        Returns:
            v: Predicted v component.
        """

        # Set h, t_min, t_max to dummy values for v prediction
        h = jnp.zeros_like(t)
        t_min = jnp.zeros_like(t)
        t_max = jnp.ones_like(t)

        v = self.u_fn(x, t, h, omega, t_min, t_max, y=y)[1]

        return v

    def v_fn(self, x, t, omega, y):
        """
        Compute both conditioned and unconditioned predicted v components.

        Args:
            x: Noisy image at time t.
            t: Current time step.
            omega: CFG scale.
            y: Class labels.

        Returns:
            v_c: Predicted v component conditioned on class labels.
            v_u: Predicted v component without class labels.
        """
        bz = x.shape[0]

        # Create duplicated batch for conditioned and unconditioned predictions
        x = jnp.concatenate([x, x], axis=0)
        y_null = jnp.array([self.num_classes] * bz)
        y = jnp.concatenate([y, y_null], axis=0)
        t = jnp.concatenate([t, t], axis=0)
        w = jnp.concatenate([omega, jnp.ones_like(omega)], axis=0)

        out = self.v_cond_fn(x, t, w, y)
        v_c, v_u = jnp.split(out, 2, axis=0)

        return v_c, v_u

    def cond_drop(self, v_t, v_g, labels):
        """
        Drop class labels with a certain probability for CFG.

        Args:
            v_t: Unguided instantaneous velocity at time t.
            v_g: Guided instantaneous velocity at time t.
            labels: Class labels for the batch.

        Returns:
            labels: Possibly dropped class labels.
            v_g: Modified guided instantaneous velocity at time t. For samples
                 with dropped labels, v_g = v_t.
        """
        bz = v_t.shape[0]

        rand_mask = (
            self.fixed["drop_uniform"]
            < self.class_dropout_prob
        )
        num_drop = jnp.sum(rand_mask).astype(jnp.int32)
        drop_mask = jnp.arange(bz)[:, None, None, None] < num_drop

        labels = jnp.where(
            drop_mask.reshape(bz),
            self.num_classes,
            labels,
        )
        v_g = jnp.where(drop_mask, v_t, v_g)

        return labels, v_g

    def guidance_fn(self, v_t, z_t, t, r, y, fm_mask, w, t_min, t_max):
        """
        Compute the guided velocity v_g using classifier-free guidance.

        Args:
            v_t: Unguided instantaneous velocity at time t.
            z_t: Noisy image at time t.
            t, r: Two time steps.
            y: Class labels.
            fm_mask: Mask for t=r samples, i.e., flow matching samples.
            t_min, t_max: Guidance interval.
            w: CFG scale.

        Returns:
            v_g: Guided instantaneous velocity at time t, as target for training.
            v_c: Conditioned instantaneous velocity at time t, for jvp computation.
        """

        # compute CFG target
        v_c, v_u = self.v_fn(z_t, t, w, y=y)
        v_g_fm = v_t + (1 - 1 / w) * (v_c - v_u)

        w = jnp.where((t >= t_min) & (t <= t_max), w, 1.0)

        v_c = self.v_cond_fn(z_t, t, w, y=y)
        v_g = v_t + (1 - 1 / w) * (v_c - v_u)

        # For flow matching samples, there is no CFG interval
        v_g = jnp.where(fm_mask, v_g_fm, v_g) 

        return v_g, v_c

    #######################################################
    #               Forward Pass and Loss                 #
    #######################################################

    def forward(self, images, labels):
        """
        Forward process of improved MeanFlow and compute loss.

        Args:
            images: A batch of images, shape (B, H, W, C).
            labels: Corresponding class labels, shape (B,).
        
        Returns:
            loss: Scalar loss value.
            dict_losses: Dictionary of individual loss components.
        """
        x = images.astype(self.dtype)
        bz = images.shape[0]

        # Instantaneous velocity computation
        t, r, fm_mask = self.sample_tr(bz)

        e = self.fixed["noise"]
        z_t = (1 - t) * x + t * e
        v_t = e - x

        # Sample CFG scale and interval
        t_min, t_max = self.sample_cfg_interval(bz, fm_mask)
        omega = self.sample_cfg_scale(bz)

        # Compute guided velocity v_g and conditioned velocity v_c
        v_g, v_c = self.guidance_fn(
            v_t, z_t, t, r, labels, fm_mask, omega, t_min, t_max
        )

        # Cond dropout (dropout class labels)
        labels, v_g = self.cond_drop(v_t, v_g, labels)

        # Warped u-function for jvp computation
        def u_fn(z_t, t, r):
            return self.u_fn(z_t, t, t - r, omega, t_min, t_max, y=labels)

        dtdt = jnp.ones_like(t)
        dtdr = jnp.zeros_like(t)

        # Different from original MeanFlow, we use predicted v in the jvp
        u, du_dt, v = jax.jvp(u_fn, (z_t, t, r), (v_c, dtdt, dtdr), has_aux=True)

        # Our compound function V = u + (t - r) * du/dt
        V = u + (t - r) * jax.lax.stop_gradient(du_dt)

        v_g = jax.lax.stop_gradient(v_g)

        def adp_wt_fn(loss):
            adp_wt = (loss + self.norm_eps) ** self.norm_p
            return loss / jax.lax.stop_gradient(adp_wt)

        # improved MeanFlow objective is conceptually v-loss
        loss_u = jnp.sum((V - v_g) ** 2, axis=(1, 2, 3))
        loss_u = adp_wt_fn(loss_u)

        # auxiliary v-head loss
        loss_v = jnp.sum((v - v_g) ** 2, axis=(1, 2, 3))
        loss_v = adp_wt_fn(loss_v)

        loss = loss_u + loss_v
        loss = loss.mean()  # mean over batch

        dict_losses = {
            "loss": loss,
            "loss_u": jnp.mean((V - v_g) ** 2),
            "loss_v": jnp.mean((v - v_g) ** 2),
        }

        return loss, dict_losses

