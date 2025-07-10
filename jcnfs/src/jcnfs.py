import equinox as eqx
from diffrax import diffeqsolve, Dopri5, Tsit5
import functools
from functools import partial

import jax.random as jrnd

import optax
from optax import ema 
import chex 

from ofdft_nflows.equiv_flows import CNF
from ofdft_nflows.functionals import *
from ofdft_nflows.utils import *
from ofdft_nflows.promolecular_dist import ProMolecularDensity
from ofdft_nflows.eqx_ode import * 

jax.config.update("jax_enable_x64", True)

@chex.dataclass
class F_values:
    energy: chex.ArrayDevice
    kin: chex.ArrayDevice
    vnuc: chex.ArrayDevice
    hart: chex.ArrayDevice
    xc: chex.ArrayDevice

mol_name = 'H2' 
Ne,atoms,z,coords = coordinates(mol_name)
mol = {'coords': coords, 'z': z}
mu = coords
z_one_hot = one_hot_encode(z)
png = jrnd.PRNGKey(0)
_, key = jrnd.split(png)

@functools.partial(jax.vmap, in_axes=(None,0,0), out_axes=0)
def forward(model,x,t):
  return model(x,t)

def grad_loss(model, z_and_logpz):
  solver = Tsit5()
  x, log_px, _score = fwd_ode(model,z_and_logpz,solver)
  bs = int(x.shape[0]/2)
  den_all, x_all,score_all = jnp.exp(log_px), x, _score
  score, scorep = score_all[:bs], score_all[bs:]
  den, denp = den_all[:bs], den_all[bs:]
  x, xp = x_all[:bs], x_all[bs:]

  # evaluate all the functionals locally F[x_i, \rho(x_i)]
  e_t = thomas_fermi(den,score,Ne) + weizsacker(den, score, Ne)
  e_external = nuclear_potential(x, Ne, mol)
  e_h = coulomb_potential(x, xp,Ne)
  e_xc = lda(den,score,Ne) + b88_x_e(den, score, Ne) + correlation_vwn_c_e(den, Ne) 

  e = e_t + e_external + e_h + e_xc

  energy = jnp.mean(e)

  f_values = F_values(energy=energy,
                            kin=jnp.mean(e_t),
                            vnuc=jnp.mean(e_external),
                            hart=jnp.mean(e_h),
                            xc = jnp.mean(e_xc)
                            )

  return energy, f_values

@eqx.filter_jit
def train_step(flow_model, optimizer_state, x_and_logpx):
    # Compute loss and gradients
    loss, grads = eqx.filter_value_and_grad(grad_loss, has_aux=True)(flow_model, x_and_logpx)

    # Update the model parameters
    updates, optimizer_state = optimizer.update(grads, optimizer_state,flow_model)
    flow_model = eqx.apply_updates(flow_model, updates)

    return loss, flow_model, optimizer_state
batch_size = 512
data_dim = 3
flow_model = CNF(data_dim,batch_size,mu,z_one_hot,key)
lr = 3e-4
epochs = 1000
_lr = optax.warmup_cosine_decay_schedule(
                init_value=lr,
                peak_value=lr,
                warmup_steps=150,
                decay_steps=epochs,
                end_value=1E-5,
            )
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(lr, weight_decay=1e-5)
)
optimizer_state = optimizer.init(eqx.filter(flow_model, eqx.is_array))

energies_ema = ema(decay=0.99)
energies_state = energies_ema.init(
        F_values(energy=jnp.array(0.), kin=jnp.array(0.), vnuc=jnp.array(0.), hart=jnp.array(0.), xc=jnp.array(0.)))


base_dist = ProMolecularDensity(z.ravel(), mu)

gen_batches = batch_generator(key, batch_size, base_dist)
# Training loop
key = jax.random.PRNGKey(0)
for itr in range(500):
    key, subkey = jax.random.split(key)
    _,key = jrnd.split(key)
    batch = next(gen_batches)

    # Perform a training step
    loss, flow_model, optimizer_state = train_step(flow_model, optimizer_state, batch)
    loss_epoch, losses = loss
    energies_i_ema, energies_state = energies_ema.update(
            losses, energies_state)
    ei_ema = energies_i_ema.energy

    r_ema = {'epoch': itr,
                 'E': energies_i_ema.energy,
                 'T': energies_i_ema.kin, 'V': energies_i_ema.vnuc, 'H': energies_i_ema.hart

                 }
    print( r_ema)
    # xt = jnp.linspace(-4.5, 4.5, 1000)
    # yz = jnp.zeros((xt.shape[0], 2))
    # zt = lax.concatenate((yz, xt[:, None]), 1)
    # rho_true_1d = m.prob(m, zt)
    # rho_predicted_1d = rho_(flow_model, zt)
    # fig, ax = plt.subplots()
    # ax.plot(xt, rho_true_1d, label='Exact Density')
    # ax.plot(xt, Ne*rho_predicted_1d, label='Predicted Density')
    # ax.plot(xt, Ne*rho_predicted_1d[::-1], label='Predicted Density')
    # ax.set_title('Exact vs. Predicted 1D Density along Z-axis')
    # ax.set_xlabel('Z-axis [Bohr]')
    # ax.set_ylabel('Density')
    # ax.legend()
    # plt.show()
    # plt.close(fig)
