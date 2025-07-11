import argparse
import json
import os
from typing import Optional
import flax 
import jax
import jax.numpy as jnp
import equinox as eqx
from diffrax import diffeqsolve, Dopri5, Tsit5, AbstractSolver
import optax
from optax import ema 
import chex
import pandas as pd
import time 
import orbax.checkpoint as ocp
from flax.training import checkpoints
import matplotlib.pyplot as plt
# Local imports
from ofdft_nflows.equiv_flows import CNF
from ofdft_nflows.functionals import _kinetic, _nuclear, _hartree, _exchange_correlation 
from ofdft_nflows.utils import *
from ofdft_nflows.promolecular_dist import ProMolecularDensity
from ofdft_nflows.eqx_ode import *
from ofdft_nflows.dft_distrax import DFTDistribution 

jax.config.update("jax_enable_x64", True)

@chex.dataclass
class F_values:
    energy: chex.ArrayDevice
    kin: chex.ArrayDevice
    vnuc: chex.ArrayDevice
    hart: chex.ArrayDevice
    xc: chex.ArrayDevice

def rho_rev(model, x):
    data_dim = 3 
    zt = jnp.concatenate((x, jnp.zeros((x.shape[0], 1)), jnp.zeros((x.shape[0], data_dim))), axis=1)
    z0, logp_jac = rev_ode(model, zt,solver)
    logp_x = base_dist.log_prob(z0)[:, None] - logp_jac
    return jnp.exp(logp_x) 

def training(mol_name: str,
            tw_kin: str = 'tf-w',
            n_pot: str = 'np',
            h_pot: str = 'hp',
            x_pot: str = 'lda',
            c_pot: str = 'vwn_c_e', 
            batch_size: int = 256,
            epochs: int = 100,
            lr: float = 1E-5,
            bool_load_params: bool = False,
            scheduler_type: str = 'ones',
            solver_type: SolverType = 'tsit5',
            prior_dist: Optional[ProMolecularDensity] = None):
    """
    Main training function.
    
    Args:
        mol_name: Name of molecule (e.g., 'H2')
        tw_kin: Kinetic energy functional type ('TF', 'TF-W', etc.)
        v_pot: Nuclear potential type ('NP', etc.)
        h_pot: Hartree potential type ('HP', etc.)
        x_pot: Exchange potential type ('LDA', etc.)
        c_pot: Correlation potential type ('VWN_C_E', etc.)
        batch_size: Batch size for training
        epochs: Number of training epochs
        lr: Learning rate
        bool_load_params: Whether to load pretrained parameters
        scheduler_type: Learning rate scheduler type
        solver_type: ODE solver type ('Tsit5', 'Dopri5')
        prior_dist: Prior distribution for initialization
    """
    # Set the checkpoints 
    options = ocp.CheckpointManagerOptions(  
    save_interval_steps=10,  
    step_prefix='step'  
    )
    ckpt_mgr = ocp.CheckpointManager(
    CKPT_DIR,
    ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
    options=options
    )

    # Get molecular coordinates and properties
    Ne, atoms, z, coords = coordinates(mol_name)
    mol = {'coords': coords, 'z': z}
    mu = coords
    z_one_hot = one_hot_encode(z)
    m = DFTDistribution(atoms, coords)
    normalization_array = (m.coords, m.weights)
    # Initialize random key
    png = jrnd.PRNGKey(0)
    _, key = jrnd.split(png)

    # Initialize prior distribution if not provided
    if prior_dist is None:
        prior_dist = ProMolecularDensity(z.ravel(), mu)

    # Initialize model
    data_dim = 3
    flow_model = CNF(data_dim, batch_size, mu, z_one_hot, key)

    # Set up solver
    solver = get_solver(solver_type) 

    # Set up optimizer and scheduler
    _lr = get_scheduler(
    epochs=epochs,
    sched_type=scheduler_type,
    lr=lr
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(_lr, weight_decay=1e-5)
    )
    optimizer_state = optimizer.init(eqx.filter(flow_model, eqx.is_array))

    # Initialize EMA for tracking energies
    energies_ema = ema(decay=0.99)
    energies_state = energies_ema.init(
        F_values(energy=jnp.array(0.), kin=jnp.array(0.), 
                vnuc=jnp.array(0.), hart=jnp.array(0.), xc=jnp.array(0.)))

    # Set up batch generator
    gen_batches = batch_generator(key, batch_size, prior_dist)

    t_functional = _kinetic(tw_kin)
    n_functional = _nuclear(n_pot)
    h_functional = _hartree(h_pot)
    x_functional = _exchange_correlation(x_pot)
    c_functional = _exchange_correlation(c_pot)

    @jax.jit
    def rho_rev(model, x):
        data_dim = 3 
        zt = jnp.concatenate((x, jnp.zeros((x.shape[0], 1)), jnp.zeros((x.shape[0], data_dim))), axis=1)
        z0, logp_jac = rev_ode(model, zt,solver)
        logp_x = prior_dist.log_prob(z0) - logp_jac
        return jnp.exp(logp_x)

    def grad_loss(model, z_and_logpz):
        x, log_px, _score = fwd_ode(model,z_and_logpz,solver)
        bs = int(x.shape[0]/2)
        den_all, x_all,score_all = jnp.exp(log_px), x, _score
        score, scorep = score_all[:bs], score_all[bs:]
        den, denp = den_all[:bs], den_all[bs:]
        x, xp = x_all[:bs], x_all[bs:]

        # evaluate all the functionals locally F[x_i, \rho(x_i)]
        t_e = t_functional(den,score,Ne) 
        n_e = n_functional(x, Ne, mol)
        h_e = h_functional(x, xp,Ne)
        xc_e = x_functional(den,score,Ne) + c_functional(den,Ne) 
        
        e = t_e + n_e + h_e + xc_e 

        energy = jnp.mean(e)
        
        f_values = F_values(energy=energy,
                            kin=jnp.mean(t_e),
                            vnuc=jnp.mean(n_e),
                            hart=jnp.mean(h_e),
                            xc = jnp.mean(xc_e)
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

    df = pd.DataFrame()
    df_ema = pd.DataFrame() 
    # Training loop
    for itr in range(epochs):
        batch = next(gen_batches)
        start_time = time.time() 
        # Perform a training step
        loss, flow_model, optimizer_state = train_step(flow_model, optimizer_state, batch)
        end_time = time.time()
        elapsed_time_seconds = end_time - start_time

        loss_epoch, losses = loss
        norm_val = compute_integral(
            flow_model, normalization_array, rho_rev, Ne)

        energies_i_ema, energies_state = energies_ema.update(
            losses, energies_state)
        ei_ema = energies_i_ema.energy
        
        r_ = {'epoch': itr,
              'E': loss_epoch,'T': losses.kin,
              'V': losses.vnuc, 'H': losses.hart,
              'XC': losses.xc,'t': elapsed_time_seconds 
              }
        r_ema = {'epoch': itr,
                 'E': energies_i_ema.energy,'T': energies_i_ema.kin,
                 'V': energies_i_ema.vnuc,'H': energies_i_ema.hart,
                 'XC':energies_i_ema.xc,'t': elapsed_time_seconds,
                 'I': norm_val
                 }
        print(r_ema)
        # Save models
        save_args = flax.training.orbax_utils.save_args_from_target(flow_model)
        ckpt_mgr.save(
            step=itr,  
            items={'model': flow_model},
            save_kwargs={'save_args': {'model': save_args}}  
        )
        #PLOTTING
        if itr % 20 == 0 or itr <= 25:
            # 1D Figure
            fig, ax1 = plt.subplots(1, 1)
            xt = jnp.linspace(-4.5, 4.5, 1000)
            yz = jnp.zeros((xt.shape[0], 2))
            zt = lax.concatenate((yz, xt[:, None]), 1)
            rho_pred = rho_rev(flow_model, zt)

            if itr == 0:
                rho_exact = m.prob(m, zt)
                norm_dft = jnp.vdot(m.weights, m.prob(m, m.coords))

            plt.clf()
            fig, ax = plt.subplots()
            ax.text(0.075, 0.92,
                    f'({itr}):  E = {ei_ema:.3f}', transform=ax1.transAxes, va='top', fontsize=10)
            plt.plot(xt, rho_exact,
                     color='k', ls=":", label=r"$\hat{\rho}_{DFT}(x)$" % norm_dft)
            plt.plot(xt, Ne*rho_pred,
                     color='tab:blue', label=r'$N_{e}\;\rho_{NF}(x)$')
            plt.xlabel('X [Bhor]')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{FIG_DIR}/epoch_rho_z_{itr}.png')
            plt.close(fig)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mol_name", type=str, default='H2',
                      help="molecule name")
    parser.add_argument("--epochs", type=int, default=50, 
                      help="training epochs")
    parser.add_argument("--bs", type=int, default=64,
                      help="batch size")
    parser.add_argument("--params", type=bool, default=False,
                      help="load pre-trained model")
    parser.add_argument("--lr", type=float, default=3E-4,
                      help="learning rate")
    parser.add_argument("--kin", type=str, default='tf-w',
                      help="Kinetic energy functional")
    parser.add_argument("--nuc", type=str, default='nuclear_potential',
                      help="Nuclear Potential energy functional")
    parser.add_argument("--hart", type=str, default='coulomb',
                      help="Hartree energy functional")
    parser.add_argument("--x", type=str, default='lda_w_b88_x_e',
                      help="Exchange energy functional")
    parser.add_argument("--c", type=str, default='pw92_c',
                      help="Correlation energy functional")
    parser.add_argument("--sched", type=str, default='mix',
                      help="Hartree integral scheduler")
    parser.add_argument("--solver", type=str, default='tsit5',
                      help="ODE solver")
    args = parser.parse_args()

    global RESULTS_DIR
    global CKPT_DIR
    global FIG_DIR
    
    RESULTS_DIR = f"Results/{args.mol_name}_{args.c.upper()}_{args.solver.upper()}"
    if args.sched.lower() not in ['c', 'const']:
        RESULTS_DIR += f"_sched_{args.sched.upper()}"
    CKPT_DIR = f"{RESULTS_DIR}/Checkpoints"
    FIG_DIR = f"{RESULTS_DIR}/Figures"

    RESULTS_DIR = os.path.abspath(RESULTS_DIR)
    CKPT_DIR = os.path.abspath(CKPT_DIR) 
    cwd = os.getcwd()
    rwd = os.path.join(cwd, RESULTS_DIR)
    if not os.path.exists(rwd):
        os.makedirs(rwd)
    fwd = os.path.join(cwd, FIG_DIR)
    if not os.path.exists(fwd):
        os.makedirs(fwd)

    # Save job parameters
    job_params = {
        'mol_name': args.mol_name,
        'epochs': args.epochs,
        'batch_size': args.bs,
        'lr': args.lr,
        'kin': args.kin,
        'v_nuc': args.nuc,
        'h_pot': args.hart,
        'x_pot': args.x,
        'c_pot': args.c,
        'sched': args.sched,
        'solver': args.solver
    }

    with open(f"{RESULTS_DIR}/job_params.json", "w") as outfile:
        json.dump(job_params, outfile, indent=4)
    
    # Run training
    training(
        mol_name=args.mol_name,
        tw_kin=args.kin,
        n_pot=args.nuc,
        h_pot=args.hart,
        x_pot=args.x,
        c_pot=args.c,
        batch_size=args.bs,
        epochs=args.epochs,
        lr=args.lr,
        bool_load_params=args.params,
        scheduler_type=args.sched,
        solver_type=args.solver
    )

if __name__ == "__main__":
    main()
