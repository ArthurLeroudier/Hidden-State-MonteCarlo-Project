import jax.numpy as jnp
import matplotlib.pyplot as plt

from data.load_data import load_wta

match_times, match_player_indices, match_results, players_id_to_name_dict, players_name_to_id_dict = load_wta()

grad_step_size = 1e-3 
min_prob: float = 1e-10

class Filter():
    def __init__(self,
                 tau, 
                 sigma0,
                 players_id_to_name_dict,
                 S = 500,
                 s=1):
        self.tau = tau
        self.sigma0 = sigma0
        self.S = S
        self.s = s
        
        self.n_players = len(players_id_to_name_dict)

        self.x = [[] for pid in range(self.n_players)] #empty list

        self.t = [[0.] for i in range(self.n_players)]

        self.smoothed = [[] for pid in range(self.n_players)]

        skills_index = jnp.reshape(jnp.linspace(0, S - 1, S), (S, 1))
        omegas = jnp.pi * skills_index / (2 * S)
        self.Lambda = jnp.cos(2 * omegas)
        self.Lambda = jnp.diag(self.Lambda[:,0]) - jnp.eye(S)
        self.Psi_inv = jnp.sqrt(2 / S) * jnp.cos(jnp.transpose(omegas) * (2 * (skills_index + 1) - 1))
        self.Psi = self.Psi_inv.at[:,0].set(self.Psi_inv[:, 0] * jnp.sqrt(1 / 2))
        self.Psi = jnp.transpose(self.Psi_inv)

        self.Gk = jnp.array([[jnp.exp(self.logG(i, j)) for i in range(self.S)] for j in range(self.S)])


    def times_M(self, pi, dt, tau):
        pi1 = jnp.dot(pi,self.Psi_inv)
        e = jnp.exp(-tau*dt*self.Lambda)
        pi2 = jnp.dot(pi1, e)
        pi3 = jnp.dot(pi2, self.Psi)
        return(pi3)
    

    def propagate(self, pid, t):

        dt = t - self.t[pid][-1]
        Predict_i = self.times_M(self.x[pid][-1], dt, self.tau)
        self.t[pid].append(t)

        return(Predict_i)
    
    def logG(self, xw, xl):
        """
        xw: x of the winner
        wl: x of the loser
        log G(x_w, x_l) = log sigmoid((x_w - x_l) / s)
        """
        z = (xw - xl) / self.s
        return -jnp.logaddexp(0.0, -z)

    def update_match(self, t, winner_id, loser_id):
        predict_winner = self.propagate(winner_id, t)
        predict_loser = self.propagate(loser_id, t)
        
        f = jnp.dot(jnp.transpose(predict_winner), predict_loser)
        

        f = jnp.multiply(f, self.Gk)  
        f = f/jnp.sum(f) #Normalize joint filter to get distribution

        self.x[winner_id].append(jnp.transpose(jnp.dot(f,jnp.ones(self.S))))
        self.x[loser_id].append(jnp.dot(jnp.ones(self.S), f))

        

    # Loop
    def run(self, match_times, match_player_indices):
        print('filtering')

        #Initialization of the filtering for t=0
        x0 = jnp.zeros(self.S)
        x0 = x0.at[self.S//2].set(x0[self.S//2] + 1/2)
        x0 = x0.at[(self.S+1)//2].set(x0[(self.S+1)//2] + 1/2)
        x0 = self.times_M(x0, dt=self.sigma0, tau=1)
        x0 = x0.at[x0 < min_prob].set(min_prob)
        x0 = x0/jnp.sum(x0)
        for pid in range(self.n_players):
            self.x[pid].append(x0)

        for k in range(len(match_times)):
            t = float(match_times[k])
            w_id = int(match_player_indices[k, 0])
            l_id = int(match_player_indices[k, 1])

            self.update_match(t, w_id, l_id)

    
    def posterior_mean(self):
        return jnp.mean(self.x)

    def smooth_step(self, pid, k):
        dt = self.t[pid][k] - self.t[pid][k-1]
        filter_k = self.x[pid][k]
        smooth = self.smoothed[pid][-1]

        predict = self.times_M(filter_k, dt, self.tau)
        
        smooth = jnp.divide(smooth,predict)
        smooth = self.times_M(smooth, dt, self.tau)
        smooth = jnp.multiply(filter_k, smooth)

        #get a proper distribution
        smooth = smooth.at[smooth < min_prob].set(min_prob)
        smooth = smooth/jnp.sum(smooth)

        self.smoothed[pid].append(smooth)

    def smoothing(self, match_times, match_player_indices):
        print('smoothing')

        for pid in range(self.n_players):
            K = len(self.x[pid])
            self.smoothed[pid].append(self.x[pid][-1])
            for k in range(K-1,-1,-1):
                self.smooth_step(pid,k)

        #likelihood computation
        llh = 0
        for k in range(len(match_times)):
            t = float(match_times[k])
            w_id = int(match_player_indices[k, 0])
            l_id = int(match_player_indices[k, 1])

            k_w = self.t[w_id].index(t) #index of skill distribution for winner at time t
            k_l = self.t[l_id].index(t)

            xw = sum([i*self.x[w_id][k_w][i] for i in range(self.S)])/self.S #get expected skill of winner at time t
            xl = sum([i*self.x[l_id][k_l][i] for i in range(self.S)])/self.S 

            llh += self.logG(xw, xl)
        print(llh)
        return llh

    
    
def new_theta(tau, sigma0, match_times, match_player_indices, players_id_to_name_dict):

    filter = Filter(tau= tau, sigma0 = sigma0, players_id_to_name_dict=players_id_to_name_dict)
    filter.run(match_times, match_player_indices)
    llh = filter.smoothing(match_times=match_times, match_player_indices=match_player_indices)
    print('filtered and smoothed')
    
    new_sigma0 = 0
    for pid in range(filter.n_players):
        mean_x = 0
        for x in range(filter.S):
            mean_x += x*filter.smoothed[pid][0][x]
        mean_x = mean_x/filter.S
        new_sigma0 += mean_x**2
    new_sigma0 = new_sigma0/filter.n_players #empirical initial variance
    print('new sigma0')
    print(new_sigma0)

    dQ2 = 0
    for pid in range(filter.n_players): #compute necessary quantities for gradient G2 computation
        K = len(filter.x[pid])
        for k in range(1, K):
            F_ik = jnp.dot(filter.x[pid][k-1], filter.Psi_inv)
            S_ik = jnp.dot(filter.smoothed[pid][k], filter.Psi_inv)

            dt = filter.t[pid][k] - filter.t[pid][k-1]
            e = jnp.exp(-filter.tau * dt * filter.Lambda )
            Lambda_k = dt*jnp.dot(filter.Lambda ,e)

            N_ik = jnp.dot(F_ik, Lambda_k)
            N_ik = jnp.dot(N_ik, jnp.transpose(S_ik))
            D_ik = jnp.dot(F_ik, filter.Lambda)
            D_ik = jnp.dot(D_ik, jnp.transpose(S_ik))

            dQ2 += N_ik/D_ik

    new_tau = tau + grad_step_size*tau*dQ2
    print('new tau')
    print(new_tau)

    return (new_sigma0, new_tau, llh)

def EM(tau, sigma0, match_times, match_player_indices, players_id_to_name_dict= players_id_to_name_dict, steps=1000):
    all_tau = [tau]
    all_sigma0 = [sigma0]
    all_llh = []
    last_tau = tau
    last_sigma0 = sigma0
    for i in range(steps):
        last_tau, last_sigma0, llh = new_theta(last_tau, last_sigma0, match_times, match_player_indices, players_id_to_name_dict)
        all_tau.append(last_tau)
        all_sigma0.append(last_sigma0)
        print(llh)
        all_llh.append(llh)