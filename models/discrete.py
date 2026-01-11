import numpy as np
import matplotlib.pyplot as plt

from data.load_data_numpy import load_wta_numpy

match_times, match_player_indices, match_results, players_id_to_name_dict, players_name_to_id_dict = load_wta_numpy()
n_players = len(players_id_to_name_dict)

grad_step_size = 1e-3 
min_prob: float = 1e-10

class Filter():
    def __init__(self,
                 tau, 
                 sigma0,
                 S = 500,
                 s=1):
        self.tau = tau
        self.sigma0 = sigma0
        self.S = S
        self.s = s

        self.x = [[] for pid in range(n_players)] #empty list

        self.t = [[0.] for i in range(n_players)]

        self.smoothed = [[] for pid in range(n_players)]

        skills_index = np.reshape(np.linspace(0, S - 1, S), (S, 1))
        omegas = np.pi * skills_index / (2 * S)
        self.Lambda = np.cos(2 * omegas)
        self.Lambda = np.diag(self.Lambda[:,0]) - np.eye(S)
        self.Psi_inv = np.sqrt(2 / S) * np.cos(np.transpose(omegas) * (2 * (skills_index + 1) - 1))
        self.Psi_inv[:,0] = self.Psi_inv[:, 0] * np.sqrt(1 / 2)
        self.Psi = np.transpose(self.Psi_inv)


    def times_M(self, pi, dt, tau):
        pi1 = np.dot(pi,self.Psi_inv)
        e = np.exp(-tau*dt*self.Lambda)
        pi2 = np.dot(pi1, e)
        pi3 = np.dot(pi2, self.Psi)
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
        return -np.logaddexp(0.0, -z)

    def update_match(self, t, winner_id, loser_id):
        predict_winner = self.propagate(winner_id, t)
        predict_loser = self.propagate(loser_id, t)
        
        f = np.dot(np.transpose(predict_winner), predict_loser)
        Gk = np.array([[np.exp(self.logG(i, j)) for i in range(self.S)] for j in range(self.S)])

        f = np.multiply(f, Gk)  
        f = f/np.sum(f) #Normalize joint filter to get distribution

        self.x[winner_id].append(np.transpose(np.dot(f,np.ones(self.S))))
        self.x[loser_id].append(np.dot(np.ones(self.S), f))

        

    # Loop
    def run(self, match_times, match_player_indices):
        means_hist = np.empty((len(match_times), n_players), dtype=float)

        #Initialization of the filtering for t=0
        x0 = np.zeros(self.S)
        x0[self.S//2] += 1/2
        x0[(self.S+1)//2] += 1/2
        x0 = self.times_M(x0, dt=self.sigma0, tau=1)
        x0[x0 < min_prob] = min_prob
        x0 = x0/np.sum(x0)
        for pid in range(n_players):
            self.x[pid].append(x0)

        for k in range(len(match_times)):
            t = float(match_times[k])
            w_id = int(match_player_indices[k, 0])
            l_id = int(match_player_indices[k, 1])

            self.update_match(t, w_id, l_id)

    
    def posterior_mean(self):
        return np.mean(self.x)

    def smooth_step(self, pid, k):
        dt = self.t[pid][k] - self.t[pid][k-1]
        filter_k = self.x[pid][k]
        smooth = self.smoothed[pid][-1]

        predict = self.times_M(filter_k, dt, self.tau)
        
        smooth = np.divide(smooth,predict)
        smooth = self.times_M(smooth, dt, self.tau)
        smooth = np.multiply(filter_k, smooth)

        #get a proper distribution
        smooth[smooth < min_prob] = min_prob
        smooth = smooth/np.sum(smooth)

        self.smoothed[pid].append(smooth)

    def smoothing(self, match_times, match_players_indices):
        for pid in range(n_players):
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
        return llh

    
    
def new_theta(tau, sigma0, match_times, match_player_indices):
    filter = Filter(tau= tau, sigma0 = sigma0)
    filter.run(match_times, match_player_indices)
    llh = filter.smoothing(match_times=match_times, match_players_indices=match_player_indices)
    
    new_sigma0 = 0
    for pid in range(n_players):
        mean_x = 0
        for x in range(filter.S):
            mean_x += x*filter.smoothed[pid][0][x]
        mean_x = mean_x/filter.S
        new_sigma0 += mean_x**2
    new_sigma0 = new_sigma0/n_players #empirical initial variance

    N = [[] for pid in range(n_players)]
    D = [[] for pid in range(n_players)]

    for pid in range(n_players): #compute necessary quantities for gradient G2 computation
        K = len(filter.x[pid])
        for k in range(1, K):
            F_ik = np.dot(filter.x[pid][k-1], filter.Psi_inv)
            S_ik = np.dot(filter.smoothed[pid][k], filter.Psi_inv)

            dt = filter.t[pid][k] - filter.t[pid][k-1]
            e = np.exp(filter.tau * dt * filter.Lambda )
            Lambda_k = dt*np.dot(filter.Lambda ,e)

            N_ik = np.dot(F_ik, Lambda_k)
            N_ik = np.dot(N_ik, np.transpose(S_ik))
            D_ik = np.dot(F_ik, filter.Lambda)
            D_ik = np.dot(D_ik, np.transpose(S_ik))

            N[pid].append(N_ik)
            D[pid].append(D_ik)

    dQ2 = sum([np.sum(np.divide(N[pid],D[pid])) for pid in range(n_players)])
    new_tau = np.exp(np.log(tau) + grad_step_size*tau*dQ2)

    return (new_sigma0, new_tau, llh)

def EM(tau, sigma0, match_times, match_player_indices, steps=1000):
    all_tau = [tau]
    all_sigma0 = [sigma0]
    all_llh = []
    last_tau = tau
    last_sigma0 = sigma0
    for i in range(steps):
        last_tau, last_sigma0, llh = new_theta(last_tau, last_sigma0, match_times, match_player_indices)
        all_tau.append(last_tau)
        all_sigma0.append(last_sigma0)
        all_llh.append(llh)