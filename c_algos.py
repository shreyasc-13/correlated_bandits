import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import operator
import matplotlib.pyplot as plt
plt.style.use('data/plots_paper.mplstyle')

class algs:
    def __init__(self, exp='genre', p=0.0, padval=0.0):

        assert(exp in ['genre', 'movie', 'book'])
        self.exp_name = exp
        self.p = p
        self.padval = padval

        #Load data
        self.tables = pd.read_pickle(f'preproc/{self.exp_name}s/{self.exp_name}_tables_train.pkl')
        self.test_data = pd.read_pickle(f'preproc/{self.exp_name}s/test_data_usercount')
        self.true_means_test = pd.read_pickle(f'preproc/{self.exp_name}s/true_means_test')

        self.numArms = len(self.tables.keys())
        self.optArm = np.argmax(self.true_means_test)

    def generate_sample(self, arm):

        d = self.test_data[test_data[f'{self.exp_name}_col'] == arm]
        reward = d['Rating'].sample(n=1, replace = True)

        return reward

    def ThompsonSample(self, empiricalMean, numPulls, beta):
        numArms = self.numArms
        sampleArm = np.zeros(numArms)

        var_ = beta/(numPulls + 1.)
        std_ = np.sqrt(var_)

        mean_ = empiricalMean

        sampleArm = np.random.normal(mean_, std_)

        return sampleArm

    def UCB(self, num_iterations, T):

        numArms = self.numArms
        optArm = self.optArm
        true_means_test = self.true_means_test
        tables = self.tables

        B = [5.] * numArms

        avg_ucb_regret = np.zeros((num_iterations, T))
        for iteration in tqdm(range(num_iterations)):
            UCB_pulls = np.zeros(numArms)
            UCB_index = np.zeros(numArms)
            UCB_empReward = np.zeros(numArms)
            UCB_sumReward = np.zeros(numArms)

            UCB_index[:] = np.inf
            UCB_empReward[:] = np.inf

            ucb_regret = np.zeros(T)
            for t in range(T):
                if t < numArms:
                    UCB_kt = t
                else:
                    UCB_kt = np.argmax(UCB_index)

                UCB_reward = self.generate_sample(UCB_kt)

                UCB_pulls[UCB_kt] = UCB_pulls[UCB_kt] + 1

                UCB_sumReward[UCB_kt] = UCB_sumReward[UCB_kt] + UCB_reward
                UCB_empReward[UCB_kt] = UCB_sumReward[UCB_kt]/float(UCB_pulls[UCB_kt])

                for k in range(numArms):
                    if UCB_pulls[k] > 0:
                        UCB_index[k] = UCB_empReward[k] + B[k]*np.sqrt(2. * np.log(t+1)/ UCB_pulls[k])

                if t == 0:
                    ucb_regret[t] = true_means_test[optArm] - true_means_test[UCB_kt]
                else:
                    ucb_regret[t] = ucb_regret[t-1] + true_means_test[optArm] - true_means_test[UCB_kt]

            avg_ucb_regret[iteration, :] = ucb_regret

        return avg_ucb_regret

    def TS(self, num_iterations, T):

        numArms = self.numArms
        optArm = self.optArm
        true_means_test = self.true_means_test
        tables = self.tables

        beta = 4.

        avg_ts_regret = np.zeros((num_iterations, T))
        for iteration in tqdm(range(num_iterations)):
            numPulls = np.zeros(numArms)
            empReward = np.zeros(numArms)

            ts_regret = np.zeros(T)
            for t in range(T):
                #Initialise by pulling each arm once
                if t < numArms:
                    numPulls[t] += 1
                    assert numPulls[t] == 1

                    reward = generate_sample(t)
                    empReward[t] = reward

                    if t != 0:
                        ts_regret[t] = ts_regret[t-1] + true_means_test[optArm] - true_means_test[t]

                    continue

                thompson = self.ThompsonSample(empReward,numPulls,beta)
                next_arm = np.argmax(thompson)

                #Generate reward, update pulls and empirical reward
                reward = generate_sample( next_arm )
                empReward[next_arm] = (empReward[next_arm]*numPulls[next_arm] + reward)/(numPulls[next_arm] + 1)
                numPulls[next_arm] = numPulls[next_arm] + 1

                #Evaluate regret
                ts_regret[t] = ts_regret[t-1] + true_means_test[optArm] - true_means_test[next_arm]

            avg_ts_regret[iteration, :] = ts_regret

        return avg_ts_regret

    def C_UCB(self, num_iterations, T):

        numArms = self.numArms
        optArm = self.optArm
        true_means_test = self.true_means_test
        tables = self.tables

        B = [5.] * numArms

        avg_cucb_regret = np.zeros((num_iterations, T))
        for iteration in tqdm(range(num_iterations)):
            pulls = np.zeros(numArms)
            empReward = np.zeros(numArms)
            sumReward = np.zeros(numArms)
            Index = dict(zip(range(numArms), [np.inf]*numArms))

            empReward[:] = np.inf

            empPseudoReward = np.zeros((numArms, numArms)) #(i,j) empPseudoReward of arm $i$ wrt arm $j$.
            sumPseudoReward = np.zeros((numArms, numArms))

            empPseudoReward[:,:] = np.inf


            cucb_regret = np.zeros(T)
            for t in range(T):

                #add to set \ell for arms with pulls >t/K
                bool_ell = pulls >= (float(t)/numArms)

                max_mu_hat = np.max(empReward[bool_ell])

                if empReward[bool_ell].shape[0] == 1:
                    secmax_mu_hat = max_mu_hat
                else:
                    temp = empReward[bool_ell]
                    temp[::-1].sort()
                    secmax_mu_hat = temp[1]
                argmax_mu_hat = np.where(empReward == max_mu_hat)[0][0]

                #Set of competitive arms - update through the run
                min_phi = np.min(empPseudoReward[:, bool_ell], axis=1)

                comp_set = set()
                #Adding back the argmax arm
                comp_set.add(argmax_mu_hat)

                for arm in range(numArms):
                    if arm != argmax_mu_hat and min_phi[arm] >= max_mu_hat:
                        comp_set.add(arm)
                    elif arm == argmax_mu_hat and min_phi[arm] >= secmax_mu_hat:
                        comp_set.add(arm)

                if t < numArms:
                    k_t = t %numArms
                elif len(comp_set)==0:
                    #UCB for empty comp set
                    k_t = max(Index.iteritems(), key=operator.itemgetter(1))[0]
                else:
                    comp_Index = {ind: Index[ind] for ind in comp_set}
                    k_t = max(comp_Index.iteritems(), key=operator.itemgetter(1))[0]

                pulls[k_t] = pulls[k_t] + 1

                reward = generate_sample(k_t)

                #Update \mu_{k_t}
                sumReward[k_t] = sumReward[k_t] + reward
                empReward[k_t] = sumReward[k_t]/float(pulls[k_t])

                #Pseudo-reward updates
                pseudoRewards = tables[k_t][reward-1,:] #(zero-indexed)

                sumPseudoReward[:,k_t] = sumPseudoReward[:,k_t] + pseudoRewards
                empPseudoReward[:,k_t] = np.divide(sumPseudoReward[:,k_t], float(pulls[k_t]))

                #Diagonal elements of pseudorewards
                empPseudoReward[np.arange(numArms), np.arange(numArms)] = empReward

                #Update UCB+LCB indices: Using pseduorewards
                for k in range(numArms):
                    if(pulls[k] > 0):
                        #UCB index
                        Index[k] = empReward[k] + B[k]*np.sqrt(2. * np.log(t+1)/pulls[k])

                #Regret calculation
                if t == 0:
                    cucb_regret[t] = true_means_test[optArm] - true_means_test[k_t]
                else:
                    cucb_regret[t] = cucb_regret[t-1] + true_means_test[optArm] - true_means_test[k_t]

            avg_cucb_regret[iteration, :] = cucb_regret

        return avg_cucb_regret

    def C_TS(self, num_iterations, T):

        numArms = self.numArms
        optArm = self.optArm
        true_means_test = self.true_means_test
        tables = self.tables

        B = [5.] * numArms

        avg_tsc_regret = np.zeros((num_iterations, T))

        beta = 4. #since sigma was taken as 2
        for iteration in tqdm(range(num_iterations)):
            TSC_pulls = np.zeros(numArms)

            TSC_empReward = np.zeros(numArms)
            TSC_sumReward = np.zeros(numArms)

            TSC_empReward[:] = np.inf

            TSC_empPseudoReward = np.zeros((numArms, numArms)) #(i,j) empPseudoReward of arm $i$ wrt arm $j$.
            TSC_sumPseudoReward = np.zeros((numArms, numArms))

            TSC_empPseudoReward[:,:] = np.inf


            tsc_regret = np.zeros(T)

            for t in range(T):

                #add to set \ell for arms with pulls >t/K
                bool_ell = TSC_pulls >= (float(t)/numArms)

                max_mu_hat = np.max(TSC_empReward[bool_ell])

                if TSC_empReward[bool_ell].shape[0] == 1:
                    secmax_mu_hat = max_mu_hat
                else:
                    temp = TSC_empReward[bool_ell]
                    temp[::-1].sort()
                    secmax_mu_hat = temp[1]
                argmax_mu_hat = np.where(TSC_empReward == max_mu_hat)[0][0]

                #Set of competitive arms - update through the run
                min_phi = np.min(TSC_empPseudoReward[:, bool_ell], axis=1)

                comp_set = set()
                #Adding the argmax arm
                comp_set.add(argmax_mu_hat)

                for arm in range(numArms):
                    if arm != argmax_mu_hat and min_phi[arm] >= max_mu_hat:
                        comp_set.add(arm)
                    elif arm == argmax_mu_hat and min_phi[arm] >= secmax_mu_hat:
                        comp_set.add(arm)

                if t < numArms:
                    k_t = t #%numArms
                else:
                    #Thompson Sampling
                    thompson = ThompsonSample(TSC_empReward, TSC_pulls, beta)
                    comp_values = {ind: thompson[ind] for ind in comp_set}
                    k_t = max(comp_values.iteritems(), key=operator.itemgetter(1))[0]

                TSC_pulls[k_t] = TSC_pulls[k_t] + 1

                reward = generate_sample(k_t)

                #Update \mu_{k_t}
                TSC_sumReward[k_t] = TSC_sumReward[k_t] + reward
                TSC_empReward[k_t] = TSC_sumReward[k_t]/float(TSC_pulls[k_t])

                #Pseudo-reward updates
                TSC_pseudoRewards = tables[k_t][reward-1,:] #(zero-indexed)

                TSC_sumPseudoReward[:,k_t] = TSC_sumPseudoReward[:,k_t] + TSC_pseudoRewards
                TSC_empPseudoReward[:,k_t] = np.divide(TSC_sumPseudoReward[:,k_t], float(TSC_pulls[k_t]))

                #Regret calculation
                if t == 0:
                    tsc_regret[t] = true_means_test[optArm] - true_means_test[k_t]
                else:
                    tsc_regret[t] = tsc_regret[t-1] + true_means_test[optArm] - true_means_test[k_t]

            avg_tsc_regret[iteration, :] = tsc_regret

        return avg_tsc_regret

    def run(self, num_iterations=20, T=5000):

        avg_ucb_regret = self.UCB(num_iterations, T)
        avg_ts_regret = self.TS(num_iterations, T)
        avg_cucb_regret = self.C_UCB(num_iterations, T)
        avg_cts_regret = self.C_TS(num_iterations, T)

        #mean cumulative regret
        self.plot_av_ucb = np.mean(avg_ucb_regret, axis = 0)
        self.plot_av_ts = np.mean(avg_ts_regret, axis = 0)
        self.plot_av_cucb = np.mean(avg_cucb_regret, axis=0)
        self.plot_av_cts = np.mean(avg_cts_regret, axis=0)

        #std dev over runs
        self.plot_std_ucb = np.sqrt(np.var(avg_ucb_regret, axis = 0))
        self.plot_std_ts = np.sqrt(np.var(avg_ts_regret, axis = 0))
        self.plot_std_cucb = np.sqrt(np.var(avg_cucb_regret, axis=0))
        self.plot_std_cts = np.sqrt(np.var(avg_cts_regret, axis=0))

        self.save_data()


    def edit_data(self):

        if self.exp_name == 'genre':
            #code only masks values as done in the paper
            genre_tables = pd.read_pickle(f'preproc/{self.exp_name}s/genre_tables_train.pkl')
            p = self.p
            for genre in range(18):
                for row in range(genre_tables[genre].shape[0]):
                    row_len = int(genre_tables[genre].shape[1])
                    genre_tables[genre][row][np.random.choice(np.arange(row_len), \
                                                              size= int(p*row_len),\
                                                             replace = False)] = 5.
            #restore reference columns
            for genre in range(18):
                genre_tables[genre][:, genre] = np.arange(1,6)

            self.tables = genre_tables

        elif self.exp_name == 'movie':
            #code only pads entries as done in the paper
            movie_tables = pd.read_pickle(f'preproc/{self.exp_name}s/movie_tables_train.pkl')
            pad_val = self.padval
            for movie in range(50): #top 50 movies picked in preproc
                movie_tables[movie] += pad_val
                movie_tables[movie][movie_tables[movie] > 5] = 5.
                movie_tables[movie][:,movie] = np.arange(1,6)

            self.tables = movie_tables

        elif self.exp_name == 'book':
            book_tables = pd.read_pickle(f'preproc/{self.exp_name}s/book_tables_train.pkl')
            p = self.p
            pad_val = self.padval
            for book in range(25): #top 25 books picked in preproc
                for row in range(book_tables[book].shape[0]):
                    row_len = int(book_tables[book].shape[1])
                    book_tables[book][row][np.random.choice(np.arange(row_len), \
                                                              size= int(p*row_len),\
                                                             replace = False)] = 5.

                book_tables[book] += pad_val
                book_tables[book][book_tables[book] > 5] = 5.

                book_tables[book][:, book] = np.arange(1,6)

            self.tables = book_tables

    def save_data(self):

        algs = ['ucb', 'ts', 'cucb', 'cts']
        for alg in algs:
            np.save(f'plot_arrays/{self.exp_name}s/plot_av_{alg}_p{self.p:.2f}_pad{self.padval:.2f}', getattr(self, f'plot_av_{alg}'))
            np.save(f'plot_arrays/{self.exp_name}s/plot_std_{alg}_p{self.p:.2f}_pad{self.padval:.2f}', getattr(self, f'plot_std_{alg}'))


    def plot(self):

        spacing = 400
        #Means
        plt.plot(range(0,5000)[::spacing], self.plot_av_ucb[::spacing], label='UCB', color='red', marker='+')
        plt.plot(range(0,5000)[::spacing], self.plot_av_ts[::spacing], label='TS', color='yellow', marker='o')
        plt.plot(range(0,5000)[::spacing], self.plot_av_cucb[::spacing], label='C-UCB', color='blue', marker='^')
        plt.plot(range(0,5000)[::spacing], self.plot_av_cts[::spacing], label='C-TS', color='black', marker='x')
        #Confidence bounds
        plt.fill_between(range(0,5000)[::spacing], (self.plot_av_ucb + self.plot_std_ucb)[::spacing], (self.plot_av_ucb - self.plot_std_ucb)[::spacing], alpha=0.3, facecolor='r')
        plt.fill_between(range(0,5000)[::spacing], (self.plot_av_ts + self.plot_std_ts)[::spacing], (self.plot_av_ts - self.plot_std_ts)[::spacing], alpha=0.3, facecolor='y')
        plt.fill_between(range(0,5000)[::spacing], (self.plot_av_cucb + self.plot_std_cucb)[::spacing], (self.plot_av_cucb - self.plot_std_cucb)[::spacing], alpha=0.3, facecolor='b')
        plt.fill_between(range(0,5000)[::spacing], (self.plot_av_cts + self.plot_std_cts)[::spacing], (self.plot_av_cts - self.plot_std_cts)[::spacing], alpha=0.3, facecolor='k')

        plt.legend()
        plt.grid(True, axis='y')
        plt.xlabel('Number of Rounds')
        plt.ylabel('Cumulative Regret')

        plt.savefig(f'data/plots/{self.exp_name}_p{self.p:.2f}_pad{self.padval:.2f}.pdf')

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', dest='exp', type=str, default='genre', help="Experiment to run (genre, movie, book)")
    parser.add_argument('--num_iterations', dest='num_iterations', type=int, default=20, help="Number of iterations of each run")
    parser.add_argument('--T', dest='T', type=int, default=5000, help="Number of rounds")
    parser.add_argument('--p', dest='p', type=float, default=0.0, help="Fraction of table entries to mask")
    parser.add_argument('--padval', dest='padval', type=float, default=0.0, help="Padding value for table entries")

    return parser.parse_args()

def main(args):
    args = parse_arguments()

    bandit_obj = algs(args.exp, p=args.p, padval=args.padval)
    bandit_obj.edit_data()
    bandit_obj.run(args.num_iterations, args.T)
    bandit_obj.plot()

if __name__ == '__main__':
    main(sys.argv)
