import tensorflow as tf
import numpy as np

class MOGP(object):
    # x = n X n_dim
    # y = n X n_output
    # wwt = n_output X n_output
    # kxx_inv = (n * n_output) X (n * n_output)
    def __init__(self, n_dim, n_output, n_factor, x, y):
        self.n_dim, self.n_output, self.n_factor = n_dim, n_output, n_factor
        self.param = np.ones(n_dim + 3 + n_output * (n_factor + 1))
        self.x = x
        self.y = y.reshape((x.shape[0] * n_output, 1))
        self.wwt = np.eye(n_output)
        self.kXX = np.eye(x.shape[0])
        self.kXX_B = np.eye(x.shape[0] * n_output)
        self.kXX_B_inv = np.eye(x.shape[0] * n_output)
        self.update_wwt()
        self.update_kxx_inv()

    def mean(self):
        return self.param[0]

    def noise(self):
        return self.param[1]

    def signal(self):
        return self.param[2]

    def ls(self):
        return self.param[3: 3 + self.n_dim]

    def w(self):
        offset = 3 + self.n_dim
        w_vec = self.param[offset: offset + self.n_output * self.n_factor]
        return w_vec.reshape((self.n_output, self.n_factor))

    def update_wwt(self):
        self.wwt = self.w() * self.w().transpose() + self.kappa()

    def kappa(self):
        offset = 3 + self.n_dim + self.n_output * self.n_factor
        k_vec = self.param[offset: offset + self.n_output]
        return np.dot(k_vec, np.eye(self.n_output))

    def k(self, x1, x2):
        trnorms1 = np.mat([(v * v.T)[0, 0] for v in x1]).T
        trnorms2 = np.mat([(v * v.T)[0, 0] for v in x2]).T
        k1 = trnorms1 * np.mat(np.ones((x2.shape[0], 1), dtype=np.float64)).T
        k2 = np.mat(np.ones((x1.shape[0], 1), dtype=np.float64)) * trnorms2.T
        k = k1 + k2
        k -= 2 * np.mat(x1 * x2.T)
        k *= - 1. / (2 * np.power(self.ls(), 2))
        return (self.signal() ** 2) * np.exp(k)

    def dk_signal(self, krbf):
        return krbf * (2.0 / self.signal())

    def dk_ls(self, x1, x2, krbf, i):
        ai = np.power((x1[:, i] - x2[:, i].transpose()), 2) / (self.ls()[i] ** 3)
        return krbf * ai

    def update_kxx_inv(self):
        self.kXX = self.k(self.x, self.x)
        self.kXX_B = np.kron(self.kXX, self.wwt)
        self.kXX_B_inv = np.linalg.inv(self.kXX_B + (self.noise() ** 2) * np.eye(self.kXX_B.shape[0]))

    def kterms(self, xt):
        t = dict()
        t['xt'] = xt
        t['kxtX'] = self.k(xt, self.x)
        t['kxtxt'] = self.k(xt, xt)
        t['kxtX_B'] = np.kron(t['kxtX'], self.wwt)
        t['kxtxt_B'] = np.kron(t['kxtxt'], self.wwt)
        return t

    def predict(self, t):
        m = t['kxtX_B'].dot(self.kXX_B_inv).dot(self.y)
        v = t['kxtxt'] - t['kxtX'].dot(self.kXX_B_inv).dot(t['kxtX'].transpose())

        return m, v

    def dm(self, t):
        kXX_B_inv_y = self.kXX_B_inv.dot(self.y)
        kxtX_B_kXX_B_inv = t['kxtX_B'].dot(self.kXX_B_inv)
        dkxtX_ds_B = np.kron(self.dk_signal(t['kxtX']), self.wwt)
        dkXX_ds_B = np.kron(self.dk_signal(self.kXX), self.wwt)
        dm_ds = (dkxtX_ds_B - kxtX_B_kXX_B_inv.dot(dkXX_ds_B)).dot(kXX_B_inv_y)

        dm_dl = np.zeros((self.n_dim, self.n_factor))
        for i in range(self.n_dim):
            dkxtX_dli_B = np.kron(self.dk_ls(t['xt'], self.x, t['kxtX'], i), self.wwt)
            dkXX_dli_B = np.kron(self.dk_ls(self.x, self.x, self.kXX, i), self.wwt)
            dm_dl[i] = (dkxtX_dli_B - kxtX_B_kXX_B_inv.dot(dkXX_dli_B)).dot(kXX_B_inv_y)

        dm_dn = -2 * self.noise() * kxtX_B_kXX_B_inv.dot(kXX_B_inv_y)

        dm_dw = np.zeros((self.n_factor, self.n_factor, self.n_factor))
        for i in range(self.w().shape[0]):
            for j in range(self.w().shape[1]):
                Eij = np.zeros(self.wwt.shape)
                Eij[i, j] = 1
                kxtX_Eij = np.kron(t['kxtX'], Eij)
                kXX_Eij = np.kron(self.kXX, Eij)
                dm_dw[i, j] = (kxtX_Eij - kxtX_B_kXX_B_inv.dot(kXX_Eij)).dot(kXX_B_inv_y)

        return dm_ds, dm_dn, dm_dl, dm_dw

    def dv(self, t):
        kXX_B_inv_y = self.kXX_B_inv.dot(self.y)
        kxtX_B_kXX_B_inv = t['kxtX_B'].dot(self.kXX_B_inv)
        dkxtX_ds_B = np.kron(self.dk_signal(t['kxtX']), self.wwt)
        dkXX_ds_B = np.kron(self.dk_signal(self.kXX), self.wwt)
        dm_ds = (dkxtX_ds_B - kxtX_B_kXX_B_inv.dot(dkXX_ds_B)).dot(kXX_B_inv_y)

        dm_dl = np.zeros((self.n_dim, self.n_factor))
        for i in range(self.n_dim):
            dkxtX_dli_B = np.kron(self.dk_ls(t['xt'], self.x, t['kxtX'], i), self.wwt)
            dkXX_dli_B = np.kron(self.dk_ls(self.x, self.x, self.kXX, i), self.wwt)
            dm_dl[i] = (dkxtX_dli_B - kxtX_B_kXX_B_inv.dot(dkXX_dli_B)).dot(kXX_B_inv_y)

        dm_dn = -2 * self.noise() * kxtX_B_kXX_B_inv.dot(kXX_B_inv_y)

        dm_dw = np.zeros((self.n_factor, self.n_factor, self.n_factor))
        for i in range(self.w().shape[0]):
            for j in range(self.w().shape[1]):
                Eij = np.zeros(self.wwt.shape)
                Eij[i, j] = 1
                kxtX_Eij = np.kron(t['kxtX'], Eij)
                kXX_Eij = np.kron(self.kXX, Eij)
                dm_dw[i, j] = (kxtX_Eij - kxtX_B_kXX_B_inv.dot(kXX_Eij)).dot(kXX_B_inv_y)

        return dm_ds, dm_dn, dm_dl, dm_dw