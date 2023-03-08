# Adapted from https://github.com/MichalisLazarou/iLPC/
import torch as th
from torch import nn

import config
from classifiers.simpleshot import initial_weights


@config.record_defaults(prefix="ilpc")
class ILPC(nn.Module):
    def __init__(self, n_queries, n_ways, k=20, alpha=0.8, best_samples=3, denoising_iterations=50, tau=3.0,
                 sinkhorn_iter=1, gamma=3.0):
        super(ILPC, self).__init__()

        self.n_queries = n_queries
        self.n_ways = n_ways

        self.k = k
        self.alpha = alpha
        self.gamma = gamma
        self.best_samples = best_samples
        self.denoising_iterations = denoising_iterations
        self.tau = tau
        self.sinkhorn_iter = sinkhorn_iter
        self.unbalanced = False
        self.pseudo_labelling_max_iter = 20
        self.optimal_transport_max_iter = 100

        self.label_denoising_start_lr = 0.1
        self.label_denoising_end_lr = 0.1
        self.label_denoising_cycle = 10  # number of epochs

    def forward(self):
        raise NotImplementedError(f"Use 'ILPC.fit_predict(...)' instead.")

    def update_pseudo_labels(self, support, support_ys, query):
        n_support = support.shape[0]
        n_query = query.shape[0]
        n_features = n_support + n_query

        X = th.cat((support, query), dim=0)
        labels = th.zeros(X.shape[0], device=config.DEVICE).long()
        labels[:n_support] = support_ys
        labeled_idx = th.arange(n_support, device=config.DEVICE)
        unlabeled_idx = th.arange(n_features, device=config.DEVICE)

        # Create the graph
        ip = X @ X.t()
        ip_k, _ = th.topk(ip, k=(self.k + 1))
        # NN graph
        W = (ip >= ip_k.min(dim=1, keepdim=True).values).float()

        W = (W * nn.functional.relu(ip)) ** self.gamma
        # Symmetrize
        W = W + W.t()
        # Normalize
        W.fill_diagonal_(0.0)
        S = W.sum(dim=1)
        S[S == 0] = 1
        D = th.diag(1. / th.sqrt(S))
        Wn = D @ W @ D

        # Initialize the y vector for each class (eq 5 from the paper, normalized with the class size)
        # and apply label propagation
        Z = th.zeros((n_features, self.n_ways), device=config.DEVICE)
        A = th.eye(n_features, device=config.DEVICE) - self.alpha * Wn
        for i in range(self.n_ways):
            cur_idx = labeled_idx[th.where(labels[labeled_idx] == i)]
            y = th.zeros((n_features,), device=config.DEVICE)
            y[cur_idx] = 1.0
            f = th.linalg.solve(A, y)
            Z[:, i] = f

        # Handle numerical errors
        Z[Z < 0] = 0
        # --------try to filter-----------
        z_amax = -1 * th.amax(Z, 1)[n_support:]
        # -----------trying filtering--------
        # Compute the weight for each instance based on the entropy (eq 11 from the paper)
        probs_l1 = nn.functional.normalize(Z, 1)
        probs_l1[probs_l1 < 0] = 0
        p_labels = th.argmax(probs_l1, 1)
        p_labels[labeled_idx] = labels[labeled_idx]
        return p_labels[n_support:], probs_l1[n_support:], z_amax

    def compute_optimal_transport(self, M, n_samples, epsilon=1e-6):
        # r is the P we discussed in paper r.shape = n_runs x total_queries, all entries = 1
        r = th.ones(1, M.shape[0], device=config.DEVICE)
        # r = r * weights
        # c = th.ones(1, M.shape[1]) * int(M.shape[0]/M.shape[1])
        # c = th.Tensor(n_samples, device=config.DEVICE)
        idx = th.where(n_samples <= 0)
        # if opt.unbalanced == True:
        #     c = th.FloatTensor(n_samples)
        #     idx = np.where(c.detach().cpu().numpy() <= 0)
        #     if len(idx[0]) > 0:
        #         M[:, idx[0]] = th.zeros(M.shape[0], 1)

        # M = M.to(device=config.DEVICE)
        # r = r.to(device=config.DEVICE)
        # c = c.to(device=config.DEVICE)
        M = th.unsqueeze(M, dim=0)
        n_runs, n, m = M.shape
        P = M

        u = th.zeros(n_runs, n, device=config.DEVICE)
        iters = 1
        for i in range(self.sinkhorn_iter):
            P = th.pow(P, self.tau)
            while th.max(th.abs(u - P.sum(2))) > epsilon:
                u = P.sum(2)
                P *= (r / u).view((n_runs, -1, 1))
                P *= (n_samples / P.sum(1)).view((n_runs, 1, -1))
                if len(idx[0]) > 0:
                    P[P != P] = 0
                if iters == self.optimal_transport_max_iter:
                    break
                iters = iters + 1
        P = th.squeeze(P)
        best_per_class = th.argmax(P, 0)
        if M.shape[1] == 1:
            P = P[None, ...]
        labels = th.argmax(P, 1)
        return P, labels, best_per_class

    def weight_imprinting(self, X, Y, model):
        imprinted = th.zeros(self.n_ways, X.shape[1])
        for i in range(self.n_ways):
            idx = th.where(Y == i)
            tmp = th.mean(X[idx], dim=0)
            imprinted[i, :] = tmp
        imprinted = nn.functional.normalize(imprinted, dim=1, p=2)
        model.weight.data = imprinted
        return model

    @th.enable_grad()
    def label_denoising(self, support, support_ys, query, query_ys_pred):
        all_embeddings = th.cat((support, query), dim=0)
        input_dim = all_embeddings.shape[1]
        all_ys = th.cat((support_ys, query_ys_pred), dim=0)
        output_size = self.n_ways

        step_size_lr = (self.label_denoising_start_lr - self.label_denoising_end_lr) / self.label_denoising_cycle
        lambda1 = lambda x: self.label_denoising_start_lr - (x % self.label_denoising_cycle) * step_size_lr

        o2u = nn.Linear(input_dim, output_size).to(device=config.DEVICE)
        o2u.weight.data = initial_weights(support[None, ...], support_ys[None, ...], normalize=True)[0]
        # o2u = self.weight_imprinting(all_embeddings[:support.shape[0]], support_ys, o2u)

        optimizer = th.optim.SGD(o2u.parameters(), 1, momentum=0.9, weight_decay=5e-4)
        scheduler_lr = th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss_statistics = th.zeros(all_ys.shape, device=config.DEVICE)
        lr_progression = []
        for epoch in range(self.denoising_iterations):
            output = o2u(all_embeddings)
            optimizer.zero_grad()
            loss_each = criterion(output, all_ys)
            loss_all = th.mean(loss_each)
            loss_all.backward()
            loss_statistics += (loss_each.detach() / self.denoising_iterations)
            optimizer.step()
            scheduler_lr.step()
            lr_progression.append(optimizer.param_groups[0]['lr'])
        return loss_statistics, lr_progression

    def rank_per_class(self, rank, ys_pred, no_keep):
        list_indices = []
        list_ys = []
        for i in range(self.n_ways):
            cur_idx = th.nonzero(ys_pred == i, as_tuple=True)
            y = th.ones((self.n_ways,), device=config.DEVICE) * i
            class_rank = rank[cur_idx]
            # class_rank_sorted = sp.stats.rankdata(class_rank, method='ordinal')
            class_rank_sorted = th.argsort(class_rank) + 1
            class_rank_sorted[class_rank_sorted > no_keep] = 0
            indices = th.nonzero(class_rank_sorted, as_tuple=True)
            list_indices.append(cur_idx[0][indices[0]])
            list_ys.append(y)
        idxs = th.cat(list_indices, dim=0)
        ys = th.cat(list_ys, dim=0)
        return idxs, ys

    @staticmethod
    def remaining_labels(n_samples, selected_samples):
        for i in range(len(n_samples)):
            occurrences = th.count_nonzero(selected_samples == i)
            n_samples[i] = n_samples[i] - occurrences

    @th.no_grad()
    def iter_balanced_trans(self, support_features, support_ys, query_features):
        n_support = support_features.shape[0]
        n_query = query_features.shape[0]
        n_features = n_support + n_query
        n_samples = th.tensor(self.n_ways * [self.n_queries], device=config.DEVICE)

        # Record order at which queries are added, so we can re-order the predictions to preserve the original order.
        query_added_order = []
        # Indices of remaining queries. Initialize to all queries
        remaining_query_idx = th.arange(n_query, device=config.DEVICE)

        for j in range(n_query):
            # Do ILPC stuff
            query_ys_pred, probs, weights = self.update_pseudo_labels(support_features, support_ys, query_features)
            P, query_ys_pred, indices = self.compute_optimal_transport(probs, n_samples=n_samples)
            loss_statistics, _ = self.label_denoising(support_features, support_ys, query_features, query_ys_pred)
            un_loss_statistics = loss_statistics[support_ys.shape[0]:].detach()
            # rank = sp.stats.rankdata(un_loss_statistics, method='ordinal')
            rank = th.argsort(un_loss_statistics) + 1

            # Figure out which queries to add to the augmented support set
            indices, ys = self.rank_per_class(rank, query_ys_pred, self.best_samples)
            if len(indices) < 5:
                break

            pseudo_mask = th.isin(th.arange(query_features.shape[0], device=config.DEVICE), indices)
            pseudo_features, query_features = query_features[pseudo_mask], query_features[~pseudo_mask]
            pseudo_ys, query_ys_pred = query_ys_pred[pseudo_mask], query_ys_pred[~pseudo_mask]

            add_indices = remaining_query_idx[pseudo_mask]
            query_added_order.append(add_indices)
            remaining_query_idx = remaining_query_idx[~pseudo_mask]

            # Add pseudo-labelled queries to the support set.
            support_features = th.cat((support_features, pseudo_features), dim=0)
            support_ys = th.cat((support_ys, pseudo_ys), dim=0)

            self.remaining_labels(n_samples, pseudo_ys)

            if support_features.shape[0] == n_features:
                break

        support_ys = th.cat((support_ys, query_ys_pred), dim=0)
        query_ys_pred = support_ys[n_support:]

        # Reorder predictions to original order.
        query_added_order.append(remaining_query_idx)
        query_added_order = th.cat(query_added_order, dim=0)

        original_query_order_idx = th.argsort(query_added_order)
        query_pred = query_ys_pred[original_query_order_idx]

        assert query_pred.shape[0] == n_query, f"Unexpected shape for query predictions: {query_ys_pred.shape[0]}." \
                                               f" Expected: {n_query}"
        return query_pred

    def fit_predict(self, support_features, support_labels, query_features):
        query_predictions = []
        for ep, (sf, sl, qf) in enumerate(zip(support_features, support_labels, query_features)):
            # Iterate episodes since the algorithm is hard to vectorize over episodes...
            qp = self.iter_balanced_trans(support_features=sf, support_ys=sl, query_features=qf)
            query_predictions.append(qp)
        return th.stack(query_predictions, dim=0)
