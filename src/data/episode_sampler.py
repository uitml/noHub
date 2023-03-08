import logging
import torch as th

logger = logging.getLogger(__name__)


class EpisodeSampler:
    def __init__(self, features, labels, n_ways, n_shots, n_queries, n_episodes, batch_size=None, fix_labels=True):
        self.features = features
        self.labels = labels

        self.n_ways = n_ways
        self.n_shots = n_shots
        self.n_queries = n_queries
        self.n_episodes = n_episodes
        self.fix_labels = fix_labels

        if batch_size is None:
            self.batch_size = n_episodes
        elif batch_size > n_episodes:
            logger.warning(f"Got episode_batch_size ({batch_size}) > n_episodes ({n_episodes}). Using "
                           f"episode_batch_size = {n_episodes} instead.")
            self.batch_size = n_episodes
        else:
            assert (n_episodes % batch_size) == 0, "n_episodes must be divisible by episode_batch_size."
            self.batch_size = batch_size

        self.count = 0
        self.n_iter = self.n_episodes // self.batch_size

        with th.no_grad():
            self.unique_labels = th.unique(labels)
            self.label_sample_weights = th.ones_like(self.unique_labels) / len(self.unique_labels)
            self.label_sample_weights = self.label_sample_weights.expand(self.batch_size, -1)

    def __len__(self):
        return self.n_iter

    @staticmethod
    def _fix_labels(episode_labels, sampled_labels):
        """
        Convert labels from arbitrary integers to [0, 1, ..., n_ways]
        """
        return th.argmax((sampled_labels[:, :, None] == episode_labels[:, None, :]).long(), dim=2)

    @th.no_grad()
    def __next__(self):
        if self.count >= self.n_iter:
            raise StopIteration

        # Sample episode classes
        episode_label_idx = th.multinomial(self.label_sample_weights, replacement=False, num_samples=self.n_ways)
        episode_labels = self.unique_labels[episode_label_idx]

        # Sample support and queries
        weights = (episode_labels[:, :, None] == self.labels[None, None, :]).float()
        # Reshape weights so that they are compatible with th.multinomial.
        weights = weights.view(self.batch_size * self.n_ways, -1)
        idx = th.multinomial(weights, num_samples=(self.n_shots + self.n_queries), replacement=False)
        idx = idx.view(self.batch_size, self.n_ways, -1)

        # Slice feature and label tensors
        sampled_features = self.features[idx]
        sampled_labels = self.labels[idx]

        # Split into support and query
        support_labels = sampled_labels[:, :, :self.n_shots].reshape(self.batch_size, self.n_ways * self.n_shots)
        support_features = sampled_features[:, :, :self.n_shots]\
            .reshape(self.batch_size, self.n_ways * self.n_shots, -1)

        query_labels = sampled_labels[:, :, self.n_shots:].reshape(self.batch_size, self.n_ways * self.n_queries)
        query_features = sampled_features[:, :, self.n_shots:]\
            .reshape(self.batch_size, self.n_ways * self.n_queries, -1)

        if self.fix_labels:
            support_labels = self._fix_labels(episode_labels, support_labels)
            query_labels = self._fix_labels(episode_labels, query_labels)

        self.count += 1
        return (support_features, support_labels), (query_features, query_labels)

    def __iter__(self):
        return self


if __name__ == '__main__':
    import numpy as np

    # Test sampler
    data = np.stack([np.linspace(0, 5, 50, endpoint=False) for _ in range(2)], axis=1)
    labels = np.floor(data[:, 0])
    data = th.from_numpy(data)
    labels = th.from_numpy(labels)
    sampler = EpisodeSampler(features=data, labels=labels, n_ways=2, n_shots=5, n_queries=2, n_episodes=3, batch_size=3)
    for (sf, sl), (qf, ql) in sampler:
        print(f"{sf.shape=}")
        print(f"{sl.shape=}")
        print(f"{qf.shape=}")
        print(f"{ql.shape=}")
