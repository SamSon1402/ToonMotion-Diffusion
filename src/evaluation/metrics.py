"""FID, diversity, multimodality, R-Precision for motion generation."""

import numpy as np
from typing import List
from scipy import linalg


class MotionMetrics:
    @staticmethod
    def frechet_distance(mu1, sigma1, mu2, sigma2):
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))

    @staticmethod
    def compute_fid(real_features, gen_features):
        mu_r, sigma_r = real_features.mean(0), np.cov(real_features, rowvar=False)
        mu_g, sigma_g = gen_features.mean(0), np.cov(gen_features, rowvar=False)
        return MotionMetrics.frechet_distance(mu_r, sigma_r, mu_g, sigma_g)

    @staticmethod
    def diversity(motions, num_samples=200):
        if len(motions) < 2:
            return 0.0
        feats = np.array([m.flatten() for m in motions])
        n = min(num_samples, len(feats))
        idx = np.random.choice(len(feats), n, replace=False)
        total, count = 0.0, 0
        for i in range(n):
            for j in range(i + 1, n):
                total += np.linalg.norm(feats[idx[i]] - feats[idx[j]])
                count += 1
        return total / max(count, 1)

    @staticmethod
    def multimodality(motions_per_prompt):
        total, count = 0.0, 0
        for motions in motions_per_prompt:
            if len(motions) < 2:
                continue
            feats = [m.flatten() for m in motions]
            for i in range(len(feats)):
                for j in range(i + 1, len(feats)):
                    total += np.linalg.norm(np.array(feats[i]) - np.array(feats[j]))
                    count += 1
        return total / max(count, 1)

    @staticmethod
    def r_precision(text_features, motion_features, top_k=3):
        n = len(text_features)
        correct = 0
        for i in range(n):
            sims = text_features @ motion_features[i]
            if i in np.argsort(sims)[-top_k:]:
                correct += 1
        return correct / n