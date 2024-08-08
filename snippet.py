# ## Simulate the noise with Gaussians
# delta = (outs - outs.mean(0, keepdim=True)).float()
# covs = (delta[:, :, None, :] * delta[:, :, :, None]).mean(0)
# res = torch.linalg.eigh(covs)
# covs_ss = res.eigenvalues.sqrt()
# covs_ee = res.eigenvectors
# covs_dirs = covs_ee * covs_ss[:, None, :]

# ## Confirm that sampling these independent directions yields a similar accuracy to what we see in practice
# noise = torch.randn(200, 10000, 10, device=delta.device)
# outs_sim = outs.mean(0) + torch.einsum('sik,ilk -> sil', noise, covs_dirs)
# print((outs_sim.argmax(2) == labels).float().mean())

# ## The columns of covs_ee are the directions of independent variation
# xx = delta[:, 55] @ covs_ee[55]
# cov = (xx[:, None, :] * xx[:, :, None]).mean(0)
# plt.imshow(cov.cpu())
# plt.show()

## To sample from the example-wise Gaussian approximations logit distributions, do as follows
# ## Confirm that sampling these independent directions now yields the same covariances
# noise = torch.randn(1000000, 5, 10, device=delta.device)
# outs_sim = torch.einsum('sik,ilk -> sil', noise, covs_dirs[:5])
# xx = outs_sim[:, :]
# covs2 = (xx[:, :, None, :] * xx[:, :, :, None]).mean(0)
# (covs[:5] - covs2[:5]).abs().sum() / covs2[:5].abs().sum() # should be small

# plt.imshow(covs[1].cpu())
# plt.show()



# # Random, Loss, SelfInfluence, Diff, Oracle
# vv = [281, 261, 251, 252, 253, 257, 271, 272, 273, 274, 275, 291, 296,
#       282, 262, 162, 160, 161, 201, 211, 221, 225, 229, 233, 292, 297,
#       283, 263, 163, 152, 157, 202, 212, 222, 226, 230, 234, 293, 298,
#       284, 264, 164, 153, 158, 203, 213, 223, 227, 231, 235, 294, 299,
#       285, 265, 254, 255, 256, 258, 276, 277, 278, 279, 280, 295, 300,]

# yy1 = []
# yy2 = []
# for v in vv:
#     outs = load('masks/best_mask_v%d' % v)[:320]
#     if len(outs) < 320:
#         print(v, len(outs))
#     yy1.append(((outs.argmax(2) == labels)).float().mean().item())
#     yy2.append((outs.float().mean(0).argmax(1) == labels).float().mean().item())
    
# obj = dict(yy1=yy1, yy2=yy2)
# torch.save(obj, 'fig_active_learning.pt')





# # score = torch.randn_like(score)

# # score = loss02
# # score = self_influence
# # score = diff

# mask = (margin(out0, labels) < 2)
# datamodel_m = get_datamodel_m(datamodel)
# score = datamodel_m[:, mask].sum(1)

# x = 0.975
# mask = (score > score.float().quantile(1-x).item())
# v = 300
# torch.save(mask, '/workspace/elastic-airbench/masks/best_influence_v%d.pt' % v)

