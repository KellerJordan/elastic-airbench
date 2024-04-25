Changes relative to `airbench94.py`:

Functional changes:
* Removed test-time augmentation (-0.8% accuracy), label smoothing (-0.2% accuracy), lookahead optimization, and progressive freezing of the whitening layer bias.
* Reduced learning rate from 11.5 to 5.0 (which is optimal given removal of label smoothing).
* Rounded weight decay to 0.15, batch size to 1000, and epochs to 10.0.

Under-the-hood changes:
* Replaced the dataloader with an "infinite iterator" variant which ensures that every example is seen every epoch (rather than potentially being dropped if the batch size does not divide the example count)
* This dataloader also accepts random seeds for the data ordering, data augmentation, and model initialization, enabling detailed study of these factors.
* The total batch count uses stochastic rounding to ensure that the expected number of iterations is strictly equal to a specified multiple of the number of training examples. This is a subtle point that is very important for certain experiments in data influence.
