# Fields & Metrics Reference

This page defines the main configuration fields and recorded values used by the app.

The exact fields available depend on the selected problem family, data source, parameterization, optimizer, and metric/logging options.

## Core experiment fields

**problem_family**  
The experiment family. The main families are PSD sensing and matrix completion.

**data_source**  
The source of the data. Matrix completion can use synthetic data, MovieLens data, or a registered custom triplets CSV. PSD sensing uses synthetic data.

**seed**  
Controls random choices such as initialization, synthetic data generation, sampling, and noise.

**n**  
The main problem dimension. For square matrix problems, this is the matrix size.

**n_cols**  
The number of columns for rectangular matrix completion problems. When omitted, the problem is treated as square.

**m**  
The number of measurements or observed entries.

**r_star**  
The target rank used to generate or describe the underlying low-rank structure.

**factor_rank**  
The rank used by the learned factorized model.

**noise_std**  
The standard deviation of added noise, when noise is used.

## Problem-specific fields

**measurement_type**  
The measurement model used for PSD sensing.

**psd_regime**  
The PSD sensing regime, such as commuting or non-commuting.

**noncommutativity_level**  
Controls the strength of non-commuting structure in PSD sensing when that regime is used.

**sampling_regime**  
The sampling pattern for matrix completion, such as uniform or biased sampling.

**coherence_level**  
Controls coherence in synthetic matrix completion problems.

**mc_spectrum_type**  
The singular-value spectrum used for synthetic matrix completion.

**mc_spectrum_decay**  
Controls the decay rate when the synthetic spectrum is not flat.

**mc_factor_balance**  
Controls whether the synthetic matrix-completion factors are balanced or unbalanced.

**mc_factor_imbalance_level**  
Controls the strength of factor imbalance when unbalanced factors are used.

## Model and initialization fields

**parameterization**  
The representation used for optimization. Different parameterizations can represent related solutions while producing different optimization behavior.

**init_type**  
The initialization family.

**init_scale**  
The scale of the initialization.

## Optimization fields

**optimizer**  
The optimization method used for training.

**lr**  
The learning rate.

**lr_schedule_type**  
The learning-rate schedule.

**lr_decay_gamma**  
The decay multiplier used by schedules that reduce the learning rate.

**lr_step_size**  
The step interval used by step-based schedules.

**lr_min**  
The minimum learning rate allowed by the schedule.

**batch_size**  
The minibatch size for stochastic optimizers.

**momentum_beta**  
The momentum coefficient for momentum-based methods, when used.

**adam_beta1, adam_beta2, adam_eps**  
Adam optimizer parameters, when used.

**preconditioner_eps**  
Stability constant used by preconditioned optimizers, when used.

## Explicit regularization fields

**explicit_reg_type**  
The explicit regularization method.

**explicit_reg_strength**  
The strength of the explicit regularization term.

## Runtime and stopping fields

**steps**  
The number of optimization steps requested for the run. For convergence-based stopping, this acts as the maximum step cap.

**log_every**  
How often metrics are recorded.

**stopping_mode**  
The stopping policy for the run.

**grad_norm_tol**  
Gradient-norm threshold used by convergence-based stopping.

**objective_tol**  
Objective-change threshold used by convergence-based stopping.

**divergence_norm_threshold**  
Threshold used to detect divergent parameter growth.

**early_stopping_enabled**  
Whether early stopping is enabled for supported data sources.

**early_stopping_metric**  
The validation metric used for early stopping.

**early_stopping_patience**  
How long training can continue without sufficient validation improvement.

**early_stopping_min_delta**  
Minimum validation improvement required to reset patience.

**device_preference**  
Requested device behavior, such as automatic selection.

**dtype_preference**  
Requested numeric precision behavior.

## Tracking fields

**track_eigs**  
Whether eigenvalue or singular-value tracking is enabled.

**num_eigs**  
The number of eigenvalues or singular values to track.

**run_name**  
Optional display name for the run.

**notes**  
Optional user notes attached to the run.

**group_id**  
Identifier used to group related runs.

**seed_group_id**  
Identifier used for runs that belong to the same seed group.

## Common metric conventions

Metrics recorded over time are trajectory metrics.

Summary metrics usually use the `final_` prefix when they refer to the last recorded value.

For example:

- `objective` is the objective over training
- `final_objective` is the final recorded objective
- `gradient_norm` is the gradient norm over training
- `final_gradient_norm` is the final recorded gradient norm

## Loss and objective metrics

**train_loss**  
The training loss.

**val_loss**  
The validation loss for data sources with validation splits.

**test_loss**  
The test loss for data sources with test splits.

**observed_loss**  
The loss on observed entries in matrix completion.

**data_loss**  
The data-fit part of the loss.

**explicit_reg_term**  
The explicit regularization contribution.

**objective**  
The optimized objective, usually combining data loss and explicit regularization.

**objective_delta**  
The change in objective between recorded steps.

## Reconstruction and prediction metrics

**relative_reconstruction_error / rel_recon_error**  
Normalized reconstruction error against the target matrix. `rel_recon_error` may appear as a shorter alias. This is mainly used when a ground-truth synthetic target is available.

**val_rmse**  
Validation root mean squared error.

**test_rmse**  
Test root mean squared error.

**best_val_rmse**  
Best validation RMSE observed during training.

**best_test_rmse_at_best_val**  
Test RMSE at the step where validation performance was best.

## Optimization diagnostics

**gradient_norm**  
Magnitude of the gradient.

**update_norm**  
Magnitude of the parameter update.

**parameter_norm**  
Magnitude of the learned parameters.

**lr**  
Learning rate at the recorded step.

These metrics help identify convergence, instability, and schedule behavior.

## Matrix-structure metrics

**frobenius_norm_X**  
Frobenius norm of the learned matrix.

**effective_rank**  
A rank-like summary based on the spread of singular values.

**stable_rank**  
A norm-based rank measure.

**trace_X**  
Trace of the learned matrix when applicable.

**min_eigenvalue_X**  
Smallest tracked eigenvalue of the learned matrix when applicable.

**negative_eigen_mass**  
Amount of negative eigenvalue mass. This is useful for checking whether a learned matrix has left the expected PSD structure.

**eig_1, eig_2, ...**  
Tracked eigenvalues.

**sv_1, sv_2, ...**  
Tracked singular values.

**target_sv_1, target_sv_2, ...**  
Tracked target singular values for synthetic problems with ground truth.

## Factor metrics

**factor_u_norm**  
Norm of the learned left factor.

**factor_v_norm**  
Norm of the learned right factor.

**factor_norm_ratio**  
Ratio between factor norms.

**factor_log_norm_ratio**  
Log-scaled factor norm ratio.

These metrics are useful for matrix completion runs where factor balance is part of the behavior being inspected.

## Target-structure metrics

**target_row_coherence**  
Row coherence of the target matrix.

**target_col_coherence**  
Column coherence of the target matrix.

**target_condition_number**  
Condition number of the target matrix.

**target_effective_rank**  
Effective rank of the target matrix.

**target_stable_rank**  
Stable rank of the target matrix.

These describe the synthetic target, which helps interpret whether a run is easy, hard, well-conditioned, coherent, or rank-concentrated.

## Alignment metrics

**subspace_alignment**  
Alignment between the learned subspace and the target subspace for symmetric PSD problems. Higher values indicate stronger alignment.

**subspace_projection_distance**  
Projection distance between learned and target subspaces for symmetric PSD problems. Lower values indicate closer subspaces.

**left_subspace_alignment**  
Alignment between learned and target left subspaces for matrix completion.

**right_subspace_alignment**  
Alignment between learned and target right subspaces for matrix completion.

**mean_subspace_alignment**  
Average of left and right subspace alignment.

**left_subspace_projection_distance**  
Projection distance for the left subspace.

**right_subspace_projection_distance**  
Projection distance for the right subspace.

**mean_subspace_projection_distance**  
Average of left and right projection distance.

## Sweep fields and compare axes

Sweeps vary one or two selected fields while keeping the remaining configuration fixed. A 1D sweep has one varied field. A full-grid 2D sweep has two varied fields and is shown as a heatmap in Results.

Compare audits use expected-change axes rather than raw field names. Common axes include optimizer, learning rate, parameterization, explicit regularization, initialization, training budget, and random seed. More technical axes include learning-rate schedule, batch size, stopping mode, PSD regime, noncommutativity level, coherence level, sampling regime, matrix-completion spectrum, and factor-balance settings.

Some compare axes map to one field, such as `learning_rate` mapping to `lr`. Others map to several fields. For example, `learning_rate_schedule` includes `lr_schedule_type`, `lr_decay_gamma`, `lr_step_size`, and `lr_min`; `training_budget` includes `steps`, `grad_norm_tol`, and `objective_tol`; `optimizer` includes optimizer-family hyperparameters such as momentum, Adam parameters, and preconditioner epsilon.

Bookkeeping fields such as `group_id` and `seed_group_id` are provenance fields. They should not by themselves make an otherwise controlled comparison invalid.

## Sweep metrics

Sweep views use selected summary metrics to compare runs.

For synthetic matrix completion, common examples include:

- final relative reconstruction error
- final objective
- final Frobenius norm
- factor norm ratio
- subspace alignment
- subspace projection distance

For real observed-rating data, common examples include:

- final validation RMSE
- final test RMSE
- best validation RMSE
- test RMSE at best validation step
- final objective

Availability depends on problem family, data source, and metric logging settings. Metrics that are unavailable or entirely missing for a sweep are not offered as plot choices.

For 1D sweeps, aggregate metrics are shown as curves over the varied field. For full-grid 2D sweeps, aggregate metrics are shown as heatmaps over the two varied fields. In both cases, the aggregate values should be read together with seed counts, run counts, and warning/failure status.

## Next steps

Go to **Reproducibility & Provenance** for saved configs, seeds, metadata, and exported artifacts.

Go to **Reading Results** for guidance on interpreting these fields in runs and sweeps.
