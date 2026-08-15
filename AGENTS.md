# Squisher Agent Guide

## CLI-First Workflows

- For every operational step supported by a packaged CLI, invoke that CLI directly. Prefer installed entry points such as `squisher-deconv`, `lightsheet`, and `lightsheet-stitch`; in an uninstalled development checkout, use the project package runner to invoke the same entry point.
- Defer to packaged CLIs whenever they cover the requested work. Wrapper or ad hoc scripts are acceptable when the requested workflow differs substantially from what the CLIs can express, but they should invoke existing CLI capabilities rather than reimplement them. Shell setup may activate the environment, set environment variables, orchestrate distinct CLI steps, and pipe CLI output to a log.
- If a recurring or production workflow lacks a necessary option or subcommand, add it to the owning packaged CLI with focused tests, then run that CLI. Do not use a one-off script or downstream metadata rewrite to emulate a missing CLI option.
- Direct Python APIs may implement the portions of a substantially different workflow that the packaged CLIs cannot express; otherwise limit them to tests, focused diagnostics, and profiling. Do not use `python -m` as a substitute for an available console script; in a development checkout, invoke that entry point through the project package runner.
- For long-running CLI jobs, run the installed executable in tmux with stdout and stderr logged. Record the exact invocation, then verify that the pane or process and persistent log show early progress; do not report a job as started from session existence alone.

## Runtime Environments

- Run every Squisher fusion and OME-Zarr pyramid-writing entrypoint—including CLI or module invocations, wrappers, retries, and worker processes—in the `multi` Conda environment. Never invoke these operations through `seq`; its Zarr/Fsspec stack fails on local sharded writes.
- Launch long fusion jobs from an explicit Bash activation of `multi`. Before any output write, log and verify that the actual job interpreter is `/home/chaichontat/miniforge3/envs/multi/bin/python`; abort if it differs. A separate dry run, parent-shell activation, inherited tmux environment, or executable-name check is insufficient. Use `seq` only for registration commands specifically verified to require its working CUDA stack.

## Experiment Notebook

- Use the `$lab-notebook` skill for ongoing experiments, multi-step data-processing runs, benchmarks, debugging investigations, and long-running jobs. Create or update the project-local notebook at the start, record reproducible inputs, commands, parameters, outputs, failures, decisions, and process or log locations as work progresses, and append the final result before responding.
