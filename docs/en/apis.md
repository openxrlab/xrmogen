# API
## run_mogen

Input: args, run arguments

Purpose: Generate a model based on running parameters, train or test actions

## train_mogen
Input: cfg, mmcv.Config

Purpose: According to the running parameters, train the action generation model

## test_mogen
Input: cfg, mmcv.Config

Purpose: Generate a model for test actions based on running parameters

## parse_args
Input: args, run arguments

Purpose: Convert running parameters to mmcv.Config
