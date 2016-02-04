# FbAppsCollateralDamage

Simulates a social network friendship graph where users may or may not install an app, and analyses how many users have a friend with the App installed.

## Parameters

Parameters of the simulation:

1. Number of nodes
2. Expected_degree
3. Expected fraction of users with the app installed
4. Graph_model (Barabási-Albért, Erdos-Renyi, or Watts-Strogatz)
5. App adoption_model (uniform or prop_friends_installed)

## A simulation

For every configuration of parameters, the code: 

1. Generates num_graphs graphs of n nodes with the given graph model and expected degree
2. Assigns installations given adoption model and desired fraction of users
3. Computes how many users have a friend with the app installed

## Execution

To execute, run:

```
python ifipsec.py
```
