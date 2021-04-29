import random

random.seed(12345)

param_grid = {
    "activation": ["tanh", "relu", "none"],
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "epochs": [50, 100, 150, 200, 250, 300, 350, 400],
    "lr": [1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6],
    "nhidden": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
    "nlayers": [1, 2, 3, 4],
}

for i in range(50):
    # randomly pick hyper-parameters
    params = {k: random.choice(v) for k, v in param_grid.items()}
    print('\t'.join([str(param) for param in params.values()]))
    params['nhidden'] = ' '.join([str(params['nhidden'])] * params['nlayers'])

    # Generate bash scripts e.g., "run1.sh"
    script = f"""
    python3 src/models/train_model.py \\
       --activation {params['activation']} \\
       --dropout {params['dropout']} \\
       --epochs {params['epochs']} \\
       --lr {params['lr']} \\
       --nhidden {params['nhidden']} \\
       --featureless 
    """
    with open(f'src/scripts/temp/run{i}.sh', mode="w") as f:
        print(script, file=f)
