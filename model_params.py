
model_params= {

    'trial_1': {
        'epoch': 100,
        'optimizer': 'adam',
        'num_units': 64,
        'is_gru': True,
        'num_heads': 4
    },

    'trial_2': {
        'epoch': 100,
        'optimizer': 'RMSprop',
        'num_units': 64,
        'is_gru': False,
        'num_heads': 4
    },

    'trial_3': {
        'epoch': 100,
        'optimizer': 'adam',
        'num_units': 256,
        'is_gru': False,
        'num_heads': 8
    },

    'trial_4': {
        'epoch': 100,
        'optimizer': 'adam',
        'num_units': 256,
        'is_gru': True,
        'num_heads': 8
    },

    'trial_5': {
        'epoch': 100,
        'optimizer': 'adam',
        'num_units': 512,
        'is_gru': True,
        'num_heads': 8
    }

}