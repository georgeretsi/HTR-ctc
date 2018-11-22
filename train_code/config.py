
classes = '_!"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
cdict = {c:i for i,c in enumerate(classes)}
icdict = {i:c for i,c in enumerate(classes)}

cnn_cfg = [(2, 32), 'M', (4, 64), 'M', (6, 128), 'M', (2, 256)]
rnn_cfg = (256, 1)  # (hidden , num_layers)

max_epochs = 20
batch_size = 1
iter_size = 16
# fixed_size

save_model_name = '../saved_models/htr_net_line_t.pt'
#load_model_name = None
load_model_name = '../saved_models/htr_net_line.pt'
