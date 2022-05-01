import os


def neq_load_customized(model, pretrained_dict):
    ''' load pre-trained model in a not-equal way,
    when new model has been partially modified '''
    model_dict = model.state_dict()
    tmp = {}
    print('\n=======Check Weights Loading======')
    print('Weights not used from pretrained file:')
    for k, v in pretrained_dict.items():
        if k in model_dict:
            tmp[k] = v
        else:
            print(k)
    print('---------------------------')
    print('Weights not loaded into new model:')
    for k, v in model_dict.items():
        if k not in pretrained_dict:
            print(k)
    print('===================================\n')
    del pretrained_dict
    model_dict.update(tmp)
    del tmp
    model.load_state_dict(model_dict)
    return model


def set_path(args):
    if args.resume:
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        exp_path = 'log_{args.prefix}/{args.dataset}-{args.img_dim}_{0}_{args.model}_bs{args.batch_size}_lr{1}_seq{args.num_seq}_pred{args.pred_step}_len{args.seq_len}_ds{args.ds}_train-{args.train_what}{2}'.format(
            'r%s' % args.net[6::],
            args.old_lr if args.old_lr is not None else args.lr,
            '_pt=%s' %
            args.pretrain.replace('/', '-') if args.pretrain else '',
            args=args)
    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    return img_path, model_path

def create_full_mask(pred1, pred2):
    '''Pred1 and Pred2 are 2 torch tensors (Bx540x540)
    which will be combined to create the full mask (Bx540x960)'''
    
    #Tensor on the left
    tenLeft = pred1[:,:,:420]
    
    #Tensor on the right
    tenRight = pred2[:,:,120:]

    #Tensor in the middle, take element-wise max of pred1, pred2
    tenMid = torch.maximum(pred1[:,:,420:], pred2[:,:,:120])

    full_mask = torch.cat((tenLeft,tenMid, tenRight),dim=2)

    return full_mask
