import random
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import pickle

from utils import Full_Batch_Experiment, Mini_Batch_Experiment, Step_Size_Experiment
from models import get_model_name
import copy


def train_pytorch_schedulers(train_loader, test_loader, num_layers, nodes, initial_lr, epochs, opt, size, model_num=False):
    print('PYTORCH')
    model_name = get_model_name(model_num)
    if model_name == "linear":
        root_dir = f"./results/{model_name}/Updated-pytorch/{opt.type}/{opt.dataset}/{epochs}/{num_layers}__{nodes}"
        save_dir = f"./pkl/{model_name}/Updated-pytorch/{opt.type}/{opt.dataset}/{epochs}/{num_layers}__{nodes}"
    else:
        root_dir = f"./results/{model_name}/Updated-pytorch/{opt.type}/{opt.dataset}/{epochs}"
        save_dir = f"./pkl/{model_name}/Updated-pytorch/{opt.type}/{opt.dataset}/{epochs}"
    
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    if opt.type == 'full_batch':
        print(opt.type)
        ours = Full_Batch_Experiment(opt)
        ours.train(train_loader, test_loader, num_layers, nodes, initial_lr, "ours", model_num=model_num, epochs=epochs)
        save_pkl_file(ours, "/ours.pkl", save_dir)

        ours_a = Full_Batch_Experiment(opt)
        ours_a.train(train_loader, test_loader, num_layers, nodes, initial_lr, "ours", model_num=model_num, epochs=epochs, our_lr=ours.lr*2)
        save_pkl_file(ours_a, "/ours_a.pkl", save_dir)

        ours_b = Full_Batch_Experiment(opt)
        ours_b.train(train_loader, test_loader, num_layers, nodes, initial_lr, "ours", model_num=model_num, epochs=epochs, our_lr=ours.lr/2)
        save_pkl_file(ours_b, "/ours_b.pkl", save_dir)

        step = Full_Batch_Experiment(opt)
        step.train(train_loader, test_loader, num_layers, nodes, initial_lr, "step", model_num=model_num, epochs=epochs)
        save_pkl_file(step, "/step.pkl", save_dir)

        exp = Full_Batch_Experiment(opt)
        exp.train(train_loader, test_loader, num_layers, nodes, initial_lr, "exponential", model_num=model_num, epochs=epochs)
        save_pkl_file(exp, "/exp.pkl", save_dir)

        linear = Full_Batch_Experiment(opt)
        linear.train(train_loader, test_loader, num_layers, nodes, initial_lr, "linear", model_num=model_num, epochs=epochs)
        save_pkl_file(linear, "/linear.pkl", save_dir)

    elif opt.type == 'mini_batch':
        print(opt.type)
        # ours = Mini_Batch_Experiment(opt)
        # ours.train(train_loader, test_loader, num_layers, nodes, initial_lr, "ours", model_num=model_num, epochs=epochs)
        # save_pkl_file(ours, "/ours.pkl", save_dir)

        # ours_a = Full_Batch_Experiment(opt)
        # ours_a.train(train_loader, test_loader, num_layers, nodes, initial_lr, "ours", model_num=model_num, epochs=epochs, our_lr=ours.lr*2)
        # save_pkl_file(ours_a, "/ours_a.pkl", save_dir)

        # ours_b = Full_Batch_Experiment(opt)
        # ours_b.train(train_loader, test_loader, num_layers, nodes, initial_lr, "ours", model_num=model_num, epochs=epochs, our_lr=ours.lr/2)
        # save_pkl_file(ours_b, "/ours_b.pkl", save_dir)

        # step = Mini_Batch_Experiment(opt)
        # step.train(train_loader, test_loader, num_layers, nodes, initial_lr, "step", model_num=model_num, epochs=epochs)
        # save_pkl_file(step, "/step.pkl", save_dir)

        # exp = Mini_Batch_Experiment(opt)
        # exp.train(train_loader, test_loader, num_layers, nodes, initial_lr, "exponential", model_num=model_num, epochs=epochs)
        # save_pkl_file(exp, "/exp.pkl", save_dir)

        # linear = Mini_Batch_Experiment(opt)
        # linear.train(train_loader, test_loader, num_layers, nodes, initial_lr, "linear", model_num=model_num, epochs=epochs)
        # save_pkl_file(linear, "/linear.pkl", save_dir)
        load_dir = "/mnt/SSD_2/Alok/IISc/sabharwal/autoencoders/Schedulers/distributed/pkl/vit/Updated-pytorch/mini_batch/imagenet100/100"
        # root = f'{load_dir}/exp.pkl'
        # with open(root, 'rb') as inp3:
        #     exp = pickle.load(inp3)

        root = f'{save_dir}/linear.pkl'
        with open(root, 'rb') as inp3:
            linear = pickle.load(inp3)

        root = f'{save_dir}/step.pkl'
        with open(root, 'rb') as inp3:
            step = pickle.load(inp3)

        load_dir = "/mnt/SSD_2/Alok/IISc/sabharwal/autoencoders/Schedulers/distributed/pkl/vit/step/mini_batch/imagenet100/100"
        root = f'{save_dir}/ours.pkl'
        with open(root, 'rb') as inp3:
            ours = pickle.load(inp3)

        root = f'{save_dir}/ours_a.pkl'
        with open(root, 'rb') as inp3:
            ours_a = pickle.load(inp3)

        root = f'{save_dir}/ours_b.pkl'
        with open(root, 'rb') as inp3:
            ours_b = pickle.load(inp3)

    else:
        raise ValueError('incorrect type provided')

    # ours_a = Mini_Batch_Experiment()
    # ours_a.train(train_loader, test_loader, num_layers, nodes, initial_lr, "ours", model_num=model_num, epochs=epochs, our_lr=ours.lr*2)
    # ours_b = Mini_Batch_Experiment()
    # ours_b.train(train_loader, test_loader, num_layers, nodes, initial_lr, "ours", model_num=model_num, epochs=epochs, our_lr=ours.lr/2)
    # obj_names = ["/ours_a.pkl", "/ours_b.pkl"]
    # objs = [ours_a, ours_b]
    # for i in range(len(obj_names)):
    #     save_obj_path = root_dir + obj_names[i]
    #     with open(save_obj_path, 'wb') as outp:
    #         pickle.dump(objs[i], outp, pickle.HIGHEST_PROTOCOL)
    # return

    # obj_names = ["/ours.pkl", "/ours_a.pkl", "/ours_b.pkl", "/step.pkl", "/exp.pkl", "/linear.pkl"]
    # objs = [ours, ours_a, ours_b, step, exp, linear]
    # for i in range(len(obj_names)):
    #     save_obj_path = save_dir + obj_names[i]
    #     with open(save_obj_path, 'wb') as outp:
    #         pickle.dump(objs[i], outp, pickle.HIGHEST_PROTOCOL)

    return linear, step, None, ours, ours_a, ours_b


def train_custom_schedulers(train_loader, test_loader, num_layers, nodes, initial_lr, epochs, opt, size, model_num=False):
    print("CUSTOM")
    model_name = get_model_name(model_num)
    if model_name == "linear":
        root_dir = f"./results/{model_name}/custom/{opt.type}/{opt.dataset}/{epochs}/{num_layers}__{nodes}"
        save_dir = f"./pkl/{model_name}/custom/{opt.type}/{opt.dataset}/{epochs}/{num_layers}__{nodes}"
    else:
        root_dir = f"./results/{model_name}/custom/{opt.type}/{opt.dataset}/{epochs}"
        save_dir = f"./pkl/{model_name}/custom/{opt.type}/{opt.dataset}/{epochs}"

    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    if opt.type == 'full_batch':
        print(opt.type)
        # ours = Full_Batch_Experiment(opt)
        # ours.train(train_loader, test_loader, num_layers, nodes, initial_lr, "ours", model_num, epochs=epochs)
        sqrt = Full_Batch_Experiment(opt)
        sqrt.train(train_loader, test_loader, num_layers, nodes, initial_lr, "sqrt", model_num, epochs=epochs)
        save_pkl_file(sqrt, "/sqrt.pkl", save_dir)

        inverse_time_decay = Full_Batch_Experiment(opt)
        inverse_time_decay.train(train_loader, test_loader, num_layers, nodes, initial_lr, "inverse_time_decay", model_num, epochs=epochs)
        save_pkl_file(inverse_time_decay, "/inverse_time_decay.pkl", save_dir)
        # cosine_decay = Full_Batch_Experiment()
        # cosine_decay.train(train_loader, test_loader, num_layers, nodes, initial_lr, "cosine_decay", model_num, epochs=epochs)
        exponential_decay = Full_Batch_Experiment(opt)
        exponential_decay.train(train_loader, test_loader, num_layers, nodes, initial_lr, "exponential_decay", model_num, epochs=epochs)
        save_pkl_file(exponential_decay, "/exponential_decay.pkl", save_dir)


    elif opt.type == 'mini_batch':
        print(opt.type)
        # # ours = Mini_Batch_Experiment(opt)
        # # ours.train(train_loader, test_loader, num_layers, nodes, initial_lr, "ours", model_num, epochs=epochs)
        # inverse_time_decay = Mini_Batch_Experiment(opt)
        # inverse_time_decay.train(train_loader, test_loader, num_layers, nodes, initial_lr, "inverse_time_decay", model_num, epochs=epochs)
        # save_pkl_file(inverse_time_decay, "/inverse_time_decay.pkl", save_dir)

        # exponential_decay = Mini_Batch_Experiment(opt)
        # exponential_decay.train(train_loader, test_loader, num_layers, nodes, initial_lr, "exponential_decay", model_num, epochs=epochs)
        # save_pkl_file(exponential_decay, "/exponential_decay.pkl", save_dir)
        # # cosine_decay = Mini_Batch_Experiment()
        # # cosine_decay.train(train_loader, test_loader, num_layers, nodes, initial_lr, "cosine_decay", model_num, epochs=epochs)
        # sqrt = Mini_Batch_Experiment(opt)
        # sqrt.train(train_loader, test_loader, num_layers, nodes, initial_lr, "sqrt", model_num, epochs=epochs)
        # save_pkl_file(sqrt, "/sqrt.pkl", save_dir)

        # load_dir = "/mnt/SSD_2/Alok/IISc/sabharwal/autoencoders/Schedulers/distributed/pkl/vit/custom/mini_batch/imagenet100/100"
        root = f'{save_dir}/exponential_decay.pkl'
        with open(root, 'rb') as inp3:
            exponential_decay = pickle.load(inp3)

        root = f'{save_dir}/inverse_time_decay.pkl'
        with open(root, 'rb') as inp3:
            inverse_time_decay = pickle.load(inp3)

        root = f'{save_dir}/sqrt.pkl'
        with open(root, 'rb') as inp3:
            sqrt = pickle.load(inp3)


    else:
        raise ValueError('incorrect type provided')

    # os.makedirs(root_dir, exist_ok=True)
    # print(f"saving to: {root_dir}")
    # obj_names = ["/inverse_time_decay.pkl", "/exponential_decay.pkl", "/sqrt.pkl"]
    # objs = [inverse_time_decay, exponential_decay, sqrt]
    # for i in range(len(obj_names)):
    #     save_obj_path = save_dir + obj_names[i]
    #     with open(save_obj_path, 'wb') as outp:
    #         pickle.dump(objs[i], outp, pickle.HIGHEST_PROTOCOL)

    return inverse_time_decay, exponential_decay, sqrt


def plot_combined(train_loader, test_loader, num_layers, nodes, initial_lr, epochs, opt, size, model_num=False):
    if opt.type == 'full_batch':
        print(opt.type)
    elif opt.type == 'mini_batch':
        print(opt.type)
    else:
        raise ValueError('incorrect type provided')
    
    # linear, step, _, ours_py, ours_a, ours_b = train_pytorch_schedulers(train_loader, test_loader, num_layers, nodes, initial_lr, epochs, opt, size, model_num)
    # inverse_time_decay, exponential_decay, sqrt = train_custom_schedulers(train_loader, test_loader, num_layers, nodes, initial_lr, epochs, opt, size, model_num)

    print('PYTORCH')
    model_name = get_model_name(model_num)
    if model_name == "linear":
        root_dir = f"./results/{model_name}/Updated-pytorch/{opt.type}/{opt.dataset}/{epochs}/{num_layers}__{nodes}"
        save_dir = f"./pkl/{model_name}/Updated-pytorch/{opt.type}/{opt.dataset}/{epochs}/{num_layers}__{nodes}"
    else:
        root_dir = f"./results/{model_name}/Updated-pytorch/{opt.type}/{opt.dataset}/{epochs}"
        save_dir = f"./pkl/{model_name}/Updated-pytorch/{opt.type}/{opt.dataset}/{epochs}"
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # ours = Mini_Batch_Experiment(opt)
    # ours.train(train_loader, test_loader, num_layers, nodes, initial_lr, "ours", model_num=model_num, epochs=epochs)
    # save_pkl_file(ours, "/ours.pkl", save_dir)
    # # ours = ours.cpu()
    root = f'{save_dir}/ours.pkl'
    with open(root, 'rb') as inp3:
        ours = pickle.load(inp3)

    root = f'{save_dir}/ours_a.pkl'
    with open(root, 'rb') as inp3:
        ours_a = pickle.load(inp3)

    root = f'{save_dir}/ours_b.pkl'
    with open(root, 'rb') as inp3:
        ours_b = pickle.load(inp3)

    root = f'{save_dir}/linear.pkl'
    with open(root, 'rb') as inp3:
        linear = pickle.load(inp3)

    root = f'{save_dir}/exp.pkl'
    with open(root, 'rb') as inp3:
        step = pickle.load(inp3) #---#

    # ours_a = Mini_Batch_Experiment(opt)
    # ours_a.train(train_loader, test_loader, num_layers, nodes, initial_lr, "ours", model_num=model_num, epochs=epochs, our_lr=ours.lr*2)
    # save_pkl_file(ours_a, "/ours_a.pkl", save_dir)
    # del ours_a
    # # ours_a = ours_a.cpu()

    # ours_b = Mini_Batch_Experiment(opt)
    # ours_b.train(train_loader, test_loader, num_layers, nodes, initial_lr, "ours", model_num=model_num, epochs=epochs, our_lr=ours.lr/2)
    # save_pkl_file(ours_b, "/ours_b.pkl", save_dir)
    # del ours_b
    # # ours_b = ours_b.cpu()

    # step = Full_Batch_Experiment(opt)
    # step.train(train_loader, test_loader, num_layers, nodes, initial_lr, "step", model_num=model_num, epochs=epochs)
    # save_pkl_file(step, "/step.pkl", save_dir)
    # del step
    # root = f'{save_dir}/step.pkl'
    # with open(root, 'rb') as inp3:
    #     step = pickle.load(inp3)
    # temp = Mini_Batch_Experiment(opt)
    # temp.grads = step.grads
    # temp.history = step.history
    # temp.lr = step.lr
    # del step
    # step = copy.deepcopy(temp)
    # del temp

    # exp = Mini_Batch_Experiment(opt)
    # exp.train(train_loader, test_loader, num_layers, nodes, initial_lr, "exponential", model_num=model_num, epochs=epochs)
    # save_pkl_file(exp, "/exp.pkl", save_dir)
    # del exp
    # # exp = exp.cpu()

    # linear = Mini_Batch_Experiment(opt)
    # linear.train(train_loader, test_loader, num_layers, nodes, initial_lr, "linear", model_num=model_num, epochs=epochs)
    # save_pkl_file(linear, "/linear.pkl", save_dir)
    # del linear
    # # linear = linear.cpu()
    

    print("CUSTOM")
    model_name = get_model_name(model_num)
    if model_name == "linear":
        root_dir = f"./results/{model_name}/custom/{opt.type}/{opt.dataset}/{epochs}/{num_layers}__{nodes}"
        save_dir = f"./pkl/{model_name}/custom/{opt.type}/{opt.dataset}/{epochs}/{num_layers}__{nodes}"
    else:
        root_dir = f"./results/{model_name}/custom/{opt.type}/{opt.dataset}/{epochs}"
        save_dir = f"./pkl/{model_name}/custom/{opt.type}/{opt.dataset}/{epochs}"

    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)


    inverse_time_decay = Mini_Batch_Experiment(opt)
    inverse_time_decay.train(train_loader, test_loader, num_layers, nodes, initial_lr, "inverse_time_decay", model_num, epochs=epochs)
    save_pkl_file(inverse_time_decay, "/inverse_time_decay.pkl", save_dir)
    # del inverse_time_decay
    # # inverse_time_decay = inverse_time_decay.cpu()

    exponential_decay = Mini_Batch_Experiment(opt)
    exponential_decay.train(train_loader, test_loader, num_layers, nodes, initial_lr, "exponential_decay", model_num, epochs=epochs)
    save_pkl_file(exponential_decay, "/exponential_decay.pkl", save_dir)
    # del exponential_decay
    # # exponential_decay = exponential_decay.cpu()

    sqrt = Mini_Batch_Experiment(opt)
    sqrt.train(train_loader, test_loader, num_layers, nodes, initial_lr, "sqrt", model_num, epochs=epochs)
    save_pkl_file(sqrt, "/sqrt.pkl", save_dir)
    # del sqrt

    cosine_decay = Mini_Batch_Experiment(opt)
    cosine_decay.train(train_loader, test_loader, num_layers, nodes, initial_lr, "cosine_decay", model_num, epochs=epochs)
    save_pkl_file(cosine_decay, "/cosine_decay.pkl", save_dir)
    # del cosine_decay


    model_name = get_model_name(model_num)
    if model_name == "linear":
        root_dir = f"./results/{model_name}/combined/{opt.type}/{opt.dataset}/{epochs}/{num_layers}__{nodes}"
        save_dir = f"./pkl/{model_name}/combined/{opt.type}/{opt.dataset}/{epochs}/{num_layers}__{nodes}"
    else:
        root_dir = f"./results/{model_name}/combined/{opt.type}/{opt.dataset}/{epochs}"
        save_dir = f"./pkl/{model_name}/combined/{opt.type}/{opt.dataset}/{epochs}"
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    print(f"saving to: {root_dir}")

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    rang = len(ours.grads)
    plt.yscale('log')

    sns.lineplot(x=np.arange(0, rang), y=np.array(linear.grads), color='pink', label=r'Linear')
    sns.lineplot(x=np.arange(0, rang), y=np.array(step.grads), color='orange', label=r'Step')
    #sns.lineplot(x=np.arange(0, rang), y=np.array(cyclic.grads), color='orange', label=r'One Cycle')

    sns.lineplot(x=np.arange(0, rang), y=np.array(sqrt.grads), color='magenta', label=r'Square Root')
    sns.lineplot(x=np.arange(0, rang), y=np.array(inverse_time_decay.grads), color='green', label=r'Inverse Time')
    sns.lineplot(x=np.arange(0, rang), y=np.array(cosine_decay.grads), color='brown', label=r'Cosine')
    sns.lineplot(x=np.arange(0, rang), y=np.array(exponential_decay.grads), color='cyan', label=r'Exponential')
    sns.lineplot(x=np.arange(0, rang), y=np.array(ours_a.grads), color='blue', label=f'Ours*2 ({round(ours.lr*2, 5)})')
    sns.lineplot(x=np.arange(0, rang), y=np.array(ours_b.grads), color='purple', label=f'Ours/2 ({round(ours.lr/2, 5)})')
    sns.lineplot(x=np.arange(0, rang), y=np.array(ours.grads), color='red', label=f'Ours ({round(ours.lr, 5)})')

    # plt.title('Gradient Norm vs Epochs')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Gradient Norm', fontsize=20)
    # plt.xticks(iter)
    if epochs !=20:
        plt.xticks(np.arange(0, epochs+1, epochs//5))
    else:
        plt.xticks(np.arange(0, epochs+1, epochs//4))
    plt.legend(fontsize=15)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    save_plot_path = root_dir + "/grad_norm_new.pdf"
    fig.tight_layout()
    plt.savefig(save_plot_path)
    # plt.show()

    for key in step.history:

        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(8, 6))
        rang = len(step.history[key])
        if key == 'loss' or key =="test_loss":
            plt.yscale('log')
            #plt.ylim(lower_lim, upper_lim)
        else:
            # plt.ylim(0, 1)
            plt.yscale('linear')

        sns.lineplot(x=np.arange(0, rang), y=np.array(linear.history[key]), color='pink', label=r'Linear')
        sns.lineplot(x=np.arange(0, rang), y=np.array(step.history[key]), color='orange', label=r'Step')
        #sns.lineplot(x=np.arange(0, rang), y=np.array(cyclic.history[key]), color='orange', label=r'One Cycle')

        sns.lineplot(x=np.arange(0, rang), y=np.array(sqrt.history[key]), color='magenta', label=r'Square Root')
        sns.lineplot(x=np.arange(0, rang), y=np.array(inverse_time_decay.history[key]), color='green', label=r'Inverse Time')
        sns.lineplot(x=np.arange(0, rang), y=np.array(cosine_decay.history[key]), color='brown', label=r'Cosine')
        sns.lineplot(x=np.arange(0, rang), y=np.array(exponential_decay.history[key]), color='cyan', label=r'Exponential')
        sns.lineplot(x=np.arange(0, rang), y=np.array(ours_a.history[key]), color='blue', label=f'Ours*2 ({round(ours.lr*2, 5)})')
        sns.lineplot(x=np.arange(0, rang), y=np.array(ours_b.history[key]), color='purple', label=f'Ours/2 ({round(ours.lr/2, 5)})')
        sns.lineplot(x=np.arange(0, rang), y=np.array(ours.history[key]), color='red', label=f'Ours ({round(ours.lr, 5)})')

        if key == 'loss':
            y = 'Loss'
            # plt.title(f'{y} vs Epochs')
        if key == 'train_acc':
            y = 'Train Accuracy'
            # plt.title(f'{y} vs Epochs')
        if key == 'val_acc':
            y = 'Test Accuracy'
            # plt.title(f'{y} vs Epochs')
        if key == 'test_loss':
            y = 'Test Loss'

        plt.xlabel('Epochs', fontsize=20)
        plt.ylabel(f'{y}', fontsize=20)
        # plt.xticks(iter)
        if epochs !=20:
            plt.xticks(np.arange(0, epochs+1, epochs//5))
        else:
            plt.xticks(np.arange(0, epochs+1, epochs//4))
        plt.legend(fontsize=15)
        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18)
        save_plot_path = root_dir + f"/{key}.pdf"
        fig.tight_layout()
        plt.savefig(save_plot_path)
        # plt.show()
        plt.cla()
    plt.close(fig)

    del sqrt, inverse_time_decay, exponential_decay


def plot_step(train_loader, test_loader, num_layers, nodes, initial_lr, epochs, opt, size, model_num=False):
    print("STEP SIZE exp")
    model_name = get_model_name(model_num)
    if model_name == "linear":
        root_dir = f"./results/{model_name}/step/{opt.type}/{opt.dataset}/{epochs}/{num_layers}__{nodes}"
        save_dir = f"./pkl/{model_name}/step/{opt.type}/{opt.dataset}/{epochs}/{num_layers}__{nodes}"
    else:
        root_dir = f"./results/{model_name}/step/{opt.type}/{opt.dataset}/{epochs}"
        save_dir = f"./pkl/{model_name}/step/{opt.type}/{opt.dataset}/{epochs}"
   
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # print("mini-batch-setting")

    ours = Step_Size_Experiment(True, opt)
    ours.train(train_loader, test_loader, num_layers, nodes, initial_lr, model_num, epochs=epochs)
    save_pkl_file(ours, "/ours.pkl", save_dir)

    ours_a = Step_Size_Experiment(True, opt)
    ours_a.train(train_loader, test_loader, num_layers, nodes, initial_lr, model_num, epochs=epochs, our_lr=ours.lr*2)
    save_pkl_file(ours_a, "/ours_a.pkl", save_dir)

    ours_b = Step_Size_Experiment(True, opt)
    ours_b.train(train_loader, test_loader, num_layers, nodes, initial_lr, model_num, epochs=epochs, our_lr=ours.lr/2)
    save_pkl_file(ours_b, "/ours_b.pkl", save_dir)

    # ours_a = Step_Size_Experiment(False, opt)
    # ours_a.train(train_loader, test_loader, num_layers, nodes, ours.lr*2, model_num, epochs=epochs)
    # ours_b = Step_Size_Experiment(False, opt)
    # ours_b.train(train_loader, test_loader, num_layers, nodes, ours.lr/2, model_num, epochs=epochs)

    # load_dir = "/mnt/SSD_2/Alok/IISc/sabharwal/autoencoders/Schedulers/distributed/pkl/resnet/step/mini_batch/cifar100/100"
    # if model_name == "linear":
    #     save_dir_ours = f"./pkl/{model_name}/Updated-pytorch/{opt.type}/{opt.dataset}/{epochs}/{num_layers}__{nodes}"
    # else:
    #     save_dir_ours = f"./pkl/{model_name}/Updated-pytorch/{opt.type}/{opt.dataset}/{epochs}"

    # load_dir = save_dir_ours
    # root = f'{load_dir}/ours.pkl'
    # with open(root, 'rb') as inp3:
    #     ours = pickle.load(inp3)
    # # ours.model = ours.model.cpu()
    # temp = Step_Size_Experiment(True, opt)
    # temp.grads = ours.grads
    # temp.history = ours.history
    # temp.lr = ours.lr
    # del ours
    # ours = copy.deepcopy(temp)
    
    # root = f'{load_dir}/ours_a.pkl'
    # with open(root, 'rb') as inp3:
    #     ours_a = pickle.load(inp3)
    # # ours_a.model = ours_a.model.cpu()
    # temp = Step_Size_Experiment(True, opt)
    # temp.grads = ours_a.grads
    # temp.history = ours_a.history
    # del ours_a
    # ours_a = copy.deepcopy(temp)
    
    # root = f'{load_dir}/ours_b.pkl'
    # with open(root, 'rb') as inp3:
    #     ours_b = pickle.load(inp3)
    # # ours_b.model = ours_b.model.cpu()
    # temp = Step_Size_Experiment(True, opt)
    # temp.grads = ours_b.grads
    # temp.history = ours_b.history
    # del ours_b
    # ours_b = copy.deepcopy(temp)


    # load_dir = save_dir
    # root = f'{load_dir}/a.pkl'
    # with open(root, 'rb') as inp3:
    #     a = pickle.load(inp3)
    # a.model = a.model.cpu()
    # temp = Step_Size_Experiment(True, opt)
    # temp.grads = a.grads
    # temp.history = a.history
    # del a
    # a = copy.deepcopy(temp)

    # root = f'{load_dir}/b.pkl'
    # with open(root, 'rb') as inp3:
    #     b = pickle.load(inp3)
    # b.model = b.model.cpu()
    # temp = Step_Size_Experiment(True, opt)
    # temp.grads = b.grads
    # temp.history = b.history
    # del b
    # b = copy.deepcopy(temp)
    
    # root = f'{load_dir}/c.pkl'
    # with open(root, 'rb') as inp3:
    #     c = pickle.load(inp3)
    # c.model = c.model.cpu()
    # temp = Step_Size_Experiment(True, opt)
    # temp.grads = c.grads
    # temp.history = c.history
    # del c
    # c = copy.deepcopy(temp)
    
    # root = f'{load_dir}/d.pkl'
    # with open(root, 'rb') as inp3:
    #     d = pickle.load(inp3)
    # d.model = d.model.cpu()
    # temp = Step_Size_Experiment(True, opt)
    # temp.grads = d.grads
    # temp.history = d.history
    # del d
    # d = copy.deepcopy(temp)
    
    # root = f'{load_dir}/e.pkl'
    # with open(root, 'rb') as inp3:
    #     e = pickle.load(inp3)
    # e.model = e.model.cpu()
    # temp = Step_Size_Experiment(True, opt)
    # temp.grads = e.grads
    # temp.history = e.history
    # del e
    # e = copy.deepcopy(temp)
    
    a = Step_Size_Experiment(False, opt)
    a.train(train_loader, test_loader, num_layers, nodes, 0.1, model_num, epochs=epochs)
    save_pkl_file(a, "/a.pkl", save_dir)
    del a

    b = Step_Size_Experiment(False, opt)
    b.train(train_loader, test_loader, num_layers, nodes, 0.01, model_num, epochs=epochs)
    save_pkl_file(b, "/b.pkl", save_dir)

    c = Step_Size_Experiment(False, opt)
    c.train(train_loader, test_loader, num_layers, nodes, 0.001, model_num, epochs=epochs)
    save_pkl_file(c, "/c.pkl", save_dir)
    del c

    d = Step_Size_Experiment(False, opt)
    d.train(train_loader, test_loader, num_layers, nodes, 0.0001, model_num, epochs=epochs)
    save_pkl_file(d, "/d.pkl", save_dir)
    del d

    e = Step_Size_Experiment(False, opt)
    e.train(train_loader, test_loader, num_layers, nodes, 0.00001, model_num, epochs=epochs)
    save_pkl_file(e, "/e.pkl", save_dir)
    del e
    # return 
    print(f"saved to: {root_dir}")
    extension = "pdf"
    # return

    # obj_names = ["/a.pkl", "/b.pkl", "/c.pkl", "/d.pkl", "/e.pkl"]# , "/ours.pkl", "/ours_a.pkl", "/ours_b.pkl"]
    # objs = [a, b, c, d, e]#, ours, ours_a, ours_b]
    # for idx in range(len(obj_names)):
    #     save_obj_path = save_dir + obj_names[idx]
    #     with open(save_obj_path, 'wb') as outp:
    #         pickle.dump(objs[idx], outp, pickle.HIGHEST_PROTOCOL)
    

    sns.set(style='whitegrid')
    fig, ax =plt.subplots(figsize=(8, 6))
    rang = len(b.grads)
    plt.yscale('log')
    # iter = np.linspace(0, epochs, 5)
    # sns.lineplot(x=np.arange(0, rang), y=f.grads, color="cyan", label=r'LR = 1')
    sns.lineplot(x=np.arange(0, rang), y=a.grads, color='blue', label=r'LR = 0.1') # r'$LR = \frac{0.01}{\sqrt{T}}$'
    sns.lineplot(x=np.arange(0, rang), y=b.grads, color='purple', label=r'LR = 0.01')
    sns.lineplot(x=np.arange(0, rang), y=c.grads, color='orange', label=r'LR = 0.001')
    sns.lineplot(x=np.arange(0, rang), y=d.grads, color='magenta', label=r'LR = 0.0001')
    sns.lineplot(x=np.arange(0, rang), y=e.grads, color='green', label=r'LR = 0.00001')
    # sns.lineplot(x=np.arange(0, rang), y=ours_a.grads, color='brown', label=f'Ours*2 ({round(ours.lr*2, 5)})')
    # sns.lineplot(x=np.arange(0, rang), y=ours_b.grads, color='cyan', label=f'Ours/2 ({round(ours.lr/2, 5)})')
    # sns.lineplot(x=np.arange(0, rang), y=ours.grads, color='red', label=f'Ours ({round(ours.lr, 5)})')

    # sns.lineplot(x=np.arange(0, rang), y=f.grads, color='brown', label=r'$LR = \frac{0.00001}{\sqrt{T}}$')
    sns.lineplot(x=np.arange(0, rang), y=ours_a.grads, color='brown', label=f'Ours*2 ({round((ours.lr*2), 5)})')
    sns.lineplot(x=np.arange(0, rang), y=ours_b.grads, color='cyan', label=f'Ours/2 ({round((ours.lr/2), 5)})')
    sns.lineplot(x=np.arange(0, rang), y=ours.grads, color='red', label=f'Ours ({round((ours.lr), 5)})')

    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Gradient Norm', fontsize=20)
    # plt.xticks(iter)
    if epochs !=20:
        plt.xticks(np.arange(0, epochs+1, epochs//5))
    else:
        plt.xticks(np.arange(0, epochs+1, epochs//4))
    plt.legend(fontsize=15)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    save_plot_path = root_dir + f"/grad_norm_new.{extension}"
    fig.tight_layout()
    plt.savefig(save_plot_path)

    # plt.show()

    for key in a.history:
        if key == "test_loss":
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.set(style='whitegrid')

        rang = len(a.history[key])
        if key == 'loss' or key =="test_loss":
            plt.yscale('log')
            #plt.ylim(lower_lim, upper_lim)
        else:
            # plt.ylim(0, 1)
            plt.yscale('linear')
            
        # sns.lineplot(x=np.arange(0, rang), y=f.history[key], color="cyan", label=r'LR = 1')
        sns.lineplot(x=np.arange(0, rang), y=a.history[key], color='blue', label=r'LR = 0.1')
        sns.lineplot(x=np.arange(0, rang), y=b.history[key], color='purple', label=r'LR = 0.01')
        sns.lineplot(x=np.arange(0, rang), y=c.history[key], color='orange', label=r'LR = 0.001')
        sns.lineplot(x=np.arange(0, rang), y=d.history[key], color='magenta', label=r'LR = 0.0001')
        sns.lineplot(x=np.arange(0, rang), y=e.history[key], color='green', label=r'LR = 0.00001')
        
        #  sns.lineplot(x=np.arange(0, rang), y=f.history[key], color='brown', label=r'LR = 0.1')
        # sns.lineplot(x=np.arange(0, rang), y=ours_a.history[key], color='brown', label=f'Ours*2 ({round((ours.lr*2), 3)})')
        # sns.lineplot(x=np.arange(0, rang), y=ours_b.history[key], color='cyan', label=f'Ours/2 ({round((ours.lr/2), 3)})')
        # sns.lineplot(x=np.arange(0, rang), y=ours.history[key], color='red', label=f'Ours ({round((ours.lr), 3)})')
        sns.lineplot(x=np.arange(0, rang), y=ours_a.history[key], color='brown', label=f'Ours*2 ({(round(ours.lr*2, 5))})')
        sns.lineplot(x=np.arange(0, rang), y=ours_b.history[key], color='cyan', label=f'Ours/2 ({(round(ours.lr/2, 5))})')
        sns.lineplot(x=np.arange(0, rang), y=ours.history[key], color='red', label=f'Ours ({(round(ours.lr, 5))})')

        if key == 'loss':
            y = 'Loss'
            # plt.title(f'{y} vs Epochs')
        if key == 'train_acc':
            y = 'Train Accuracy'
            # plt.title(f'{y} vs Epochs')
        if key == 'val_acc':
            y = 'Test Accuracy'
            # plt.title(f'{y} vs Epochs')
        if key == 'test_loss':
            y = 'Test Loss'
            
        plt.xlabel('Epochs', fontsize=20)
        plt.ylabel(f'{y}', fontsize=20)
        # plt.xticks(iter)
        if epochs !=20:
            plt.xticks(np.arange(0, epochs+1, epochs//5))
        else:
            plt.xticks(np.arange(0, epochs+1, epochs//4))
        plt.legend(fontsize=15)
        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18)
        save_plot_path = root_dir + f"/{key}.{extension}"
        fig.tight_layout()
        plt.savefig(save_plot_path)
        # plt.show()
        plt.cla()
    plt.close(fig)

    del a, b, c, d, e


def save_pkl_file(pkl_obj, obj_name, save_dir):
    save_obj_path = save_dir + obj_name
    with open(save_obj_path, 'wb') as outp:
        pickle.dump(pkl_obj, outp, pickle.HIGHEST_PROTOCOL)
