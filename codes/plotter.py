# plotting script for plots in the paper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

def prepare_data(df, dataset):
    data = df[df.data == dataset]
    data['eval_file'] = data['file'].apply(lambda x : str(x).split('/')[-1])
    data['task'] = data['eval_file'].apply(lambda x : x[0])
    data['num_rel'] = data['eval_file'].apply(lambda x : str(x).split('_')[0])
    data['nr'] = data.num_rel.apply(lambda x : int(x.split('.')[-1]) if x != 'nan' else 0)
    data['model'] = data.experiment_name.apply(lambda x : x.split('_data_')[0])
    data['run'] = data.experiment_name.apply(lambda x : x.split('_')[-1])
    data['emb_policy'] = data.experiment_name.apply(lambda x : x.split('_')[-2])
    return data


def plot_runs_policy_models(datas, models, model_names, ep, policy='learned', mode='test', header='', save_fl_name='', loc='lower left'):
    """
    Plot Systematic Generalization performance between models
    :param datas:
    :param models:
    :param model_names:
    :param ep:
    :param policy:
    :param mode:
    :param header:
    :param save_fl_name:
    :param loc:
    :return:
    """
    if type(ep) != list:
        ep = [ep for d in datas]
    if type(policy) != list:
        policy = [policy for d in datas]
    colors = sns.color_palette('colorblind', n_colors=len(datas))
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(16)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,10))
    ax.xaxis.get_offset_text().set_fontsize(16)
    axis_font = {'fontname':'Arial', 'size':'18'}
    for di,data in enumerate(datas):
        model = models[di]
        model_name = model_names[di]
        color = colors[di]
        e = ep[di]
        max_run = int(data['run'].max())
        xd = data[(data.model == model) & (data.epoch == e) & (data['mode'] == 'test') & (data['emb_policy'] == policy[di])]
        #print(xd['run'].value_counts())
        xdg = xd.groupby(['nr'],as_index=False).mean()
        xdg_dev = xd.groupby(['nr']).std()
        x = xdg['nr'].tolist()
        y = xdg['accuracy'].tolist()
        y_std = xdg_dev['accuracy'].tolist()
        sm_data = pd.DataFrame({'x':x, 'y':y, 'y_std':y_std})
        plt.plot('x','y',data=sm_data, label=model_name, linewidth=1.5, color=color)
        ax.fill_between(sm_data.x,  sm_data.y + sm_data.y_std, sm_data.y - sm_data.y_std, alpha=0.05,
                            edgecolor=color, facecolor=color)
    ax.legend(loc=loc, prop={'size' : 18})
    ax.set_xlabel('Relation Length',**axis_font)
    ax.set_ylabel('Accuracy', **axis_font)
    ax.set_title(header, **axis_font)
    plt.savefig(save_fl_name, bbox_inches = 'tight', pad_inches = 0)
    plt.show()

def plot_gen(data, model, max_rel=10, header='',save_fl_name=''):
    """
    Plot generalization behavior on training
    :param data:
    :param model:
    :param max_rel:
    :param header:
    :param save_fl_name:
    :return:
    """
    colors = sns.color_palette('colorblind', n_colors=max_rel-1)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(16)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,100))
    ax.xaxis.get_offset_text().set_fontsize(16)
    axis_font = {'fontname':'Arial', 'size':'18'}
    xd = data[(data.model == model) & (data['mode'] == 'test')]
    for i in range(2, max_rel):
        color = colors[i-2]
        xdg = xd[xd.nr == i].groupby(['epoch'], as_index=False).mean()
        xdg_dev = xd[xd.nr == i].groupby(['epoch']).std()
        x = xdg['epoch'].tolist()
        #print(x)
        y = xdg['accuracy'].tolist()
        y_std = xdg_dev['accuracy'].tolist()
        sm_data = pd.DataFrame({'x':x, 'y':y, 'y_std':y_std})
        plt.plot('x','y',data=sm_data, label='k = {}'.format(i), linewidth=1.5, color=color)
        ax.fill_between(sm_data.x,  sm_data.y + sm_data.y_std, sm_data.y - sm_data.y_std, alpha=0.05,
                            edgecolor=color, facecolor=color)
    ax.legend(loc='upper left', prop={'size' : 18})
    ax.set_xlabel('Number of epochs',**axis_font)
    ax.set_ylabel('Accuracy', **axis_font)
    ax.set_title(header, **axis_font)
    plt.savefig(save_fl_name, bbox_inches = 'tight', pad_inches = 0)
    plt.show()

def plot_runs_robust(data, models, ep, policy='learned', mode='test', extra_id=''):
    """
    Plot bar charts on Robust Reasoning tasks
    :param data:
    :param models:
    :param ep:
    :param policy:
    :param mode:
    :param extra_id:
    :return:
    """
    plt_df = {'x':[],'y':[],'model':[]}
    for model in models:
        max_run = int(data['run'].max())
        xd = data[(data.model == model) & (data.epoch == ep) & (data['mode'] == 'test') & (data['emb_policy'] == policy) &(data['task'] != '7')]
        #print(xd['run'].value_counts())
        xdg = xd.groupby(['num_rel'], as_index=False).mean()
        #sns.barplot(x='num_rel',y='accuracy', data=xdg, label=model + '_' + policy + extra_id)
        plt_df['x'].extend(xdg['num_rel'].tolist())
        plt_df['y'].extend(xdg['accuracy'].tolist())
        plt_df['model'].extend([model for i in range(len(xdg['accuracy']))])
    plt_df = pd.DataFrame(data=plt_df)
    sns.barplot(x='x',y='y',hue='model',data=plt_df)

def print_table_robust(data, models, ep, policy='fixed', mode='test', extra_id=''):
    """
    return the dataframe containing statistics on Robust Reasoning experiments
    :param data:
    :param models:
    :param ep:
    :param policy:
    :param mode:
    :param extra_id:
    :return:
    """
    plt_df = {'x':[],'y':[],'y_std':[],'model':[],'runs':[]}
    for model in models:
        max_run = int(data['run'].max())
        xd = data[(data.model == model) & (data.epoch == ep) & (data['mode'] == 'test') & (data['emb_policy'] == policy) &(data['task'] != '7')]
        runs = len(list(xd['run'].unique()))
        xdg = xd.groupby(['num_rel'], as_index=False).mean()
        xdg_dev = xd.groupby(['num_rel']).std()
        plt_df['x'].extend(xdg['num_rel'].tolist())
        plt_df['y'].extend(xdg['accuracy'].tolist())
        plt_df['y_std'].extend(xdg_dev['accuracy'].tolist())
        plt_df['model'].extend([model for i in range(len(xdg['accuracy']))])
        plt_df['runs'].extend([runs for i in range(len(xdg['num_rel'].tolist()))])
    plt_df = pd.DataFrame(data=plt_df)
    print(plt_df)
    return plt_df

