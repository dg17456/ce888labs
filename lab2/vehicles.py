import matplotlib

matplotlib.use('Agg')

import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np


# def permutation(statistic, error):

#########MAD FUNCTION#############################################
def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
        http://stackoverflow.com/questions/8930370/where-can-i-find-mad-mean-absolute-deviation-in-scipy
    """
    arr = np.ma.array(arr).compressed()  # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))

#########BOOTSTRAP FUNCTION#############################################
def boostrap(statistic_func, iterations, data):
    samples = np.random.choice(data, replace=True, size=[iterations, len(data)])
    # print samples.shape
    data_mean = statistic_func(data)
    vals = []
    for sample in samples:
        sta = statistic_func(sample)
        # print sta
        vals.append(sta)
    b = np.array(vals)
    # print b
    lower, upper = np.percentile(b, [2.5, 97.5])
    return data_mean, lower, upper


if __name__ == "__main__":
    dfn = pd.read_csv('./vehicles_new.csv')
    dfc = pd.read_csv('./vehicles_current.csv')

    #########NEW FLEET#############################################
    print((dfn.columns))
    sns_plot_new = sns.lmplot(dfn.columns[0], dfn.columns[1], data=dfn, fit_reg=False)

    sns_plot_new.axes[0, 0].set_ylim(0, )
    sns_plot_new.axes[0, 0].set_xlim(0, )

    sns_plot_new.savefig("scaterplot_new.png", bbox_inches='tight')
    # sns_plot_new.savefig("scaterplot_new.pdf", bbox_inches='tight')

    data_new = dfn.values.T[1]

    print((("Mean_New: %f") % (np.mean(data_new))))
    print((("Median_New: %f") % (np.median(data_new))))
    print((("Var_New: %f") % (np.var(data_new))))
    print((("std_New: %f") % (np.std(data_new))))
    print((("MAD_New: %f") % (mad(data_new))))

    plt.clf()
    sns_plot2_new = sns.distplot(data_new, bins=20, kde=False, rug=True).get_figure()

    axes_new = plt.gca()
    axes_new.set_xlabel('Miles per gallon')
    axes_new.set_ylabel('Car number')

    sns_plot2_new.savefig("histogram_new.png", bbox_inches='tight')
    # sns_plot2_new.savefig("histogram_new.pdf", bbox_inches='tight')

    ###BOOSTRAPING NEW###

    boots_new = []
    boot_new_std = []
    for i in range(100, 100000, 1000):
        boot_new = boostrap(np.mean, i, data_new)
        boots_new.append([i, boot_new[0], "mean"])
        boots_new.append([i, boot_new[1], "lower"])
        boots_new.append([i, boot_new[2], "upper"])
        boot_new_std = boostrap(np.std,i,data_new)

    print((("standard deviation_New: ", boot_new_std)))

    dfn_boot = pd.DataFrame(boots_new, columns=['Boostrap Iterations', 'Mean', "Value"])
    bot_sns_plot_new = sns.lmplot(dfn_boot.columns[0], dfn_boot.columns[1], data=dfn_boot, 	fit_reg=False, hue="Value")

    bot_sns_plot_new.axes[0, 0].set_ylim(0, )
    bot_sns_plot_new.axes[0, 0].set_xlim(0, 100000)

    bot_sns_plot_new.savefig("bootstrap_confidence_new.png", bbox_inches='tight')
    #bot_sns_plot_new.savefig("bootstrap_confidence_new.pdf", bbox_inches='tight')

    #########CURRENT FLEET#############################################

    print((dfc.columns))
    sns_plot_current = sns.lmplot(dfc.columns[0], dfc.columns[1], data=dfc, fit_reg=False)

    sns_plot_current.axes[0, 0].set_ylim(0, )
    sns_plot_current.axes[0, 0].set_xlim(0, )

    sns_plot_current.savefig("scaterplot_current.png", bbox_inches='tight')
    # sns_plot_current.savefig("scaterplot_current.pdf", bbox_inches='tight')

    data_current = dfc.values.T[1]

    print((("Mean_Current: %f") % (np.mean(data_current))))
    print((("Median_Current: %f") % (np.median(data_current))))
    print((("Var_Current: %f") % (np.var(data_current))))
    print((("std_Current: %f") % (np.std(data_current))))
    print((("MAD_Current: %f") % (mad(data_current))))

    plt.clf()
    sns_plot2_current = sns.distplot(data_current, bins=20, kde=False, rug=True).get_figure()

    axes_current = plt.gca()
    axes_current.set_xlabel('Miles per gallon')
    axes_current.set_ylabel('car number')

    sns_plot2_current.savefig("histogram_current.png", bbox_inches='tight')
    # sns_plot2_current.savefig("histogram_currnet.pdf", bbox_inches='tight')

    ###BOOSTRAPING CURRENT###

    boots_current = []
    boot_current_std = []

    for j in range(100, 100000, 1000):
        boot_current = boostrap(np.mean, j, data_current)
        boots_current.append([j, boot_current[0], "mean"])
        boots_current.append([j, boot_current[1], "lower"])
        boots_current.append([j, boot_current[2], "upper"])
        boot_current_std = boostrap(np.std,j,data_current)


    print((("standard deviation_Currnet: ", boot_current_std)))
    dfc_boot = pd.DataFrame(boots_current, columns=['Boostrap Iterations', 'Mean', "Value"])
    bot_sns_plot_current = sns.lmplot(dfc_boot.columns[0], dfc_boot.columns[1], data=dfc_boot, fit_reg=False, hue="Value")

    bot_sns_plot_current.axes[0, 0].set_ylim(0, )
    bot_sns_plot_current.axes[0, 0].set_xlim(0, 100000)

    bot_sns_plot_current.savefig("bootstrap_confidence_current.png", bbox_inches='tight')
    #bot_sns_plot_current.savefig("bootstrap_confidence_current.pdf", bbox_inches='tight')
