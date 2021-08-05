import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm 
from scipy.interpolate import splev, splrep
import warnings
warnings.filterwarnings("ignore")

def load (n_steps=500):

    mlp = []

    for i in range(n_steps):

        mlp.append(joblib.load('./../MLP_trained_models/model'+str(i+1)+'.sav'))

    return mlp

def predict_single_object (period, confidence_percentile=98, plot_all = False):

    models = load()
    preds, x_test, true_y, pmax = predict_over_all(models)

    q = []
    for model in (models):

        q.append(model.predict([[np.log10((period/pmax)+1)]]))

    q_mean = np.mean(q)
    q_up = np.percentile(q, confidence_percentile)
    q_down = np.percentile(q, 100 - confidence_percentile)

    print('Mid estimation of q for this object is ',q_mean)
    print('Upper estimation of q for this object is ',q_up)
    print('Lower estimation of q for this object is ',q_down)

    if plot_all:
        plot_all_models(preds, x_test, true_y)

def predict_over_all (models):

    filepath = './../data/final_dataset.csv'
    df = pd.read_csv(filepath)

    preds = []

    for mlp in models:

        mlp_pred = mlp.predict(df.np.values.reshape(-1,1))
        # x2,y2 = zip(*sorted(zip(df.np.values,mlp_pred),key=lambda x: x[0]))
        preds.append(mlp_pred)

    pmax = np.max(df.Period)

    return preds, df.np.values,df.nq.values, pmax



def plot_all_models (preds, x_test, true_y, confidence_percentile=98):


    bins = np.linspace(min(x_test), max(x_test), 300)

    mid = [(bins[j] + bins[j+1])/2 for j in range(len(bins)-1)]


    plt.figure(figsize=(20,15))

    bin_means = []
    up_con = confidence_percentile
    low_con = 100 - confidence_percentile

    for i in tqdm(range(len(preds))):

        digitized = np.digitize(np.asarray(x_test), bins)
        try:
            bin_means.append([np.asarray(preds[i])[digitized == j].mean() for j in range(1, len(bins))])
        except:
            pass
        x2,y2 = zip(*sorted(zip(x_test, preds[i]),key=lambda x: x[0]))

        if i == 0:
            plt.plot(x2,y2, color='grey', alpha=0.2, zorder=1, label = 'MLP fits with bootstrapping')
        else:
            plt.plot(x2,y2, color='grey', alpha=0.2, zorder=1)

            

    plt.scatter(x_test,true_y, marker='o', color='orange', zorder=4, s = 30, label = 'The dataset')
    # plt.scatter(x_test,y_test, marker='o', color='#a6cee3', zorder=4,s = 25, label = 'Test dataset')





    df_mids = pd.DataFrame({'mid': mid, 'means': (np.nanmean(bin_means, axis=0)),\
                            'ups':(np.nanpercentile(bin_means,up_con, axis=0)),\
                            'downs':(np.nanpercentile(bin_means,low_con, axis=0))})

    df_mids = df_mids.dropna().reset_index(drop=True)

    spl = splrep(df_mids.mid.values, df_mids.means,k=1, s=5e-3)
    y2 = splev(bins, spl)
    spl = splrep(df_mids.mid.values, df_mids.downs,k=1, s=1e-2)
    y3 = splev(bins, spl)
    spl = splrep(df_mids.mid.values, df_mids.ups,k=1, s=1e-2)
    y4 = splev(bins, spl)



    plt.plot(bins,y2, color='green', zorder=5, linewidth = 6, label = 'Mean MLP fit')
    plt.plot(bins,y3, color='green', zorder=5, linewidth = 3, alpha = 0.2)
    plt.plot(bins,y4, color='green', zorder=5, linewidth = 3, alpha = 0.2)

    plt.fill_between(bins, y3, y4, color= 'green', alpha = 0.2 , label = str(up_con)+'% Confidence interval')
    plt.xlabel(r'$log_{10}(\frac{p}{p_{max}}+1)$', size = 40)
    plt.ylabel(r'$q\star$', size = 40)
    # plt.xlim(min(x_test), max(x_test))
    # plt.legend()
    lgnd = plt.legend(bbox_to_anchor=(0., 1.01, 1., .102),numpoints=1, loc='lower left',
               ncol=2,fontsize=5, mode="expand", borderaxespad=0.,prop={'size': 30})

    plt.gca().tick_params(axis='both', which='major', labelsize=30)
    plt.savefig('./../Figures/plot_all_models.png')



            

if __name__ == '__main__':
    if len(sys.argv) > 1:
        period = np.float(sys.argv[1])
        if len(sys.argv)>2 :
            predict_single_object(period, plot_all=True)
        else:
            predict_single_object(period)

    else:
        print('Enter the period of the object for which you need mass ratio estimation.')
        sys.exit()








