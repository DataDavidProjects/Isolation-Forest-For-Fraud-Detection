from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap


def plot_performance_time(s,time_props,legend_props,save=False):
    s.index = ['isolation forest','dummy model','random guess']
    my_cmap = sns.light_palette("Blue", as_cmap=True)
    fig, ax = plt.subplots(layout='constrained',figsize=(10, 4))
    ax.set_title(f"{time_props}")
    ax.set_xlabel("AUC")
    colorad = ["#457b9d", "#98c1d9", "#3d5a80"]
    ax.barh(y = s.index,width = s.values, color=colorad[::-1])
    ax.bar_label(ax.containers[0],fontsize=8, padding=3)
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend([legend_props])
    if save:
        fig.savefig('auc_sliding_time_window_2.png',transparent=True)
    return fig

def plot_roc_scenarios(data_scenarios_dict,save= False):
    fig, ax = plt.subplots(layout='constrained',figsize=(6, 4))
    for n,c in enumerate(["#457b9d", "#98c1d9", "#3d5a80"]):
        n+=1
        X_test =  data_scenarios_dict[f"scenario_{n}"][1]
        y_test =  data_scenarios_dict[f"scenario_{n}"][3]
        scores = -data_scenarios_dict[f"scenario_{n}"][-1].score_samples(X_test)
        fpr, tpr, thresholds = roc_curve(y_test,scores,pos_label=1)
        auc = roc_auc_score(y_test, scores)
        ax.plot(fpr, tpr, color =  c,linestyle = "-" ,lw=2,label = f"Fraud Scenario {n}:{round(auc,3)}")

    ax.plot(np.arange(100)/100, np.arange(100)/100, '#e63946',linestyle = "--", lw=2)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f"ROC for Fraud Scenario")
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend()
    if save:
        fig.savefig(f'roc_fraud_scenarios.png',transparent=True)
    return fig

def plot_cv_results(aucs,time,color_p,save=False):
    x = np.arange(len(time)) 
    width = 0.1  
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained',figsize=(7,5))

    for attribute, measurement in auc_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=color_p[attribute],edgecolor='grey')
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('AUC')
    ax.set_title('Area of Investigation')
    ax.set_xticks(x + width, time)
    ax.legend(loc='best', ncols=3)
    ax.set_ylim(0, 1.1)
    ax.spines[['right', 'top']].set_visible(False)
    if save:
        fig.savefig('auc_features.png',transparent=True)
    return fig